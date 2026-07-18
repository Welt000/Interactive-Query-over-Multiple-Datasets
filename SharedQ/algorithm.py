from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from HD_PI import FastRegion, HDPIFast, fast_region_from_any
from SUR import filter_points_by_shared_range
from structure.common import (
    EPS,
    UtilityRange,
    as_points,
    normalize_utility,
    utility_range_from_any,
)


@dataclass(frozen=True)
class SharedQuestion:
    dataset_id: int
    left_index: int
    right_index: int
    preferred_index: int
    rejected_index: int
    score: float


@dataclass
class SharedQuestionsDatasetResult:
    dataset_id: int
    point_index: int
    candidate_count: int
    pruned_count: int
    candidate_indices: list[int] = field(default_factory=list)


@dataclass
class SharedQuestionsResult:
    dataset_results: list[SharedQuestionsDatasetResult] = field(default_factory=list)
    questions: int = 0
    elapsed: float = 0.0
    utility_range: UtilityRange | None = None
    transcript: list[SharedQuestion] = field(default_factory=list)
    candidate_history: list[int] = field(default_factory=list)
    detect_centers: list[list[float]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def point_indices(self) -> list[int]:
        return [item.point_index for item in self.dataset_results]


@dataclass
class _DatasetState:
    dataset_id: int
    points: np.ndarray
    candidates: list[int]
    resolved: bool = False
    local_center: np.ndarray | None = None
    local_radius: float = 0.0
    local_regions: dict[int, FastRegion] = field(default_factory=dict)
    local_boundary_partition_keys: set[tuple[int, int]] = field(default_factory=set)
    local_region_cache_version: tuple[int, tuple[int, ...]] | None = None
    adjacency: dict[int, tuple[int, ...]] = field(default_factory=dict)
    index_map: np.ndarray | None = None
    original_count: int | None = None
    incremental_region_updates: int = 0
    rebuilt_region_updates: int = 0


def _candidate_tuple(candidates) -> tuple[int, ...]:
    return tuple(int(candidate) for candidate in candidates)


@dataclass(frozen=True)
class _CandidateQuestion:
    dataset_id: int
    left: int
    right: int
    normal: np.ndarray
    diff: float
    score: float


class SharedQuestions:
    """Shared Questions for interactive top-1 over multiple datasets.

    The algorithm keeps one global utility range and one active candidate set
    per dataset.  Each asked comparison inserts a single global preference
    constraint and then prunes every dataset by the exact top-1 partition test.
    Question selection follows the paper's PCA/set-cover heuristic: for each
    dataset, find a locally ideal hyperplane, then choose the locally selected
    hyperplane with the largest weighted cross-dataset pruning score.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        max_questions: int = 10_000,
        max_pairs_per_dataset: int = 20_000,
        random_state: int | None = None,
        detect_workers: int = 4,
        verbose: bool = False,
        verbose_detect_points: bool = False,
        detect_point_timeout_seconds: float | None = None,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_questions = int(max_questions)
        self.max_pairs_per_dataset = int(max_pairs_per_dataset)
        self.rng = np.random.default_rng(random_state)
        self.detect_workers = max(1, int(detect_workers))
        self.verbose = bool(verbose)
        self.verbose_detect_points = bool(verbose_detect_points)
        self.detect_point_timeout_seconds = (
            None
            if detect_point_timeout_seconds is None
            else float(detect_point_timeout_seconds)
        )
        self._top1_builder = HDPIFast(candidate_mode="accurate")
        self._versioned_region_cache_lock = Lock()
        self._versioned_region_cache: dict[
            tuple[int, int, int, tuple[int, ...]],
            FastRegion,
        ] = {}

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _build_hull_adjacency(
        self,
        points: np.ndarray,
        candidates: list[int],
    ) -> dict[int, tuple[int, ...]]:
        if len(candidates) <= 1:
            return {}
        candidates = [int(idx) for idx in candidates]
        dim = points.shape[1]
        if len(candidates) <= dim:
            return {
                idx: tuple(other for other in candidates if other != idx)
                for idx in candidates
            }
        local_points = points[candidates]
        qhull_input = np.vstack([local_points, np.zeros((1, dim), dtype=float)])
        origin_local = len(local_points)
        try:
            hull = ConvexHull(qhull_input)
        except Exception:
            try:
                # High-dimensional real data frequently contains nearly
                # coplanar points. QJ perturbs only Qhull's working copy and
                # gives a conservative triangulated adjacency graph.
                hull = ConvexHull(qhull_input, qhull_options="QJ")
            except Exception:
                return {}
        adjacency_sets: dict[int, set[int]] = {idx: set() for idx in candidates}
        for simplex in hull.simplices:
            facet = [
                candidates[int(local_idx)]
                for local_idx in simplex
                if int(local_idx) != origin_local
            ]
            for pos, left in enumerate(facet):
                for right in facet[pos + 1 :]:
                    if left == right:
                        continue
                    adjacency_sets[left].add(right)
                    adjacency_sets[right].add(left)
        return {
            idx: tuple(sorted(neighbors))
            for idx, neighbors in adjacency_sets.items()
            if neighbors
        }

    def _convex_top1_candidates_and_adjacency(
        self,
        points: np.ndarray,
    ) -> tuple[list[int], dict[int, tuple[int, ...]]]:
        if len(points) <= points.shape[1] + 1:
            candidates = list(range(len(points)))
            return candidates, self._build_hull_adjacency(points, candidates)

        dim = points.shape[1]
        origin_index = len(points)
        qhull_input = np.vstack([points, np.zeros((1, dim), dtype=float)])
        try:
            hull = ConvexHull(qhull_input)
        except Exception:
            try:
                hull = ConvexHull(qhull_input, qhull_options="QJ")
            except Exception:
                candidates = list(range(len(points)))
                return candidates, {}

        candidates = sorted(int(i) for i in hull.vertices if int(i) < len(points))
        if not candidates:
            candidates = list(range(len(points)))
            return candidates, {}

        candidate_set = set(candidates)
        adjacency_sets: dict[int, set[int]] = {idx: set() for idx in candidates}
        for simplex in hull.simplices:
            facet = [
                int(local_idx)
                for local_idx in simplex
                if int(local_idx) != origin_index and int(local_idx) in candidate_set
            ]
            for pos, left in enumerate(facet):
                for right in facet[pos + 1 :]:
                    if left == right:
                        continue
                    adjacency_sets[left].add(right)
                    adjacency_sets[right].add(left)
        adjacency = {
            idx: tuple(sorted(neighbors))
            for idx, neighbors in adjacency_sets.items()
            if neighbors
        }
        return candidates, adjacency

    def search(
        self,
        datasets: Sequence[Sequence[Sequence[float]]],
        true_utility: Sequence[float],
        initial_range=None,
    ) -> SharedQuestionsResult:
        start = perf_counter()
        data_list = [as_points(points) for points in datasets]
        if not data_list:
            raise ValueError("datasets must not be empty")
        dim = data_list[0].shape[1]
        if any(data.shape[1] != dim for data in data_list):
            raise ValueError("all datasets must have the same dimension")
        if any(len(data) == 0 for data in data_list):
            raise ValueError("datasets must not contain empty point sets")

        true_u = normalize_utility(true_utility)
        if true_u.shape != (dim,):
            raise ValueError(f"utility dimension mismatch: expected {dim}")

        utility_range = utility_range_from_any(initial_range, dim)
        init_items = [
            self._convex_top1_candidates_and_adjacency(data)
            for data in data_list
        ]
        candidate_lists = [item[0] for item in init_items]
        adjacency_list = [item[1] for item in init_items]
        states = [
            _DatasetState(
                dataset_id=i,
                points=data,
                candidates=candidate_lists[i],
                adjacency=adjacency_list[i],
            )
            for i, data in enumerate(data_list)
        ]
        for state in states:
            if len(state.candidates) == 0:
                center = utility_range.center()
                state.candidates = [int(np.argmax(state.points @ center))]

        transcript: list[SharedQuestion] = []
        candidate_history = [self._active_candidate_count(states)]
        asked: set[tuple[int, int, int]] = set()

        while (
            any(len(state.candidates) > 1 for state in states)
            and len(transcript) < self.max_questions
        ):
            question = self._select_question(states, utility_range, asked)
            if question is None:
                break

            asked.add(self._question_key(question.dataset_id, question.left, question.right))
            state = states[question.dataset_id]
            left_score = float(np.dot(true_u, state.points[question.left]))
            right_score = float(np.dot(true_u, state.points[question.right]))
            if left_score >= right_score:
                preferred, rejected = question.left, question.right
            else:
                preferred, rejected = question.right, question.left

            utility_range.add_preference(state.points[preferred], state.points[rejected])
            if rejected in state.candidates:
                state.candidates = [idx for idx in state.candidates if idx != rejected]
            transcript.append(
                SharedQuestion(
                    dataset_id=question.dataset_id,
                    left_index=question.left,
                    right_index=question.right,
                    preferred_index=preferred,
                    rejected_index=rejected,
                    score=float(question.score),
                )
            )

            for target in states:
                if len(target.candidates) <= 1:
                    continue
                _, kept = filter_points_by_shared_range(
                    target.points,
                    utility_range,
                    candidates=target.candidates,
                )
                target.candidates = kept
                target.resolved = len(kept) <= 1
            candidate_history.append(self._active_candidate_count(states))

        center = utility_range.center()
        dataset_results: list[SharedQuestionsDatasetResult] = []
        for state in states:
            if len(state.candidates) == 1:
                answer = int(state.candidates[0])
            else:
                scores = state.points[state.candidates] @ center
                answer = int(state.candidates[int(np.argmax(scores))])
            dataset_results.append(
                SharedQuestionsDatasetResult(
                    dataset_id=state.dataset_id,
                    point_index=answer,
                    candidate_count=len(state.candidates),
                    pruned_count=len(state.points) - len(state.candidates),
                    candidate_indices=[int(idx) for idx in state.candidates],
                )
            )

        return SharedQuestionsResult(
            dataset_results=dataset_results,
            questions=len(transcript),
            elapsed=perf_counter() - start,
            utility_range=utility_range,
            transcript=transcript,
            candidate_history=candidate_history,
        )

    def _select_question(
        self,
        states: list[_DatasetState],
        utility_range: UtilityRange,
        asked: set[tuple[int, int, int]],
    ) -> _CandidateQuestion | None:
        middle_points = {
            state.dataset_id: self._partition_middle_points(state, utility_range)
            for state in states
            if len(state.candidates) > 1
        }
        region_cache = {
            state.dataset_id: self._state_region_cache(state, utility_range)
            for state in states
            if len(state.candidates) > 1
        }
        local_questions = [
            self._best_local_question(state, utility_range, middle_points[state.dataset_id], asked)
            for state in states
            if len(state.candidates) > 1
        ]
        local_questions = [question for question in local_questions if question is not None]
        if not local_questions:
            return self._fallback_question(states, utility_range, asked)

        total_candidates = sum(len(state.candidates) for state in states)
        best_question: _CandidateQuestion | None = None
        best_key: tuple[float, float] | None = None
        score_values = self._shared_pruning_scores(
            np.vstack([question.normal for question in local_questions]),
            states,
            region_cache,
            total_candidates,
            utility_range,
        )
        for question, score in zip(local_questions, score_values):
            candidate = _CandidateQuestion(
                dataset_id=question.dataset_id,
                left=question.left,
                right=question.right,
                normal=question.normal,
                diff=question.diff,
                score=float(score),
            )
            key = (score, -question.diff)
            if best_key is None or key > best_key:
                best_key = key
                best_question = candidate
        return best_question

    def _fallback_question(
        self,
        states: list[_DatasetState],
        utility_range: UtilityRange,
        asked: set[tuple[int, int, int]],
    ) -> _CandidateQuestion | None:
        center = utility_range.center()
        unresolved = sorted(
            (state for state in states if len(state.candidates) > 1),
            key=lambda item: len(item.candidates),
            reverse=True,
        )
        for state in unresolved:
            candidates = list(state.candidates)
            scores = state.points[candidates] @ center
            order = np.argsort(scores)
            for low_pos in range(len(order)):
                for high_pos in range(len(order) - 1, low_pos, -1):
                    left = candidates[int(order[low_pos])]
                    right = candidates[int(order[high_pos])]
                    if self._question_key(state.dataset_id, left, right) in asked:
                        continue
                    normal = state.points[left] - state.points[right]
                    if np.linalg.norm(normal) <= EPS:
                        continue
                    return _CandidateQuestion(
                        dataset_id=state.dataset_id,
                        left=int(left),
                        right=int(right),
                        normal=normal,
                        diff=0.0,
                        score=0.0,
                    )
        return None

    def _best_local_question(
        self,
        state: _DatasetState,
        utility_range: UtilityRange,
        middle_points: np.ndarray,
        asked: set[tuple[int, int, int]],
    ) -> _CandidateQuestion | None:
        if len(state.candidates) <= 1:
            return None
        ideal_normal, ideal_point = self._ideal_hyperplane(middle_points, utility_range)
        pairs = self._candidate_pairs(state, utility_range, ideal_normal, asked)
        if not pairs:
            return None

        pair_arr = np.asarray(pairs, dtype=int)
        left_points = state.points[pair_arr[:, 0]]
        right_points = state.points[pair_arr[:, 1]]
        normals = left_points - right_points
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > EPS
        if not np.any(valid):
            return None
        pair_arr = pair_arr[valid]
        normals = normals[valid]
        norms = norms[valid]

        cosine = np.abs(normals @ ideal_normal) / norms
        distance = np.abs(normals @ ideal_point) / norms
        diff = self.alpha * (1.0 - cosine) + self.beta * distance
        best_pos = int(np.argmin(diff))
        normal = normals[best_pos]
        return _CandidateQuestion(
            dataset_id=state.dataset_id,
            left=int(pair_arr[best_pos, 0]),
            right=int(pair_arr[best_pos, 1]),
            normal=normal,
            diff=float(diff[best_pos]),
            score=0.0,
        )

    def _partition_middle_points(
        self,
        state: _DatasetState,
        utility_range: UtilityRange,
    ) -> np.ndarray:
        center = utility_range.center()
        middle_points = np.empty((len(state.candidates), state.points.shape[1]), dtype=float)
        cache = self._state_region_cache(state, utility_range)
        for pos, candidate in enumerate(state.candidates):
            region = cache.get(int(candidate))
            if region is not None and region.midpoint is not None:
                middle_points[pos] = region.midpoint
            elif region is not None and region.vertices is not None and len(region.vertices) > 0:
                middle = np.mean(region.vertices, axis=0)
                total = float(np.sum(middle))
                region.midpoint = middle / total if total > EPS else None
                middle_points[pos] = region.midpoint if region.midpoint is not None else center
            else:
                middle_points[pos] = center
        return middle_points

    def _state_region_cache(
        self,
        state: _DatasetState,
        utility_range: UtilityRange,
    ) -> dict[int, FastRegion]:
        version = self._state_cache_version(state, utility_range)
        if state.local_region_cache_version == version and state.local_regions:
            cached = {
                int(candidate): state.local_regions[int(candidate)]
                for candidate in state.candidates
                if int(candidate) in state.local_regions
            }
            if len(cached) == len(state.candidates):
                return cached

        cache: dict[int, FastRegion] = {}
        for candidate in state.candidates:
            self._partition_region(
                state.points,
                int(candidate),
                state.candidates,
                utility_range,
                cache,
            )
        state.local_regions = cache
        state.local_region_cache_version = version
        return cache

    @staticmethod
    def _state_cache_version(
        state: _DatasetState,
        utility_range: UtilityRange,
    ) -> tuple[int, tuple[int, ...]]:
        return (
            len(utility_range.constraints),
            _candidate_tuple(state.candidates),
        )

    def _ideal_hyperplane(
        self,
        middle_points: np.ndarray,
        utility_range: UtilityRange,
    ) -> tuple[np.ndarray, np.ndarray]:
        dim = middle_points.shape[1]
        if len(middle_points) <= 1:
            return utility_range.center(), utility_range.center()

        centered = middle_points - np.mean(middle_points, axis=0)
        cov = centered.T @ centered
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, int(np.argmax(eigvals))]
        except np.linalg.LinAlgError:
            normal = utility_range.center()
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            normal = np.full(dim, 1.0 / np.sqrt(dim))
        else:
            normal = normal / norm

        projections = middle_points @ normal
        median_projection = float(np.median(projections))
        median_pos = int(np.argmin(np.abs(projections - median_projection)))
        return normal, middle_points[median_pos]

    def _candidate_pairs(
        self,
        state: _DatasetState,
        utility_range: UtilityRange,
        ideal_normal: np.ndarray,
        asked: set[tuple[int, int, int]],
    ) -> list[tuple[int, int]]:
        candidates = list(state.candidates)
        pair_count = len(candidates) * (len(candidates) - 1) // 2
        if pair_count <= self.max_pairs_per_dataset:
            return [
                (int(left), int(right))
                for pos, left in enumerate(candidates)
                for right in candidates[pos + 1 :]
                if self._question_key(state.dataset_id, left, right) not in asked
            ]

        center = utility_range.center()
        center_scores = state.points[candidates] @ center
        ideal_scores = state.points[candidates] @ ideal_normal
        order = np.lexsort((ideal_scores, center_scores))
        ordered = [candidates[int(i)] for i in order]

        pairs: set[tuple[int, int]] = set()
        for offset in (1, 2, 4, 8, 16, 32):
            for pos in range(0, max(0, len(ordered) - offset)):
                self._add_pair(pairs, state.dataset_id, ordered[pos], ordered[pos + offset], asked)
                if len(pairs) >= self.max_pairs_per_dataset:
                    return sorted(pairs)

        while len(pairs) < self.max_pairs_per_dataset:
            left, right = self.rng.choice(candidates, size=2, replace=False)
            self._add_pair(pairs, state.dataset_id, int(left), int(right), asked)
            if len(pairs) >= pair_count:
                break
        return sorted(pairs)

    def _shared_pruning_score(
        self,
        normal: np.ndarray,
        states: list[_DatasetState],
        region_cache: dict[int, dict[int, FastRegion]],
        total_candidates: int,
        utility_range: UtilityRange,
    ) -> float:
        if total_candidates <= 0:
            return 0.0
        total_score = 0.0
        for state in states:
            if len(state.candidates) <= 1:
                continue
            cache = region_cache.get(state.dataset_id)
            if cache is None:
                cache = self._state_region_cache(state, utility_range)
            positive = 0
            negative = 0
            for candidate in state.candidates:
                side = self._partition_side_against_hyperplane(
                    cache.get(int(candidate)),
                    normal,
                    utility_range.tol,
                )
                if side > 0:
                    positive += 1
                elif side < 0:
                    negative += 1
            weight = len(state.candidates) / total_candidates
            total_score += weight * min(positive, negative)
        return float(total_score)

    def _shared_pruning_scores(
        self,
        normals: np.ndarray,
        states: list[_DatasetState],
        region_cache: dict[int, dict[int, FastRegion]],
        total_candidates: int,
        utility_range: UtilityRange,
    ) -> np.ndarray:
        if total_candidates <= 0 or len(normals) == 0:
            return np.zeros(len(normals), dtype=float)
        normals = np.asarray(normals, dtype=float)
        total_scores = np.zeros(normals.shape[0], dtype=float)
        tol = utility_range.tol
        for state in states:
            if len(state.candidates) <= 1:
                continue
            cache = region_cache.get(state.dataset_id)
            if cache is None:
                cache = self._state_region_cache(state, utility_range)
            vertex_blocks: list[np.ndarray] = []
            lengths: list[int] = []
            for candidate in state.candidates:
                region = cache.get(int(candidate))
                if region is None or region.vertices is None or len(region.vertices) == 0:
                    continue
                vertices = np.asarray(region.vertices, dtype=float)
                vertex_blocks.append(vertices)
                lengths.append(len(vertices))
            if not vertex_blocks:
                continue
            all_vertices = np.vstack(vertex_blocks)
            values = all_vertices @ normals.T
            starts = np.cumsum([0, *lengths[:-1]])
            pos_by_vertex = values > tol
            neg_by_vertex = values < -tol
            pos_by_partition = np.logical_or.reduceat(pos_by_vertex, starts, axis=0)
            neg_by_partition = np.logical_or.reduceat(neg_by_vertex, starts, axis=0)
            positive = np.sum(pos_by_partition & ~neg_by_partition, axis=0)
            negative = np.sum(neg_by_partition & ~pos_by_partition, axis=0)
            weight = len(state.candidates) / total_candidates
            total_scores += weight * np.minimum(positive, negative)
        return total_scores

    def _partition_side_against_hyperplane(
        self,
        region: FastRegion | None,
        normal: np.ndarray,
        tol: float,
    ) -> int:
        if region is None or region.vertices is None or len(region.vertices) == 0:
            return 0
        values = region.vertices @ normal
        pos = bool(np.any(values > tol))
        neg = bool(np.any(values < -tol))
        if pos and not neg:
            return 1
        if neg and not pos:
            return -1
        return 0

    def _partition_region(
        self,
        points: np.ndarray,
        point_index: int,
        candidate_pool: list[int],
        utility_range: UtilityRange,
        cache: dict[int, FastRegion],
        adjacency: dict[int, tuple[int, ...]] | None = None,
    ) -> FastRegion | None:
        if point_index in cache:
            return self._valid_region_or_none(cache[point_index])
        version_key = self._region_version_key(points, point_index, candidate_pool, utility_range)
        if version_key is not None:
            cached = self._versioned_region_cache.get(version_key)
            if cached is not None:
                cache[point_index] = cached
                return self._valid_region_or_none(cached)

        dim = points.shape[1]
        region = fast_region_from_any(utility_range, dim, point_index=point_index)
        point = points[point_index]
        neighbor_candidates = (
            adjacency.get(int(point_index), ())
            if adjacency
            else ()
        )
        # For a convex-hull vertex, only adjacent vertices can define facets
        # of its top-1 normal cone. QJ triangulation may add redundant
        # neighbors, but using that superset remains exact and is far smaller
        # than constraining against every global candidate.
        constraint_pool = (
            neighbor_candidates
            if neighbor_candidates
            else candidate_pool
        )
        candidates = np.asarray(constraint_pool, dtype=int)
        candidates = candidates[candidates != int(point_index)]
        if len(candidates) > 0:
            region.add_leq_batch(points[candidates] - point, 0.0, owners=candidates)
        try:
            feasible = region.refine_vertices()
        except MemoryError as exc:
            raise MemoryError(
                "SharedQ partition LP ran out of memory: "
                f"dim={dim},point={point_index},"
                f"global_candidates={len(candidate_pool)},"
                f"partition_constraints={len(candidates)}"
            ) from exc
        if version_key is not None:
            self._versioned_region_cache[version_key] = region
        if not feasible:
            cache[point_index] = region
            return None
        cache[point_index] = region
        return region

    def _region_version_key(
        self,
        points: np.ndarray,
        point_index: int,
        candidate_pool: list[int],
        utility_range: UtilityRange,
    ) -> tuple[int, int, int, tuple[int, ...]] | None:
        return (
            id(points),
            int(point_index),
            len(utility_range.constraints),
            _candidate_tuple(candidate_pool),
        )

    @staticmethod
    def _valid_region_or_none(region: FastRegion) -> FastRegion | None:
        if region.vertices is None or len(region.vertices) == 0:
            return None
        return region

    def _partition_intersects_ball(
        self,
        region: FastRegion,
        center: np.ndarray,
        radius: float,
    ) -> bool:
        if region.vertices is None or len(region.vertices) == 0:
            return False
        if self._region_contains_point(region, center):
            return True
        distances = np.linalg.norm(region.vertices - center, axis=1)
        return bool(np.any(distances <= radius + 1e-8))

    def _region_contains_point(self, region: FastRegion, point: np.ndarray) -> bool:
        if abs(float(np.sum(point)) - 1.0) > 1e-6:
            return False
        normals, offsets = region.halfspace_matrix()
        if len(offsets) == 0:
            return True
        return bool(np.all(normals @ point + offsets <= 1e-7))

    def _partition_neighbors(
        self,
        points: np.ndarray,
        point_index: int,
        candidate_pool: list[int],
        region: FastRegion,
        candidate_set: set[int],
    ) -> list[int]:
        if region.vertices is None or len(region.vertices) == 0:
            return []
        if region.active_neighbors:
            return [
                int(idx)
                for idx in region.active_neighbors
                if int(idx) != int(point_index) and int(idx) in candidate_set
            ]
        point = points[point_index]
        neighbors: list[int] = []
        vertices = region.vertices
        for other in candidate_pool:
            other = int(other)
            if other == point_index or other not in candidate_set:
                continue
            values = vertices @ (points[other] - point)
            if np.any(np.abs(values) <= 1e-7):
                neighbors.append(other)
        return neighbors

    @staticmethod
    def _active_candidate_count(states: list[_DatasetState]) -> int:
        return int(sum(len(state.candidates) for state in states))

    @staticmethod
    def _question_key(dataset_id: int, left: int, right: int) -> tuple[int, int, int]:
        a, b = sorted((int(left), int(right)))
        return int(dataset_id), a, b

    @staticmethod
    def _partition_key(dataset_id: int, point_index: int) -> tuple[int, int]:
        return int(dataset_id), int(point_index)

    @staticmethod
    def _add_pair(
        pairs: set[tuple[int, int]],
        dataset_id: int,
        left: int,
        right: int,
        asked: set[tuple[int, int, int]],
    ) -> None:
        if left == right:
            return
        a, b = sorted((int(left), int(right)))
        if (dataset_id, a, b) not in asked:
            pairs.add((a, b))


class DetectDivideSharedQuestions(SharedQuestions):
    """Detect-and-Divide acceleration for Shared Questions.

    Detect selects a localized candidate set L_i around the current center of
    the global utility range. Divide then runs the shared-question selection
    only inside these localized sets. A remaining local winner is returned only
    after it is verified to dominate the full dataset throughout the current
    global utility range; otherwise another Detect round starts from the
    refined utility range.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        max_questions: int = 10_000,
        max_pairs_per_dataset: int = 20_000,
        random_state: int | None = None,
        radius: float = 0.20,
        radius_growth: float = 2.0,
        detect_round_limit: int = 100,
        divide_round_limit: int | None = None,
        random_initial_center: bool = False,
        boundary_center_strategy: str = "ray_midpoint",
        incremental_region_update: bool = True,
        detect_workers: int = 4,
        verbose: bool = False,
        verbose_detect_points: bool = False,
        detect_point_timeout_seconds: float | None = None,
        ball_intersection_mode: str = "vertex",
    ):
        if boundary_center_strategy not in {"range_center", "ray_midpoint"}:
            raise ValueError("boundary_center_strategy must be 'range_center' or 'ray_midpoint'")
        if ball_intersection_mode not in {"vertex", "exact_qp"}:
            raise ValueError("ball_intersection_mode must be 'vertex' or 'exact_qp'")
        super().__init__(
            alpha=alpha,
            beta=beta,
            max_questions=max_questions,
            max_pairs_per_dataset=max_pairs_per_dataset,
            random_state=random_state,
            detect_workers=detect_workers,
            verbose=verbose,
            verbose_detect_points=verbose_detect_points,
            detect_point_timeout_seconds=detect_point_timeout_seconds,
        )
        self.radius = float(radius)
        self.radius_growth = float(radius_growth)
        self.detect_round_limit = int(detect_round_limit)
        self.divide_round_limit = divide_round_limit
        self.random_initial_center = bool(random_initial_center)
        self.boundary_center_strategy = boundary_center_strategy
        self.incremental_region_update = bool(incremental_region_update)
        self.ball_intersection_mode = ball_intersection_mode

    def search(
        self,
        datasets: Sequence[Sequence[Sequence[float]]],
        true_utility: Sequence[float],
        initial_range=None,
    ) -> SharedQuestionsResult:
        start = perf_counter()
        data_list = [as_points(points) for points in datasets]
        if not data_list:
            raise ValueError("datasets must not be empty")
        dim = data_list[0].shape[1]
        if any(data.shape[1] != dim for data in data_list):
            raise ValueError("all datasets must have the same dimension")
        if any(len(data) == 0 for data in data_list):
            raise ValueError("datasets must not contain empty point sets")

        true_u = normalize_utility(true_utility)
        if true_u.shape != (dim,):
            raise ValueError(f"utility dimension mismatch: expected {dim}")

        utility_range = utility_range_from_any(initial_range, dim)
        init_items = [
            self._convex_top1_candidates_and_adjacency(data)
            for data in data_list
        ]
        candidate_lists = [item[0] for item in init_items]
        adjacency_list = [item[1] for item in init_items]
        global_states = [
            _DatasetState(
                dataset_id=i,
                points=data,
                candidates=candidate_lists[i],
                adjacency=adjacency_list[i],
            )
            for i, data in enumerate(data_list)
        ]
        for state in global_states:
            if len(state.candidates) == 0:
                center = utility_range.center()
                state.candidates = [int(np.argmax(state.points @ center))]

        transcript: list[SharedQuestion] = []
        candidate_history = [self._active_candidate_count(global_states)]
        detect_centers: list[list[float]] = []
        asked: set[tuple[int, int, int]] = set()
        answers: dict[int, int] = {}
        unresolved = {state.dataset_id for state in global_states}
        detect_round = 0
        next_detect_center: np.ndarray | None = None
        timing_totals = {
            "detect": 0.0,
            "divide": 0.0,
            "verify": 0.0,
        }

        while (
            unresolved
            and len(transcript) < self.max_questions
            and detect_round < self.detect_round_limit
        ):
            round_start = perf_counter()
            detect_round += 1
            if next_detect_center is not None:
                detect_center = next_detect_center
                next_detect_center = None
            elif detect_round == 1 and self.random_initial_center:
                detect_center = self.rng.dirichlet(np.ones(dim))
            else:
                detect_center = utility_range.center()
            detect_centers.append([float(x) for x in detect_center])
            self._log(
                "[SharedQ-detect-start]"
                f" round={detect_round},"
                f"center={np.array2string(detect_center, precision=6, separator=',')},"
                f"unresolved={sorted(unresolved)},"
                f"global_candidates={[len(state.candidates) for state in global_states]}"
            )
            detect_start = perf_counter()
            local_states = self._detect(
                global_states,
                unresolved,
                utility_range,
                detect_center,
            )
            detect_elapsed = perf_counter() - detect_start
            timing_totals["detect"] += detect_elapsed
            for local_state in local_states:
                self._log(
                    "[SharedQ-detect-result]"
                    f" round={detect_round},"
                    f"dataset={local_state.dataset_id + 1},"
                    f"local_candidates={len(local_state.candidates)},"
                    f"boundary_candidates={len(local_state.local_boundary_partition_keys)},"
                    f"radius={local_state.local_radius:.6f}"
                )
            self._log(
                "[SharedQ-step-time]"
                f" round={detect_round},step=detect,elapsed={detect_elapsed:.6f}"
            )
            before_questions = len(transcript)
            divide_start = perf_counter()
            boundary_center = self._divide(
                local_states,
                global_states,
                utility_range,
                true_u,
                asked,
                transcript,
                candidate_history,
            )
            divide_elapsed = perf_counter() - divide_start
            timing_totals["divide"] += divide_elapsed
            self._log(
                "[SharedQ-step-time]"
                f" round={detect_round},step=divide,elapsed={divide_elapsed:.6f},"
                f"questions_added={len(transcript) - before_questions}"
            )
            if boundary_center is not None:
                next_detect_center = boundary_center
                self._log(
                    "[SharedQ-next-center]"
                    f" round={detect_round},strategy={self.boundary_center_strategy},"
                    f"center={np.array2string(boundary_center, precision=6, separator=',')}"
                )
            verify_start = perf_counter()
            resolved_this_round = 0
            for local_state in local_states:
                if local_state.dataset_id not in unresolved:
                    continue
                if len(local_state.candidates) == 0:
                    continue
                scores = local_state.points[local_state.candidates] @ detect_center
                candidate = int(local_state.candidates[int(np.argmax(scores))])
                candidate_key = self._partition_key(local_state.dataset_id, candidate)
                touches_boundary = candidate_key in local_state.local_boundary_partition_keys
                if touches_boundary:
                    region_cache = self._state_region_cache(local_state, utility_range)
                    touches_boundary = self._partition_touches_local_boundary(
                        local_state.points,
                        candidate,
                        utility_range,
                        local_state.local_center if local_state.local_center is not None else detect_center,
                        local_state.local_radius,
                        region_cache.get(candidate),
                    )
                if not touches_boundary and self._is_guaranteed_global_top1(
                    local_state.points,
                    candidate,
                    utility_range,
                ):
                    answers[local_state.dataset_id] = candidate
                    unresolved.remove(local_state.dataset_id)
                    resolved_this_round += 1
                    global_states[local_state.dataset_id].candidates = [candidate]
                    self._log(
                        "[SharedQ-dataset-resolved]"
                        f" round={detect_round},dataset={local_state.dataset_id + 1},"
                        f"answer={candidate},remaining=1"
                    )
            verify_elapsed = perf_counter() - verify_start
            timing_totals["verify"] += verify_elapsed
            self._log(
                "[SharedQ-step-time]"
                f" round={detect_round},step=verify,elapsed={verify_elapsed:.6f}"
            )
            stop_after_round = False
            if (
                unresolved
                and len(transcript) == before_questions
                and all(
                    len(state.candidates) <= 1
                    for state in local_states
                    if state.dataset_id in unresolved
                )
            ):
                old_radius = self.radius
                expanded = self._expand_radius(global_states, unresolved)
                if expanded:
                    self._log(
                        "[SharedQ-radius-expand]"
                        f" round={detect_round},old_radius={old_radius:.6f},"
                        f"new_radius={self.radius:.6f}"
                    )
                else:
                    self._log(
                        "[SharedQ-radius-max]"
                        f" round={detect_round},radius={self.radius:.6f},"
                        f"unresolved={sorted(unresolved)}"
                    )
                    if resolved_this_round == 0:
                        stop_after_round = True

            candidate_history.append(
                sum(
                    1 if state.dataset_id not in unresolved else len(state.candidates)
                    for state in global_states
                )
            )
            self._log(
                "[SharedQ-round-done]"
                f" round={detect_round},elapsed={perf_counter() - round_start:.6f},"
                f"total_questions={len(transcript)},"
                f"global_candidates={[len(state.candidates) for state in global_states]},"
                f"unresolved={sorted(unresolved)}"
            )
            if stop_after_round:
                break

        center = utility_range.center()
        dataset_results: list[SharedQuestionsDatasetResult] = []
        for state in global_states:
            if state.dataset_id in answers:
                answer = int(answers[state.dataset_id])
                final_candidates = [answer]
            elif len(state.candidates) == 1:
                answer = int(state.candidates[0])
                final_candidates = [answer]
            else:
                _, kept = filter_points_by_shared_range(
                    state.points,
                    utility_range,
                    candidates=state.candidates,
                )
                final_candidates = kept
                scores = state.points[kept] @ center
                answer = int(kept[int(np.argmax(scores))])
            if state.index_map is not None:
                output_answer = int(state.index_map[answer])
                output_candidates = [int(state.index_map[int(idx)]) for idx in final_candidates]
            else:
                output_answer = int(answer)
                output_candidates = [int(idx) for idx in final_candidates]
            original_count = int(state.original_count) if state.original_count is not None else len(state.points)
            dataset_results.append(
                SharedQuestionsDatasetResult(
                    dataset_id=state.dataset_id,
                    point_index=output_answer,
                    candidate_count=len(final_candidates),
                    pruned_count=original_count - len(final_candidates),
                    candidate_indices=output_candidates,
                )
            )

        return SharedQuestionsResult(
            dataset_results=dataset_results,
            questions=len(transcript),
            elapsed=perf_counter() - start,
            utility_range=utility_range,
            transcript=transcript,
            candidate_history=candidate_history,
            detect_centers=detect_centers,
            timings=timing_totals,
        )

    def _detect(
        self,
        global_states: list[_DatasetState],
        unresolved: set[int],
        utility_range: UtilityRange,
        center: np.ndarray,
    ) -> list[_DatasetState]:
        pending_states = [
            state
            for state in global_states
            if state.dataset_id in unresolved
        ]
        if self.detect_workers <= 1 or len(pending_states) <= 1:
            return [
                self._detect_state(state, utility_range, center)
                for state in pending_states
            ]
        workers = min(self.detect_workers, len(pending_states))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda state: self._detect_state(state, utility_range, center),
                    pending_states,
                )
            )

    def _detect_state(
        self,
        state: _DatasetState,
        utility_range: UtilityRange,
        center: np.ndarray,
    ) -> _DatasetState:
        candidate_pool = state.candidates
        radius = min(self.radius, self._max_detect_radius())
        local_candidates, local_regions, local_boundary_candidates = self._detect_with_radius(
            state.dataset_id,
            state.points,
            candidate_pool,
            utility_range,
            center,
            radius,
            state.adjacency,
        )
        max_radius = self._max_detect_radius()
        while (
            len(local_candidates) <= 1
            and len(candidate_pool) > 1
            and radius < max_radius - EPS
        ):
            radius = min(radius * self.radius_growth, max_radius)
            local_candidates, local_regions, local_boundary_candidates = self._detect_with_radius(
                state.dataset_id,
                state.points,
                candidate_pool,
                utility_range,
                center,
                radius,
                state.adjacency,
            )
        local_state = _DatasetState(
            dataset_id=state.dataset_id,
            points=state.points,
            candidates=local_candidates,
            local_center=center.copy(),
            local_radius=float(radius),
            local_regions=local_regions,
            local_boundary_partition_keys=local_boundary_candidates,
            adjacency=state.adjacency,
            index_map=state.index_map,
            original_count=state.original_count,
        )
        local_state.local_region_cache_version = self._state_cache_version(local_state, utility_range)
        return local_state

    def _detect_with_radius(
        self,
        dataset_id: int,
        points: np.ndarray,
        candidate_pool: list[int],
        utility_range: UtilityRange,
        center: np.ndarray,
        radius: float,
        adjacency: dict[int, tuple[int, ...]] | None = None,
    ) -> tuple[list[int], dict[int, FastRegion], set[tuple[int, int]]]:
        if len(candidate_pool) <= 1:
            return (
                list(candidate_pool),
                {},
                {self._partition_key(dataset_id, candidate) for candidate in candidate_pool},
            )

        scores = points[candidate_pool] @ center
        anchor = int(candidate_pool[int(np.argmax(scores))])
        candidate_set = set(int(idx) for idx in candidate_pool)
        region_cache: dict[int, FastRegion] = {}
        boundary_candidates: set[tuple[int, int]] = set()
        local_candidates: set[int] = set()
        visited: set[int] = set()
        queue = deque([anchor])
        detect_start = perf_counter()

        while queue:
            point_start = perf_counter()
            elapsed = point_start - detect_start
            if (
                self.detect_point_timeout_seconds is not None
                and elapsed > self.detect_point_timeout_seconds
            ):
                raise TimeoutError(
                    "SharedQ detect exceeded timeout: "
                    f"dataset={dataset_id + 1},radius={radius},"
                    f"elapsed={elapsed:.6f},visited={len(visited)},"
                    f"queue={len(queue)},local_candidates={len(local_candidates)}"
                )
            point_index = queue.popleft()
            if point_index in visited:
                continue
            visited.add(point_index)
            region = self._partition_region(
                points,
                point_index,
                candidate_pool,
                utility_range,
                region_cache,
                adjacency,
            )
            if region is None:
                if self.verbose_detect_points:
                    self._log(
                        "[SharedQ-detect-point]"
                        f" dataset={dataset_id + 1},"
                        f"radius={radius:.6f},"
                        f"point={point_index},"
                        f"visited={len(visited)},"
                        f"queue={len(queue)},"
                        f"local_candidates={len(local_candidates)},"
                        f"boundary_candidates={len(boundary_candidates)},"
                        f"point_elapsed={perf_counter() - point_start:.6f},"
                        f"detect_elapsed={perf_counter() - detect_start:.6f},"
                        f"status=infeasible_region"
                    )
                elapsed = perf_counter() - detect_start
                if (
                    self.detect_point_timeout_seconds is not None
                    and elapsed > self.detect_point_timeout_seconds
                ):
                    raise TimeoutError(
                        "SharedQ detect exceeded timeout: "
                        f"dataset={dataset_id + 1},radius={radius},"
                        f"elapsed={elapsed:.6f},visited={len(visited)},"
                        f"queue={len(queue)},local_candidates={len(local_candidates)}"
                    )
                continue
            if point_index != anchor and not self._partition_intersects_ball(
                region,
                center,
                radius,
            ):
                if self.verbose_detect_points:
                    self._log(
                        "[SharedQ-detect-point]"
                        f" dataset={dataset_id + 1},"
                        f"radius={radius:.6f},"
                        f"point={point_index},"
                        f"visited={len(visited)},"
                        f"queue={len(queue)},"
                        f"local_candidates={len(local_candidates)},"
                        f"boundary_candidates={len(boundary_candidates)},"
                        f"point_elapsed={perf_counter() - point_start:.6f},"
                        f"detect_elapsed={perf_counter() - detect_start:.6f},"
                        f"status=outside_ball"
                    )
                elapsed = perf_counter() - detect_start
                if (
                    self.detect_point_timeout_seconds is not None
                    and elapsed > self.detect_point_timeout_seconds
                ):
                    raise TimeoutError(
                        "SharedQ detect exceeded timeout: "
                        f"dataset={dataset_id + 1},radius={radius},"
                        f"elapsed={elapsed:.6f},visited={len(visited)},"
                        f"queue={len(queue)},local_candidates={len(local_candidates)}"
                    )
                continue
            local_candidates.add(point_index)
            if self._partition_touches_local_boundary(
                points,
                point_index,
                utility_range,
                center,
                radius,
                region,
            ):
                boundary_candidates.add(self._partition_key(dataset_id, point_index))
            for neighbor in self._partition_neighbors(
                points,
                point_index,
                candidate_pool,
                region,
                candidate_set,
                adjacency,
            ):
                if neighbor not in visited:
                    queue.append(neighbor)
            if self.verbose_detect_points:
                self._log(
                    "[SharedQ-detect-point]"
                    f" dataset={dataset_id + 1},"
                    f"radius={radius:.6f},"
                    f"point={point_index},"
                    f"visited={len(visited)},"
                    f"queue={len(queue)},"
                    f"local_candidates={len(local_candidates)},"
                    f"boundary_candidates={len(boundary_candidates)},"
                    f"point_elapsed={perf_counter() - point_start:.6f},"
                    f"detect_elapsed={perf_counter() - detect_start:.6f},"
                    f"status=accepted"
                )
            elapsed = perf_counter() - detect_start
            if (
                self.detect_point_timeout_seconds is not None
                and elapsed > self.detect_point_timeout_seconds
            ):
                raise TimeoutError(
                    "SharedQ detect exceeded timeout: "
                    f"dataset={dataset_id + 1},radius={radius},"
                    f"elapsed={elapsed:.6f},visited={len(visited)},"
                    f"queue={len(queue)},local_candidates={len(local_candidates)}"
                )

        return sorted(local_candidates), region_cache, boundary_candidates

    def _partition_region(
        self,
        points: np.ndarray,
        point_index: int,
        candidate_pool: list[int],
        utility_range: UtilityRange,
        cache: dict[int, FastRegion],
        adjacency: dict[int, tuple[int, ...]] | None = None,
    ) -> FastRegion | None:
        if point_index in cache:
            return self._valid_region_or_none(cache[point_index])
        version_key = self._region_version_key(points, point_index, candidate_pool, utility_range)
        if version_key is not None:
            with self._versioned_region_cache_lock:
                cached = self._versioned_region_cache.get(version_key)
            if cached is not None:
                cache[point_index] = cached
                return self._valid_region_or_none(cached)

        dim = points.shape[1]
        region = fast_region_from_any(utility_range, dim, point_index=point_index)
        point = points[point_index]
        candidates = np.asarray(candidate_pool, dtype=int)
        candidates = candidates[candidates != int(point_index)]
        if len(candidates) > 0:
            region.add_leq_batch(points[candidates] - point, 0.0, owners=candidates)
        feasible = region.refine_vertices()
        if version_key is not None:
            with self._versioned_region_cache_lock:
                self._versioned_region_cache[version_key] = region
        if not feasible:
            cache[point_index] = region
            return None
        cache[point_index] = region
        return region

    def _partition_intersects_ball(
        self,
        region: FastRegion,
        center: np.ndarray,
        radius: float,
    ) -> bool:
        if self.ball_intersection_mode == "exact_qp":
            return self._partition_intersects_ball_exact_qp(region, center, radius)
        return self._partition_intersects_ball_vertex(region, center, radius)

    def _partition_intersects_ball_vertex(
        self,
        region: FastRegion,
        center: np.ndarray,
        radius: float,
    ) -> bool:
        if region.vertices is None or len(region.vertices) == 0:
            return False
        if self._region_contains_point(region, center):
            return True
        limit = (radius + 1e-8) ** 2
        distances_sq = np.sum((region.vertices - center) ** 2, axis=1)
        return bool(np.any(distances_sq <= limit))

    def _partition_intersects_ball_exact_qp(
        self,
        region: FastRegion,
        center: np.ndarray,
        radius: float,
    ) -> bool:
        if self._partition_intersects_ball_vertex(region, center, radius):
            return True
        if region.vertices is None or len(region.vertices) == 0:
            return False

        normals, offsets = region.halfspace_matrix()
        dim = region.dim
        center = np.asarray(center, dtype=float)
        limit = (radius + 1e-8) ** 2

        if region.midpoint is not None:
            x0 = np.asarray(region.midpoint, dtype=float)
        else:
            x0 = np.mean(region.vertices, axis=0)
            total = float(np.sum(x0))
            x0 = x0 / total if total > EPS else np.full(dim, 1.0 / dim)

        def objective(x: np.ndarray) -> float:
            diff = x - center
            return float(np.dot(diff, diff))

        def jacobian(x: np.ndarray) -> np.ndarray:
            return 2.0 * (x - center)

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: float(np.sum(x) - 1.0),
                "jac": lambda x: np.ones(dim, dtype=float),
            }
        ]
        if len(offsets) > 0:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, a=normals, b=offsets: -(a @ x + b),
                    "jac": lambda x, a=normals: -a,
                }
            )

        result = minimize(
            objective,
            x0,
            jac=jacobian,
            bounds=[(0.0, 1.0)] * dim,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 100, "disp": False},
        )
        if result.success and np.isfinite(result.fun):
            return bool(float(result.fun) <= limit)

        distances_sq = np.sum((region.vertices - center) ** 2, axis=1)
        return bool(np.any(distances_sq <= limit))

    def _region_contains_point(self, region: FastRegion, point: np.ndarray) -> bool:
        if abs(float(np.sum(point)) - 1.0) > 1e-6:
            return False
        normals, offsets = region.halfspace_matrix()
        if len(offsets) == 0:
            return True
        return bool(np.all(normals @ point + offsets <= 1e-7))

    def _partition_neighbors(
        self,
        points: np.ndarray,
        point_index: int,
        candidate_pool: list[int],
        region: FastRegion,
        candidate_set: set[int],
        adjacency: dict[int, tuple[int, ...]] | None = None,
    ) -> list[int]:
        if region.vertices is None or len(region.vertices) == 0:
            return []
        if region.active_neighbors:
            active = [
                int(idx)
                for idx in region.active_neighbors
                if int(idx) != int(point_index) and int(idx) in candidate_set
            ]
            if adjacency:
                graph_set = set(int(idx) for idx in adjacency.get(int(point_index), ()))
                active = [idx for idx in active if idx in graph_set]
            if active:
                return active
        point = points[point_index]
        vertices = region.vertices
        neighbor_pool = candidate_pool
        if adjacency:
            graph_neighbors = [
                int(idx)
                for idx in adjacency.get(int(point_index), ())
                if int(idx) in candidate_set
            ]
            if graph_neighbors:
                neighbor_pool = graph_neighbors
        candidates = np.asarray(neighbor_pool, dtype=int)
        if len(candidates) == 0:
            return []
        if len(candidate_set) != len(candidate_pool):
            candidates = np.asarray([idx for idx in candidates if int(idx) in candidate_set], dtype=int)
            if len(candidates) == 0:
                return []
        mask = candidates != int(point_index)
        if not np.any(mask):
            return []
        candidates = candidates[mask]
        if len(candidates) < 256:
            neighbors: list[int] = []
            for other in candidates:
                other = int(other)
                values = vertices @ (points[other] - point)
                if np.any(np.abs(values) <= 1e-7):
                    neighbors.append(other)
            return neighbors
        values = (points[candidates] - point) @ vertices.T
        neighbor_mask = np.any(np.abs(values) <= 1e-7, axis=1)
        return [int(idx) for idx in candidates[neighbor_mask]]

    def _divide(
        self,
        local_states: list[_DatasetState],
        global_states: list[_DatasetState],
        utility_range: UtilityRange,
        true_u: np.ndarray,
        asked: set[tuple[int, int, int]],
        transcript: list[SharedQuestion],
        candidate_history: list[int],
    ) -> np.ndarray | None:
        rounds = 0
        divide_normals: list[np.ndarray] = []
        state_by_dataset = {state.dataset_id: state for state in local_states}
        while (
            any(len(state.candidates) > 1 for state in local_states)
            and len(transcript) < self.max_questions
        ):
            if self.divide_round_limit is not None and rounds >= self.divide_round_limit:
                return None
            rounds += 1
            select_start = perf_counter()
            question = self._select_question(local_states, utility_range, asked)
            select_elapsed = perf_counter() - select_start
            if question is None:
                return None

            asked.add(self._question_key(question.dataset_id, question.left, question.right))
            state = state_by_dataset[question.dataset_id]
            left_score = float(np.dot(true_u, state.points[question.left]))
            right_score = float(np.dot(true_u, state.points[question.right]))
            if left_score >= right_score:
                preferred, rejected = question.left, question.right
            else:
                preferred, rejected = question.right, question.left

            self._log(
                "[SharedQ-question]"
                f" divide_round={rounds},"
                f"dataset={question.dataset_id + 1},"
                f"left={question.left},right={question.right},"
                f"left_score={left_score:.12g},right_score={right_score:.12g},"
                f"preferred={preferred},rejected={rejected},"
                f"score={question.score:.6f},"
                f"select_elapsed={select_elapsed:.6f}"
            )

            update_start = perf_counter()
            preference_normal = state.points[preferred] - state.points[rejected]
            divide_normals.append(preference_normal)
            utility_range.add_constraint(preference_normal)
            old_candidates_by_dataset = {
                target.dataset_id: _candidate_tuple(target.candidates)
                for target in local_states
            }
            global_state = global_states[question.dataset_id]
            if rejected in global_state.candidates:
                global_state.candidates = [
                    idx for idx in global_state.candidates if idx != rejected
                ]
            global_update_elapsed = perf_counter() - update_start
            transcript.append(
                SharedQuestion(
                    dataset_id=question.dataset_id,
                    left_index=question.left,
                    right_index=question.right,
                    preferred_index=preferred,
                    rejected_index=rejected,
                    score=float(question.score),
                )
            )

            prune_total_start = perf_counter()
            for target in local_states:
                prune_start = perf_counter()
                before_count = len(target.candidates)
                if rejected in target.candidates and target.dataset_id == question.dataset_id:
                    target.candidates = [idx for idx in target.candidates if idx != rejected]
                if len(target.candidates) <= 1:
                    self._log(
                        "[SharedQ-prune]"
                        f" divide_round={rounds},dataset={target.dataset_id + 1},"
                        f"before={before_count},removed={before_count - len(target.candidates)},"
                        f"remaining={len(target.candidates)},elapsed={perf_counter() - prune_start:.6f},"
                        f"mode=direct"
                    )
                    continue
                mode = "incremental_prune"
                target.candidates = self._filter_local_candidates(target, utility_range)
                old_candidates = old_candidates_by_dataset.get(target.dataset_id, ())
                mode = "filter"
                if _candidate_tuple(target.candidates) == old_candidates:
                    if self._incremental_update_local_regions(
                        target,
                        utility_range,
                        preference_normal,
                    ):
                        target.incremental_region_updates += 1
                        mode = "incremental_region"
                    else:
                        target.rebuilt_region_updates += 1
                        target.local_region_cache_version = None
                        target.local_regions = {}
                        target.local_boundary_partition_keys = set()
                        self._state_region_cache(target, utility_range)
                        mode = "rebuild_region"
                else:
                    target.local_region_cache_version = None
                    target.local_regions = {}
                    target.local_boundary_partition_keys = set()
                    target.rebuilt_region_updates += 1
                self._log(
                    "[SharedQ-prune]"
                    f" divide_round={rounds},dataset={target.dataset_id + 1},"
                    f"before={before_count},removed={before_count - len(target.candidates)},"
                    f"remaining={len(target.candidates)},elapsed={perf_counter() - prune_start:.6f},"
                    f"mode={mode}"
                )

            candidate_history.append(self._active_candidate_count(local_states))
            self._log(
                "[SharedQ-step-time]"
                f" divide_round={rounds},step=question_update,"
                f"elapsed={perf_counter() - update_start:.6f},"
                f"global_update_elapsed={global_update_elapsed:.6f},"
                f"local_prune_elapsed={perf_counter() - prune_total_start:.6f},"
                f"local_remaining={[len(target.candidates) for target in local_states]}"
            )
            boundary_start = perf_counter()
            if self._all_local_candidates_touch_boundary(local_states, utility_range):
                self._log(
                    "[SharedQ-boundary-check]"
                    f" divide_round={rounds},all_local_boundary=true,"
                    f"elapsed={perf_counter() - boundary_start:.6f}"
                )
                if self.boundary_center_strategy == "ray_midpoint":
                    return self._ray_midpoint_detect_center(
                        local_states,
                        utility_range,
                        divide_normals,
                    )
                return None
            self._log(
                "[SharedQ-boundary-check]"
                f" divide_round={rounds},all_local_boundary=false,"
                f"elapsed={perf_counter() - boundary_start:.6f}"
            )
        return None

    def _filter_local_candidates(
        self,
        state: _DatasetState,
        utility_range: UtilityRange,
    ) -> list[int]:
        local_points = state.points[state.candidates]
        _, kept_local = filter_points_by_shared_range(local_points, utility_range)
        kept = [state.candidates[int(pos)] for pos in kept_local]
        return kept if kept else list(state.candidates)

    def _incremental_update_local_regions(
        self,
        state: _DatasetState,
        utility_range: UtilityRange,
        added_constraint: np.ndarray,
    ) -> bool:
        if not self.incremental_region_update:
            return False
        if not state.local_regions:
            return False
        previous_version = (
            len(utility_range.constraints) - 1,
            _candidate_tuple(state.candidates),
        )
        if state.local_region_cache_version != previous_version:
            return False

        current_version = self._state_cache_version(state, utility_range)
        updated: dict[int, FastRegion] = {}
        boundary_keys: set[tuple[int, int]] = set()
        center = state.local_center if state.local_center is not None else utility_range.center()
        leq_normal = -np.asarray(added_constraint, dtype=float)
        current_candidate_count = max(1, len(state.candidates))

        for candidate in state.candidates:
            candidate = int(candidate)
            region = state.local_regions.get(candidate)
            if region is None:
                return False
            if region.owner_count > max(64, current_candidate_count * 4):
                return False
            next_region = region.copy()
            next_region.add_leq(leq_normal, 0.0)
            next_region.refine_vertices()
            updated[candidate] = next_region
            if self._partition_touches_local_boundary(
                state.points,
                candidate,
                utility_range,
                center,
                state.local_radius,
                next_region,
            ):
                boundary_keys.add(self._partition_key(state.dataset_id, candidate))

        state.local_regions = updated
        state.local_boundary_partition_keys = boundary_keys
        state.local_region_cache_version = current_version
        return True

    def _is_guaranteed_global_top1(
        self,
        points: np.ndarray,
        point_index: int,
        utility_range: UtilityRange,
    ) -> bool:
        vertices = utility_range.vertices()
        if len(vertices) == 0:
            return False
        point_scores = points[point_index] @ vertices.T
        max_scores = np.max(points @ vertices.T, axis=0)
        return bool(np.all(point_scores >= max_scores - utility_range.tol))

    def _all_local_candidates_touch_boundary(
        self,
        local_states: list[_DatasetState],
        utility_range: UtilityRange,
    ) -> bool:
        saw_candidate = False
        for state in local_states:
            if len(state.candidates) == 0:
                continue
            saw_candidate = True
            remaining_keys = {
                self._partition_key(state.dataset_id, candidate)
                for candidate in state.candidates
            }
            if not remaining_keys.issubset(state.local_boundary_partition_keys):
                return False
            center = (
                state.local_center
                if state.local_center is not None
                else utility_range.center()
            )
            cache = self._state_region_cache(state, utility_range)
            for candidate in state.candidates:
                region = self._partition_region(
                    state.points,
                    int(candidate),
                    state.candidates,
                    utility_range,
                    cache,
                )
                if not self._partition_touches_local_boundary(
                    state.points,
                    int(candidate),
                    utility_range,
                    center,
                    state.local_radius,
                    region,
                ):
                    return False
        return saw_candidate

    def _remaining_partition_vertex_mean(
        self,
        local_states: list[_DatasetState],
        utility_range: UtilityRange,
    ) -> np.ndarray:
        vertices: list[np.ndarray] = []
        for state in local_states:
            if len(state.candidates) == 0:
                continue
            cache = self._state_region_cache(state, utility_range)
            for candidate in state.candidates:
                region = self._partition_region(
                    state.points,
                    int(candidate),
                    state.candidates,
                    utility_range,
                    cache,
                )
                if region is None or region.vertices is None or len(region.vertices) == 0:
                    continue
                if region.midpoint is not None:
                    vertices.append(region.midpoint[None, :])
                else:
                    vertices.append(region.vertices)
        if not vertices:
            return utility_range.center()
        point = np.mean(np.vstack(vertices), axis=0)
        total = float(np.sum(point))
        if total <= EPS:
            return utility_range.center()
        point = np.clip(point / total, 0.0, 1.0)
        if utility_range.contains(point):
            return point
        return utility_range.center()

    def _ray_midpoint_detect_center(
        self,
        local_states: list[_DatasetState],
        utility_range: UtilityRange,
        divide_normals: list[np.ndarray],
    ) -> np.ndarray | None:
        if not divide_normals:
            return None
        start = self._remaining_partition_vertex_mean(local_states, utility_range)
        direction = np.sum(np.vstack(divide_normals), axis=0)
        direction = direction - np.mean(direction)
        norm = float(np.linalg.norm(direction))
        if norm <= EPS:
            return None
        direction = direction / norm

        inequalities = [np.eye(utility_range.dim)[i] for i in range(utility_range.dim)]
        inequalities.extend(np.asarray(c, dtype=float) for c in utility_range.constraints)
        t_upper = float("inf")
        for normal in inequalities:
            value = float(np.dot(normal, start))
            slope = float(np.dot(normal, direction))
            if slope < -1e-12:
                t_upper = min(t_upper, -value / slope)
        if not np.isfinite(t_upper) or t_upper <= 1e-10:
            return None
        endpoint = start + t_upper * direction
        midpoint = (start + endpoint) / 2.0
        midpoint = np.clip(midpoint, 0.0, 1.0)
        total = float(np.sum(midpoint))
        if total <= EPS:
            return None
        midpoint = midpoint / total
        if not utility_range.contains(midpoint):
            return None
        return midpoint

    def _partition_touches_local_boundary(
        self,
        points: np.ndarray,
        point_index: int,
        utility_range: UtilityRange,
        center: np.ndarray,
        radius: float,
        region: FastRegion | None = None,
    ) -> bool:
        if radius <= 0:
            return True
        if region is None:
            return True
        if region.vertices is None or len(region.vertices) == 0:
            return True
        limit = max(radius - 1e-7, 0.0) ** 2
        distances_sq = np.sum((region.vertices - center) ** 2, axis=1)
        return bool(np.any(distances_sq >= limit))

    def _max_detect_radius(self) -> float:
        return 2.0

    def _expand_radius(
        self,
        global_states: list[_DatasetState],
        unresolved: set[int],
    ) -> bool:
        if not any(state.dataset_id in unresolved for state in global_states):
            return False
        max_radius = self._max_detect_radius()
        new_radius = min(self.radius * self.radius_growth, max_radius)
        changed = new_radius > self.radius + EPS
        self.radius = new_radius
        return changed


def run_detect_divide(
    datasets: Sequence[Sequence[Sequence[float]]],
    true_utility: Sequence[float],
    initial_range=None,
    **kwargs,
) -> SharedQuestionsResult:
    return DetectDivideSharedQuestions(**kwargs).search(
        datasets,
        true_utility,
        initial_range=initial_range,
    )


__all__ = [
    "DetectDivideSharedQuestions",
    "SharedQuestion",
    "SharedQuestionsDatasetResult",
    "SharedQuestionsResult",
    "run_detect_divide",
]
