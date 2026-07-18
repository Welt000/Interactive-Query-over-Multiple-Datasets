from __future__ import annotations

from time import perf_counter
from typing import Callable, Optional, Sequence

import numpy as np

from HD_PI import FastRegion, fast_region_from_any
from structure.common import (
    Question,
    SearchResult,
    SimulatedOracle,
    as_points,
    finish_result,
)


EPS = 1e-8
CPP_PRECISION = 0.0000015


class RHFast:
    """RH / Random_half port following the C++ implementation.

    This version uses the same high-level data flow as
    `InteractiveTopk/RH/alg_one.cpp`: randomize points, incrementally compare a
    new pivot against already used points, choose the intersecting hyperplane
    closest to the current R center, update R, and stop via the
    find_possible_topk/check_possible_topk logic.
    """

    def __init__(
        self,
        max_questions: int = 10_000,
        random_state: int = 0,
        use_skyline: bool = False,
        use_skyband: bool = True,
        skyband_k: int = 1,
        acceleration: str = "sphere",
        strict_original_k1: bool = False,
    ):
        if skyband_k < 1:
            raise ValueError("skyband_k must be positive")
        if acceleration not in {"sphere", "rectangle", "none"}:
            raise ValueError("acceleration must be 'sphere', 'rectangle', or 'none'")
        self.max_questions = max_questions
        self.random_state = random_state
        self.use_skyline = use_skyline
        self.use_skyband = use_skyband
        self.skyband_k = skyband_k
        self.acceleration = acceleration
        self.strict_original_k1 = strict_original_k1

    def _dominates_same(self, p: np.ndarray, q: np.ndarray) -> bool:
        """C++ dominates_same: p >= q in every dimension and p != q."""
        return bool(np.all(p >= q - EPS) and np.any(np.abs(p - q) > EPS))

    def _dominates_same_batch(self, candidates: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Vectorized dominates_same(candidates[i], point)."""
        if len(candidates) == 0:
            return np.zeros(0, dtype=bool)
        ge_all = np.all(candidates >= point - EPS, axis=1)
        not_same = np.any(np.abs(candidates - point) > EPS, axis=1)
        return ge_all & not_same

    def _skyband_indices(self, points: np.ndarray, k: int) -> list[int]:
        """Port of InteractiveTopk::skyband.

        It keeps points that have not been dominated by at least k previously
        retained points, and deletes retained points once the current point
        raises their domination count to k.
        """
        dominated = {idx: 0 for idx in range(len(points))}
        returned: list[int] = []
        for idx, point in enumerate(points):
            if returned:
                kept_points = points[returned]
                dominated[idx] = int(np.sum(self._dominates_same_batch(kept_points, point)))
            else:
                dominated[idx] = 0

            if dominated[idx] < k:
                if returned:
                    kept_points = points[returned]
                    current_dominates = self._dominates_same_batch(
                        np.repeat(point[None, :], len(returned), axis=0),
                        kept_points,
                    )
                    next_returned: list[int] = []
                    for kept, is_dominated in zip(returned, current_dominates):
                        if is_dominated:
                            dominated[kept] = dominated.get(kept, 0) + 1
                        if dominated[kept] < k:
                            next_returned.append(kept)
                    returned = next_returned
                returned.append(idx)
        return returned

    def _nondominated_indices(self, points: np.ndarray) -> list[int]:
        return list(range(len(points)))

    def _region_center(self, region: FastRegion) -> np.ndarray:
        if region.vertices is None or len(region.vertices) == 0:
            region.refine_vertices()
        if region.vertices is not None and len(region.vertices) > 0:
            return np.mean(region.vertices, axis=0)
        return np.full(region.dim, 1.0 / region.dim)

    def _distance_to_center(self, normal: np.ndarray, center: np.ndarray) -> float:
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            return float("inf")
        return abs(float(np.dot(normal, center))) / norm

    def _region_vertices(self, region: FastRegion) -> np.ndarray:
        if region.vertices is None or len(region.vertices) == 0:
            if not region.refine_vertices():
                return np.empty((0, region.dim))
        return region.vertices if region.vertices is not None else np.empty((0, region.dim))

    def _check_bounding_sphere(self, region: FastRegion, normal: np.ndarray) -> int:
        vertices = self._region_vertices(region)
        if len(vertices) == 0:
            return -2
        center = np.mean(vertices, axis=0)
        radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
        norm = float(np.linalg.norm(normal))
        if norm <= EPS:
            return -2

        value = float(np.dot(normal, center))
        distance = abs(value) / norm
        if distance >= radius:
            return 1 if value >= 0 else -1
        return -2

    def _check_bounding_rectangle(self, region: FastRegion, normal: np.ndarray) -> int:
        vertices = self._region_vertices(region)
        if len(vertices) == 0:
            return -2
        low = np.min(vertices, axis=0)
        high = np.max(vertices, axis=0)
        max_value = float(np.sum(np.where(normal >= 0, normal * high, normal * low)))
        min_value = float(np.sum(np.where(normal >= 0, normal * low, normal * high)))
        if min_value > 0:
            return 1
        if max_value < 0:
            return -1
        return 0

    def _check_situation(self, region: FastRegion, normal: np.ndarray) -> int:
        """Port of check_situation_accelerate.

        First use the selected bounding primitive to prove a one-sided answer.
        If it is inconclusive, scan the exact extreme points with the C++
        Precision / 2 threshold.
        """
        vertices = self._region_vertices(region)
        if len(vertices) < 1:
            return -2

        if self.acceleration == "sphere":
            situation = self._check_bounding_sphere(region, normal)
            if situation in {1, -1}:
                return situation
        elif self.acceleration == "rectangle":
            situation = self._check_bounding_rectangle(region, normal)
            if situation in {1, -1}:
                return situation

        values = vertices @ normal
        tol = CPP_PRECISION / 2.0
        pos = np.any(values > tol)
        neg = np.any(values < -tol)
        if pos and neg:
            return 0
        if pos:
            return 1
        if neg:
            return -1
        return 0

    def _check_situation_batch(self, region: FastRegion, normals: np.ndarray) -> np.ndarray:
        """Vectorized check_situation_accelerate for many hyperplanes."""
        normals = np.asarray(normals, dtype=float)
        if normals.ndim == 1:
            normals = normals[None, :]
        if len(normals) == 0:
            return np.empty(0, dtype=int)

        vertices = self._region_vertices(region)
        if len(vertices) < 1:
            return np.full(len(normals), -2, dtype=int)

        relation = np.full(len(normals), -2, dtype=int)
        unresolved = np.ones(len(normals), dtype=bool)

        if self.acceleration == "sphere":
            center = np.mean(vertices, axis=0)
            radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
            norms = np.linalg.norm(normals, axis=1)
            valid = norms > EPS
            values = normals @ center
            distance = np.zeros(len(normals), dtype=float)
            distance[valid] = np.abs(values[valid]) / norms[valid]
            decided = valid & (distance >= radius)
            relation[decided & (values >= 0)] = 1
            relation[decided & (values < 0)] = -1
            unresolved &= ~decided
        elif self.acceleration == "rectangle":
            low = np.min(vertices, axis=0)
            high = np.max(vertices, axis=0)
            max_values = np.sum(np.where(normals >= 0, normals * high, normals * low), axis=1)
            min_values = np.sum(np.where(normals >= 0, normals * low, normals * high), axis=1)
            positive = min_values > 0
            negative = max_values < 0
            relation[positive] = 1
            relation[negative] = -1
            unresolved &= ~(positive | negative)

        if np.any(unresolved):
            unresolved_idx = np.flatnonzero(unresolved)
            values = vertices @ normals[unresolved_idx].T
            tol = CPP_PRECISION / 2.0
            pos = np.any(values > tol, axis=0)
            neg = np.any(values < -tol, axis=0)
            local_relation = np.zeros(len(unresolved_idx), dtype=int)
            local_relation[pos & ~neg] = 1
            local_relation[neg & ~pos] = -1
            relation[unresolved_idx] = local_relation

        return relation

    def _preprocess_indices(self, points: np.ndarray) -> list[int]:
        if self.use_skyband:
            return self._skyband_indices(points, self.skyband_k)
        if self.use_skyline:
            return self._nondominated_indices(points)
        return list(range(len(points)))

    def _add_feedback(self, region: FastRegion, preferred: np.ndarray, rejected: np.ndarray) -> bool:
        region.add_leq(rejected - preferred, 0.0)
        return region.refine_vertices()

    def _find_possible_top1(self, points: np.ndarray, region: FastRegion) -> list[int]:
        if region.vertices is None or len(region.vertices) == 0:
            if not region.refine_vertices():
                return []
        top_current: set[int] | None = None
        limit = min(len(region.vertices), 3)
        for i in range(limit):
            scores = points @ region.vertices[i]
            max_score = float(np.max(scores))
            top = {int(idx) for idx in np.flatnonzero(scores >= max_score - 1e-6)}
            if top_current is None:
                top_current = top
            else:
                top_current.intersection_update(top)
            if not top_current:
                return []
        return sorted(top_current or [])

    def _check_possible_top1(self, points: np.ndarray, region: FastRegion, candidates: Sequence[int]) -> Optional[int]:
        if region.vertices is None or len(region.vertices) == 0:
            if not region.refine_vertices():
                return None
        vertices = region.vertices
        for idx in candidates:
            p = points[idx]
            margins = vertices @ (p - points).T
            same = np.all(np.isclose(points, p, atol=EPS), axis=1)
            margins[:, same] = 0.0
            if not np.any(margins < -1e-6):
                return int(idx)
        return None

    def _fallback_answer(self, points: np.ndarray, region: FastRegion, active_points: Sequence[int]) -> int:
        center = self._region_center(region)
        return int(active_points[int(np.argmax(points[list(active_points)] @ center))])

    def _guaranteed_top1_full(self, points: np.ndarray, region: FastRegion) -> Optional[int]:
        vertices = self._region_vertices(region)
        if len(vertices) == 0:
            return None
        scores = points @ vertices.T
        winners = np.argmax(scores, axis=0)
        first = int(winners[0])
        if np.all(winners == first):
            return first
        return None

    def _exact_top1_tournament(
        self,
        points: np.ndarray,
        true_utility: Sequence[float],
        region: FastRegion,
        transcript: list[Question],
        questions: int,
        candidate_history: list[int],
        progress_callback: Callable[[dict], None] | None,
        progress_context: dict | None,
    ) -> tuple[int, int]:
        candidates = list(range(len(points)))
        oracle = SimulatedOracle(true_utility)
        best = int(candidates[0])
        for challenger in candidates[1:]:
            challenger = int(challenger)
            previous_best = best
            preferred = oracle.compare(points, previous_best, challenger)
            rejected = challenger if preferred == previous_best else previous_best
            transcript.append(Question(previous_best, challenger, preferred, rejected))
            self._add_feedback(region, points[preferred], points[rejected])
            best = int(preferred)
            questions += 1
            candidate_history.append(1)
            if progress_callback is not None:
                progress_callback(
                    {
                        **(progress_context or {}),
                        "algorithm": "RH-exact-tournament",
                        "round": questions,
                        "left": int(previous_best),
                        "right": int(challenger),
                        "preferred": int(preferred),
                        "rejected": int(rejected),
                        "before_candidates": int(len(candidates)),
                        "after_candidates": 1,
                        "pruned": int(len(candidates) - 1),
                        "remaining_questions_limit": int(self.max_questions - questions),
                    }
                )
        return best, questions

    def search(
        self,
        points: Sequence[Sequence[float]],
        true_utility: Sequence[float],
        initial_range=None,
        progress_callback: Callable[[dict], None] | None = None,
        progress_context: dict | None = None,
    ) -> SearchResult:
        data = as_points(points)
        if len(data) == 0:
            raise ValueError("points must not be empty")
        start = perf_counter()
        rng = np.random.default_rng(self.random_state)
        oracle = SimulatedOracle(true_utility)

        p_set = self._preprocess_indices(data)
        if not p_set:
            raise ValueError("preprocessing removed all points")
        rng.shuffle(p_set)
        region = fast_region_from_any(initial_range, data.shape[1])
        region.refine_vertices()

        current_use = [p_set[0]]
        candidate_history = [len(current_use)]
        transcript: list[Question] = []
        questions = 0
        answer: Optional[int] = None

        if self.strict_original_k1:
            initial_top_current = self._find_possible_top1(data[p_set], region)
            if initial_top_current:
                local_answer = self._check_possible_top1(data[p_set], region, initial_top_current)
                if local_answer is not None:
                    return finish_result(
                        start,
                        int(p_set[local_answer]),
                        questions,
                        candidate_history,
                        region,
                        transcript,
                    )

        for pivot in p_set[1:]:
            if questions >= self.max_questions:
                break
            if any(np.allclose(data[pivot], data[used], atol=EPS) for used in current_use):
                continue

            current_points = list(current_use)
            current_use.append(pivot)

            while current_points and questions < self.max_questions:
                before_current_points = len(current_points)
                before_active = len(current_use)
                center = self._region_center(region)
                opponents = np.asarray(current_points, dtype=int)
                normals = data[pivot] - data[opponents]
                relations = self._check_situation_batch(region, normals)
                intersect_mask = relations == 0
                if not np.any(intersect_mask):
                    current_points = []
                    break

                intersect_points = opponents[intersect_mask]
                intersect_normals = normals[intersect_mask]
                normal_norms = np.linalg.norm(intersect_normals, axis=1)
                normal_norms[normal_norms <= EPS] = np.inf
                distances = np.abs(intersect_normals @ center) / normal_norms
                best_pos = int(np.argmin(distances))
                opponent = int(intersect_points[best_pos])
                preferred = oracle.compare(data, pivot, opponent)
                rejected = opponent if preferred == pivot else pivot
                questions += 1
                transcript.append(Question(pivot, opponent, preferred, rejected))
                self._add_feedback(region, data[preferred], data[rejected])
                current_points = [
                    int(idx)
                    for pos, idx in enumerate(intersect_points)
                    if pos != best_pos
                ]

                top_current = self._find_possible_top1(data[p_set], region)
                mapped_top_current = [p_set[i] for i in top_current]
                if top_current:
                    if self.strict_original_k1:
                        local_answer = self._check_possible_top1(data[p_set], region, top_current)
                        answer = None if local_answer is None else int(p_set[local_answer])
                    else:
                        answer = self._check_possible_top1(data, region, mapped_top_current)
                candidate_history.append(len(current_use))
                if progress_callback is not None:
                    payload = {
                        **(progress_context or {}),
                        "algorithm": "RH",
                        "round": questions,
                        "left": int(pivot),
                        "right": int(opponent),
                        "preferred": int(preferred),
                        "rejected": int(rejected),
                        "before_candidates": int(before_active),
                        "after_candidates": int(len(current_use)),
                        "pruned": int(max(0, before_current_points - len(current_points) - 1)),
                        "intersect_before": int(before_current_points),
                        "intersect_after": int(len(current_points)),
                        "possible_top1_count": int(len(mapped_top_current)),
                        "answer_found": bool(answer is not None),
                        "remaining_questions_limit": int(self.max_questions - questions),
                    }
                    progress_callback(payload)
                if answer is not None:
                    return finish_result(start, answer, questions, candidate_history, region, transcript)

        if answer is None and self.strict_original_k1:
            raise RuntimeError("RH strict original k=1 did not find a certified top-1 before the loop ended")
        if answer is None:
            answer = self._guaranteed_top1_full(data, region)
        if answer is None:
            answer, questions = self._exact_top1_tournament(
                data,
                true_utility,
                region,
                transcript,
                questions,
                candidate_history,
                progress_callback,
                progress_context,
            )
        return finish_result(start, answer, questions, candidate_history, region, transcript)


def run_rh(
    points: Sequence[Sequence[float]],
    true_utility: Sequence[float],
    initial_range=None,
    **kwargs,
) -> SearchResult:
    progress_callback = kwargs.pop("progress_callback", None)
    progress_context = kwargs.pop("progress_context", None)
    return RHFast(**kwargs).search(
        points,
        true_utility,
        initial_range,
        progress_callback=progress_callback,
        progress_context=progress_context,
    )
