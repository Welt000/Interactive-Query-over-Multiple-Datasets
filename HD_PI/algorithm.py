from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from structure.common import (
    Question,
    SearchResult,
    SimulatedOracle,
    as_points,
    finish_result,
)


EPS = 1e-8


@dataclass
class FastRegion:
    dim: int
    halfspaces: list[tuple[np.ndarray, float]]
    point_index: int
    vertices: np.ndarray | None = None
    midpoint: np.ndarray | None = None
    constraint_owners: list[int | None] = field(default_factory=list)
    active_halfspace_indices: tuple[int, ...] = ()
    active_neighbors: tuple[int, ...] = ()
    owner_count: int = 0
    _matrix_cache: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def initial(cls, dim: int, point_index: int = -1) -> "FastRegion":
        halfspaces: list[tuple[np.ndarray, float]] = []
        for i in range(dim):
            normal = np.zeros(dim)
            normal[i] = -1.0
            halfspaces.append((normal, 0.0))
        return cls(
            dim=dim,
            halfspaces=halfspaces,
            point_index=point_index,
            constraint_owners=[None] * len(halfspaces),
        )

    def copy(self) -> "FastRegion":
        return FastRegion(
            dim=self.dim,
            halfspaces=[(n.copy(), float(b)) for n, b in self.halfspaces],
            point_index=self.point_index,
            vertices=None if self.vertices is None else self.vertices.copy(),
            midpoint=None if self.midpoint is None else self.midpoint.copy(),
            constraint_owners=list(self.constraint_owners),
            active_halfspace_indices=tuple(self.active_halfspace_indices),
            active_neighbors=tuple(self.active_neighbors),
            owner_count=self.owner_count,
        )

    def add_leq(self, normal: np.ndarray, offset: float = 0.0) -> None:
        if np.linalg.norm(normal) <= EPS:
            return
        self.halfspaces.append((np.asarray(normal, dtype=float), float(offset)))
        self.constraint_owners.append(None)
        self.vertices = None
        self.midpoint = None
        self.active_halfspace_indices = ()
        self.active_neighbors = ()
        self._matrix_cache = None

    def add_leq_batch(
        self,
        normals: np.ndarray,
        offset: float = 0.0,
        owners: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        normals = np.asarray(normals, dtype=float)
        if normals.ndim != 2 or normals.shape[1] != self.dim:
            raise ValueError(f"normals must have shape (n, {self.dim})")
        if len(normals) == 0:
            return
        owner_arr = None
        if owners is not None:
            owner_arr = np.asarray(owners, dtype=object)
            if owner_arr.shape[0] != normals.shape[0]:
                raise ValueError("owners length must match normals")
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > EPS
        if not np.any(valid):
            return
        self.halfspaces.extend((normal.copy(), float(offset)) for normal in normals[valid])
        if owner_arr is None:
            self.constraint_owners.extend([None] * int(np.sum(valid)))
        else:
            self.constraint_owners.extend(
                None if owner is None else int(owner) for owner in owner_arr[valid]
            )
            self.owner_count += int(np.sum(valid))
        self.vertices = None
        self.midpoint = None
        self.active_halfspace_indices = ()
        self.active_neighbors = ()
        self._matrix_cache = None

    def halfspace_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix_cache is None:
            if not self.halfspaces:
                self._matrix_cache = (
                    np.empty((0, self.dim), dtype=float),
                    np.empty(0, dtype=float),
                )
            else:
                normals = np.vstack([normal for normal, _ in self.halfspaces]).astype(float, copy=False)
                offsets = np.asarray([offset for _, offset in self.halfspaces], dtype=float)
                self._matrix_cache = (normals, offsets)
        return self._matrix_cache

    def _bounded_halfspaces(self) -> np.ndarray:
        rows = [[*normal.tolist(), offset] for normal, offset in self.halfspaces]
        rows.append([*np.ones(self.dim).tolist(), -1.0])
        return np.asarray(rows, dtype=float)

    def _interior_point(self) -> np.ndarray | None:
        rows = self._bounded_halfspaces()
        a = rows[:, : self.dim]
        b = rows[:, self.dim]
        norms = np.linalg.norm(a, axis=1)
        norms[norms <= EPS] = 1.0

        a_ub = np.hstack([a, norms[:, None]])
        b_ub = -b
        c = np.zeros(self.dim + 1)
        c[-1] = -1.0
        bounds = [(None, None)] * self.dim + [(0.0, None)]
        result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success or result.x[-1] <= 1e-10:
            return None
        return np.asarray(result.x[: self.dim], dtype=float)

    def refine_vertices(self) -> bool:
        interior = self._interior_point()
        if interior is None:
            self.vertices = np.empty((0, self.dim))
            self.midpoint = None
            return False
        rows = self._bounded_halfspaces()
        try:
            hs = HalfspaceIntersection(rows, interior)
        except Exception:
            self.vertices = np.empty((0, self.dim))
            self.midpoint = None
            self.active_halfspace_indices = ()
            self.active_neighbors = ()
            return False
        vertices = hs.intersections
        if len(vertices) == 0:
            self.vertices = np.empty((0, self.dim))
            self.midpoint = None
            self.active_halfspace_indices = ()
            self.active_neighbors = ()
            return False
        clean = np.clip(vertices, 0.0, 1.0)
        self.vertices = np.unique(np.round(clean, 12), axis=0)
        if len(self.vertices) > 0:
            midpoint = np.mean(self.vertices, axis=0)
            total = float(np.sum(midpoint))
            self.midpoint = midpoint / total if total > EPS else None
        else:
            self.midpoint = None
        self._update_active_halfspaces(hs)
        return len(self.vertices) > 0

    def _update_active_halfspaces(self, hs: HalfspaceIntersection) -> None:
        active: set[int] = set()
        halfspace_count = len(self.halfspaces)
        dual_facets = getattr(hs, "dual_facets", None)
        if dual_facets is not None:
            for facet in dual_facets:
                for idx in facet:
                    idx_int = int(idx)
                    if 0 <= idx_int < halfspace_count:
                        active.add(idx_int)
        if not active and self.vertices is not None and len(self.vertices) > 0:
            normals, offsets = self.halfspace_matrix()
            values = self.vertices @ normals.T + offsets
            active.update(int(idx) for idx in np.where(np.any(np.abs(values) <= 1e-7, axis=0))[0])
        self.active_halfspace_indices = tuple(sorted(active))
        if len(self.constraint_owners) == halfspace_count:
            owners = {
                int(self.constraint_owners[idx])
                for idx in active
                if self.constraint_owners[idx] is not None
            }
            self.active_neighbors = tuple(sorted(owners))
        else:
            self.active_neighbors = ()

    def classify(self, normal: np.ndarray) -> int:
        if self.vertices is None:
            if not self.refine_vertices():
                return -2
        values = self.vertices @ normal
        pos = np.any(values > EPS)
        neg = np.any(values < -EPS)
        if pos and neg:
            return 0
        if pos:
            return 1
        if neg:
            return -1
        return 0


def fast_region_from_any(initial_range, dim: int, point_index: int = -1) -> FastRegion:
    """Return a FastRegion initialized from UtilityRange or FastRegion input."""
    if initial_range is None:
        return FastRegion.initial(dim, point_index=point_index)
    if isinstance(initial_range, FastRegion):
        region = initial_range.copy()
        region.point_index = point_index
        return region

    constraints = getattr(initial_range, "constraints", None)
    if constraints is None:
        raise TypeError("initial_range must be UtilityRange or FastRegion-like")

    region = FastRegion.initial(dim, point_index=point_index)
    for constraint in constraints:
        region.add_leq(-np.asarray(constraint, dtype=float), 0.0)
    return region


@dataclass
class ChooseItem:
    left: int
    right: int
    normal: np.ndarray
    positive_side: set[int]
    negative_side: set[int]
    intersect_case: set[int]

    def even_score(self, beta: float) -> float:
        return min(len(self.positive_side), len(self.negative_side)) - beta * len(self.intersect_case)

    def can_split(self) -> bool:
        return bool(self.negative_side or self.intersect_case) and bool(self.positive_side or self.intersect_case)


class HDPIFast:
    """HD-PI with Qhull halfspace intersections and incremental choose-item table.

    This backend mirrors the C++ `HDPI_sampling` / `HDPI_accurate` structure more
    closely than `HD-PI.py`: partitions are Qhull halfspace intersections,
    choose-items persist across rounds, and feedback incrementally updates
    positive/negative/intersect memberships.
    """

    def __init__(
        self,
        max_questions: int = 10_000,
        beta: float = 0.01,
        candidate_mode: str = "sampling",
        sample_count: int = 1024,
        max_partition_candidates: int = 160,
        random_state: int = 0,
    ):
        if candidate_mode not in {"accurate", "sampling", "skyline"}:
            raise ValueError("candidate_mode must be 'accurate', 'sampling', or 'skyline'")
        self.max_questions = max_questions
        self.beta = beta
        self.candidate_mode = candidate_mode
        self.sample_count = sample_count
        self.max_partition_candidates = max_partition_candidates
        self.random_state = random_state

    def _nondominated_indices(self, points: np.ndarray) -> list[int]:
        return list(range(len(points)))

    def _skyline_from_indices(self, points: np.ndarray, indices: list[int]) -> list[int]:
        return list(indices)

    def _convex_top1_candidates(self, points: np.ndarray) -> list[int]:
        """Port of C++ find_top1 + skyline_c used by HDPI_accurate."""
        if len(points) <= points.shape[1] + 1:
            return self._nondominated_indices(points)

        dim = points.shape[1]
        origin = np.zeros((1, dim), dtype=float)
        qhull_input = np.vstack([points, origin])
        try:
            hull = ConvexHull(qhull_input)
        except Exception:
            return self._nondominated_indices(points)

        hull_indices = sorted(int(i) for i in hull.vertices if int(i) < len(points))
        if not hull_indices:
            return self._nondominated_indices(points)
        return self._skyline_from_indices(points, hull_indices)

    def _sample_top1_candidates(self, points: np.ndarray, rng: np.random.Generator) -> list[int]:
        dim = points.shape[1]
        samples = [np.full(dim, 1.0 / dim)]
        samples.extend(np.eye(dim))
        for _ in range(max(0, self.sample_count - len(samples))):
            samples.append(rng.dirichlet(np.ones(dim)))
        winners = {int(np.argmax(points @ u)) for u in samples}
        candidates = sorted(winners)
        if len(candidates) > self.max_partition_candidates:
            center = np.full(dim, 1.0 / dim)
            scores = points[candidates] @ center
            order = np.argsort(scores)[::-1][: self.max_partition_candidates]
            candidates = sorted(int(candidates[int(i)]) for i in order)
        return candidates

    def _candidate_points(self, points: np.ndarray, rng: np.random.Generator) -> list[int]:
        if self.candidate_mode == "sampling":
            return self._sample_top1_candidates(points, rng)
        if self.candidate_mode == "accurate":
            return self._convex_top1_candidates(points)
        return self._nondominated_indices(points)

    def _construct_regions(
        self,
        points: np.ndarray,
        candidates: list[int],
        base_region: FastRegion,
    ) -> tuple[list[FastRegion], list[int]]:
        regions: list[FastRegion] = []
        choose_points: list[int] = []
        for idx in candidates:
            region = base_region.copy()
            region.point_index = idx
            p = points[idx]
            for other in candidates:
                if other == idx:
                    continue
                region.add_leq(points[other] - p, 0.0)
            if region.refine_vertices():
                regions.append(region)
                choose_points.append(idx)
        return regions, choose_points

    def _region_side_matrix(self, regions: list[FastRegion], normals: np.ndarray) -> np.ndarray:
        sides = np.full((len(regions), len(normals)), -2, dtype=np.int8)
        for region_id, region in enumerate(regions):
            if region.vertices is None:
                region.refine_vertices()
            if region.vertices is None or len(region.vertices) == 0:
                continue
            values = region.vertices @ normals.T
            pos = np.any(values > EPS, axis=0)
            neg = np.any(values < -EPS, axis=0)
            region_sides = np.zeros(len(normals), dtype=np.int8)
            region_sides[pos & ~neg] = 1
            region_sides[neg & ~pos] = -1
            sides[region_id] = region_sides
        return sides

    def _build_choose_items(self, points: np.ndarray, regions: list[FastRegion], choose_points: list[int]) -> list[ChooseItem]:
        items: list[ChooseItem] = []
        pair_meta: list[tuple[int, int]] = []
        normals: list[np.ndarray] = []
        for pos_i, left in enumerate(choose_points):
            for right in choose_points[pos_i + 1 :]:
                pair_meta.append((left, right))
                normals.append(points[left] - points[right])

        if not normals:
            return items

        normal_arr = np.vstack(normals)
        side_matrix = self._region_side_matrix(regions, normal_arr)
        for pair_id, (left, right) in enumerate(pair_meta):
            item = ChooseItem(left, right, normal_arr[pair_id], set(), set(), set())
            sides = side_matrix[:, pair_id]
            item.positive_side = set(int(i) for i in np.flatnonzero(sides == 1))
            item.negative_side = set(int(i) for i in np.flatnonzero(sides == -1))
            item.intersect_case = set(int(i) for i in np.flatnonzero(sides == 0))
            if item.can_split():
                items.append(item)
        return items

    def _select_item(self, items: list[ChooseItem]) -> int:
        best_idx = 0
        best_score = -float("inf")
        for i, item in enumerate(items):
            score = item.even_score(self.beta)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _remove_region_ids(self, items: list[ChooseItem], removed: set[int]) -> None:
        if not removed:
            return
        for item in items:
            item.positive_side.difference_update(removed)
            item.negative_side.difference_update(removed)
            item.intersect_case.difference_update(removed)

    def _insert_side(self, item: ChooseItem, region_id: int, side: int) -> None:
        item.positive_side.discard(region_id)
        item.negative_side.discard(region_id)
        item.intersect_case.discard(region_id)
        if side == 1:
            item.positive_side.add(region_id)
        elif side == -1:
            item.negative_side.add(region_id)
        elif side == 0:
            item.intersect_case.add(region_id)

    def _modify_choose_table(
        self,
        items: list[ChooseItem],
        regions: list[FastRegion],
        item_idx: int,
        keep_positive: bool,
        preference_normal_leq: np.ndarray,
    ) -> tuple[list[ChooseItem], set[int]]:
        chosen = items[item_idx]
        removed = set(chosen.negative_side if keep_positive else chosen.positive_side)
        considered_before_refine = set(range(len(regions))) - removed
        self._remove_region_ids(items, removed)

        refined_ids = set(chosen.intersect_case) & considered_before_refine
        infeasible: set[int] = set()
        for region_id in refined_ids:
            regions[region_id].add_leq(preference_normal_leq, 0.0)
            if not regions[region_id].refine_vertices():
                infeasible.add(region_id)
        self._remove_region_ids(items, infeasible)

        refined_ids.difference_update(infeasible)
        for region_id in refined_ids:
            region = regions[region_id]
            for i, item in enumerate(items):
                if i == item_idx:
                    continue
                if region_id in item.intersect_case:
                    side = region.classify(item.normal)
                    if side != 0:
                        self._insert_side(item, region_id, side)

        items.pop(item_idx)
        items = [item for item in items if item.can_split()]
        return items, removed | infeasible

    def _guaranteed_top1(self, points: np.ndarray, region: FastRegion) -> int | None:
        if region.vertices is None or len(region.vertices) == 0:
            if not region.refine_vertices():
                return None
        vertices = region.vertices
        if vertices is None or len(vertices) == 0:
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
        candidates = self._nondominated_indices(points)
        if not candidates:
            candidates = list(range(len(points)))
        oracle = SimulatedOracle(true_utility)
        best = int(candidates[0])
        for challenger in candidates[1:]:
            before_count = len(candidates)
            challenger = int(challenger)
            previous_best = best
            preferred = oracle.compare(points, previous_best, challenger)
            rejected = challenger if preferred == previous_best else previous_best
            transcript.append(Question(best, challenger, preferred, rejected))
            region.add_leq(points[rejected] - points[preferred], 0.0)
            region.refine_vertices()
            best = int(preferred)
            questions += 1
            candidate_history.append(1)
            if progress_callback is not None:
                progress_callback(
                    {
                        **(progress_context or {}),
                        "algorithm": "HD-PI-exact-tournament",
                        "round": questions,
                        "left": int(previous_best),
                        "right": int(challenger),
                        "preferred": int(preferred),
                        "rejected": int(rejected),
                        "before_candidates": int(before_count),
                        "after_candidates": 1,
                        "pruned": int(before_count - 1),
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
        base_region = fast_region_from_any(initial_range, data.shape[1])
        base_region.refine_vertices()
        shared_region = base_region.copy()

        candidates = self._candidate_points(data, rng)
        regions, choose_points = self._construct_regions(data, candidates, base_region)
        considered = set(range(len(regions)))
        items = self._build_choose_items(data, regions, choose_points)
        candidate_history = [len(considered)]
        transcript: list[Question] = []
        questions = 0

        while len(considered) > 1 and items and questions < self.max_questions:
            before_count = len(considered)
            item_idx = self._select_item(items)
            item = items[item_idx]
            item_even_score = item.even_score(self.beta)
            item_positive_count = len(item.positive_side)
            item_negative_count = len(item.negative_side)
            item_intersect_count = len(item.intersect_case)
            preferred = oracle.compare(data, item.left, item.right)
            rejected = item.right if preferred == item.left else item.left
            keep_positive = preferred == item.left
            preference_normal = data[rejected] - data[preferred]
            transcript.append(Question(item.left, item.right, preferred, rejected))
            questions += 1
            shared_region.add_leq(preference_normal, 0.0)
            shared_region.refine_vertices()

            items, removed = self._modify_choose_table(
                items,
                regions,
                item_idx,
                keep_positive,
                preference_normal,
            )
            considered.difference_update(removed)
            candidate_history.append(len(considered))
            if progress_callback is not None:
                payload = {
                    **(progress_context or {}),
                    "algorithm": "HD-PI",
                    "round": questions,
                    "left": int(item.left),
                    "right": int(item.right),
                    "preferred": int(preferred),
                    "rejected": int(rejected),
                    "before_candidates": int(before_count),
                    "after_candidates": int(len(considered)),
                    "pruned": int(before_count - len(considered)),
                    "removed_regions": int(len(removed)),
                    "even_score": float(item_even_score),
                    "positive_side": int(item_positive_count),
                    "negative_side": int(item_negative_count),
                    "intersect_case": int(item_intersect_count),
                    "remaining_questions_limit": int(self.max_questions - questions),
                }
                progress_callback(payload)

        if considered:
            if len(considered) == 1:
                chosen_region = min(considered)
                answer = regions[chosen_region].point_index
            else:
                answer = self._guaranteed_top1(data, shared_region)
                if answer is None:
                    answer, questions = self._exact_top1_tournament(
                        data,
                        true_utility,
                        shared_region,
                        transcript,
                        questions,
                        candidate_history,
                        progress_callback,
                        progress_context,
                    )
        else:
            utility = np.asarray(true_utility, dtype=float)
            utility = utility / np.sum(utility)
            answer = int(np.argmax(data @ utility))

        # Keep SearchResult shape compatible. Fast backend does not expose the
        # old UtilityRange object, so `utility_range` is returned as None.
        return SearchResult(
            point_index=answer,
            questions=questions,
            candidate_history=candidate_history,
            elapsed=perf_counter() - start,
            utility_range=shared_region,
            transcript=transcript,
        )


def run_hd_pi(
    points: Sequence[Sequence[float]],
    true_utility: Sequence[float],
    initial_range=None,
    **kwargs,
) -> SearchResult:
    progress_callback = kwargs.pop("progress_callback", None)
    progress_context = kwargs.pop("progress_context", None)
    return HDPIFast(**kwargs).search(
        points,
        true_utility,
        initial_range,
        progress_callback=progress_callback,
        progress_context=progress_context,
    )
