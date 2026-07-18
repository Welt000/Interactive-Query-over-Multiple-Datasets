from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter
from collections import deque
from functools import reduce
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-9


def as_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError("points must be a 2-D array-like object")
    return arr


def normalize_utility(utility: Sequence[float]) -> np.ndarray:
    u = np.asarray(utility, dtype=float)
    if u.ndim != 1:
        raise ValueError("utility must be a 1-D vector")
    if np.any(u < -EPS):
        raise ValueError("utility must be non-negative")
    total = float(np.sum(u))
    if total <= EPS:
        raise ValueError("utility vector must have positive sum")
    return u / total


def nondominated_indices(points: np.ndarray, block_size: int = 512) -> List[int]:
    """Return points not dominated under the usual larger-is-better convention."""
    data = as_points(points)
    n = len(data)
    if n == 0:
        return []

    dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = data[start:end]
        ge_all = np.all(data[None, :, :] >= block[:, None, :] - EPS, axis=2)
        gt_any = np.any(data[None, :, :] > block[:, None, :] + EPS, axis=2)
        dominated[start:end] = np.any(ge_all & gt_any, axis=1)
    return [int(i) for i in np.flatnonzero(~dominated)]


def skyline_indices(
    points: np.ndarray,
    compare_block: int = 256,
    max_block: int = 8192,
) -> List[int]:
    """Return exact skyline indices under larger-is-better dominance.

    Points are scanned in descending coordinate-sum order. Any strict dominator
    of the current point must have a larger coordinate sum, so it has already
    been considered. Accepted skyline points are stored in blocks. The active
    block starts at compare_block capacity, doubles until max_block, then a new
    block is created. This keeps early-stop behavior without repeated vstack.
    """
    data = as_points(points)
    n = len(data)
    if n == 0:
        return []

    dim = data.shape[1]
    sums = np.sum(data, axis=1)
    order = np.argsort(-sums, kind="mergesort")
    skyline_out: List[int] = []
    min_block = max(1, min(int(compare_block), n))
    max_block = max(min_block, min(int(max_block), n))
    capacity = min_block
    skyline_points = np.empty((capacity, dim), dtype=float)
    skyline_sums = np.empty(capacity, dtype=float)
    skyline_count = 0
    frozen_blocks: list[np.ndarray] = []
    frozen_sums: list[np.ndarray] = []

    def dominated_by(block: np.ndarray, block_sums: np.ndarray, point: np.ndarray, point_sum: float) -> bool:
        ge_all = np.all(block >= point - EPS, axis=1)
        strict_sum = block_sums > point_sum + EPS
        return bool(np.any(ge_all & strict_sum))

    for raw_idx in order:
        idx = int(raw_idx)
        point = data[idx]
        point_sum = float(sums[idx])
        dominated = False

        for block, block_sums in zip(frozen_blocks, frozen_sums):
            if dominated_by(block, block_sums, point, point_sum):
                dominated = True
                break
        if not dominated and skyline_count > 0:
            dominated = dominated_by(
                skyline_points[:skyline_count],
                skyline_sums[:skyline_count],
                point,
                point_sum,
            )

        if not dominated:
            if skyline_count >= capacity:
                if capacity < max_block:
                    new_capacity = min(max(capacity * 2, capacity + 1), max_block)
                    next_points = np.empty((new_capacity, dim), dtype=float)
                    next_sums = np.empty(new_capacity, dtype=float)
                    next_points[:skyline_count] = skyline_points[:skyline_count]
                    next_sums[:skyline_count] = skyline_sums[:skyline_count]
                    skyline_points = next_points
                    skyline_sums = next_sums
                    capacity = new_capacity
                else:
                    frozen_blocks.append(skyline_points[:skyline_count].copy())
                    frozen_sums.append(skyline_sums[:skyline_count].copy())
                    capacity = min_block
                    skyline_points = np.empty((capacity, dim), dtype=float)
                    skyline_sums = np.empty(capacity, dtype=float)
                    skyline_count = 0
            skyline_out.append(idx)
            skyline_points[skyline_count] = point
            skyline_sums[skyline_count] = point_sum
            skyline_count += 1

    return sorted(skyline_out)


def skyline(
    points: Sequence[Sequence[float]],
    *,
    return_indices: bool = False,
    compare_block: int = 256,
    max_block: int = 8192,
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    """Return the skyline points of a dataset.

    If return_indices is true, also return the original row indices of the
    retained points.
    """
    data = as_points(points)
    keep = skyline_indices(data, compare_block=compare_block, max_block=max_block)
    skyline_data = data[keep]
    if return_indices:
        return skyline_data, keep
    return skyline_data


@dataclass(frozen=True)
class Question:
    left: int
    right: int
    preferred: int
    rejected: int


@dataclass
class SearchResult:
    point_index: int
    questions: int
    candidate_history: List[int]
    elapsed: float
    utility_range: "UtilityRange"
    transcript: List[Question] = field(default_factory=list)


class SimulatedOracle:
    """Pairwise-comparison oracle used by experiments.

    The original algorithms ask a real user. For repeatable experiments we keep
    the user's hidden utility vector and answer each comparison by dot product.
    """

    def __init__(self, utility: Sequence[float]):
        self.utility = normalize_utility(utility)

    def compare(self, points: np.ndarray, left: int, right: int) -> int:
        left_score = float(np.dot(self.utility, points[left]))
        right_score = float(np.dot(self.utility, points[right]))
        if left_score >= right_score:
            return left
        return right

    def choose_best(self, points: np.ndarray, indices: Sequence[int]) -> int:
        if len(indices) == 0:
            raise ValueError("indices must not be empty")
        scores = points[list(indices)] @ self.utility
        return int(indices[int(np.argmax(scores))])


class UtilityRange:
    """Feasible utility region R = {u >= 0, sum(u)=1, A u >= 0}.

    The implementation avoids SciPy so the baseline can run in the current
    workspace. For low-dimensional experiments it enumerates polytope vertices
    by activating d-1 inequalities together with sum(u)=1.
    """

    def __init__(
        self,
        dim: int,
        constraints: Optional[Iterable[Sequence[float]]] = None,
        tol: float = 1e-8,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.tol = tol
        self.constraints: List[np.ndarray] = []
        self._vertices_cache: tuple[int, np.ndarray] | None = None
        if constraints is not None:
            for c in constraints:
                self.add_constraint(c)

    def copy(self) -> "UtilityRange":
        return UtilityRange(self.dim, [c.copy() for c in self.constraints], self.tol)

    def add_constraint(self, normal: Sequence[float]) -> None:
        normal_arr = np.asarray(normal, dtype=float)
        if normal_arr.shape != (self.dim,):
            raise ValueError(f"constraint must have shape ({self.dim},)")
        if np.linalg.norm(normal_arr) <= self.tol:
            return
        self.constraints.append(normal_arr)
        self._vertices_cache = None

    def add_preference(self, preferred: np.ndarray, rejected: np.ndarray) -> None:
        self.add_constraint(preferred - rejected)

    def _all_inequalities(self) -> np.ndarray:
        base = np.eye(self.dim)
        if not self.constraints:
            return base
        return np.vstack([base, np.vstack(self.constraints)])

    def contains(self, u: Sequence[float]) -> bool:
        u_arr = np.asarray(u, dtype=float)
        if u_arr.shape != (self.dim,):
            return False
        if abs(float(np.sum(u_arr)) - 1.0) > 1e-6:
            return False
        if np.any(u_arr < -self.tol):
            return False
        return all(float(np.dot(c, u_arr)) >= -self.tol for c in self.constraints)

    def vertices(self, max_combinations: int = 200_000) -> np.ndarray:
        if self.dim == 1:
            return np.array([[1.0]])
        if self._vertices_cache is not None and self._vertices_cache[0] == max_combinations:
            return self._vertices_cache[1].copy()

        inequalities = self._all_inequalities()
        active_count = self.dim - 1
        rows = range(len(inequalities))
        vertices: List[np.ndarray] = []

        for checked, active in enumerate(combinations(rows, active_count), start=1):
            if checked > max_combinations:
                break
            matrix = np.vstack([np.ones(self.dim), inequalities[list(active)]])
            rhs = np.zeros(self.dim)
            rhs[0] = 1.0
            try:
                u = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                continue
            if self.contains(u):
                vertices.append(np.clip(u, 0.0, 1.0))

        if not vertices:
            result = np.empty((0, self.dim))
        else:
            rounded = np.round(np.vstack(vertices), decimals=10)
            result = np.unique(rounded, axis=0)
        self._vertices_cache = (max_combinations, result.copy())
        return result

    def is_feasible(self) -> bool:
        if self.contains(np.full(self.dim, 1.0 / self.dim)):
            return True
        return len(self.vertices()) > 0

    def center(self) -> np.ndarray:
        vertices = self.vertices()
        if len(vertices) > 0:
            c = np.mean(vertices, axis=0)
            total = float(np.sum(c))
            if total > EPS:
                return c / total
        return np.full(self.dim, 1.0 / self.dim)

    def linear_minmax(self, normal: Sequence[float]) -> Tuple[float, float]:
        normal_arr = np.asarray(normal, dtype=float)
        vertices = self.vertices()
        if len(vertices) == 0:
            c = self.center()
            value = float(np.dot(normal_arr, c))
            return value, value
        values = vertices @ normal_arr
        return float(np.min(values)), float(np.max(values))

    def intersects_hyperplane(self, normal: Sequence[float]) -> bool:
        mn, mx = self.linear_minmax(normal)
        return mn <= self.tol and mx >= -self.tol


def utility_range_from_any(initial_range, dim: int) -> UtilityRange:
    """Return a UtilityRange copy from either UtilityRange or FastRegion-like input.

    Fast HD-PI/RH store homogeneous user-feedback constraints as halfspaces
    normal * u + offset <= 0.  UH algorithms use UtilityRange constraints in
    the equivalent form normal * u >= 0.
    """
    if initial_range is None:
        return UtilityRange(dim)
    if isinstance(initial_range, UtilityRange):
        return initial_range.copy()

    halfspaces = getattr(initial_range, "halfspaces", None)
    if halfspaces is None:
        raise TypeError("initial_range must be UtilityRange or FastRegion-like")

    utility_range = UtilityRange(dim)
    for normal, offset in halfspaces:
        if abs(float(offset)) > 1e-8:
            raise ValueError("UtilityRange cannot represent non-homogeneous halfspaces")
        utility_range.add_constraint(-np.asarray(normal, dtype=float))
    return utility_range


def can_be_top1(points: np.ndarray, point_index: int, utility_range: UtilityRange) -> bool:
    test_range = utility_range.copy()
    p = points[point_index]
    for j, q in enumerate(points):
        if j == point_index:
            continue
        test_range.add_constraint(p - q)
    return test_range.is_feasible()


def guaranteed_top1(points: np.ndarray, utility_range: UtilityRange) -> Optional[int]:
    """Return a point that is top-1 for every utility vector in R, if one exists."""
    vertices = utility_range.vertices()
    if len(vertices) == 0:
        return None
    scores = points @ vertices.T
    winners = np.argmax(scores, axis=0)
    first = int(winners[0])
    if np.all(winners == first):
        return first
    return None


def prune_top1_candidates(
    points: np.ndarray,
    candidates: Sequence[int],
    utility_range: UtilityRange,
    exact_limit: int = 250,
) -> List[int]:
    """Keep points whose top-1 partition still intersects the utility range.

    For larger candidate sets we use vertices of R as a conservative accelerator:
    any point that wins at one current vertex is kept, and then exact feasibility
    is applied to the reduced set.
    """
    candidate_list = list(dict.fromkeys(candidates))
    if len(candidate_list) <= 1:
        return candidate_list

    if len(candidate_list) > exact_limit:
        vertices = utility_range.vertices()
        if len(vertices) > 0:
            winner_set = set()
            scores = points[candidate_list] @ vertices.T
            for col in range(scores.shape[1]):
                winner_set.add(candidate_list[int(np.argmax(scores[:, col]))])
            candidate_list = sorted(winner_set)

    kept = [
        idx
        for idx in candidate_list
        if can_be_top1(points, idx, utility_range)
    ]
    return kept if kept else candidate_list


def hyperplane_prune_candidates(
    points: np.ndarray,
    candidates: Sequence[int],
    utility_range: UtilityRange,
) -> List[int]:
    """Prune candidates dominated throughout the current utility range.

    This mirrors the source UH implementation's `dom` pruning: p dominates q
    under R if u·(p-q) is non-negative for every extreme utility vector in R.
    """
    candidate_list = list(dict.fromkeys(candidates))
    if len(candidate_list) <= 1:
        return candidate_list
    vertices = utility_range.vertices()
    if len(vertices) == 0:
        return candidate_list

    scores = points[candidate_list] @ vertices.T
    dominated = np.zeros(len(candidate_list), dtype=bool)
    block_size = 512
    for start in range(0, len(candidate_list), block_size):
        end = min(start + block_size, len(candidate_list))
        block_scores = scores[start:end]
        dominates_block = np.all(
            scores[None, :, :] >= block_scores[:, None, :] - utility_range.tol,
            axis=2,
        )
        local = np.arange(end - start)
        dominates_block[local, start + local] = False
        dominated[start:end] = np.any(dominates_block, axis=1)

    kept = [candidate_list[int(i)] for i in np.flatnonzero(~dominated)]
    return kept if kept else candidate_list


class _RTreeNode:
    def __init__(self, dim: int, max_entries: int = 36, parent: "_RTreeNode | None" = None):
        self.dim = dim
        self.max_entries = max_entries
        self.parent = parent
        self.entries: list[tuple[int, np.ndarray] | _RTreeNode] = []
        self.is_leaf = True
        self.min_point = np.full(dim, np.inf)
        self.max_point = np.full(dim, -np.inf)

    def insert_entry(self, entry: tuple[int, np.ndarray] | "_RTreeNode") -> None:
        self.entries.append(entry)
        self.update_mbr_local()

    def update_mbr_local(self) -> None:
        self.min_point = np.full(self.dim, np.inf)
        self.max_point = np.full(self.dim, -np.inf)
        if not self.entries:
            return
        if self.is_leaf:
            coords = np.vstack([entry[1] for entry in self.entries])  # type: ignore[index]
            self.min_point = np.min(coords, axis=0)
            self.max_point = np.max(coords, axis=0)
        else:
            mins = np.vstack([entry.min_point for entry in self.entries])  # type: ignore[union-attr]
            maxs = np.vstack([entry.max_point for entry in self.entries])  # type: ignore[union-attr]
            self.min_point = np.min(mins, axis=0)
            self.max_point = np.max(maxs, axis=0)

    def update_mbr(self) -> None:
        self.update_mbr_local()
        if self.parent is not None:
            self.parent.update_mbr()


class _RTree:
    def __init__(self, dim: int, max_entries: int = 36):
        self.dim = dim
        self.max_entries = max_entries
        self.root = _RTreeNode(dim, max_entries)

    def insert(self, idx: int, coord: np.ndarray) -> None:
        node = self._choose_leaf(self.root, coord)
        node.insert_entry((int(idx), coord.copy()))
        node.update_mbr()
        if len(node.entries) > self.max_entries:
            self._split_node(node)

    def _choose_leaf(self, node: _RTreeNode, coord: np.ndarray) -> _RTreeNode:
        if node.is_leaf:
            return node
        best_child = None
        min_increase = float("inf")
        for child in node.entries:
            child_node = child  # type: ignore[assignment]
            before = self._area(child_node.min_point, child_node.max_point)
            after_min = np.minimum(coord, child_node.min_point)
            after_max = np.maximum(coord, child_node.max_point)
            increase = self._area(after_min, after_max) - before
            if increase < min_increase:
                min_increase = increase
                best_child = child_node
        return self._choose_leaf(best_child, coord)  # type: ignore[arg-type]

    def _split_node(self, node: _RTreeNode) -> None:
        sibling = _RTreeNode(self.dim, self.max_entries, node.parent)
        half = len(node.entries) // 2
        if node.is_leaf:
            node.entries.sort(key=lambda entry: float(entry[1][0]))  # type: ignore[index]
        else:
            node.entries.sort(key=lambda entry: float(entry.min_point[0] + entry.max_point[0]))  # type: ignore[union-attr]
            sibling.is_leaf = False
        sibling.entries = node.entries[half:]
        node.entries = node.entries[:half]
        for entry in sibling.entries:
            if isinstance(entry, _RTreeNode):
                entry.parent = sibling

        if node.parent is not None:
            node.parent.entries.append(sibling)
            if len(node.parent.entries) > self.max_entries:
                self._split_node(node.parent)
        else:
            new_root = _RTreeNode(self.dim, self.max_entries)
            new_root.is_leaf = False
            new_root.entries = [node, sibling]
            node.parent = new_root
            sibling.parent = new_root
            self.root = new_root
        node.update_mbr()
        sibling.update_mbr()

    def _area(self, min_point: np.ndarray, max_point: np.ndarray) -> float:
        lengths = np.maximum(max_point - min_point, 0.0)
        return float(reduce(lambda x, y: x * y, lengths, 1.0))


def _dominates_in_range(
    p1: np.ndarray,
    p2: np.ndarray,
    vertices: np.ndarray,
    tol: float = 1e-9,
) -> bool:
    if len(vertices) == 0:
        return False
    return bool(np.all(vertices @ (p1 - p2) >= -tol))


def _dominates_scores(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    tol: float = 1e-9,
) -> bool:
    return bool(np.all(left_scores >= right_scores - tol))


def rtree_prune_candidates(
    points: np.ndarray,
    candidates: Sequence[int],
    utility_range: UtilityRange,
) -> List[int]:
    """Prune candidates with the source UH RTree skyline routine.

    This ports `HyperplaneSet.rtree_prune`: build an RTree over the current
    skyline set, traverse node MBRs, and keep only points not dominated over all
    extreme utilities of the current range.
    """
    candidate_list = list(dict.fromkeys(int(idx) for idx in candidates))
    if len(candidate_list) <= 1:
        return candidate_list

    vertices = utility_range.vertices()
    if len(vertices) == 0:
        return candidate_list

    data = as_points(points)
    tree = _RTree(data.shape[1])
    for idx in candidate_list:
        tree.insert(idx, data[idx])

    point_scores = {idx: vertices @ data[idx] for idx in candidate_list}

    queue: deque[_RTreeNode] = deque([tree.root])
    skyline: list[int] = []
    while queue:
        node = queue.popleft()
        if not node.is_leaf:
            dominated = False
            node_scores = vertices @ node.max_point
            for idx in skyline:
                if _dominates_scores(point_scores[idx], node_scores):
                    dominated = True
                    break
            if not dominated:
                for child in node.entries:
                    queue.append(child)  # type: ignore[arg-type]
        else:
            for entry in node.entries:
                idx = entry[0]  # type: ignore[index]
                dominated = False
                for kept in skyline:
                    if _dominates_scores(point_scores[kept], point_scores[idx]):
                        dominated = True
                        break
                if not dominated:
                    skyline = [
                        kept
                        for kept in skyline
                        if not _dominates_scores(point_scores[idx], point_scores[kept])
                    ]
                    skyline.append(int(idx))

    return skyline if skyline else candidate_list


def candidate_hyperplanes(candidates: Sequence[int]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    cand = list(candidates)
    for pos, left in enumerate(cand):
        for right in cand[pos + 1:]:
            pairs.append((left, right))
    return pairs


def apply_user_feedback(
    points: np.ndarray,
    utility_range: UtilityRange,
    oracle: SimulatedOracle,
    left: int,
    right: int,
) -> Question:
    preferred = oracle.compare(points, left, right)
    rejected = right if preferred == left else left
    utility_range.add_preference(points[preferred], points[rejected])
    return Question(left=left, right=right, preferred=preferred, rejected=rejected)


def finish_result(
    start: float,
    point_index: int,
    questions: int,
    candidate_history: List[int],
    utility_range: UtilityRange,
    transcript: List[Question],
) -> SearchResult:
    return SearchResult(
        point_index=point_index,
        questions=questions,
        candidate_history=candidate_history,
        elapsed=perf_counter() - start,
        utility_range=utility_range,
        transcript=transcript,
    )
