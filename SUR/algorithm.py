from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import linprog, minimize

from HD_PI import run_hd_pi
from RH import run_rh
from UH_Random import run_uh_random
from UH_Simplex import run_uh_simplex
from HD_PI import FastRegion
from structure.common import (
    Question,
    SearchResult,
    UtilityRange,
    as_points,
    utility_range_from_any,
)


AlgorithmRunner = Callable[..., SearchResult]
SUR_VERTEX_ENUMERATION_LIMIT = 5_000_000


def _range_constraint_count(range_like) -> int:
    if range_like is None:
        return 0
    constraints = getattr(range_like, "constraints", None)
    if constraints is not None:
        return len(constraints)
    halfspaces = getattr(range_like, "halfspaces", None)
    if halfspaces is not None:
        return len(halfspaces)
    return -1


def _range_vertex_combination_count(range_like, dim: int) -> int:
    if range_like is None:
        return 0
    active_count = max(0, dim - 1)
    if isinstance(range_like, UtilityRange):
        constraint_count = len(getattr(range_like, "constraints", []))
        return comb(dim + constraint_count, active_count)

    halfspaces = getattr(range_like, "halfspaces", None)
    if halfspaces is not None:
        valid_count = 0
        for normal, _offset in halfspaces:
            normal_arr = np.asarray(normal, dtype=float)
            if normal_arr.shape == (dim,):
                valid_count += 1
        return comb(valid_count, active_count)

    constraints = getattr(range_like, "constraints", None)
    if constraints is not None:
        return comb(dim + len(constraints), active_count)
    return 0


def _range_too_complex_for_utility_prune(range_like, dim: int) -> tuple[bool, int]:
    combination_count = _range_vertex_combination_count(range_like, dim)
    return combination_count > SUR_VERTEX_ENUMERATION_LIMIT, combination_count


def _format_progress_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        if len(value) > 8:
            head = ",".join(str(item) for item in value[:8])
            return f"[{head},...;n={len(value)}]"
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _print_sur_progress(payload: dict) -> None:
    preferred = payload.get("preferred", "")
    rejected = payload.get("rejected", "")
    extras = []
    for key in (
        "selected",
        "intersect_before",
        "intersect_after",
        "possible_top1_count",
        "answer_found",
        "removed_regions",
        "direct_choice_pruned",
        "update_elapsed",
        "rtree_prune_elapsed",
        "round_elapsed",
    ):
        if key in payload:
            extras.append(f"{key}={_format_progress_value(payload[key])}")
    extra_text = "," + ",".join(extras) if extras else ""
    print(
        "[SUR-interaction]"
        f" algorithm={payload.get('algorithm')},"
        f"u={payload.get('utility_id')},"
        f"dataset={payload.get('dataset_id')},"
        f"round={payload.get('round')},"
        f"choice={preferred}>{rejected},"
        f"candidates={payload.get('before_candidates')}->{payload.get('after_candidates')},"
        f"pruned={payload.get('pruned')},"
        f"remaining_budget={payload.get('remaining_questions_limit')}"
        f"{extra_text}",
        flush=True,
    )


def _print_sur_point_prune(payload: dict) -> None:
    status = "KEEP" if payload.get("kept") else "PRUNE"
    print(
        "[SUR-point-prune]"
        f" u={payload.get('utility_id')},"
        f"dataset={payload.get('dataset_id')},"
        f"point={payload.get('point_index')},"
        f"status={status},"
        f"reason={payload.get('reason')}",
        flush=True,
    )


def _print_sur_neighbor_partition(payload: dict) -> None:
    status = "KEEP" if payload.get("kept") else "SKIP"
    print(
        "[SUR-neighbor-partition]"
        f" u={payload.get('utility_id')},"
        f"dataset={payload.get('dataset_id')},"
        f"partition={payload.get('point_index')},"
        f"status={status},"
        f"reason={payload.get('reason')},"
        f"visited={payload.get('visited_count')}/{payload.get('pool_count')},"
        f"kept={payload.get('kept_count')},"
        f"queue={payload.get('queue_count')},"
        f"regions={payload.get('region_count')},"
        f"elapsed={payload.get('elapsed'):.6f}",
        flush=True,
    )


@dataclass
class SharedDatasetResult:
    dataset_id: int
    point_index: int
    local_point_index: int
    questions: int
    elapsed: float
    kept_count: int
    pruned_count: int
    result: SearchResult


@dataclass
class SharedUtilityRangeResult:
    dataset_results: list[SharedDatasetResult] = field(default_factory=list)
    total_questions: int = 0
    total_elapsed: float = 0.0
    utility_range: object | None = None

    @property
    def point_indices(self) -> list[int]:
        return [item.point_index for item in self.dataset_results]


@dataclass
class NeighborPruneStats:
    strategy: str
    pool_count: int
    visited_count: int
    region_count: int
    anchor_index: int | None


def _algorithm_runner(algorithm: str | AlgorithmRunner) -> AlgorithmRunner:
    if callable(algorithm):
        return algorithm

    key = algorithm.strip().lower().replace("_", "-")
    runners = {
        "hd-pi": run_hd_pi,
        "hdpi": run_hd_pi,
        "rh": run_rh,
        "uh-simplex": run_uh_simplex,
        "uh-simple": run_uh_simplex,
        "uh-random": run_uh_random,
        "uh-ramdom": run_uh_random,
    }
    if key not in runners:
        raise ValueError(f"unknown algorithm: {algorithm}")
    return runners[key]


def _range_halfspaces(range_like, dim: int) -> tuple[list[np.ndarray], list[float]]:
    normals: list[np.ndarray] = []
    offsets: list[float] = []

    if range_like is None:
        return normals, offsets

    if isinstance(range_like, UtilityRange):
        for constraint in range_like.constraints:
            normals.append(-np.asarray(constraint, dtype=float))
            offsets.append(0.0)
        return normals, offsets

    halfspaces = getattr(range_like, "halfspaces", None)
    if halfspaces is None:
        utility_range = utility_range_from_any(range_like, dim)
        return _range_halfspaces(utility_range, dim)

    for normal, offset in halfspaces:
        normal_arr = np.asarray(normal, dtype=float)
        if normal_arr.shape != (dim,):
            raise ValueError(f"range halfspace dimension mismatch: expected {dim}")
        normals.append(normal_arr)
        offsets.append(float(offset))
    return normals, offsets


def top1_partition_intersects_range(
    points: Sequence[Sequence[float]],
    point_index: int,
    utility_range,
    tol: float = 1e-9,
) -> bool:
    """Return whether point_index can be top-1 inside utility_range.

    The checked partition is:
        {u: u >= 0, sum(u)=1, p_i * u >= p_j * u for every j}
    intersected with the shared utility range accumulated from earlier
    datasets.  A feasible LP means the point's top-1 cell should remain.
    """
    data = as_points(points)
    dim = data.shape[1]
    if point_index < 0 or point_index >= len(data):
        raise IndexError("point_index out of range")

    normals, offsets = _range_halfspaces(utility_range, dim)
    return _top1_partition_intersects_range_prepared(
        data,
        point_index,
        np.asarray(normals, dtype=float) if normals else np.empty((0, dim)),
        np.asarray(offsets, dtype=float) if offsets else np.empty(0),
        tol=tol,
    )


def _top1_partition_intersects_range_prepared(
    data: np.ndarray,
    point_index: int,
    range_normals: np.ndarray,
    range_offsets: np.ndarray,
    tol: float = 1e-9,
) -> bool:
    dim = data.shape[1]
    p = data[point_index]
    top1_normals = data - p
    if len(range_normals) > 0:
        a_ub = np.vstack([range_normals, top1_normals])
        b_ub = np.concatenate([-range_offsets, np.zeros(len(data), dtype=float)])
    else:
        a_ub = top1_normals
        b_ub = np.zeros(len(data), dtype=float)
    a_eq = np.ones((1, dim), dtype=float)
    b_eq = np.array([1.0], dtype=float)
    bounds = [(0.0, None)] * dim
    result = linprog(
        np.zeros(dim),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    return bool(result.success and result.fun <= tol)


def _shared_range_vertices(utility_range, dim: int) -> np.ndarray:
    if utility_range is None:
        return np.empty((0, dim))
    if isinstance(utility_range, UtilityRange):
        constraint_count = len(getattr(utility_range, "constraints", []))
        active_count = max(0, dim - 1)
        total_combinations = comb(dim + constraint_count, active_count)
        if total_combinations > SUR_VERTEX_ENUMERATION_LIMIT:
            return np.empty((0, dim))
        return utility_range.vertices(max_combinations=total_combinations)

    halfspaces = getattr(utility_range, "halfspaces", None)
    if halfspaces is not None:
        normals: list[np.ndarray] = []
        offsets: list[float] = []
        for normal, offset in halfspaces:
            normal_arr = np.asarray(normal, dtype=float)
            if normal_arr.shape != (dim,):
                continue
            normals.append(normal_arr)
            offsets.append(float(offset))
        if not normals:
            return UtilityRange(dim).vertices()

        vertices: list[np.ndarray] = []
        normal_matrix = np.vstack(normals)
        offset_arr = np.asarray(offsets, dtype=float)
        for active in combinations(range(len(normals)), dim - 1):
            matrix = np.vstack([np.ones(dim), normal_matrix[list(active)]])
            rhs = np.zeros(dim)
            rhs[0] = 1.0
            rhs[1:] = -offset_arr[list(active)]
            try:
                point = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                continue
            if abs(float(np.sum(point)) - 1.0) > 1e-7:
                continue
            if np.any(point < -1e-7):
                continue
            if np.all(normal_matrix @ point + offset_arr <= 1e-7):
                vertices.append(np.clip(point, 0.0, 1.0))
        if vertices:
            return np.unique(np.round(np.vstack(vertices), 12), axis=0)
        return np.empty((0, dim))

    vertices_attr = getattr(utility_range, "vertices", None)
    if callable(vertices_attr):
        try:
            vertex_arr = np.asarray(vertices_attr(), dtype=float)
        except TypeError:
            vertex_arr = np.empty((0, dim))
        if vertex_arr.ndim == 2 and vertex_arr.shape[1] == dim:
            return vertex_arr
    elif vertices_attr is not None:
        vertex_arr = np.asarray(vertices_attr, dtype=float)
        if vertex_arr.ndim == 2 and vertex_arr.shape[1] == dim:
            return vertex_arr

    converted = utility_range_from_any(utility_range, dim)
    return converted.vertices()


def _candidate_winners_at_vertices(
    points: np.ndarray,
    candidate_list: list[int],
    vertices: np.ndarray,
) -> list[int]:
    if len(vertices) == 0 or len(candidate_list) == 0:
        return []
    scores = points[candidate_list] @ vertices.T
    winners = {
        candidate_list[int(np.argmax(scores[:, col]))]
        for col in range(scores.shape[1])
    }
    return sorted(winners)


def _range_contains_vertex(utility_range, vertex: np.ndarray, dim: int, tol: float = 1e-7) -> bool:
    point = np.asarray(vertex, dtype=float)
    if point.shape != (dim,):
        return False
    if abs(float(np.sum(point)) - 1.0) > 1e-5:
        return False
    if np.any(point < -tol):
        return False
    if isinstance(utility_range, UtilityRange):
        return utility_range.contains(point)

    halfspaces = getattr(utility_range, "halfspaces", None)
    if halfspaces is not None:
        return all(
            float(np.dot(np.asarray(normal, dtype=float), point) + float(offset)) <= tol
            for normal, offset in halfspaces
        )

    converted = utility_range_from_any(utility_range, dim)
    return converted.contains(point)


def _partition_region_on_pool(
    points: np.ndarray,
    point_index: int,
    candidate_pool: list[int],
    cache: dict[int, FastRegion],
) -> FastRegion | None:
    if point_index in cache:
        return cache[point_index]

    dim = points.shape[1]
    region = FastRegion.initial(dim, point_index=point_index)
    point = points[point_index]
    for other in candidate_pool:
        other = int(other)
        if other == point_index:
            continue
        region.add_leq(points[other] - point, 0.0)
    if not region.refine_vertices():
        cache[point_index] = region
        return None
    cache[point_index] = region
    return region


def _partition_has_vertex_in_range(
    region: FastRegion,
    utility_range,
    dim: int,
) -> bool:
    if region.vertices is None or len(region.vertices) == 0:
        return False
    return any(_range_contains_vertex(utility_range, vertex, dim) for vertex in region.vertices)


def _point_in_prepared_range(
    point: np.ndarray,
    range_normals: np.ndarray,
    range_offsets: np.ndarray,
    tol: float = 1e-7,
) -> bool:
    if np.any(point < -tol):
        return False
    if abs(float(np.sum(point)) - 1.0) > tol:
        return False
    if len(range_normals) == 0:
        return True
    return bool(np.all(range_normals @ point + range_offsets <= tol))


def _project_to_simplex(point: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(point, dtype=float), 0.0)
    total = float(np.sum(clipped))
    if total <= 1e-12:
        return np.full(len(clipped), 1.0 / len(clipped), dtype=float)
    return clipped / total


def _partition_envelope_sphere(
    region: FastRegion,
) -> tuple[np.ndarray, float] | None:
    if region.vertices is None or len(region.vertices) == 0:
        return None
    center = np.mean(region.vertices, axis=0)
    radius = float(np.max(np.linalg.norm(region.vertices - center, axis=1)))
    return center, radius


def _partition_sphere_intersects_range_prepared(
    region: FastRegion,
    range_normals: np.ndarray,
    range_offsets: np.ndarray,
    tol: float = 1e-7,
) -> bool:
    sphere = _partition_envelope_sphere(region)
    if sphere is None:
        return False
    center, radius = sphere

    if _point_in_prepared_range(center, range_normals, range_offsets, tol=tol):
        return True

    dim = len(center)
    x0 = _project_to_simplex(center)
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: float(np.sum(x) - 1.0),
            "jac": lambda x: np.ones_like(x),
        }
    ]
    for normal, offset in zip(range_normals, range_offsets):
        n = np.asarray(normal, dtype=float)
        b = float(offset)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, n=n, b=b: float(-(np.dot(n, x) + b)),
                "jac": lambda x, n=n, b=b: -n,
            }
        )

    def objective(x: np.ndarray) -> float:
        diff = x - center
        return float(np.dot(diff, diff))

    def objective_jac(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - center)

    result = minimize(
        objective,
        x0,
        jac=objective_jac,
        bounds=[(0.0, 1.0)] * dim,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 100, "disp": False},
    )
    if not result.success:
        return True
    return bool(result.fun <= (radius + tol) ** 2)


def _partition_neighbors_on_pool(
    points: np.ndarray,
    point_index: int,
    candidate_pool: list[int],
    region: FastRegion,
    candidate_set: set[int],
) -> list[int]:
    if region.vertices is None or len(region.vertices) == 0:
        return []
    point = points[point_index]
    vertices = region.vertices
    neighbors: list[int] = []
    for other in candidate_pool:
        other = int(other)
        if other == point_index or other not in candidate_set:
            continue
        values = vertices @ (points[other] - point)
        if np.any(np.abs(values) <= 1e-7):
            neighbors.append(other)
    return neighbors


def _shared_range_center(utility_range, dim: int) -> np.ndarray:
    vertices = _shared_range_vertices(utility_range, dim)
    if len(vertices) > 0:
        center = np.mean(vertices, axis=0)
        total = float(np.sum(center))
        if total > 1e-12:
            return center / total
    return utility_range_from_any(utility_range, dim).center()


def _top1_candidate_pool(points: np.ndarray, candidates: Iterable[int] | None = None) -> list[int]:
    if candidates is not None:
        pool = list(dict.fromkeys(int(idx) for idx in candidates))
        return pool

    pool = list(range(len(points)))
    return list(dict.fromkeys(int(idx) for idx in pool))


def filter_points_by_range_vertex_dominance(
    points: Sequence[Sequence[float]],
    utility_range,
    candidates: Iterable[int] | None = None,
    block_size: int = 512,
    tol: float = 1e-12,
) -> tuple[np.ndarray, list[int]]:
    """Condition B: prune p if another q is better at every range vertex."""
    data = as_points(points)
    candidate_list = list(range(len(data))) if candidates is None else list(dict.fromkeys(int(i) for i in candidates))
    if utility_range is None or len(candidate_list) <= 1:
        return data[candidate_list], candidate_list

    vertices = _shared_range_vertices(utility_range, data.shape[1])
    if len(vertices) == 0:
        return data[candidate_list], candidate_list

    scores = data[candidate_list] @ vertices.T
    n = len(candidate_list)
    dominated = np.zeros(n, dtype=bool)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = scores[start:end]
        strictly_better_all_vertices = np.all(scores[None, :, :] > block[:, None, :] + tol, axis=2)
        dominated[start:end] = np.any(strictly_better_all_vertices, axis=1)

    kept = [candidate_list[int(i)] for i in np.flatnonzero(~dominated)]
    return data[kept], kept


def filter_points_by_shared_range_neighbors(
    points: Sequence[Sequence[float]],
    utility_range,
    candidates: Iterable[int] | None = None,
    partition_log_callback: Callable[[dict], None] | None = None,
    partition_log_context: dict | None = None,
    max_elapsed_seconds: float | None = None,
) -> tuple[np.ndarray, list[int], NeighborPruneStats]:
    """Local SUR pruning by expanding neighboring top-1 partitions.

    Start from the point that is top-1 at the midpoint of the shared utility
    range vertices.  When no candidate list is supplied, this path uses the
    full input point set, then expands through SharedQ-style neighboring partitions.
    A point is kept when the envelope sphere of its partition intersects the
    shared range.
    """
    data = as_points(points)
    if candidates is None:
        candidate_pool = list(range(len(data)))
    else:
        candidate_pool = _top1_candidate_pool(data, candidates)
    if utility_range is None or len(candidate_pool) <= 1:
        return (
            data[candidate_pool],
            candidate_pool,
            NeighborPruneStats(
                strategy="neighbor",
                pool_count=len(candidate_pool),
                visited_count=len(candidate_pool),
                region_count=0,
                anchor_index=candidate_pool[0] if candidate_pool else None,
            ),
        )

    dim = data.shape[1]
    center = _shared_range_center(utility_range, dim)
    range_normals_list, range_offsets_list = _range_halfspaces(utility_range, dim)
    range_normals = (
        np.asarray(range_normals_list, dtype=float)
        if range_normals_list
        else np.empty((0, dim), dtype=float)
    )
    range_offsets = (
        np.asarray(range_offsets_list, dtype=float)
        if range_offsets_list
        else np.empty(0, dtype=float)
    )
    scores = data[candidate_pool] @ center
    anchor = int(candidate_pool[int(np.argmax(scores))])
    candidate_set = set(candidate_pool)
    region_cache: dict[int, FastRegion] = {}
    kept: set[int] = set()
    visited: set[int] = set()
    queue: deque[int] = deque([anchor])
    prune_start = perf_counter()

    def emit_partition(point_index: int, kept_partition: bool, reason: str) -> None:
        if partition_log_callback is None:
            return
        partition_log_callback(
            {
                **(partition_log_context or {}),
                "point_index": int(point_index),
                "kept": bool(kept_partition),
                "reason": reason,
                "visited_count": int(len(visited)),
                "pool_count": int(len(candidate_pool)),
                "kept_count": int(len(kept)),
                "queue_count": int(len(queue)),
                "region_count": int(len(region_cache)),
                "elapsed": float(perf_counter() - prune_start),
            }
        )

    while queue:
        if max_elapsed_seconds is not None and perf_counter() - prune_start >= max_elapsed_seconds:
            break
        point_index = int(queue.popleft())
        if point_index in visited:
            continue
        visited.add(point_index)

        region = _partition_region_on_pool(data, point_index, candidate_pool, region_cache)
        if region is None:
            emit_partition(point_index, False, "empty_partition")
            continue
        if not _partition_sphere_intersects_range_prepared(region, range_normals, range_offsets):
            emit_partition(point_index, False, "sphere_outside_range")
            continue

        kept.add(point_index)
        neighbor_count = 0
        for neighbor in _partition_neighbors_on_pool(
            data,
            point_index,
            candidate_pool,
            region,
            candidate_set,
        ):
            if neighbor not in visited:
                queue.append(neighbor)
                neighbor_count += 1
        emit_partition(point_index, True, f"sphere_intersects_range,neighbors={neighbor_count}")
        if max_elapsed_seconds is not None and perf_counter() - prune_start >= max_elapsed_seconds:
            break

    if not kept:
        kept = {anchor}

    kept_list = sorted(kept)
    return (
        data[kept_list],
        kept_list,
        NeighborPruneStats(
            strategy="neighbor",
            pool_count=len(candidate_pool),
            visited_count=len(visited),
            region_count=len(region_cache),
            anchor_index=anchor,
        ),
    )


def filter_points_by_shared_range(
    points: Sequence[Sequence[float]],
    utility_range,
    candidates: Iterable[int] | None = None,
    point_log_callback: Callable[[dict], None] | None = None,
    point_log_context: dict | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Remove points whose top-1 partition does not intersect utility_range.

    This is the exact Shared Utility Range pruning rule: every remaining
    non-obvious candidate is checked by LP.
    """
    data = as_points(points)
    candidate_list = list(range(len(data))) if candidates is None else list(candidates)
    if utility_range is None or len(candidate_list) <= 1:
        return data[candidate_list], candidate_list

    def emit(point_index: int, kept: bool, reason: str) -> None:
        if point_log_callback is None:
            return
        point_log_callback({
            **(point_log_context or {}),
            "point_index": int(point_index),
            "kept": bool(kept),
            "reason": reason,
        })

    range_normals_list, range_offsets_list = _range_halfspaces(utility_range, data.shape[1])
    range_normals = (
        np.asarray(range_normals_list, dtype=float)
        if range_normals_list
        else np.empty((0, data.shape[1]), dtype=float)
    )
    range_offsets = (
        np.asarray(range_offsets_list, dtype=float)
        if range_offsets_list
        else np.empty(0, dtype=float)
    )

    if len(candidate_list) <= 1:
        for idx in candidate_list:
            emit(idx, True, "single_candidate")
        return data[candidate_list], candidate_list

    vertices = _shared_range_vertices(utility_range, data.shape[1])
    vertex_winners = set(_candidate_winners_at_vertices(data, candidate_list, vertices))

    kept = sorted(vertex_winners)
    for idx in kept:
        emit(idx, True, "range_vertex_winner")
    exact_candidates = [idx for idx in candidate_list if idx not in vertex_winners]
    for idx in exact_candidates:
        intersects = _top1_partition_intersects_range_prepared(
            data,
            idx,
            range_normals,
            range_offsets,
        )
        emit(idx, intersects, "top1_partition_lp")
        if intersects:
            kept.append(idx)

    if not kept:
        shared_range = utility_range_from_any(utility_range, data.shape[1])
        center = shared_range.center()
        kept = [candidate_list[int(np.argmax(data[candidate_list] @ center))]]
        emit(kept[0], True, "range_center_fallback")
    return data[kept], kept


class SharedUtilityRange:
    """Sequential baseline wrapper for multiple datasets with one shared user.

    After each dataset, the utility range learned by the wrapped algorithm is
    reused as the initial range for the next dataset.  Before running the next
    dataset, every point whose top-1 partition does not intersect the current
    shared range is removed.
    """

    def __init__(
        self,
        algorithm: str | AlgorithmRunner,
        algorithm_kwargs: dict | None = None,
        prune_by_shared_range: bool = True,
        verbose: bool = False,
        verbose_points: bool = False,
        utility_id: int | None = None,
        prune_strategy: str = "exact",
    ):
        if prune_strategy not in {"neighbor", "exact", "vertex_dominance"}:
            raise ValueError("prune_strategy must be one of: neighbor, exact, vertex_dominance")
        self.runner = _algorithm_runner(algorithm)
        self.algorithm_kwargs = dict(algorithm_kwargs or {})
        self.prune_by_shared_range = prune_by_shared_range
        self.verbose = verbose
        self.verbose_points = verbose_points
        self.utility_id = utility_id
        self.prune_strategy = prune_strategy

    def search(
        self,
        datasets: Sequence[Sequence[Sequence[float]]],
        true_utility: Sequence[float],
        initial_range=None,
    ) -> SharedUtilityRangeResult:
        shared_range = initial_range
        output = SharedUtilityRangeResult(utility_range=shared_range)
        total_start = perf_counter()

        for dataset_id, points in enumerate(datasets):
            dataset_start = perf_counter()
            data = as_points(points)
            if len(data) == 0:
                raise ValueError("datasets must not contain empty point sets")

            if self.verbose:
                print(
                    "[SUR-dataset-start]"
                    f" u={self.utility_id if self.utility_id is not None else ''},"
                    f"dataset={dataset_id + 1}/{len(datasets)},"
                    f"n={len(data)},"
                    f"shared_constraints={_range_constraint_count(shared_range)}",
                    flush=True,
                )

            prune_start = perf_counter()
            prune_stats: NeighborPruneStats | None = None
            range_for_dataset = shared_range
            range_reset = False
            range_combination_count = 0
            if self.prune_by_shared_range and shared_range is not None:
                range_reset, range_combination_count = _range_too_complex_for_utility_prune(
                    shared_range,
                    data.shape[1],
                )
                if range_reset:
                    range_for_dataset = None
                    if self.verbose:
                        print(
                            "[SUR-range-reset]"
                            f" u={self.utility_id if self.utility_id is not None else ''},"
                            f"dataset={dataset_id + 1},"
                            f"combinations={range_combination_count},"
                            f"limit={SUR_VERTEX_ENUMERATION_LIMIT},"
                            f"old_constraints={_range_constraint_count(shared_range)},"
                            "action=skip_utility_prune_and_restart_from_simplex",
                            flush=True,
                        )

            if self.prune_by_shared_range and range_for_dataset is not None:
                if self.prune_strategy == "neighbor":
                    filtered, kept_indices, prune_stats = filter_points_by_shared_range_neighbors(
                        data,
                        range_for_dataset,
                        partition_log_callback=_print_sur_neighbor_partition if self.verbose else None,
                        partition_log_context={
                            "utility_id": self.utility_id if self.utility_id is not None else "",
                            "dataset_id": dataset_id + 1,
                        },
                    )
                elif self.prune_strategy == "vertex_dominance":
                    filtered, kept_indices = filter_points_by_range_vertex_dominance(
                        data,
                        range_for_dataset,
                    )
                else:
                    filtered, kept_indices = filter_points_by_shared_range(
                        data,
                        range_for_dataset,
                        point_log_callback=_print_sur_point_prune if self.verbose_points else None,
                        point_log_context={
                            "utility_id": self.utility_id if self.utility_id is not None else "",
                            "dataset_id": dataset_id + 1,
                        },
                    )
            else:
                kept_indices = list(range(len(data)))
                filtered = data
            prune_elapsed = perf_counter() - prune_start

            if self.verbose:
                stat_text = ""
                if prune_stats is not None:
                    stat_text = (
                        f",strategy={prune_stats.strategy},"
                        f"pool={prune_stats.pool_count},"
                        f"visited={prune_stats.visited_count},"
                        f"regions={prune_stats.region_count},"
                        f"anchor={prune_stats.anchor_index}"
                    )
                print(
                    "[SUR-prune]"
                    f" u={self.utility_id if self.utility_id is not None else ''},"
                    f"dataset={dataset_id + 1},"
                    f"kept={len(kept_indices)},"
                    f"pruned={len(data) - len(kept_indices)},"
                    f"elapsed={prune_elapsed:.6f},"
                    f"shared_constraints={_range_constraint_count(shared_range)},"
                    f"range_reset={int(range_reset)},"
                    f"range_combinations={range_combination_count}"
                    f"{stat_text}",
                    flush=True,
                )

            if len(filtered) == 1:
                local_result = SearchResult(
                    point_index=0,
                    questions=0,
                    candidate_history=[1],
                    elapsed=0.0,
                    utility_range=range_for_dataset,
                    transcript=[],
                )
            else:
                run_kwargs = dict(self.algorithm_kwargs)
                if self.verbose:
                    run_kwargs["progress_callback"] = _print_sur_progress
                    run_kwargs["progress_context"] = {
                        "utility_id": self.utility_id if self.utility_id is not None else "",
                        "dataset_id": dataset_id + 1,
                    }
                local_result = self.runner(
                    filtered,
                    true_utility,
                    initial_range=range_for_dataset,
                    **run_kwargs,
                )

            shared_range = local_result.utility_range
            local_point_index = int(local_result.point_index)
            point_index = int(kept_indices[local_point_index])
            dataset_elapsed = perf_counter() - dataset_start
            dataset_result = SharedDatasetResult(
                dataset_id=dataset_id,
                point_index=point_index,
                local_point_index=local_point_index,
                questions=int(local_result.questions),
                elapsed=float(dataset_elapsed),
                kept_count=len(kept_indices),
                pruned_count=len(data) - len(kept_indices),
                result=local_result,
            )
            output.dataset_results.append(dataset_result)
            output.total_questions += dataset_result.questions
            output.utility_range = shared_range
            if self.verbose:
                history = ",".join(str(x) for x in local_result.candidate_history[-8:])
                if len(local_result.candidate_history) > 8:
                    history = "..." + history
                print(
                    "[SUR-dataset-done]"
                    f" u={self.utility_id if self.utility_id is not None else ''},"
                    f"dataset={dataset_id + 1},"
                    f"questions={dataset_result.questions},"
                    f"result_global={point_index},"
                    f"kept={dataset_result.kept_count},"
                    f"pruned={dataset_result.pruned_count},"
                    f"elapsed={dataset_elapsed:.6f},"
                    f"history={history},"
                    f"shared_constraints={_range_constraint_count(shared_range)}",
                    flush=True,
                )

        output.total_elapsed = perf_counter() - total_start
        return output


def run_shared_utility_range(
    datasets: Sequence[Sequence[Sequence[float]]],
    true_utility: Sequence[float],
    algorithm: str | AlgorithmRunner = "HD-PI",
    initial_range=None,
    prune_by_shared_range: bool = True,
    verbose: bool = False,
    verbose_points: bool = False,
    utility_id: int | None = None,
    prune_strategy: str = "exact",
    **algorithm_kwargs,
) -> SharedUtilityRangeResult:
    return SharedUtilityRange(
        algorithm=algorithm,
        algorithm_kwargs=algorithm_kwargs,
        prune_by_shared_range=prune_by_shared_range,
        verbose=verbose,
        verbose_points=verbose_points,
        utility_id=utility_id,
        prune_strategy=prune_strategy,
    ).search(datasets, true_utility, initial_range=initial_range)


__all__ = [
    "SharedDatasetResult",
    "SharedUtilityRange",
    "SharedUtilityRangeResult",
    "filter_points_by_shared_range",
    "filter_points_by_shared_range_neighbors",
    "filter_points_by_range_vertex_dominance",
    "run_shared_utility_range",
    "top1_partition_intersects_range",
]
