from __future__ import annotations

import contextlib
import io
import random
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Sequence

import numpy as np

from structure.common import (
    Question,
    SearchResult,
    UtilityRange,
    as_points,
    finish_result,
    normalize_utility,
    utility_range_from_any,
)


from structure import constant as original_constant
from structure import uh as original_uh
from structure.hyperplane import Hyperplane as OriginalHyperplane
from structure.hyperplane_set import HyperplaneSet
from structure.point import Point as OriginalPoint
from structure.point_set import PointSet as OriginalPointSet


def _to_original_point_set(points: np.ndarray, pre_skyline: bool = True) -> OriginalPointSet:
    original_points = [
        OriginalPoint(points.shape[1], id=i, coord=points[i].copy())
        for i in range(len(points))
    ]
    point_set = OriginalPointSet(P=original_points)
    if not pre_skyline:
        return point_set
    with contextlib.redirect_stdout(io.StringIO()):
        skyline = point_set.skyline()
    for pos, point in enumerate(skyline.points):
        point.original_id = int(point.id) if point.id is not None and point.id >= 0 else pos
        point.id = pos
    return skyline


def _utility_to_original_point(utility: Sequence[float], dim: int) -> OriginalPoint:
    u = normalize_utility(utility)
    if u.shape != (dim,):
        raise ValueError(f"utility dimension mismatch: expected {dim}")
    point = OriginalPoint(dim)
    point.coord = u.copy()
    return point


def _initial_original_range(dim: int, initial_range) -> tuple[HyperplaneSet, UtilityRange]:
    utility_range = utility_range_from_any(initial_range, dim)
    hset = HyperplaneSet(dim)
    for constraint in utility_range.constraints:
        hset.hyperplanes.append(
            OriginalHyperplane(dim=dim, norm=-np.asarray(constraint, dtype=float), offset=0.0)
        )
    with contextlib.redirect_stdout(io.StringIO()):
        hset.set_ext_pts()
    return hset, utility_range


def _current_best(points: list[OriginalPoint], candidates: list[int], hset: HyperplaneSet) -> int:
    return int(original_uh.get_current_best_pt(points, candidates, hset))


@contextlib.contextmanager
def _suppress_original_question_files():
    original_method = OriginalPointSet.printMiddleSelection
    OriginalPointSet.printMiddleSelection = lambda *args, **kwargs: None
    try:
        yield
    finally:
        OriginalPointSet.printMiddleSelection = original_method


def run_original_uh(
    points: Sequence[Sequence[float]],
    true_utility: Sequence[float],
    initial_range: Optional[UtilityRange] = None,
    *,
    mode: str,
    s: int = 2,
    epsilon: float = 0.0,
    max_rounds: int = 1000,
    random_state: int = 0,
    pre_skyline: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
    progress_context: dict | None = None,
    **_: object,
) -> SearchResult:
    """Run the source `pythonProject/uh.py` workflow under the local API.

    The source PointSet, HyperplaneSet, update_ext_vec, RTree pruning, and
    frame_fast code are used directly.  Stopping remains the local top-1
    baseline condition: continue until one candidate remains or max_rounds is
    reached.
    """
    if s < 2:
        raise ValueError("s must be at least 2")
    data = as_points(points)
    if len(data) == 0:
        raise ValueError("points must not be empty")

    cmp_option = {
        "random": original_constant.RANDOM,
        "simplex": original_constant.SIMPlEX,
    }[mode]
    random.seed(int(random_state))

    point_set = _to_original_point_set(data, pre_skyline=pre_skyline)
    original_points = point_set.points
    if not original_points:
        raise ValueError("skyline is empty")

    start = perf_counter()
    hset, utility_range = _initial_original_range(original_points[0].dim, initial_range)
    original_u = _utility_to_original_point(true_utility, original_points[0].dim)

    candidates = list(range(len(original_points)))
    current_best = _current_best(original_points, candidates, hset)
    last_best = -1
    frame: list[OriginalPoint] = []
    candidate_history = [len(candidates)]
    transcript: list[Question] = []
    eliminated_by_choice: set[int] = set()
    questions = 0

    while len(candidates) > 1 and questions < max_rounds:
        round_start = perf_counter()
        if eliminated_by_choice:
            candidates = [idx for idx in candidates if idx not in eliminated_by_choice]
            if len(candidates) <= 1:
                break
        if current_best not in candidates and candidates:
            current_best = _current_best(original_points, candidates, hset)
        questions += 1
        candidates.sort()
        before_candidates = len(candidates)
        candidates_before_round = list(candidates)
        recorded_selection: list[int] = []
        source_generate_s = original_uh.generate_S

        def recording_generate_s(*args, **kwargs):
            candidate_count = len(args[1]) if len(args) > 1 else 0
            current_best_arg = int(args[3]) if len(args) > 3 else -1
            try:
                selected = source_generate_s(*args, **kwargs)
            except (IndexError, ValueError):
                selected = []
                if len(args) > 1:
                    for pos, candidate_idx in enumerate(args[1]):
                        if int(candidate_idx) == current_best_arg:
                            selected.append(pos)
                            break
            target_size = min(int(s), candidate_count)
            if len(selected) < target_size:
                selected_set = {int(pos) for pos in selected}
                for pos in range(candidate_count):
                    if pos not in selected_set:
                        selected.append(pos)
                        selected_set.add(pos)
                    if len(selected) >= target_size:
                        break
            recorded_selection.clear()
            recorded_selection.extend(int(pos) for pos in selected)
            return selected

        original_uh.generate_S = recording_generate_s
        update_start = perf_counter()
        try:
            with _suppress_original_question_files(), contextlib.redirect_stdout(io.StringIO()):
                candidates, last_best, current_best = original_uh.update_ext_vec(
                    original_points,
                    candidates,
                    original_u,
                    s,
                    hset,
                    current_best,
                    last_best,
                    frame,
                    cmp_option,
                    questions,
                    "dataset",
                    epsilon,
                )
        finally:
            original_uh.generate_S = source_generate_s
        update_elapsed = perf_counter() - update_start

        if len(recorded_selection) == 0:
            break

        selected_candidates = [int(candidates_before_round[position]) for position in recorded_selection]
        max_position = recorded_selection[0]
        max_value = -float("inf")
        for position in recorded_selection:
            value = float(original_u.dot_prod(original_points[candidates_before_round[position]]))
            if value > max_value:
                max_value = value
                max_position = position

        favorite_candidate = int(candidates_before_round[max_position])
        favorite_id = int(getattr(original_points[favorite_candidate], "original_id", original_points[favorite_candidate].id))
        rejected_candidates: list[int] = []
        for position in recorded_selection:
            if position == max_position:
                continue
            rejected_candidate = int(candidates_before_round[position])
            rejected_candidates.append(rejected_candidate)
            rejected_id = int(getattr(original_points[rejected_candidate], "original_id", original_points[rejected_candidate].id))
            utility_range.add_preference(data[favorite_id], data[rejected_id])
            transcript.append(
                Question(
                    left=favorite_id,
                    right=rejected_id,
                    preferred=favorite_id,
                    rejected=rejected_id,
                )
            )
        eliminated_by_choice.update(rejected_candidates)
        if eliminated_by_choice:
            candidates = [idx for idx in candidates if idx not in eliminated_by_choice]
        rtree_prune_elapsed = 0.0
        if len(candidates) > 1:
            rtree_prune_start = perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                candidates, _ = hset.rtree_prune(
                    original_points,
                    candidates,
                    original_constant.NO_BOUND,
                )
            rtree_prune_elapsed = perf_counter() - rtree_prune_start
            if eliminated_by_choice:
                candidates = [idx for idx in candidates if idx not in eliminated_by_choice]
        candidate_history.append(len(candidates))
        if progress_callback is not None:
            selected_ids = [
                int(getattr(original_points[idx], "original_id", original_points[idx].id))
                for idx in selected_candidates
            ]
            payload = {
                **(progress_context or {}),
                "algorithm": f"UH-{mode}",
                "round": questions,
                "selected": selected_ids,
                "preferred": int(favorite_id),
                "rejected": [
                    int(getattr(original_points[idx], "original_id", original_points[idx].id))
                    for idx in selected_candidates
                    if idx != favorite_candidate
                ],
                "before_candidates": int(before_candidates),
                "after_candidates": int(len(candidates)),
                "pruned": int(before_candidates - len(candidates)),
                "direct_choice_pruned": int(sum(1 for idx in rejected_candidates if idx in candidates_before_round)),
                "remaining_questions_limit": int(max_rounds - questions),
                "update_elapsed": float(update_elapsed),
                "rtree_prune_elapsed": float(rtree_prune_elapsed),
                "round_elapsed": float(perf_counter() - round_start),
            }
            progress_callback(payload)

        if current_best not in candidates and candidates:
            current_best = _current_best(original_points, candidates, hset)

    if len(candidates) == 1:
        answer_local = int(candidates[0])
    else:
        answer_local = _current_best(original_points, candidates, hset)
    answer = int(getattr(original_points[answer_local], "original_id", original_points[answer_local].id))
    return finish_result(start, answer, questions, candidate_history, utility_range, transcript)


def run_original_uh_random(
    points: Sequence[Sequence[float]],
    true_utility: Sequence[float],
    initial_range: Optional[UtilityRange] = None,
    **kwargs,
) -> SearchResult:
    return run_original_uh(points, true_utility, initial_range, mode="random", **kwargs)


def run_original_uh_simplex(
    points: Sequence[Sequence[float]],
    true_utility: Sequence[float],
    initial_range: Optional[UtilityRange] = None,
    **kwargs,
) -> SearchResult:
    return run_original_uh(points, true_utility, initial_range, mode="simplex", **kwargs)
