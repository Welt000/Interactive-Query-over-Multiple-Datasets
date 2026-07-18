from __future__ import annotations

from typing import Callable, Optional, Sequence

from structure.common import SearchResult, UtilityRange
from structure.original_uh import run_original_uh_simplex


class UHSimplex:
    """Thin wrapper over the source `pythonProject/uh.py` UH-Simplex."""

    def __init__(
        self,
        s: int = 2,
        epsilon: float = 0.0,
        max_rounds: int = 1000,
        exact_prune_limit: int = 250,
        max_frame_rays: Optional[int] = None,
        use_glpk_frame: bool = True,
        random_state: int = 0,
        pre_skyline: bool = True,
        progress_callback: Optional[Callable[[dict], None]] = None,
        progress_context: Optional[dict] = None,
    ):
        self.s = s
        self.epsilon = epsilon
        self.max_rounds = max_rounds
        self.exact_prune_limit = exact_prune_limit
        self.max_frame_rays = max_frame_rays
        self.use_glpk_frame = use_glpk_frame
        self.random_state = random_state
        self.pre_skyline = pre_skyline
        self.progress_callback = progress_callback
        self.progress_context = progress_context

    def search(
        self,
        points: Sequence[Sequence[float]],
        true_utility: Sequence[float],
        initial_range: Optional[UtilityRange] = None,
    ) -> SearchResult:
        return run_original_uh_simplex(
            points,
            true_utility,
            initial_range=initial_range,
            s=self.s,
            epsilon=self.epsilon,
            max_rounds=self.max_rounds,
            random_state=self.random_state,
            pre_skyline=self.pre_skyline,
            progress_callback=self.progress_callback,
            progress_context=self.progress_context,
        )


def run_uh_simplex(
    points: Sequence[Sequence[float]],
    true_utility: Sequence[float],
    initial_range: Optional[UtilityRange] = None,
    **kwargs,
) -> SearchResult:
    return UHSimplex(**kwargs).search(points, true_utility, initial_range)


__all__ = ["UHSimplex", "run_uh_simplex"]
