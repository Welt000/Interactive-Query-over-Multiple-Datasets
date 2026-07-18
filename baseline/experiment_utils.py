from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from HD_PI import run_hd_pi
from RH import run_rh
from UH_Random import run_uh_random
from UH_Simplex import run_uh_simplex


ALGORITHMS = {
    "HD-PI": run_hd_pi,
    "RH": run_rh,
    "UH-Simplex": run_uh_simplex,
    "UH-Random": run_uh_random,
}


def _split_numbers(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split()]


def load_dataset(path: Path, drop_first_column: bool = False, auto_id_column: bool = True) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(_split_numbers(line))

    if not rows:
        raise ValueError(f"empty dataset: {path}")

    first = rows[0]
    if len(first) == 2 and all(abs(x - round(x)) < 1e-9 for x in first):
        n = int(round(first[0]))
        d = int(round(first[1]))
        has_time_row = (
            len(rows) >= 2
            and len(rows[1]) in {1, 2}
            and len(rows) - 2 == n
            and all(len(row) == d for row in rows[2:])
        )
        body_start = 2 if has_time_row else 1
        body = rows[body_start:]
        if len(body) == n and all(len(row) == d for row in body):
            data = np.asarray(body, dtype=float)
        else:
            data = np.asarray(rows, dtype=float)
    else:
        data = np.asarray(rows, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"dataset must be a 2-D table: {path}")

    if drop_first_column or (auto_id_column and _looks_like_id_column(data)):
        data = data[:, 1:]
    return data


def load_skyline_time(path: Path) -> float:
    """Return skyline preprocessing time stored on the second line."""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(_split_numbers(line))
            if len(rows) >= 2:
                break
    if len(rows) >= 2 and len(rows[0]) == 2 and len(rows[1]) == 1:
        return float(rows[1][0])
    return 0.0


def _looks_like_id_column(data: np.ndarray) -> bool:
    if data.shape[0] < 2 or data.shape[1] <= 1:
        return False
    ids = data[:, 0]
    zero_based = np.arange(len(data), dtype=float)
    one_based = zero_based + 1.0
    return bool(np.allclose(ids, zero_based) or np.allclose(ids, one_based))


def parse_utility_text(text: str) -> list[np.ndarray]:
    utilities = []
    for part in text.replace(";", "\n").splitlines():
        stripped = part.strip()
        if stripped:
            utilities.append(np.asarray(_split_numbers(stripped), dtype=float))
    return utilities


def normalize_utility(u: Sequence[float], dim: int) -> np.ndarray:
    arr = np.asarray(u, dtype=float)
    if arr.shape != (dim,):
        raise ValueError(f"utility dimension mismatch: expected {dim}, got {arr.shape}")
    if np.any(arr < 0):
        raise ValueError("utility values must be non-negative")
    total = float(np.sum(arr))
    if total <= 0:
        raise ValueError("utility vector must have positive sum")
    return arr / total


def parse_algorithms(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(ALGORITHMS)
    aliases = {
        "hdpi": "HD-PI",
        "hd-pi": "HD-PI",
        "rh": "RH",
        "uh-simplex": "UH-Simplex",
        "uh-simple": "UH-Simplex",
        "uh-random": "UH-Random",
        "uh-ramdom": "UH-Random",
    }
    algorithms = []
    for raw in text.replace(";", ",").split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unknown algorithm: {raw}")
        algorithms.append(aliases[key])
    return list(dict.fromkeys(algorithms))


def algorithm_kwargs(algorithm: str, args: argparse.Namespace, seed: int) -> dict:
    if algorithm == "HD-PI":
        return {
            "max_questions": args.max_questions,
            "candidate_mode": args.hdpi_candidate_mode,
            "sample_count": args.hdpi_sample_count,
            "max_partition_candidates": args.hdpi_max_partition_candidates,
            "beta": getattr(args, "hdpi_beta", 0.01),
            "random_state": seed,
        }
    if algorithm == "RH":
        return {
            "max_questions": args.max_questions,
            "random_state": seed,
            "strict_original_k1": getattr(args, "rh_strict_original_k1", False),
        }
    if algorithm == "UH-Simplex":
        return {
            "s": args.uh_s,
            "max_rounds": args.max_questions,
            "exact_prune_limit": args.uh_exact_prune_limit,
            "max_frame_rays": args.uh_max_frame_rays,
            "use_glpk_frame": not args.no_glpk_frame,
            "random_state": seed,
        }
    if algorithm == "UH-Random":
        return {
            "s": args.uh_s,
            "max_rounds": args.max_questions,
            "exact_prune_limit": args.uh_exact_prune_limit,
            "random_state": seed,
        }
    raise ValueError(f"unknown algorithm: {algorithm}")


def evaluate_prediction(data: np.ndarray, utility: np.ndarray, idx: int, tol: float) -> tuple[int, float, float, float, bool]:
    scores = data @ utility
    truth = int(np.argmax(scores))
    truth_score = float(scores[truth])
    score = float(scores[idx]) if 0 <= idx < len(data) else float("nan")
    regret = truth_score - score
    correct = idx == truth or abs(regret) <= tol
    return truth, truth_score, score, regret, correct
