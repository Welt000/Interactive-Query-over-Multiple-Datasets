from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from SharedQ import run_detect_divide
from SUR import run_shared_utility_range
from baseline.experiment_utils import (
    ALGORITHMS,
    algorithm_kwargs,
    evaluate_prediction,
    load_dataset,
    load_skyline_time,
    normalize_utility,
    parse_algorithms,
    parse_utility_text,
)


def radius_from_rho(rho: float, dim: int) -> float:
    rho = float(rho)
    dim = int(dim)
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if dim == 2:
        return rho / math.sqrt(2.0)
    if dim == 3:
        return math.sqrt(rho * math.sqrt(3.0) / (2.0 * math.pi))
    if dim == 4:
        return (rho / (4.0 * math.pi)) ** (1.0 / 3.0)
    if dim == 5:
        return (rho * math.sqrt(5.0) / (12.0 * math.pi * math.pi)) ** (1.0 / 4.0)
    if dim == 6:
        return (
            rho * math.sqrt(6.0) / (64.0 * math.pi * math.pi)
        ) ** (1.0 / 5.0)
    if dim == 7:
        return (
            rho * math.sqrt(7.0) / (120.0 * math.pi**3)
        ) ** (1.0 / 6.0)
    raise ValueError(
        "rho-based radius is only defined for dimensions 2 through 7"
    )


DEFAULT_UTILITIES = (
    "0.62,0.18,0.12,0.08;"
    "0.08,0.62,0.18,0.12;"
    "0.12,0.08,0.62,0.18;"
    "0.18,0.12,0.08,0.62;"
    "0.30,0.25,0.35,0.10"
)


@dataclass
class DetailRow:
    method: str
    radius: float | str
    rho: float | str
    alpha: float | str
    beta: float | str
    alpha_beta_ratio: float | str
    utility_id: int
    dataset_id: int
    dataset_path: str
    n: int
    d: int
    truth_index: int
    result_index: int
    questions: int
    elapsed: float
    correct: bool
    regret: float
    score: float
    truth_score: float
    candidate_count: int
    pruned_count: int
    detect_rounds: int | str
    detect_time: float | str
    divide_time: float | str


@dataclass
class SummaryRow:
    method: str
    radius: float | str
    rho: float | str
    alpha: float | str
    beta: float | str
    alpha_beta_ratio: float | str
    status: str
    utilities: int
    datasets: int
    runs: int
    avg_total_questions_per_utility: float
    avg_total_elapsed_per_utility: float
    accuracy: float
    avg_regret: float
    avg_candidate_count: float
    avg_pruned_count: float
    avg_detect_rounds: float | str
    avg_detect_time_per_utility: float | str
    avg_divide_time_per_utility: float | str


def write_rows(path: Path, rows: list[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def append_detail_rows(path: Path, rows: list[DetailRow]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(
    method: str,
    radius: float | str,
    rho: float | str,
    rows: list[DetailRow],
    total_questions: list[int],
    total_elapsed: list[float],
    detect_rounds: list[int] | None,
    detect_times: list[float] | None,
    divide_times: list[float] | None,
    utility_count: int,
    dataset_count: int,
    shared_preprocess_elapsed: float = 0.0,
    alpha: float | str = "",
    beta: float | str = "",
    alpha_beta_ratio: float | str = "",
    status: str = "ok",
) -> SummaryRow:
    return SummaryRow(
        method=method,
        radius=radius,
        rho=rho,
        alpha=alpha,
        beta=beta,
        alpha_beta_ratio=alpha_beta_ratio,
        status=status,
        utilities=utility_count,
        datasets=dataset_count,
        runs=len(rows),
        avg_total_questions_per_utility=float(np.mean(total_questions)),
        avg_total_elapsed_per_utility=float(
            float(np.mean(total_elapsed)) + float(shared_preprocess_elapsed)
        ),
        accuracy=float(np.mean([row.correct for row in rows])),
        avg_regret=float(np.mean([row.regret for row in rows])),
        avg_candidate_count=float(np.mean([row.candidate_count for row in rows])),
        avg_pruned_count=float(np.mean([row.pruned_count for row in rows])),
        avg_detect_rounds="" if detect_rounds is None else float(np.mean(detect_rounds)),
        avg_detect_time_per_utility="" if detect_times is None else float(np.mean(detect_times)),
        avg_divide_time_per_utility="" if divide_times is None else float(np.mean(divide_times)),
    )


def make_row(
    method: str,
    radius: float | str,
    rho: float | str,
    alpha: float | str,
    beta: float | str,
    alpha_beta_ratio: float | str,
    utility_id: int,
    dataset_id: int,
    dataset_path: Path,
    data: np.ndarray,
    utility: np.ndarray,
    result_index: int,
    questions: int,
    elapsed: float,
    candidate_count: int,
    pruned_count: int,
    detect_rounds: int | str,
    detect_time: float | str,
    divide_time: float | str,
    tolerance: float,
) -> DetailRow:
    truth, truth_score, score, regret, correct = evaluate_prediction(
        data,
        utility,
        result_index,
        tolerance,
    )
    return DetailRow(
        method=method,
        radius=radius,
        rho=rho,
        alpha=alpha,
        beta=beta,
        alpha_beta_ratio=alpha_beta_ratio,
        utility_id=utility_id,
        dataset_id=dataset_id,
        dataset_path=str(dataset_path),
        n=len(data),
        d=data.shape[1],
        truth_index=truth,
        result_index=result_index,
        questions=int(questions),
        elapsed=float(elapsed),
        correct=bool(correct),
        regret=float(regret),
        score=float(score),
        truth_score=float(truth_score),
        candidate_count=int(candidate_count),
        pruned_count=int(pruned_count),
        detect_rounds=detect_rounds,
        detect_time=detect_time,
        divide_time=divide_time,
    )


def run_baseline(
    datasets: list[np.ndarray],
    dataset_paths: list[Path],
    utilities: list[np.ndarray],
    args: argparse.Namespace,
    detail_path: Path,
) -> tuple[list[DetailRow], list[SummaryRow]]:
    rows: list[DetailRow] = []
    summaries: list[SummaryRow] = []
    for algorithm in args.algorithms_parsed:
        algorithm_rows: list[DetailRow] = []
        total_questions: list[int] = []
        total_elapsed: list[float] = []
        runner = ALGORITHMS[algorithm]
        method = f"baseline_{algorithm}"
        for utility_id, utility in enumerate(utilities):
            global_utility_id = int(getattr(args, "utility_id_offset", 0)) + utility_id
            display_utility_id = global_utility_id + 1
            utility_questions = 0
            utility_elapsed = 0.0
            utility_rows: list[DetailRow] = []
            for dataset_id, data in enumerate(datasets):
                seed = args.seed + global_utility_id * 1009 + dataset_id * 917 + sum(ord(c) for c in algorithm)
                start = perf_counter()
                result = runner(
                    data,
                    utility,
                    initial_range=None,
                    **algorithm_kwargs(algorithm, args, seed),
                )
                elapsed = perf_counter() - start
                questions = int(getattr(result, "questions", 0))
                result_index = int(getattr(result, "point_index", getattr(result, "index", -1)))
                row = make_row(
                    method,
                    "",
                    "",
                    "",
                    "",
                    "",
                    global_utility_id,
                    dataset_id,
                    dataset_paths[dataset_id],
                    data,
                    utility,
                    result_index,
                    questions,
                    elapsed,
                    len(data),
                    0,
                    "",
                    "",
                    "",
                    args.tolerance,
                )
                utility_rows.append(row)
                utility_questions += questions
                utility_elapsed += elapsed
                print(
                    f"{method},u={display_utility_id},dataset={dataset_id + 1},"
                    f"dataset_name={dataset_paths[dataset_id].name},"
                    f"questions={questions},elapsed={elapsed:.6f}",
                    flush=True,
                )
            append_detail_rows(detail_path, utility_rows)
            rows.extend(utility_rows)
            algorithm_rows.extend(utility_rows)
            total_questions.append(utility_questions)
            total_elapsed.append(utility_elapsed)
            print(
                f"{method},u={display_utility_id},questions={utility_questions},"
                f"elapsed={utility_elapsed:.6f}",
                flush=True,
            )
        summaries.append(summarize(
            method,
            "",
            "",
            algorithm_rows,
            total_questions,
            total_elapsed,
            None,
            None,
            None,
            len(utilities),
            len(datasets),
            getattr(args, "skyline_total_elapsed", 0.0),
            "",
            "",
            "",
        ))
    return rows, summaries


def run_shared_range(
    datasets: list[np.ndarray],
    dataset_paths: list[Path],
    utilities: list[np.ndarray],
    args: argparse.Namespace,
    detail_path: Path,
) -> tuple[list[DetailRow], list[SummaryRow]]:
    rows: list[DetailRow] = []
    summaries: list[SummaryRow] = []
    for algorithm in args.algorithms_parsed:
        algorithm_rows: list[DetailRow] = []
        total_questions: list[int] = []
        total_elapsed: list[float] = []
        method = f"shared_utility_range_{algorithm}"
        for utility_id, utility in enumerate(utilities):
            global_utility_id = int(getattr(args, "utility_id_offset", 0)) + utility_id
            display_utility_id = global_utility_id + 1
            seed = args.seed + global_utility_id * 1009 + sum(ord(c) for c in algorithm)
            result = run_shared_utility_range(
                datasets,
                utility,
                algorithm=algorithm,
                verbose=args.verbose_sur,
                verbose_points=args.sur_verbose_points,
                utility_id=display_utility_id,
                prune_strategy=args.sur_prune_strategy,
                **algorithm_kwargs(algorithm, args, seed),
            )
            utility_rows: list[DetailRow] = []
            for item in result.dataset_results:
                data = datasets[item.dataset_id]
                item_questions = int(item.questions)
                item_elapsed = float(item.elapsed)
                utility_rows.append(
                    make_row(
                        method,
                        "",
                        "",
                        "",
                        "",
                        "",
                        global_utility_id,
                        item.dataset_id,
                        dataset_paths[item.dataset_id],
                        data,
                        utility,
                        int(item.point_index),
                        item_questions,
                        item_elapsed,
                        int(item.kept_count),
                        int(item.pruned_count),
                        "",
                        "",
                        "",
                        args.tolerance,
                    )
                )
                print(
                    f"{method},u={display_utility_id},dataset={item.dataset_id + 1},"
                    f"dataset_name={dataset_paths[item.dataset_id].name},"
                    f"questions={item_questions},elapsed={item_elapsed:.6f},"
                    f"kept={int(item.kept_count)},pruned={int(item.pruned_count)}",
                    flush=True,
                )
            append_detail_rows(detail_path, utility_rows)
            rows.extend(utility_rows)
            algorithm_rows.extend(utility_rows)
            total_questions.append(int(result.total_questions))
            total_elapsed.append(float(result.total_elapsed))
            print(
                f"{method},u={display_utility_id},questions={result.total_questions},"
                f"elapsed={result.total_elapsed:.6f}",
                flush=True,
            )
        summaries.append(summarize(
            method,
            "",
            "",
            algorithm_rows,
            total_questions,
            total_elapsed,
            None,
            None,
            None,
            len(utilities),
            len(datasets),
            getattr(args, "skyline_total_elapsed", 0.0),
            "",
            "",
            "",
        ))
    return rows, summaries


def run_sharedq(
    method: str,
    radius: float | str,
    datasets: list[np.ndarray],
    dataset_paths: list[Path],
    utilities: list[np.ndarray],
    args: argparse.Namespace,
    detail_path: Path,
) -> tuple[list[DetailRow], SummaryRow]:
    rows: list[DetailRow] = []
    total_questions: list[int] = []
    total_elapsed: list[float] = []
    detect_rounds: list[int] = []
    detect_times: list[float] = []
    divide_times: list[float] = []
    sharedq_alpha = float(args.alpha)
    sharedq_beta = float(args.beta)
    sharedq_ratio = "" if abs(sharedq_beta) <= 1e-12 else sharedq_alpha / sharedq_beta
    sharedq_rho = getattr(args, "rho", "")
    for utility_id, utility in enumerate(utilities):
        global_utility_id = int(getattr(args, "utility_id_offset", 0)) + utility_id
        display_utility_id = global_utility_id + 1
        common_kwargs = {
            "alpha": args.alpha,
            "beta": args.beta,
            "max_questions": args.max_questions,
            "max_pairs_per_dataset": args.max_pairs_per_dataset,
            "random_state": args.seed + global_utility_id * 1009,
        }
        result = run_detect_divide(
            datasets,
            utility,
            radius=float(radius),
            radius_growth=args.radius_growth,
            detect_round_limit=args.detect_round_limit,
            divide_round_limit=args.divide_round_limit,
            random_initial_center=getattr(args, "sharedq_random_initial_u", getattr(args, "dd_random_initial_u", False)),
            boundary_center_strategy=getattr(args, "sharedq_boundary_center", getattr(args, "dd_boundary_center", "ray_midpoint")),
            detect_workers=getattr(args, "sharedq_detect_workers", getattr(args, "dd_detect_workers", 4)),
            verbose=getattr(args, "verbose_sharedq", getattr(args, "verbose_dd", False)),
            verbose_detect_points=getattr(args, "sharedq_verbose_detect_points", getattr(args, "dd_verbose_detect_points", False)),
            detect_point_timeout_seconds=getattr(
                args,
                "sharedq_detect_point_timeout_seconds",
                getattr(args, "dd_detect_point_timeout_seconds", None),
            ),
            **common_kwargs,
        )
        utility_rounds = len(result.detect_centers)
        utility_detect_time = float(getattr(result, "timings", {}).get("detect", 0.0))
        utility_divide_time = float(getattr(result, "timings", {}).get("divide", 0.0))
        utility_rows: list[DetailRow] = []
        for item in result.dataset_results:
            data = datasets[item.dataset_id]
            utility_rows.append(
                make_row(
                    method,
                    radius,
                    sharedq_rho,
                    sharedq_alpha,
                    sharedq_beta,
                    sharedq_ratio,
                    global_utility_id,
                    item.dataset_id,
                    dataset_paths[item.dataset_id],
                    data,
                    utility,
                    int(item.point_index),
                    int(result.questions),
                    float(result.elapsed),
                    int(item.candidate_count),
                    int(item.pruned_count),
                    utility_rounds,
                    utility_detect_time,
                    utility_divide_time,
                    args.tolerance,
                )
            )
            print(
                f"{method},rho={sharedq_rho},radius={radius},alpha={sharedq_alpha},beta={sharedq_beta},"
                f"alpha_beta_ratio={sharedq_ratio},u={display_utility_id},"
                f"dataset={item.dataset_id + 1},"
                f"dataset_name={dataset_paths[item.dataset_id].name},"
                f"questions={int(result.questions)},elapsed={float(result.elapsed):.6f},"
                f"candidate_count={int(item.candidate_count)},pruned={int(item.pruned_count)},"
                f"detect_rounds={utility_rounds},"
                f"detect_time={utility_detect_time:.6f},divide_time={utility_divide_time:.6f}",
                flush=True,
            )
        append_detail_rows(detail_path, utility_rows)
        rows.extend(utility_rows)
        total_questions.append(int(result.questions))
        total_elapsed.append(float(result.elapsed))
        detect_rounds.append(int(utility_rounds))
        detect_times.append(utility_detect_time)
        divide_times.append(utility_divide_time)
        print(
            f"{method},rho={sharedq_rho},radius={radius},alpha={sharedq_alpha},beta={sharedq_beta},"
            f"alpha_beta_ratio={sharedq_ratio},u={display_utility_id},"
            f"questions={result.questions},"
            f"elapsed={result.elapsed:.6f},"
            f"detect_rounds={utility_rounds},"
            f"detect_time={utility_detect_time:.6f},divide_time={utility_divide_time:.6f}",
            flush=True,
        )
    return rows, summarize(
        method,
        radius,
        sharedq_rho,
        rows,
        total_questions,
        total_elapsed,
        detect_rounds,
        detect_times,
        divide_times,
        len(utilities),
        len(datasets),
        0.0,
        sharedq_alpha,
        sharedq_beta,
        sharedq_ratio,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run final IMD experiments: Baseline, SUR, and SharedQ.")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--utilities", default=DEFAULT_UTILITIES)
    parser.add_argument("--rhos", default="0.2,0.4,0.6")
    parser.add_argument("--algorithms", default="hdpi", help="Algorithms for baseline/SUR.")
    parser.add_argument(
        "--methods",
        default="baseline,shared_utility_range,sharedq",
        help="Comma-separated subset of baseline, shared_utility_range, sharedq.",
    )
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--utility-id-offset",
        type=int,
        default=0,
        help="Zero-based utility id offset used when this process runs a subset of utilities.",
    )
    parser.add_argument("--max-questions", type=int, default=10000)
    parser.add_argument("--max-pairs-per-dataset", type=int, default=20000)
    parser.add_argument("--radius-growth", type=float, default=2.0)
    parser.add_argument("--detect-round-limit", type=int, default=100)
    parser.add_argument("--divide-round-limit", type=int)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--alpha-beta-ratios",
        default="",
        help=(
            "Comma-separated alpha/beta ratios for SharedQ. "
            "When set, beta is kept fixed and alpha=ratio*beta."
        ),
    )
    parser.add_argument(
        "--sharedq-random-initial-u",
        "--dd-random-initial-u",
        dest="sharedq_random_initial_u",
        action="store_true",
    )
    parser.add_argument(
        "--sharedq-boundary-center",
        "--dd-boundary-center",
        dest="sharedq_boundary_center",
        choices=["range_center", "ray_midpoint"],
        default="ray_midpoint",
    )
    parser.add_argument(
        "--sharedq-detect-workers",
        "--dd-detect-workers",
        dest="sharedq_detect_workers",
        type=int,
        default=4,
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--hdpi-candidate-mode", choices=["sampling", "accurate", "skyline"], default="accurate")
    parser.add_argument("--hdpi-sample-count", type=int, default=2048)
    parser.add_argument("--hdpi-max-partition-candidates", type=int, default=180)
    parser.add_argument("--hdpi-beta", type=float, default=0.01)
    parser.add_argument(
        "--rh-strict-original-k1",
        action="store_true",
        help="Run RH with only the original find_possible_topk/check_possible_topk stop condition at k=1.",
    )
    parser.add_argument("--uh-s", type=int, default=2)
    parser.add_argument("--uh-exact-prune-limit", type=int, default=250)
    parser.add_argument("--uh-max-frame-rays", type=int)
    parser.add_argument("--no-glpk-frame", action="store_true")
    parser.add_argument(
        "--verbose-sur",
        action="store_true",
        help="Print SUR pruning and every simulated user choice immediately.",
    )
    parser.add_argument(
        "--sur-verbose-points",
        action="store_true",
        help="Print every point-level keep/prune decision during exact SUR pruning.",
    )
    parser.add_argument(
        "--sur-prune-strategy",
        choices=["neighbor", "exact", "vertex_dominance"],
        default="exact",
        help="SUR pruning strategy after a shared range is available.",
    )
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--detail-csv", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_paths = [Path(item) for item in args.datasets]
    datasets = [load_dataset(path) for path in dataset_paths]
    args.skyline_times = [load_skyline_time(path) for path in dataset_paths]
    args.skyline_total_elapsed = float(np.sum(args.skyline_times))
    dim = datasets[0].shape[1]
    utilities = [normalize_utility(u, dim) for u in parse_utility_text(args.utilities)]
    rhos = [float(item.strip()) for item in args.rhos.split(",") if item.strip()]
    alpha_beta_ratios = [
        float(item.strip())
        for item in getattr(args, "alpha_beta_ratios", "").split(",")
        if item.strip()
    ]
    if not alpha_beta_ratios:
        alpha_beta_ratios = [float(args.alpha) / float(args.beta)] if abs(float(args.beta)) > 1e-12 else [float(args.alpha)]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    args.algorithms_parsed = parse_algorithms(args.algorithms)
    summary_path = Path(args.summary_csv)
    detail_path = Path(args.detail_csv)

    summary_rows: list[SummaryRow] = []
    if "baseline" in methods:
        _, summaries = run_baseline(datasets, dataset_paths, utilities, args, detail_path)
        summary_rows.extend(summaries)
        write_rows(summary_path, summaries)
    if "shared_utility_range" in methods:
        _, summaries = run_shared_range(datasets, dataset_paths, utilities, args, detail_path)
        summary_rows.extend(summaries)
        write_rows(summary_path, summaries)
    if "sharedq" in methods or "dd" in methods:
        for rho in rhos:
            radius = radius_from_rho(rho, dim)
            for ratio in alpha_beta_ratios:
                old_alpha = args.alpha
                old_rho = getattr(args, "rho", "")
                args.alpha = float(ratio) * float(args.beta)
                args.rho = float(rho)
                _, summary = run_sharedq(
                    "sharedq",
                    radius,
                    datasets,
                    dataset_paths,
                    utilities,
                    args,
                    detail_path,
                )
                args.alpha = old_alpha
                args.rho = old_rho
                summary_rows.append(summary)
                write_rows(summary_path, [summary])

    for row in summary_rows:
        values = asdict(row)
        print(",".join(str(values[key]) for key in values), flush=True)


if __name__ == "__main__":
    main()
