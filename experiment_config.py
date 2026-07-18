from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import runpy
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from baseline.experiment_final import (
    DEFAULT_UTILITIES,
    SummaryRow,
    radius_from_rho,
    run_baseline,
    run_sharedq,
    run_shared_range,
    write_rows,
)
from baseline.experiment_utils import (
    load_dataset,
    load_skyline_time,
    normalize_utility,
    parse_algorithms,
    parse_utility_text,
)
from preprocess_skyline import (
    iter_dataset_files as iter_skyline_dataset_files,
    process_file as process_skyline_file,
)

SHAREDQ_METHOD = "sharedq"
SHAREDQ_METHOD_NAMES = {"dd", "sharedq"}


def canonical_method(method: str) -> str:
    method_lower = method.strip().lower()
    return SHAREDQ_METHOD if method_lower in SHAREDQ_METHOD_NAMES else method_lower


@dataclass
class ExperimentConfig:
    # Dataset batch script. A .txt file uses one dataset folder per line.
    # A .py file may define DATASET_DIRS = ["4d_100k_10", ...].
    dataset_script: str = r"./experiment_datasets.py"
    original_dataset_root: str = r"./original_datasets"
    after_skyline_root: str = r"./after_skyline_datasets"
    results_root: str = r"./results"

    # These three fields are filled automatically for one dataset folder.
    source_dataset_dir: str = ""
    skyline_dataset_dir: str = ""
    output_dir: str = ""
    dimension: int | None = None

    # Skyline preprocessing.
    preprocess_skyline: bool = True
    preprocess_overwrite: bool = True
    preprocess_recursive: bool = False
    preprocess_mode: str = "files"  # files, chunks, serial
    preprocess_file_workers: int = 0
    preprocess_chunk_workers: int = 0
    preprocess_chunks_per_worker: int = 2
    skyline_min_block: int = 128
    skyline_max_block: int = 4096

    # Dataset order. Default: correlated -> independent -> anti-correlated.
    # Set shuffle_datasets=True to randomize the order with dataset_shuffle_seed.
    shuffle_datasets: bool = False
    dataset_shuffle_seed: int = 20260606

    # Experiments: choose from baseline, SUR, SharedQ.
    methods: tuple[str, ...] = ("Baseline", "SUR", "SharedQ")
    algorithms: tuple[str, ...] = ("HD-PI", "RH", "UH-Simplex", "UH-Random")
    # Utility vectors. If random_utility_count > 0, utilities_txt is ignored.
    utilities_txt: str = DEFAULT_UTILITIES
    random_utility_count: int = 10
    seed: int = 20260601

    # Timeout / parallel parameters.
    timeout_seconds: int = 3600
    parallel_u_workers: int = 5
    sur_resample_timeouts: bool = True
    sur_max_replacement_attempts_multiplier: int = 5

    # SharedQ parameters.
    # SharedQ radius is computed from rho and dimension by radius_from_rho().
    rhos: tuple[float, ...] = (0.1,)
    alpha: float = 1
    beta: float = 1.0
    # For SharedQ parameter sweep: beta is fixed, alpha = ratio * beta.
    alpha_beta_ratios: tuple[float, ...] = (1,)
    radius_growth: float = 2.0
    detect_round_limit: int = 100
    divide_round_limit: int | None = None
    sharedq_random_initial_u: bool = True
    sharedq_boundary_center: str = "ray_midpoint"
    # Fair comparison default: parallelize across utilities only.
    # Increase this only when explicitly evaluating SharedQ's internal parallelism.
    sharedq_detect_workers: int = 1
    verbose_sharedq: bool = False
    sharedq_verbose_detect_points: bool = False
    sharedq_detect_point_timeout_seconds: float | None = 180.0

    # Shared utility range parameters.
    sur_prune_strategy: str = "vertex_dominance"
    verbose_sur: bool = False
    sur_verbose_points: bool = False

    # Algorithm parameters.
    max_questions: int = 10000
    max_pairs_per_dataset: int = 20000
    tolerance: float = 1e-9
    hdpi_candidate_mode: str = "accurate"
    hdpi_sample_count: int = 2048
    hdpi_max_partition_candidates: int = 180
    hdpi_beta: float = 0.01
    rh_strict_original_k1: bool = False
    uh_s: int = 2
    uh_exact_prune_limit: int = 250
    uh_max_frame_rays: int | None = None
    no_glpk_frame: bool = False


CONFIG = ExperimentConfig()


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw.strip().strip('"'))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _natural_path_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _dataset_type_rank(path: Path) -> int:
    tokens = {
        token
        for token in re.split(r"[^a-zA-Z0-9]+", path.stem.lower())
        if token
    }
    if tokens.intersection({"anti", "anticorrelated", "anti-correlated"}):
        return 2
    if tokens.intersection({"corr", "cor", "correlated"}):
        return 0
    if tokens.intersection({"ind", "independent"}):
        return 1
    return 3


def dataset_default_order_key(path: Path) -> tuple[int, tuple[object, ...]]:
    return _dataset_type_rank(path), _natural_path_key(path)


def _is_metadata_txt(path: Path) -> bool:
    stem = path.stem.strip().lower()
    return stem in {"readme", "metadata"} or stem.startswith("readme_")


def read_dataset_paths(dataset_dir: str | Path) -> list[Path]:
    dir_path = Path(dataset_dir)
    if not dir_path.is_absolute():
        dir_path = (CODE_DIR / dir_path).resolve()

    if not dir_path.exists():
        raise FileNotFoundError(f"dataset directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"dataset path is not a directory: {dir_path}")

    paths = sorted(
        (path for path in dir_path.glob("*.txt") if not _is_metadata_txt(path)),
        key=dataset_default_order_key,
    )

    if not paths:
        raise ValueError(f"dataset directory has no txt files: {dir_path}")

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing dataset files:\n" + "\n".join(missing))

    return [path.resolve() for path in paths]


def shuffle_dataset_paths(dataset_paths: list[Path], seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset_paths))
    return [dataset_paths[i] for i in indices]


def dataset_dir_for_method(config: ExperimentConfig, method: str) -> str:
    method_lower = canonical_method(method)
    if method_lower == SHAREDQ_METHOD:
        if not config.source_dataset_dir:
            raise ValueError("source_dataset_dir is not set for SharedQ")
        return config.source_dataset_dir
    if method_lower in {"baseline", "sur", "shared_utility_range"}:
        if not config.skyline_dataset_dir:
            raise ValueError("skyline_dataset_dir is not set for Baseline/SUR")
        return config.skyline_dataset_dir
    raise ValueError(f"unknown method for dataset dir: {method}")


def experiment_dimension(config: ExperimentConfig) -> int:
    if config.dimension is None:
        raise ValueError("experiment dimension has not been inferred yet")
    return int(config.dimension)


def get_dataset_paths(config: ExperimentConfig, method: str) -> list[Path]:
    dataset_paths = read_dataset_paths(dataset_dir_for_method(config, method))
    if config.shuffle_datasets:
        dataset_paths = shuffle_dataset_paths(
            dataset_paths,
            seed=config.dataset_shuffle_seed,
        )
    return dataset_paths


def load_datasets_checked(config: ExperimentConfig, method: str) -> tuple[list[Path], list[np.ndarray]]:
    dataset_paths = get_dataset_paths(config, method)
    datasets = [load_dataset(path) for path in dataset_paths]
    if not datasets:
        raise ValueError("no datasets loaded")

    method_lower = canonical_method(method)
    source_dim = int(datasets[0].shape[1])
    for dataset_id, data in enumerate(datasets):
        if data.shape[1] != source_dim:
            raise ValueError(
                f"dataset {dataset_id} has dimension {data.shape[1]}, "
                f"expected {source_dim}: {dataset_paths[dataset_id]}"
            )

    datasets = [data.copy() for data in datasets]
    inferred_dim = source_dim
    if config.dimension is not None and int(config.dimension) != inferred_dim:
        raise ValueError(
            f"configured dimension={config.dimension}, but loaded dimension={inferred_dim}"
        )
    return dataset_paths, datasets


def build_utilities(config: ExperimentConfig) -> list[np.ndarray]:
    dim = experiment_dimension(config)
    if config.random_utility_count > 0:
        rng = np.random.default_rng(config.seed)
        return [
            normalize_utility(rng.dirichlet(np.ones(dim)), dim)
            for _ in range(config.random_utility_count)
        ]
    return [
        normalize_utility(utility, dim)
        for utility in parse_utility_text(config.utilities_txt)
    ]


def sample_replacement_utilities(
    config: ExperimentConfig,
    start_index: int,
    count: int,
) -> list[np.ndarray]:
    if count <= 0:
        return []
    dim = experiment_dimension(config)
    rng = np.random.default_rng(config.seed)
    draws = [
        normalize_utility(rng.dirichlet(np.ones(dim)), dim)
        for _ in range(start_index + count)
    ]
    return draws[start_index:]


def make_args(config: ExperimentConfig) -> argparse.Namespace:
    algorithms_text = ",".join(config.algorithms)
    return argparse.Namespace(
        algorithms_parsed=parse_algorithms(algorithms_text),
        seed=config.seed,
        utility_id_offset=0,
        max_questions=config.max_questions,
        max_pairs_per_dataset=config.max_pairs_per_dataset,
        radius_growth=config.radius_growth,
        detect_round_limit=config.detect_round_limit,
        divide_round_limit=config.divide_round_limit,
        alpha=config.alpha,
        beta=config.beta,
        sharedq_random_initial_u=config.sharedq_random_initial_u,
        sharedq_boundary_center=config.sharedq_boundary_center,
        sharedq_detect_workers=config.sharedq_detect_workers,
        verbose_sharedq=config.verbose_sharedq,
        sharedq_verbose_detect_points=config.sharedq_verbose_detect_points,
        sharedq_detect_point_timeout_seconds=config.sharedq_detect_point_timeout_seconds,
        tolerance=config.tolerance,
        hdpi_candidate_mode=config.hdpi_candidate_mode,
        hdpi_sample_count=config.hdpi_sample_count,
        hdpi_max_partition_candidates=config.hdpi_max_partition_candidates,
        hdpi_beta=config.hdpi_beta,
        rh_strict_original_k1=config.rh_strict_original_k1,
        uh_s=config.uh_s,
        uh_exact_prune_limit=config.uh_exact_prune_limit,
        uh_max_frame_rays=config.uh_max_frame_rays,
        no_glpk_frame=config.no_glpk_frame,
        verbose_sur=config.verbose_sur,
        sur_verbose_points=config.sur_verbose_points,
        sur_prune_strategy=config.sur_prune_strategy,
    )


def make_sharedq_args(config: ExperimentConfig) -> argparse.Namespace:
    """
    Build args for SharedQ without binding it to HD-PI/RH/UH-* algorithms.

    run_sharedq does not use algorithms_parsed, but make_args requires a valid
    algorithm name because Baseline/SUR share the same argparse namespace.
    Therefore, we keep a valid placeholder internally and expose SharedQ as its
    own independent method in logs / output filenames.
    """
    placeholder_algorithm = config.algorithms[0] if config.algorithms else "HD-PI"
    return make_args(replace(config, algorithms=(placeholder_algorithm,)))


def write_config(
    output_dir: Path,
    config: ExperimentConfig,
    dataset_paths: list[Path],
    utilities: list[np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["dataset_paths"] = [str(path) for path in dataset_paths]
    payload["utilities"] = [[float(x) for x in utility] for utility in utilities]
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _safe_name(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def append_csv_file(src_path: Path, dst_path: Path) -> None:
    """
    Merge a temporary detail csv into the global detail csv.
    If the destination already exists, skip the temporary file header.
    """
    if not src_path.exists():
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with src_path.open("r", encoding="utf-8") as src:
        lines = src.readlines()

    if not lines:
        return

    if dst_path.exists():
        lines = lines[1:]

    if not lines:
        return

    with dst_path.open("a", encoding="utf-8") as dst:
        dst.writelines(lines)


def append_summary_row(summary_path: Path, row: Any) -> None:
    """
    Append only the newest aggregated summary row.
    If the file does not exist, write the header first.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    row_dict = asdict(row)
    need_header = not summary_path.exists()

    with summary_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dict.keys()))
        if need_header:
            writer.writeheader()
        writer.writerow(row_dict)


def append_sharedq_param_metric_rows(output_dir: Path, rows: list[Any]) -> None:
    """
    Write one compact SharedQ metric row per (rho, alpha/beta) parameter setting.
    This makes SharedQ parameter sweeps easy to inspect without de-duplicating
    dataset-level detail rows.
    """
    if not rows:
        return

    metric_path = output_dir / "sharedq_param_metrics.csv"
    need_header = not metric_path.exists()
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    with metric_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "rho",
                "radius",
                "alpha",
                "beta",
                "alpha_beta_ratio",
                "status",
                "utilities",
                "questions",
                "elapsed",
                "detect_rounds",
                "detect_time",
                "divide_time",
            ],
        )
        if need_header:
            writer.writeheader()
        for row in rows:
            values = asdict(row)
            if canonical_method(str(values.get("method", ""))) != SHAREDQ_METHOD:
                continue
            writer.writerow(
                {
                    "method": values.get("method", ""),
                    "rho": values.get("rho", ""),
                    "radius": values.get("radius", ""),
                    "alpha": values.get("alpha", ""),
                    "beta": values.get("beta", ""),
                    "alpha_beta_ratio": values.get("alpha_beta_ratio", ""),
                    "status": values.get("status", "ok"),
                    "utilities": values.get("utilities", ""),
                    "questions": values.get("avg_total_questions_per_utility", ""),
                    "elapsed": values.get("avg_total_elapsed_per_utility", ""),
                    "detect_rounds": values.get("avg_detect_rounds", ""),
                    "detect_time": values.get("avg_detect_time_per_utility", ""),
                    "divide_time": values.get("avg_divide_time_per_utility", ""),
                }
            )


def write_timeout_marker(
    output_dir: Path,
    method: str,
    algorithm: str,
    utility_index: int,
    rho: float | None,
    radius: float | None,
    alpha_beta_ratio: float | None,
    timeout_seconds: int,
) -> None:
    """
    Record timeout information in the output folder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timeout_csv = output_dir / "timeout_records.csv"
    need_header = not timeout_csv.exists()

    with timeout_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if need_header:
            writer.writerow(
                [
                    "method",
                    "algorithm",
                    "utility_index",
                    "rho",
                    "radius",
                    "alpha_beta_ratio",
                    "timeout_seconds",
                    "status",
                ]
            )
        writer.writerow(
            [
                method,
                algorithm,
                utility_index,
                "" if rho is None else rho,
                "" if radius is None else radius,
                "" if alpha_beta_ratio is None else alpha_beta_ratio,
                timeout_seconds,
                "timeout",
            ]
        )

    marker_name = (
        f"timeout_{_safe_name(method)}_{_safe_name(algorithm)}"
        f"_u{utility_index}"
    )
    if radius is not None:
        marker_name += f"_r{radius}"
    if rho is not None:
        marker_name += f"_rho{rho}"
    if alpha_beta_ratio is not None:
        marker_name += f"_ab{alpha_beta_ratio}"
    marker_name += ".txt"

    marker_path = output_dir / marker_name
    with marker_path.open("w", encoding="utf-8") as handle:
        handle.write("status=timeout\n")
        handle.write(f"method={method}\n")
        handle.write(f"algorithm={algorithm}\n")
        handle.write(f"utility_index={utility_index}\n")
        handle.write(f"rho={rho}\n")
        handle.write(f"radius={radius}\n")
        handle.write(f"alpha_beta_ratio={alpha_beta_ratio}\n")
        handle.write(f"timeout_seconds={timeout_seconds}\n")


def run_one_experiment_job(
    config: ExperimentConfig,
    method: str,
    algorithm: str,
    utility_index: int,
    utility: np.ndarray,
    rho: float | None,
    radius: float | None,
    alpha_beta_ratio: float | None,
    detail_tmp_path: Path,
    queue: mp.Queue,
) -> None:
    """
    Run one minimal experiment unit in a subprocess:
    method x algorithm x utility.
    """
    try:
        utilities = [utility]
        method_lower = method.strip().lower()
        dataset_paths, datasets = load_datasets_checked(config, method_lower)
        if method_lower == SHAREDQ_METHOD:
            skyline_total_elapsed = 0.0
        else:
            skyline_total_elapsed = float(np.sum([load_skyline_time(path) for path in dataset_paths]))

        if method_lower == "baseline":
            single_config = replace(config, algorithms=(algorithm,))
            args = make_args(single_config)
            args.utility_id_offset = int(utility_index)
            args.skyline_total_elapsed = skyline_total_elapsed

            _, summaries = run_baseline(
                datasets,
                dataset_paths,
                utilities,
                args,
                detail_tmp_path,
            )
            queue.put(("ok", summaries))

        elif method_lower in {"sur", "shared_utility_range"}:
            single_config = replace(config, algorithms=(algorithm,))
            args = make_args(single_config)
            args.utility_id_offset = int(utility_index)
            args.skyline_total_elapsed = skyline_total_elapsed

            _, summaries = run_shared_range(
                datasets,
                dataset_paths,
                utilities,
                args,
                detail_tmp_path,
            )
            queue.put(("ok", summaries))

        elif method_lower == SHAREDQ_METHOD:
            if radius is None:
                raise ValueError("SharedQ method requires radius")

            args = make_sharedq_args(config)
            args.utility_id_offset = int(utility_index)
            args.rho = "" if rho is None else float(rho)
            if alpha_beta_ratio is not None:
                args.alpha = float(alpha_beta_ratio) * float(args.beta)

            _, summary = run_sharedq(
                SHAREDQ_METHOD,
                radius,
                datasets,
                dataset_paths,
                utilities,
                args,
                detail_tmp_path,
            )
            queue.put(("ok", [summary]))

        else:
            raise ValueError(f"unknown method: {method}")

    except TimeoutError:
        queue.put(("timeout", traceback.format_exc()))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def run_u_batch_with_timeout(
    config: ExperimentConfig,
    method: str,
    algorithm: str,
    jobs: list[tuple[int, np.ndarray, float | None, float | None, float | None]],
    output_dir: Path,
    detail_path: Path,
) -> tuple[str, list[Any]]:
    """
    Run a batch of utilities in parallel.

    jobs item format:
        (utility_index, utility, rho, radius, alpha_beta_ratio)

    Returns:
        ("ok", summaries)
        ("timeout", summaries_finished_before_timeout)
        ("error", error_messages)
    """
    tmp_dir = output_dir / "_tmp_detail"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    processes: list[mp.Process] = []
    queues: list[mp.Queue] = []
    job_infos: list[tuple[int, np.ndarray, float | None, float | None, float | None, Path]] = []

    for utility_index, utility, rho, radius, alpha_beta_ratio in jobs:
        rho_part = "" if rho is None else f"_rho{rho}"
        radius_part = "" if radius is None else f"_r{radius}"
        ratio_part = "" if alpha_beta_ratio is None else f"_ab{alpha_beta_ratio}"
        detail_tmp_path = tmp_dir / (
            f"detail_{_safe_name(method)}_{_safe_name(algorithm)}"
            f"_u{utility_index}{rho_part}{radius_part}{ratio_part}.csv"
        )

        if detail_tmp_path.exists():
            detail_tmp_path.unlink()

        queue: mp.Queue = mp.Queue()
        process = mp.Process(
            target=run_one_experiment_job,
            args=(
                config,
                method,
                algorithm,
                utility_index,
                utility,
                rho,
                radius,
                alpha_beta_ratio,
                detail_tmp_path,
                queue,
            ),
        )
        process.start()

        processes.append(process)
        queues.append(queue)
        job_infos.append((utility_index, utility, rho, radius, alpha_beta_ratio, detail_tmp_path))

    start_time = time.time()
    timed_out_indices: list[int] = []

    while True:
        if all(not process.is_alive() for process in processes):
            break

        elapsed = time.time() - start_time
        if elapsed > config.timeout_seconds:
            timed_out_indices = [
                i for i, process in enumerate(processes) if process.is_alive()
            ]

            for i in timed_out_indices:
                processes[i].terminate()

            for i in timed_out_indices:
                processes[i].join()

            for i in timed_out_indices:
                utility_index, _utility, rho, radius, alpha_beta_ratio, _detail_tmp_path = job_infos[i]
                write_timeout_marker(
                    output_dir=output_dir,
                    method=method,
                    algorithm=algorithm,
                    utility_index=utility_index,
                    rho=rho,
                    radius=radius,
                    alpha_beta_ratio=alpha_beta_ratio,
                    timeout_seconds=config.timeout_seconds,
                )

            break

        time.sleep(1)

    for process in processes:
        if process.is_alive():
            process.join()

    timeout_happened = bool(timed_out_indices)
    worker_reported_timeout = False
    error_happened = False
    error_messages: list[str] = []
    summaries: list[Any] = []

    timed_out_index_set = set(timed_out_indices)

    for i, (queue, (utility_index, _utility, rho, radius, alpha_beta_ratio, detail_tmp_path)) in enumerate(
        zip(queues, job_infos)
    ):
        if i in timed_out_index_set:
            continue

        if queue.empty():
            error_happened = True
            error_messages.append(
                f"subprocess ended without returning result: "
                f"method={method}, algorithm={algorithm}, u={utility_index}, "
                f"rho={rho}, radius={radius}, alpha_beta_ratio={alpha_beta_ratio}"
            )
            continue

        status, payload = queue.get()

        if status == "ok":
            summaries.extend(payload)
            append_csv_file(detail_tmp_path, detail_path)
        elif status == "timeout":
            worker_reported_timeout = True
            write_timeout_marker(
                output_dir=output_dir,
                method=method,
                algorithm=algorithm,
                utility_index=utility_index,
                rho=rho,
                radius=radius,
                alpha_beta_ratio=alpha_beta_ratio,
                timeout_seconds=config.timeout_seconds,
            )
            print(
                f"TIMEOUT: method={method}, algorithm={algorithm}, "
                f"u={utility_index}, rho={rho}, radius={radius}, "
                f"alpha_beta_ratio={alpha_beta_ratio}. "
                f"Subprocess reported a detect timeout.",
                flush=True,
            )
        elif status == "error":
            error_happened = True
            error_messages.append(str(payload))
        else:
            error_happened = True
            error_messages.append(
                f"unknown status={status}, method={method}, algorithm={algorithm}, "
                f"u={utility_index}, rho={rho}, radius={radius}, alpha_beta_ratio={alpha_beta_ratio}"
            )

    if timeout_happened or worker_reported_timeout:
        return "timeout", summaries
    if error_happened:
        return "error", error_messages
    return "ok", summaries


def chunked(items: list[Any], chunk_size: int):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def aggregate_algorithm_summary(rows: list[Any]) -> Any | None:
    """
    Aggregate all utility-level summaries of the same method + algorithm into one row.

    Numeric fields whose names contain question / time / runtime / elapsed / seconds /
    duration are averaged. Other fields are copied from the first row.
    """
    if not rows:
        return None

    first = rows[0]
    values_list = [asdict(row) for row in rows]
    result = dict(values_list[0])

    average_keywords = (
        "question",
        "time",
        "runtime",
        "elapsed",
        "seconds",
        "duration",
    )

    for key in list(result.keys()):
        key_lower = key.lower()
        should_average = any(keyword in key_lower for keyword in average_keywords)
        if not should_average:
            continue

        column_values = []
        for values in values_list:
            value = values.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                column_values.append(float(value))

        if column_values:
            result[key] = float(np.mean(column_values))

    for key in list(result.keys()):
        key_lower = key.lower()
        if key_lower in {"utility", "utility_id", "utility_index", "u", "u_index"}:
            result[key] = "all"
        elif key_lower in {"utilities", "utility_count"}:
            result[key] = len(rows)
        elif key_lower == "runs":
            numeric_runs = [
                int(values[key])
                for values in values_list
                if isinstance(values.get(key), (int, float, np.integer, np.floating))
            ]
            if numeric_runs:
                result[key] = int(np.sum(numeric_runs))

    return type(first)(**result)


def aggregate_algorithm_summaries_by_sharedq_params(rows: list[Any]) -> list[Any]:
    """
    Keep SharedQ summaries separated by rho and alpha/beta ratio so parameter
    sweeps preserve one question/time row per configuration.
    """
    if not rows:
        return []

    grouped: dict[tuple[Any, Any], list[Any]] = {}
    for row in rows:
        values = asdict(row)
        rho = values.get("rho", "")
        radius = values.get("radius", "")
        ratio = values.get("alpha_beta_ratio", "")
        grouped.setdefault((rho, radius, ratio), []).append(row)

    result: list[Any] = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), str(item[1]))):
        summary = aggregate_algorithm_summary(grouped[key])
        if summary is not None:
            result.append(summary)
    return result


def make_sharedq_parameter_status_summary(
    rho: float,
    radius: float,
    alpha_beta_ratio: float,
    beta: float,
    status: str,
    completed_rows: list[Any],
    dataset_count: int,
) -> SummaryRow:
    """
    Build a final summary row for one SharedQ parameter setting when it did not
    finish normally. If some utility summaries are already available, keep
    their partial averages and only replace the status field.
    """
    if completed_rows:
        partial_summary = aggregate_algorithm_summary(completed_rows)
        if partial_summary is not None:
            return replace(partial_summary, status=status)

    alpha = float(alpha_beta_ratio) * float(beta)
    return SummaryRow(
        method=SHAREDQ_METHOD,
        radius=radius,
        rho=rho,
        alpha=alpha,
        beta=beta,
        alpha_beta_ratio=alpha_beta_ratio,
        status=status,
        utilities=0,
        datasets=dataset_count,
        runs=0,
        avg_total_questions_per_utility="",
        avg_total_elapsed_per_utility="",
        accuracy="",
        avg_regret="",
        avg_candidate_count="",
        avg_pruned_count="",
        avg_detect_rounds="",
        avg_detect_time_per_utility="",
        avg_divide_time_per_utility="",
    )


def _read_dataset_script_value(script_path: Path) -> list[str]:
    if script_path.suffix.lower() == ".py":
        payload = runpy.run_path(str(script_path))
        for key in ("DATASET_DIRS", "DATASETS", "dataset_dirs", "datasets"):
            value = payload.get(key)
            if value is not None:
                return [str(item) for item in value]
        raise ValueError(
            f"python dataset script must define DATASET_DIRS or DATASETS: {script_path}"
        )

    names: list[str] = []
    with script_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    if not names:
        raise ValueError(f"dataset script has no active dataset folders: {script_path}")
    return names


def read_dataset_dirs_from_script(config: ExperimentConfig) -> list[Path]:
    script_path = _resolve_path(config.dataset_script, CODE_DIR)
    if not script_path.exists():
        raise FileNotFoundError(f"dataset script not found: {script_path}")

    original_root = _resolve_path(config.original_dataset_root, CODE_DIR)
    dirs: list[Path] = []
    for raw in _read_dataset_script_value(script_path):
        candidate = Path(str(raw).strip().strip('"'))
        if not candidate.is_absolute():
            direct = (script_path.parent / candidate).resolve()
            rooted = (original_root / candidate).resolve()
            candidate = direct if direct.exists() else rooted
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"dataset folder from script not found: {candidate}")
        dirs.append(candidate)

    seen: set[str] = set()
    unique_dirs: list[Path] = []
    for path in dirs:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_dirs.append(path)
    return unique_dirs


def _preprocess_file_job(
    job: tuple[
        Path,
        Path,
        Path,
        bool,
        str,
        int,
        int,
        int,
        int,
        int,
    ],
) -> dict:
    (
        path,
        input_dir,
        output_dir,
        overwrite,
        mode,
        chunk_workers,
        chunks_per_worker,
        min_block,
        max_block,
        _ordinal,
    ) = job
    return process_skyline_file(
        path,
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=overwrite,
        mode=mode,
        chunk_workers=chunk_workers,
        chunks_per_worker=chunks_per_worker,
        min_block=min_block,
        max_block=max_block,
    )


def preprocess_dataset_folder(
    source_dir: Path,
    skyline_dir: Path,
    config: ExperimentConfig,
) -> None:
    if not config.preprocess_skyline:
        return

    files = iter_skyline_dataset_files(source_dir, recursive=config.preprocess_recursive)
    if not files:
        raise ValueError(f"no dataset txt files found for skyline preprocessing: {source_dir}")

    skyline_dir.mkdir(parents=True, exist_ok=True)
    mode = config.preprocess_mode.strip().lower()
    if mode not in {"files", "chunks", "serial"}:
        raise ValueError(f"unknown preprocess_mode: {config.preprocess_mode}")

    print(
        f"preprocess skyline start dataset={source_dir.name}, files={len(files)}, "
        f"mode={mode}",
        flush=True,
    )

    cpu_count = os.cpu_count() or 1
    if mode == "files":
        workers = (
            int(config.preprocess_file_workers)
            if int(config.preprocess_file_workers) > 0
            else min(len(files), cpu_count)
        )
        jobs = [
            (
                path,
                source_dir,
                skyline_dir,
                bool(config.preprocess_overwrite),
                "serial",
                1,
                1,
                int(config.skyline_min_block),
                int(config.skyline_max_block),
                ordinal,
            )
            for ordinal, path in enumerate(files, start=1)
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_preprocess_file_job, job) for job in jobs]
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                print(
                    f"skyline done [{completed}/{len(files)}] "
                    f"{Path(result['input']).name} {result['before']}->{result['after']} "
                    f"elapsed={float(result['elapsed']):.6f}",
                    flush=True,
                )
        return

    chunk_workers = (
        int(config.preprocess_chunk_workers)
        if int(config.preprocess_chunk_workers) > 0
        else max(cpu_count - 1, 1)
    )
    for ordinal, path in enumerate(files, start=1):
        result = _preprocess_file_job(
            (
                path,
                source_dir,
                skyline_dir,
                bool(config.preprocess_overwrite),
                mode,
                chunk_workers,
                int(config.preprocess_chunks_per_worker),
                int(config.skyline_min_block),
                int(config.skyline_max_block),
                ordinal,
            )
        )
        print(
            f"skyline done [{ordinal}/{len(files)}] "
            f"{Path(result['input']).name} {result['before']}->{result['after']} "
            f"elapsed={float(result['elapsed']):.6f}",
            flush=True,
        )


def infer_experiment_dimension(
    source_dir: Path,
    config: ExperimentConfig,
) -> int:
    paths = read_dataset_paths(source_dir)
    data = load_dataset(paths[0])
    return int(data.shape[1])


def config_for_dataset_folder(
    base_config: ExperimentConfig,
    source_dir: Path,
) -> ExperimentConfig:
    skyline_root = _resolve_path(base_config.after_skyline_root, CODE_DIR)
    results_root = _resolve_path(base_config.results_root, CODE_DIR)
    dataset_name = source_dir.name
    skyline_dir = skyline_root / dataset_name
    output_dir = results_root / f"{dataset_name}_result"
    dimension = infer_experiment_dimension(source_dir, base_config)

    return replace(
        base_config,
        source_dataset_dir=str(source_dir),
        skyline_dataset_dir=str(skyline_dir),
        output_dir=str(output_dir),
        dimension=dimension,
    )


def run_algorithm_all_utilities(
    config: ExperimentConfig,
    method: str,
    algorithm: str,
    utilities: list[np.ndarray],
    output_dir: Path,
    detail_path: Path,
    dataset_count: int,
) -> tuple[bool, list[Any]]:
    """
    Run all utilities for one method + algorithm.

    Returns:
        (algorithm_timeout_or_error, summary_rows_from_finished_utilities)
    """
    method_lower = method.strip().lower()
    algorithm_summary_rows: list[Any] = []

    if method_lower == SHAREDQ_METHOD:
        alpha_beta_ratios = tuple(config.alpha_beta_ratios) or (
            (float(config.alpha) / float(config.beta))
            if abs(float(config.beta)) > 1e-12
            else float(config.alpha),
        )
        for rho in config.rhos:
            radius = radius_from_rho(float(rho), experiment_dimension(config))
            for alpha_beta_ratio in alpha_beta_ratios:
                parameter_rows: list[Any] = []
                parameter_status = "ok"
                utility_jobs = [
                    (utility_index, utility, float(rho), radius, float(alpha_beta_ratio))
                    for utility_index, utility in enumerate(utilities)
                ]

                for batch in chunked(utility_jobs, config.parallel_u_workers):
                    print(
                        f"running batch method={method}, algorithm={algorithm}, "
                        f"rho={rho}, radius={radius}, alpha_beta_ratio={alpha_beta_ratio}, "
                        f"u={[job[0] for job in batch]}",
                        flush=True,
                    )

                    status, payload = run_u_batch_with_timeout(
                        config=config,
                        method=method,
                        algorithm=algorithm,
                        jobs=batch,
                        output_dir=output_dir,
                        detail_path=detail_path,
                    )

                    if status == "ok":
                        parameter_rows.extend(payload)
                        print(
                            f"done batch method={method}, algorithm={algorithm}, "
                            f"rho={rho}, radius={radius}, alpha_beta_ratio={alpha_beta_ratio}, "
                            f"u={[job[0] for job in batch]}",
                            flush=True,
                        )
                    elif status == "timeout":
                        parameter_rows.extend(payload)
                        parameter_status = "timeout"
                        print(
                            f"TIMEOUT: method={method}, algorithm={algorithm}, "
                            f"rho={rho}, radius={radius}, alpha_beta_ratio={alpha_beta_ratio}, "
                            f"batch_u={[job[0] for job in batch]}. "
                            f"Record this parameter status and continue.",
                            flush=True,
                        )
                        break
                    else:
                        parameter_status = "error"
                        print(
                            f"ERROR: method={method}, algorithm={algorithm}, "
                            f"rho={rho}, radius={radius}, alpha_beta_ratio={alpha_beta_ratio}, "
                            f"batch_u={[job[0] for job in batch]}",
                            flush=True,
                        )
                        for message in payload:
                            print(message, flush=True)
                        break

                if parameter_status == "ok":
                    parameter_summary = aggregate_algorithm_summary(parameter_rows)
                    if parameter_summary is not None:
                        algorithm_summary_rows.append(replace(parameter_summary, status="ok"))
                else:
                    algorithm_summary_rows.append(
                        make_sharedq_parameter_status_summary(
                            rho=float(rho),
                            radius=float(radius),
                            alpha_beta_ratio=float(alpha_beta_ratio),
                            beta=float(config.beta),
                            status=parameter_status,
                            completed_rows=parameter_rows,
                            dataset_count=dataset_count,
                        )
                    )

    elif method_lower in {"sur", "shared_utility_range"} and config.sur_resample_timeouts:
        target_success_count = len(utilities)
        completed_rows: list[Any] = []
        timeout_count = 0
        replacement_attempts = 0
        max_replacement_attempts = max(
            target_success_count,
            target_success_count * max(1, int(config.sur_max_replacement_attempts_multiplier)),
        )

        def run_single_sur_job(
            utility_index: int,
            utility: np.ndarray,
            *,
            replacement: bool,
        ) -> tuple[str, list[Any]]:
            label = "replacement" if replacement else "initial"
            print(
                f"running {label} method={method}, algorithm={algorithm}, u={utility_index}",
                flush=True,
            )
            status, payload = run_u_batch_with_timeout(
                config=config,
                method=method,
                algorithm=algorithm,
                jobs=[(utility_index, utility, None, None, None)],
                output_dir=output_dir,
                detail_path=detail_path,
            )
            if status == "ok":
                print(
                    f"done {label} method={method}, algorithm={algorithm}, u={utility_index}",
                    flush=True,
                )
            elif status == "timeout":
                print(
                    f"TIMEOUT: {label} method={method}, algorithm={algorithm}, "
                    f"u={utility_index}. Continue with next utility.",
                    flush=True,
                )
            else:
                print(
                    f"ERROR: {label} method={method}, algorithm={algorithm}, u={utility_index}",
                    flush=True,
                )
                for message in payload:
                    print(message, flush=True)
            return status, payload

        for utility_index, utility in enumerate(utilities):
            status, payload = run_single_sur_job(
                utility_index,
                utility,
                replacement=False,
            )
            if status == "ok":
                completed_rows.extend(payload)
            elif status == "timeout":
                timeout_count += 1
            else:
                return True, completed_rows

        next_replacement_index = len(utilities)
        while timeout_count > 0 and replacement_attempts < max_replacement_attempts:
            replacement_utility = sample_replacement_utilities(
                config,
                next_replacement_index,
                1,
            )[0]
            replacement_utility_index = next_replacement_index
            next_replacement_index += 1
            replacement_attempts += 1
            status, payload = run_single_sur_job(
                replacement_utility_index,
                replacement_utility,
                replacement=True,
            )
            if status == "ok":
                completed_rows.extend(payload)
                timeout_count -= 1
            elif status == "timeout":
                print(
                    f"replacement timeout remains method={method}, algorithm={algorithm}, "
                    f"remaining={timeout_count}",
                    flush=True,
                )
            else:
                return True, completed_rows

        if timeout_count > 0:
            print(
                f"TIMEOUT: method={method}, algorithm={algorithm}, "
                f"unfilled_replacements={timeout_count}. Mark algorithm timeout.",
                flush=True,
            )
            if completed_rows:
                algorithm_summary = aggregate_algorithm_summary(completed_rows)
                if algorithm_summary is not None:
                    algorithm_summary_rows.append(replace(algorithm_summary, status="timeout"))
            return True, algorithm_summary_rows

        algorithm_summary = aggregate_algorithm_summary(completed_rows)
        if algorithm_summary is not None:
            algorithm_summary_rows.append(replace(algorithm_summary, status="ok"))

    else:
        utility_jobs = [
            (utility_index, utility, None, None, None)
            for utility_index, utility in enumerate(utilities)
        ]

        for batch in chunked(utility_jobs, config.parallel_u_workers):
            print(
                f"running batch method={method}, algorithm={algorithm}, "
                f"u={[job[0] for job in batch]}",
                flush=True,
            )

            status, payload = run_u_batch_with_timeout(
                config=config,
                method=method,
                algorithm=algorithm,
                jobs=batch,
                output_dir=output_dir,
                detail_path=detail_path,
            )

            if status == "ok":
                algorithm_summary_rows.extend(payload)
                print(
                    f"done batch method={method}, algorithm={algorithm}, "
                    f"u={[job[0] for job in batch]}",
                    flush=True,
                )
            elif status == "timeout":
                algorithm_summary_rows.extend(payload)
                print(
                    f"TIMEOUT: method={method}, algorithm={algorithm}, "
                    f"batch_u={[job[0] for job in batch]}. "
                    f"Skip this algorithm under this method.",
                    flush=True,
                )
                return True, algorithm_summary_rows
            else:
                print(
                    f"ERROR: method={method}, algorithm={algorithm}, "
                    f"batch_u={[job[0] for job in batch]}",
                    flush=True,
                )
                for message in payload:
                    print(message, flush=True)
                return True, algorithm_summary_rows

    return False, algorithm_summary_rows


def run_one_dataset_folder(config: ExperimentConfig) -> None:
    # Required on Windows when using multiprocessing.
    mp.freeze_support()

    source_dir = _resolve_path(config.source_dataset_dir, CODE_DIR)
    skyline_dir = _resolve_path(config.skyline_dataset_dir, CODE_DIR)
    preprocess_dataset_folder(source_dir, skyline_dir, config)

    output_dir = _resolve_path(config.output_dir, CODE_DIR)
    summary_path = output_dir / "experiment_summary.csv"
    detail_path = output_dir / "experiment_detail.csv"
    sharedq_param_metrics_path = output_dir / "sharedq_param_metrics.csv"

    utilities = build_utilities(config)
    methods = tuple(config.methods)
    algorithms = tuple(config.algorithms)
    method_dataset_counts: dict[str, int] = {}
    config_dataset_paths: list[Path] = []
    for method in methods:
        method_lower = canonical_method(method)
        if method_lower not in {"baseline", "sur", "shared_utility_range", SHAREDQ_METHOD}:
            continue
        paths = get_dataset_paths(config, method_lower)
        method_dataset_counts[method_lower] = len(paths)
        config_dataset_paths.extend(paths)

    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        summary_path.unlink()
    if detail_path.exists():
        detail_path.unlink()
    if sharedq_param_metrics_path.exists():
        sharedq_param_metrics_path.unlink()

    tmp_dir = output_dir / "_tmp_detail"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    seen_paths: set[str] = set()
    unique_config_dataset_paths: list[Path] = []
    for path in config_dataset_paths:
        key = str(path)
        if key not in seen_paths:
            seen_paths.add(key)
            unique_config_dataset_paths.append(path)

    write_config(output_dir, config, unique_config_dataset_paths, utilities)

    summary_rows: list[Any] = []

    for method in methods:
        method_lower = canonical_method(method)
        if method_lower not in {"baseline", "sur", "shared_utility_range", SHAREDQ_METHOD}:
            print(f"skip unknown method={method}", flush=True)
            continue

        algorithms_for_method = ("SharedQ",) if method_lower == SHAREDQ_METHOD else algorithms

        for algorithm in algorithms_for_method:
            print(f"start method={method}, algorithm={algorithm}", flush=True)

            algorithm_failed, algorithm_summary_rows = run_algorithm_all_utilities(
                config=config,
                method=method,
                algorithm=algorithm,
                utilities=utilities,
                output_dir=output_dir,
                detail_path=detail_path,
                dataset_count=method_dataset_counts.get(
                    method_lower,
                    len(unique_config_dataset_paths),
                ),
            )

            if not algorithm_failed:
                if method_lower == SHAREDQ_METHOD:
                    aggregated_summaries = algorithm_summary_rows
                else:
                    aggregated_summary = aggregate_algorithm_summary(algorithm_summary_rows)
                    aggregated_summaries = (
                        [] if aggregated_summary is None else [aggregated_summary]
                    )

                for aggregated_summary in aggregated_summaries:
                    append_summary_row(summary_path, aggregated_summary)
                    summary_rows.append(aggregated_summary)

                    values = asdict(aggregated_summary)
                    print(
                        "latest_summary="
                        + ",".join(str(values[key]) for key in values),
                        flush=True,
                    )
                if method_lower == SHAREDQ_METHOD:
                    append_sharedq_param_metric_rows(output_dir, aggregated_summaries)

                print(
                    f"summary updated after method={method}, algorithm={algorithm}",
                    flush=True,
                )
            else:
                if algorithm_summary_rows:
                    for row in algorithm_summary_rows:
                        append_summary_row(summary_path, row)
                        summary_rows.append(row)
                        values = asdict(row)
                        print(
                            "latest_summary="
                            + ",".join(str(values[key]) for key in values),
                            flush=True,
                        )
                    if method_lower == SHAREDQ_METHOD:
                        append_sharedq_param_metric_rows(output_dir, algorithm_summary_rows)
                if algorithm_summary_rows:
                    print(
                        f"summary updated with timeout/error status: "
                        f"method={method}, algorithm={algorithm}",
                        flush=True,
                    )
                else:
                    print(
                        f"summary skipped because timeout/error occurred: "
                        f"method={method}, algorithm={algorithm}",
                        flush=True,
                    )

    print(f"summary_csv={summary_path}", flush=True)
    print(f"detail_csv={detail_path}", flush=True)
    print(f"sharedq_param_metrics_csv={sharedq_param_metrics_path}", flush=True)
    print(f"timeout_csv={output_dir / 'timeout_records.csv'}", flush=True)


def main(config: ExperimentConfig = CONFIG) -> None:
    # Required on Windows when using multiprocessing.
    mp.freeze_support()

    dataset_dirs = read_dataset_dirs_from_script(config)
    print(f"dataset_script={_resolve_path(config.dataset_script, CODE_DIR)}", flush=True)
    print(f"dataset_count={len(dataset_dirs)}", flush=True)

    for index, source_dir in enumerate(dataset_dirs, start=1):
        dataset_config = config_for_dataset_folder(config, source_dir)
        print(
            f"start dataset_folder [{index}/{len(dataset_dirs)}] "
            f"name={source_dir.name}, dimension={dataset_config.dimension}, "
            f"source={dataset_config.source_dataset_dir}, "
            f"skyline={dataset_config.skyline_dataset_dir}, "
            f"output={dataset_config.output_dir}",
            flush=True,
        )
        run_one_dataset_folder(dataset_config)
        print(
            f"done dataset_folder [{index}/{len(dataset_dirs)}] "
            f"name={source_dir.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
