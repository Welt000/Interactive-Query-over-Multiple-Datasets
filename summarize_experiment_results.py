from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"
DEFAULT_SYNTHETIC_SUMMARY_DIR = DEFAULT_RESULTS_ROOT / "summaries"
DEFAULT_REAL_SUMMARY_DIR = DEFAULT_RESULTS_ROOT / "real_summaries"

QUESTION_COL = "avg_total_questions_per_utility"
TIME_COL = "avg_total_elapsed_per_utility"

METHOD_ORDER = [
    "baseline_HD-PI",
    "baseline_RH",
    "baseline_UH-Simplex",
    "baseline_UH-Random",
    "shared_utility_range_HD-PI",
    "shared_utility_range_RH",
    "shared_utility_range_UH-Simplex",
    "shared_utility_range_UH-Random",
    "sharedq",
]

METHOD_ALIASES = {
    "sharedq": "sharedq",
    "SharedQ": "sharedq",
}

REAL_FAMILY_PREFIXES = [
    "Google_Playstore",
    "Amazon",
    "Realtor",
    "Airbnb",
    "NBA",
    "Used_Car",
    "Vehicles",
]


@dataclass(frozen=True)
class ResultDir:
    name: str
    path: Path
    summary_path: Path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], first_col: str, methods: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[first_col, *methods])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path}", flush=True)


def parse_float(value: object) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        result = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def is_ok_row(row: dict[str, str]) -> bool:
    status = str(row.get("status", "ok")).strip().lower()
    return status in {"", "ok"}


def sorted_methods(methods: Iterable[str]) -> list[str]:
    unique = list(dict.fromkeys(METHOD_ALIASES.get(method, method) for method in methods))
    known = [method for method in METHOD_ORDER if method in unique]
    extra = sorted(method for method in unique if method not in METHOD_ORDER)
    return known + extra


def choose_row(rows: list[dict[str, str]], method: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if METHOD_ALIASES.get(str(row.get("method", "")), str(row.get("method", ""))) == method
        and is_ok_row(row)
    ]
    if not candidates:
        return None
    if method == "sharedq":
        preferred = [
            row
            for row in candidates
            if (parse_float(row.get("rho")) in {None, 0.1})
            and (parse_float(row.get("alpha_beta_ratio")) in {None, 1.0})
        ]
        if preferred:
            return preferred[-1]
    return candidates[-1]


def metric_table(
    records: list[tuple[str, Path]],
    first_col: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    collected_methods: list[str] = []
    question_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []

    for label, summary_path in records:
        rows = read_csv_rows(summary_path)
        methods = sorted_methods(row.get("method", "") for row in rows if row.get("method"))
        collected_methods.extend(methods)

        question_row: dict[str, object] = {first_col: label}
        time_row: dict[str, object] = {first_col: label}
        for method in methods:
            chosen = choose_row(rows, method)
            if not chosen:
                continue
            questions = parse_float(chosen.get(QUESTION_COL))
            elapsed = parse_float(chosen.get(TIME_COL))
            if questions is not None:
                question_row[method] = questions
            if elapsed is not None:
                time_row[method] = elapsed
        question_rows.append(question_row)
        time_rows.append(time_row)

    methods = sorted_methods(collected_methods)
    for row in [*question_rows, *time_rows]:
        for method in methods:
            row.setdefault(method, "")
    return question_rows, time_rows, methods


def collect_result_dirs(results_root: Path) -> list[ResultDir]:
    found: dict[str, ResultDir] = {}
    summary_paths = sorted(
        results_root.rglob("experiment_summary.csv"),
        key=lambda path: (len(path.relative_to(results_root).parts), str(path).lower()),
    )
    for summary_path in summary_paths:
        parent = summary_path.parent
        parts = set(parent.relative_to(results_root).parts)
        if parts.intersection({"summaries", "real_summaries", "figures", "parameters_summary"}):
            continue
        if any(part.startswith("_") for part in parts):
            continue
        found.setdefault(parent.name, ResultDir(parent.name, parent, summary_path))
    return list(found.values())


def synthetic_kind(name: str) -> tuple[str, object] | None:
    match = re.fullmatch(r"(\d+)d_100k_10_result", name)
    if match:
        dim = int(match.group(1))
        return "dimension", (dim, f"{dim}d")

    match = re.fullmatch(r"4d_100k_(\d+)_result", name)
    if match:
        count = int(match.group(1))
        return "dataset_count", (count, str(count))

    match = re.fullmatch(r"4d_([^_]+)_10_result", name)
    if match and "ratio" not in name:
        size = match.group(1)
        return "dataset_size", (size_key(size), size)

    match = re.fullmatch(r"4d_100k_ratio_(\d+)_(\d+)_(\d+)_result", name)
    if match:
        ratio = ":".join(match.groups())
        return "data_type_ratio", (tuple(int(x) for x in match.groups()), ratio)

    return None


def size_key(label: str) -> float:
    text = label.lower()
    if text.endswith("k"):
        return float(text[:-1]) * 1_000
    if text.endswith("w"):
        return float(text[:-1]) * 10_000
    if text.endswith("m"):
        return float(text[:-1]) * 1_000_000
    return float(text)


def real_family_and_dim(name: str) -> tuple[str, int] | None:
    for family in REAL_FAMILY_PREFIXES:
        if name.startswith(family):
            match = re.search(r"_(\d+)d(?:_|_?result|_?results)", name)
            if match:
                return family, int(match.group(1))
    return None


def summarize_parameter_file(
    source_path: Path,
    output_prefix: Path,
    variable_col: str,
    variable_label: str,
) -> None:
    rows = [row for row in read_csv_rows(source_path) if is_ok_row(row)]
    if not rows:
        return

    methods = sorted_methods(row.get("method", "") for row in rows if row.get("method"))
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    order: list[tuple[float, str]] = []
    for row in rows:
        raw = row.get(variable_col, "")
        value = parse_float(raw)
        if value is None:
            continue
        label = f"{value:g}"
        if label not in grouped:
            grouped[label] = {}
            order.append((value, label))
        grouped[label][row.get("method", "")] = row

    question_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    for _, label in sorted(order):
        question_row: dict[str, object] = {variable_label: label}
        time_row: dict[str, object] = {variable_label: label}
        for method in methods:
            row = grouped[label].get(method)
            if not row:
                continue
            questions = parse_float(row.get(QUESTION_COL) or row.get("questions"))
            elapsed = parse_float(row.get(TIME_COL) or row.get("elapsed"))
            if questions is not None:
                question_row[method] = questions
            if elapsed is not None:
                time_row[method] = elapsed
        question_rows.append(question_row)
        time_rows.append(time_row)

    write_csv(output_prefix.with_name(output_prefix.name + "_questions.csv"), question_rows, variable_label, methods)
    write_csv(output_prefix.with_name(output_prefix.name + "_time.csv"), time_rows, variable_label, methods)


def summarize_synthetic(results_root: Path, output_dir: Path, result_dirs: list[ResultDir]) -> None:
    parameter_dir = results_root / "parameters_summary"
    rho_source = parameter_dir / "rho_summary.csv"
    radius_source = parameter_dir / "r_summary.csv"
    if rho_source.exists():
        summarize_parameter_file(rho_source, output_dir / "01_rho", "rho", "rho")
    elif radius_source.exists():
        rows = read_csv_rows(radius_source)
        variable_col = "rho" if rows and any(row.get("rho") for row in rows) else "radius"
        variable_label = variable_col
        summarize_parameter_file(radius_source, output_dir / f"01_{variable_label}", variable_col, variable_label)

    ab_source = parameter_dir / "ab_summary.csv"
    if ab_source.exists():
        summarize_parameter_file(
            ab_source,
            output_dir / "02_alpha_beta_ratio",
            "alpha_beta_ratio",
            "alpha_beta_ratio",
        )

    groups: dict[str, list[tuple[object, str, Path]]] = {
        "dataset_count": [],
        "dimension": [],
        "dataset_size": [],
        "data_type_ratio": [],
    }
    for result_dir in result_dirs:
        if real_family_and_dim(result_dir.name):
            continue
        name = result_dir.name

        match = re.fullmatch(r"(\d+)d_100k_10_result", name)
        if match:
            dim = int(match.group(1))
            groups["dimension"].append((dim, f"{dim}d", result_dir.summary_path))

        match = re.fullmatch(r"4d_100k_(\d+)_result", name)
        if match:
            count = int(match.group(1))
            groups["dataset_count"].append((count, str(count), result_dir.summary_path))

        match = re.fullmatch(r"4d_([^_]+)_10_result", name)
        if match and "ratio" not in name:
            size = match.group(1)
            groups["dataset_size"].append((size_key(size), size, result_dir.summary_path))

        match = re.fullmatch(r"4d_100k_ratio_(\d+)_(\d+)_(\d+)_result", name)
        if match:
            ratio = ":".join(match.groups())
            groups["data_type_ratio"].append(
                (tuple(int(x) for x in match.groups()), ratio, result_dir.summary_path)
            )

    specs = [
        ("03_dataset_count", "dataset_count", "dataset_count"),
        ("04_dimension", "dimension", "dimension"),
        ("05_dataset_size", "dataset_size", "dataset_size"),
        ("06_data_type_ratio", "data_type_ratio", "type_ratio"),
    ]
    for file_prefix, group_name, first_col in specs:
        records = [(label, path) for _sort, label, path in sorted(groups[group_name], key=lambda item: item[0])]
        if not records:
            continue
        question_rows, time_rows, methods = metric_table(records, first_col)
        write_csv(output_dir / f"{file_prefix}_questions.csv", question_rows, first_col, methods)
        write_csv(output_dir / f"{file_prefix}_time.csv", time_rows, first_col, methods)


def summarize_real(output_dir: Path, result_dirs: list[ResultDir]) -> None:
    groups: dict[str, list[tuple[int, str, Path]]] = {}
    for result_dir in result_dirs:
        family_dim = real_family_and_dim(result_dir.name)
        if not family_dim:
            continue
        family, dim = family_dim
        groups.setdefault(family, []).append((dim, f"{dim}d", result_dir.summary_path))

    for family, records_raw in sorted(groups.items()):
        records = [(label, path) for _dim, label, path in sorted(records_raw)]
        question_rows, time_rows, methods = metric_table(records, "dimension")
        write_csv(output_dir / f"{family}_questions.csv", question_rows, "dimension", methods)
        write_csv(output_dir / f"{family}_time.csv", time_rows, "dimension", methods)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment_summary.csv files.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--summary-dir", default=str(DEFAULT_SYNTHETIC_SUMMARY_DIR))
    parser.add_argument("--real-summary-dir", default=str(DEFAULT_REAL_SUMMARY_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    summary_dir = Path(args.summary_dir).resolve()
    real_summary_dir = Path(args.real_summary_dir).resolve()

    result_dirs = collect_result_dirs(results_root)
    summarize_synthetic(results_root, summary_dir, result_dirs)
    summarize_real(real_summary_dir, result_dirs)
    print(f"done result_dirs={len(result_dirs)}", flush=True)


if __name__ == "__main__":
    main()
