from __future__ import annotations

import csv
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from baseline.experiment_final import radius_from_rho, run_sharedq, run_shared_range
from baseline.experiment_utils import load_dataset, normalize_utility
from experiment_config import CONFIG, make_args
from preprocess_skyline import process_file


ORIGINAL_ROOT = BASE_DIR / "original_datasets"
SKYLINE_ROOT = BASE_DIR / "after_skyline_datasets"
OUTPUT_ROOT = BASE_DIR / "results" / "category_gap_search"
GENERATED_ORIGINAL_ROOT = ORIGINAL_ROOT / "_category_gap_search"
GENERATED_SKYLINE_ROOT = SKYLINE_ROOT / "_category_gap_search"

DIM = 4
DATASET_COUNT = 10
UTILITY_COUNT = 10
SEED = 20260601
RHO = 0.1


@dataclass(frozen=True)
class SourceSpec:
    name: str
    csv_path: Path
    category_column: str


SOURCES = [
    SourceSpec("Amazon_single_category_4d_10", ORIGINAL_ROOT / "amz_uk_processed_data.csv", "categoryName"),
    SourceSpec("Google_single_category_4d_10", ORIGINAL_ROOT / "Google-Playstore.csv", "Category"),
    SourceSpec("Realtor_single_state_4d_10", ORIGINAL_ROOT / "realtor-data.zip.csv", "state"),
]


def parse_float(value: str) -> float | None:
    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def parse_size_mb(value: str) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() == "varies with device":
        return None
    unit = text[-1].lower()
    number_text = text[:-1] if unit in {"k", "m", "g"} else text
    number = parse_float(number_text)
    if number is None or number <= 0:
        return None
    if unit == "k":
        return number / 1024.0
    if unit == "g":
        return number * 1024.0
    return number


def minmax_log(value: float, lo: float, hi: float) -> float:
    logged = math.log1p(value)
    if hi <= lo:
        return 1.0
    return (logged - lo) / (hi - lo)


def minmax_linear(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0
    return (value - lo) / (hi - lo)


def clean_category(value: str) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def safe_name(text: str) -> str:
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_", "&", "/"}:
            keep.append("_")
    name = "".join(keep).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name[:80] or "category"


def amazon_valid(row: dict[str, str]) -> tuple[str, tuple[float, float, float, float]] | None:
    category = clean_category(row.get("categoryName", ""))
    stars = parse_float(row.get("stars", ""))
    reviews = parse_float(row.get("reviews", ""))
    price = parse_float(row.get("price", ""))
    bought = parse_float(row.get("boughtInLastMonth", "")) or 0.0
    if category is None or stars is None or reviews is None or price is None:
        return None
    if stars <= 0 or reviews <= 0 or price <= 0 or bought < 0:
        return None
    return category, (price, stars, reviews, bought)


def google_valid(row: dict[str, str]) -> tuple[str, tuple[float, float, float, float]] | None:
    category = clean_category(row.get("Category", ""))
    rating = parse_float(row.get("Rating", ""))
    rating_count = parse_float(row.get("Rating Count", ""))
    installs = parse_float(row.get("Maximum Installs", ""))
    size_mb = parse_size_mb(row.get("Size", ""))
    if category is None or rating is None or rating_count is None or installs is None or size_mb is None:
        return None
    if rating <= 0 or rating_count <= 0 or installs <= 0 or size_mb <= 0:
        return None
    return category, (rating, rating_count, installs, size_mb)


def realtor_valid(row: dict[str, str]) -> tuple[str, tuple[float, float, float, float]] | None:
    state = clean_category(row.get("state", ""))
    price = parse_float(row.get("price", ""))
    bath = parse_float(row.get("bath", ""))
    acre_lot = parse_float(row.get("acre_lot", ""))
    house_size = parse_float(row.get("house_size", ""))
    if state is None or price is None or bath is None or acre_lot is None or house_size is None:
        return None
    if price <= 0 or bath <= 0 or acre_lot <= 0 or house_size <= 0:
        return None
    return state, (price, house_size, acre_lot, bath)


def valid_parser(source_name: str):
    if source_name.startswith("Amazon"):
        return amazon_valid
    if source_name.startswith("Google"):
        return google_valid
    if source_name.startswith("Realtor"):
        return realtor_valid
    raise ValueError(source_name)


def first_pass(spec: SourceSpec) -> tuple[Counter[str], list[tuple[float, float]]]:
    parser = valid_parser(spec.name)
    counts: Counter[str] = Counter()
    mins = [float("inf")] * DIM
    maxs = [float("-inf")] * DIM

    with spec.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = parser(row)
            if parsed is None:
                continue
            category, values = parsed
            counts[category] += 1
            for i, value in enumerate(values):
                target = math.log1p(value) if i in log_dimensions(spec.name) else value
                mins[i] = min(mins[i], target)
                maxs[i] = max(maxs[i], target)
    return counts, list(zip(mins, maxs))


def log_dimensions(source_name: str) -> set[int]:
    if source_name.startswith("Amazon"):
        return {0, 2, 3}
    if source_name.startswith("Google"):
        return {1, 2, 3}
    if source_name.startswith("Realtor"):
        return {0, 1, 2}
    return set()


def transform(source_name: str, values: tuple[float, float, float, float], ranges: list[tuple[float, float]]) -> tuple[float, ...]:
    if source_name.startswith("Amazon"):
        price, stars, reviews, bought = values
        return (
            1.0 - minmax_log(price, *ranges[0]),
            stars / 5.0,
            minmax_log(reviews, *ranges[2]),
            minmax_log(bought, *ranges[3]),
        )
    if source_name.startswith("Google"):
        rating, rating_count, installs, size_mb = values
        return (
            rating / 5.0,
            minmax_log(rating_count, *ranges[1]),
            minmax_log(installs, *ranges[2]),
            1.0 - minmax_log(size_mb, *ranges[3]),
        )
    if source_name.startswith("Realtor"):
        price, house_size, acre_lot, bath = values
        return (
            1.0 - minmax_log(price, *ranges[0]),
            minmax_log(house_size, *ranges[1]),
            minmax_log(acre_lot, *ranges[2]),
            minmax_linear(bath, *ranges[3]),
        )
    raise ValueError(source_name)


def write_dataset(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(points)} {points.shape[1]}\n")
        np.savetxt(handle, points, fmt="%.12g")


def generate_source(spec: SourceSpec) -> tuple[Path, Path, list[str]]:
    counts, ranges = first_pass(spec)
    selected = [category for category, _count in counts.most_common(DATASET_COUNT)]
    selected_set = set(selected)
    parser = valid_parser(spec.name)
    points_by_category: dict[str, dict[tuple[float, ...], None]] = defaultdict(dict)

    with spec.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = parser(row)
            if parsed is None:
                continue
            category, values = parsed
            if category not in selected_set:
                continue
            point = transform(spec.name, values, ranges)
            points_by_category[category][tuple(round(x, 12) for x in point)] = None

    original_dir = GENERATED_ORIGINAL_ROOT / spec.name
    skyline_dir = GENERATED_SKYLINE_ROOT / spec.name
    if original_dir.exists():
        shutil.rmtree(original_dir)
    if skyline_dir.exists():
        shutil.rmtree(skyline_dir)
    original_dir.mkdir(parents=True, exist_ok=True)
    skyline_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for index, category in enumerate(selected, start=1):
        points = np.asarray(list(points_by_category[category].keys()), dtype=float)
        name = f"{index:02d}_{safe_name(category)}.txt"
        path = original_dir / name
        write_dataset(path, points)
        skyline_result = process_file(
            path,
            original_dir,
            skyline_dir,
            overwrite=True,
            mode="serial",
            chunk_workers=1,
            chunks_per_worker=1,
            min_block=128,
            max_block=4096,
        )
        summary_rows.append(
            {
                "dataset": name,
                "category": category,
                "valid_rows": counts[category],
                "unique_points": len(points),
                "skyline_points": skyline_result.get("skyline_points", ""),
                "skyline_elapsed": skyline_result.get("elapsed", ""),
            }
        )
        print(
            f"generated {spec.name} {index}/{DATASET_COUNT} category={category} "
            f"unique={len(points)} skyline={skyline_result.get('skyline_points')}",
            flush=True,
        )

    with (original_dir / "dataset_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    shutil.copyfile(original_dir / "dataset_summary.csv", skyline_dir / "dataset_summary.csv")
    return original_dir, skyline_dir, selected


def build_utilities() -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [normalize_utility(rng.dirichlet(np.ones(DIM)), DIM) for _ in range(UTILITY_COUNT)]


def run_source(spec: SourceSpec, original_dir: Path, skyline_dir: Path) -> dict[str, float | str]:
    output_dir = OUTPUT_ROOT / spec.name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_paths = sorted(original_dir.glob("*.txt"))
    skyline_paths = [skyline_dir / path.name for path in original_paths]
    original_datasets = [load_dataset(path) for path in original_paths]
    skyline_datasets = [load_dataset(path) for path in skyline_paths]
    utilities = build_utilities()

    args = make_args(CONFIG)
    args.algorithms_parsed = ["HD-PI"]
    args.seed = SEED
    args.hdpi_candidate_mode = "accurate"
    args.hdpi_beta = 0.01
    args.sharedq_detect_workers = 1
    args.verbose_sur = False
    args.sur_verbose_points = False
    args.rho = RHO
    args.alpha = 1.0
    args.beta = 1.0
    args.skyline_total_elapsed = 0.0

    print(f"start SUR-HDPI {spec.name}", flush=True)
    start = perf_counter()
    _sur_rows, sur_summaries = run_shared_range(
        skyline_datasets,
        skyline_paths,
        utilities,
        args,
        output_dir / "sur_detail.csv",
    )
    sur_wall = perf_counter() - start
    sur_summary = sur_summaries[0]

    radius = radius_from_rho(RHO, DIM)
    print(f"start SharedQ {spec.name} radius={radius}", flush=True)
    start = perf_counter()
    _sharedq_rows, sharedq_summary = run_sharedq(
        "sharedq",
        radius,
        original_datasets,
        original_paths,
        utilities,
        args,
        output_dir / "sharedq_detail.csv",
    )
    sharedq_wall = perf_counter() - start

    result = {
        "source": spec.name,
        "sur_questions": float(sur_summary.avg_total_questions_per_utility),
        "sharedq_questions": float(sharedq_summary.avg_total_questions_per_utility),
        "question_gap": float(sur_summary.avg_total_questions_per_utility - sharedq_summary.avg_total_questions_per_utility),
        "sur_elapsed": float(sur_summary.avg_total_elapsed_per_utility),
        "sharedq_elapsed": float(sharedq_summary.avg_total_elapsed_per_utility),
        "sur_wall": sur_wall,
        "sharedq_wall": sharedq_wall,
    }
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []
    for spec in SOURCES:
        print(f"=== {spec.name} ===", flush=True)
        original_dir, skyline_dir, _selected = generate_source(spec)
        result = run_source(spec, original_dir, skyline_dir)
        all_results.append(result)
        print(f"result {result}", flush=True)

    best = max(all_results, key=lambda row: float(row["question_gap"]))
    summary_path = OUTPUT_ROOT / "all_sources_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"summary={summary_path}", flush=True)
    print(f"best={best}", flush=True)


if __name__ == "__main__":
    main()
