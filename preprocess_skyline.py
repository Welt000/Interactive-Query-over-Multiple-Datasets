from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np


EPS = 1e-9
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class SkylinePreprocessConfig:
    dataset_dir: str = r"./original_datasets/Realtor_5d_10"
    output_dir: str = r"./after_skyline_datasets/Realtor_5d_10"
    recursive: bool = False
    overwrite: bool = True
    mode: str = "files"
    file_workers: int = 0
    chunk_workers: int = 0
    chunks_per_worker: int = 2
    min_block: int = 128
    max_block: int = 4096
    limit: int | None = None


CONFIG = SkylinePreprocessConfig()


def _resolve_path(raw: str | Path, base_dir: Path = BASE_DIR) -> Path:
    path = Path(str(raw).strip().strip('"'))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _split_numbers(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split()]


def load_dataset(path: Path) -> np.ndarray:
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
    if len(first) == 2 and all(abs(x - round(x)) < EPS for x in first):
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
            return np.asarray(body, dtype=float)

    return np.asarray(rows, dtype=float)


def read_skyline_time(path: Path) -> float:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(_split_numbers(line))
            if len(rows) >= 2:
                break
    if len(rows) >= 2 and len(rows[0]) == 2 and len(rows[1]) >= 1:
        return float(rows[1][0])
    return 0.0


def write_dataset(path: Path, points: np.ndarray, skyline_elapsed: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(points)} {points.shape[1]}\n")
        handle.write(f"{skyline_elapsed:.12g}\n")
        np.savetxt(handle, points, fmt="%.12g")


def skyline_indices_serial(
    points: np.ndarray,
    min_block: int = 128,
    max_block: int = 4096,
) -> list[int]:
    """Exact skyline indices under larger-is-better dominance."""
    data = np.asarray(points, dtype=float)
    if data.ndim != 2:
        raise ValueError("points must be a 2-D array")
    n = len(data)
    if n == 0:
        return []

    sums = np.sum(data, axis=1)
    order = np.argsort(-sums, kind="mergesort")
    skyline: list[int] = []
    dim = data.shape[1]
    min_block = max(1, min(int(min_block), n))
    max_block = max(min_block, min(int(max_block), n))
    capacity = min_block
    skyline_points = np.empty((capacity, dim), dtype=float)
    skyline_sums = np.empty(capacity, dtype=float)
    skyline_count = 0
    frozen_blocks: list[np.ndarray] = []
    frozen_sums: list[np.ndarray] = []

    def dominated_by(
        block: np.ndarray,
        block_sums: np.ndarray,
        point: np.ndarray,
        point_sum: float,
    ) -> bool:
        ge_all = np.all(block >= point - EPS, axis=1)
        strict = block_sums > point_sum + EPS
        return bool(np.any(ge_all & strict))

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

        if dominated:
            continue

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

        skyline.append(idx)
        skyline_points[skyline_count] = point
        skyline_sums[skyline_count] = point_sum
        skyline_count += 1

    return sorted(skyline)


def _chunk_skyline_worker(args: tuple[np.ndarray, np.ndarray, int, int]) -> np.ndarray:
    indices, points, min_block, max_block = args
    keep_local = skyline_indices_serial(
        points,
        min_block=min_block,
        max_block=max_block,
    )
    return indices[np.asarray(keep_local, dtype=int)]


def skyline_indices_chunk_parallel(
    points: np.ndarray,
    chunk_workers: int,
    chunks_per_worker: int,
    min_block: int,
    max_block: int,
) -> list[int]:
    data = np.asarray(points, dtype=float)
    n = len(data)
    if n == 0:
        return []
    if chunk_workers <= 1:
        return skyline_indices_serial(data, min_block=min_block, max_block=max_block)

    chunk_count = max(chunk_workers * max(1, chunks_per_worker), chunk_workers)
    chunk_count = min(chunk_count, n)
    index_chunks = [
        chunk for chunk in np.array_split(np.arange(n, dtype=int), chunk_count)
        if len(chunk) > 0
    ]
    jobs = [(indices, data[indices], min_block, max_block) for indices in index_chunks]

    local_keeps: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=chunk_workers) as pool:
        for keep in pool.map(_chunk_skyline_worker, jobs):
            if len(keep) > 0:
                local_keeps.append(keep)

    if not local_keeps:
        return []

    candidates = np.unique(np.concatenate(local_keeps).astype(int))
    final_local = skyline_indices_serial(
        data[candidates],
        min_block=min_block,
        max_block=max_block,
    )
    return sorted(int(candidates[i]) for i in final_local)


def looks_like_dataset(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                values = _split_numbers(line)
                return len(values) >= 2 and all(np.isfinite(values))
    except Exception:
        return False
    return False


def iter_dataset_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*.txt") if recursive else input_dir.glob("*.txt")
    return sorted(
        path
        for path in iterator
        if "after_skyline" not in path.parts
        and looks_like_dataset(path)
    )


def output_path_for(path: Path, input_dir: Path, output_dir: Path) -> Path:
    try:
        relative = path.relative_to(input_dir)
    except ValueError:
        relative = Path(path.name)
    return output_dir / relative


def process_file(
    path: Path,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    mode: str,
    chunk_workers: int,
    chunks_per_worker: int,
    min_block: int,
    max_block: int,
) -> dict:
    out_path = output_path_for(path, input_dir, output_dir)
    if out_path.exists() and not overwrite:
        skyline = load_dataset(out_path)
        return {
            "status": "skip",
            "input": str(path),
            "output": str(out_path),
            "before": None,
            "after": len(skyline),
            "elapsed": read_skyline_time(out_path),
        }

    data = load_dataset(path)
    skyline_start = perf_counter()
    if mode == "chunks":
        keep = skyline_indices_chunk_parallel(
            data,
            chunk_workers=chunk_workers,
            chunks_per_worker=chunks_per_worker,
            min_block=min_block,
            max_block=max_block,
        )
    else:
        keep = skyline_indices_serial(data, min_block=min_block, max_block=max_block)
    elapsed = perf_counter() - skyline_start
    skyline = data[keep]
    write_dataset(out_path, skyline, elapsed)
    return {
        "status": "done",
        "input": str(path),
        "output": str(out_path),
        "before": len(data),
        "after": len(skyline),
        "elapsed": elapsed,
    }


def _file_worker(
    args: tuple[Path, Path, Path, bool, int, int],
) -> dict:
    path, input_dir, output_dir, overwrite, min_block, max_block = args
    return process_file(
        path,
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=overwrite,
        mode="serial",
        chunk_workers=1,
        chunks_per_worker=1,
        min_block=min_block,
        max_block=max_block,
    )


def print_result(result: dict, current: int, total: int) -> None:
    before = "existing" if result["before"] is None else str(result["before"])
    print(
        f"{result['status']} [{current}/{total}] "
        f"{result['input']} -> {result['output']} "
        f"{before}->{result['after']} elapsed={result['elapsed']:.3f}s",
        flush=True,
    )


def build_parser(config: SkylinePreprocessConfig = CONFIG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact skyline preprocessing for txt datasets.")
    parser.add_argument("--input-dir", default=config.dataset_dir)
    parser.add_argument("--output-dir", default=config.output_dir)
    parser.add_argument("--recursive", action="store_true", default=config.recursive)
    parser.add_argument("--overwrite", action="store_true", default=config.overwrite)
    parser.add_argument("--mode", choices=["files", "chunks", "serial"], default=config.mode)
    parser.add_argument("--file-workers", type=int, default=config.file_workers)
    parser.add_argument("--chunk-workers", type=int, default=config.chunk_workers)
    parser.add_argument("--chunks-per-worker", type=int, default=config.chunks_per_worker)
    parser.add_argument("--min-block", "--compare-block", dest="min_block", type=int, default=config.min_block)
    parser.add_argument("--max-block", type=int, default=config.max_block)
    parser.add_argument("--limit", type=int, default=config.limit)
    return parser


def main() -> None:
    args = build_parser(CONFIG).parse_args()
    input_dir = _resolve_path(args.input_dir)
    output_dir = _resolve_path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    files = iter_dataset_files(input_dir, recursive=args.recursive)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"no dataset txt files found under {input_dir}", flush=True)
        return

    cpu_count = os.cpu_count() or 1
    if args.mode == "files":
        file_workers = args.file_workers if args.file_workers > 0 else min(len(files), cpu_count)
        jobs = [
            (
                path,
                input_dir,
                output_dir,
                args.overwrite,
                args.min_block,
                args.max_block,
            )
            for path in files
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=file_workers) as pool:
            future_to_path = {pool.submit(_file_worker, job): job[0] for job in jobs}
            for future in as_completed(future_to_path):
                completed += 1
                print_result(future.result(), completed, len(files))
        return

    chunk_workers = args.chunk_workers if args.chunk_workers > 0 else max(cpu_count - 1, 1)
    for current, path in enumerate(files, start=1):
        result = process_file(
            path,
            input_dir=input_dir,
            output_dir=output_dir,
            overwrite=args.overwrite,
            mode=args.mode,
            chunk_workers=chunk_workers,
            chunks_per_worker=args.chunks_per_worker,
            min_block=args.min_block,
            max_block=args.max_block,
        )
        print_result(result, current, len(files))


if __name__ == "__main__":
    main()
