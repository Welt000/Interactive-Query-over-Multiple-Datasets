"""Profile one shared utility on ten datasets: HDPI vs PCA-SharedQ."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import SharedQ.algorithm as sharedq_module
from HD_PI.algorithm import FastRegion, HDPIFast
from SharedQ.algorithm import SharedQuestions
from baseline.experiment_utils import load_dataset
from experiment_config import dataset_default_order_key


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--utility-id', type=int, default=0)
    args = parser.parse_args()
    output = Path(os.environ.get(
        'SHAREDQ_DIAG_OUT', str(ROOT / 'results' / 'diagnostics' / 'hdpi_vs_pca_u0')
    )).resolve()
    source = ROOT / 'results' / '4d_100k_10_result'
    config = json.loads((source / 'config.json').read_text(encoding='utf-8-sig'))
    paths = sorted((ROOT / 'original_datasets' / '4d_100k_10').glob('*.txt'),
                   key=dataset_default_order_key)
    if len(paths) != 10:
        raise RuntimeError(f'expected 10 datasets, found {len(paths)}')
    data = [load_dataset(path) for path in paths]
    utility = np.asarray(config['utilities'][args.utility_id], dtype=float)
    seed = int(config['seed']) + args.utility_id * 1009
    params = config
    result = {
        'utility_id': args.utility_id,
        'datasets': [path.name for path in paths],
        'dataset_shapes': [list(points.shape) for points in data],
        'utility': utility.tolist(),
        'timeout_seconds': 3600,
        'sharedq': None,
        'hdpi_per_dataset': [],
    }

    # Measure shared initialization, PCA question selection, range pruning,
    # and the HDPI-style region clipping separately.
    sq_stats = defaultdict(float)
    sq_counts = defaultdict(int)
    phase = {'name': 'sharedq'}
    original_refine = FastRegion.refine_vertices
    original_filter = sharedq_module.filter_points_by_shared_range

    def timed_refine(region, *args, **kwargs):
        start = time.perf_counter()
        answer = original_refine(region, *args, **kwargs)
        sq_stats['refine_seconds'] += time.perf_counter() - start
        sq_counts['refine_calls'] += 1
        return answer

    def timed_filter(*args, **kwargs):
        start = time.perf_counter()
        answer = original_filter(*args, **kwargs)
        sq_stats['filter_seconds'] += time.perf_counter() - start
        sq_counts['filter_calls'] += 1
        return answer

    FastRegion.refine_vertices = timed_refine
    sharedq_module.filter_points_by_shared_range = timed_filter
    try:
        algorithm = SharedQuestions(
            alpha=float(params['alpha']), beta=float(params['beta']),
            max_questions=int(params['max_questions']),
            max_pairs_per_dataset=int(params['max_pairs_per_dataset']),
            random_state=seed, hyperplane_reduction=True,
            partition_construction='hdpi', detect_workers=1,
        )

        def wrap(name, timer_key, count_key=None, result_counter=None):
            original = getattr(algorithm, name)

            def measured(*args, **kwargs):
                start = time.perf_counter()
                value = original(*args, **kwargs)
                sq_stats[timer_key] += time.perf_counter() - start
                if count_key:
                    sq_counts[count_key] += 1
                if result_counter:
                    result_counter(value, args, kwargs)
                return value
            setattr(algorithm, name, measured)

        def candidates_found(value, args, kwargs):
            sq_counts['hull_candidates_total'] += len(value[0])
            sq_counts['datasets_initialized'] += 1
            sq_counts[f"hull_candidates_dataset_{sq_counts['datasets_initialized'] - 1}"] = len(value[0])

        def cache_built(value, args, kwargs):
            state, utility_range = args
            expected = algorithm._state_cache_version(state, utility_range)
            # The cache's version is updated after construction; compare its
            # previous state before the call using the pre-call marker below.
            if phase.get('cache_was_stale'):
                sq_counts['partition_regions_built'] += len(value)

        def count_pairs(value, args, kwargs):
            sq_counts['candidate_pairs_generated'] += len(value)

        def count_scored_normals(value, args, kwargs):
            normals = args[0]
            sq_counts['cross_dataset_normals_scored'] += int(len(normals))

        wrap('_convex_top1_candidates_and_adjacency', 'candidate_generation_seconds',
             'hull_calls', candidates_found)
        wrap('_ideal_hyperplane', 'pca_geometry_seconds', 'pca_geometry_calls')
        wrap('_candidate_pairs', 'pca_pair_generation_seconds', 'pca_pair_generation_calls',
             count_pairs)
        wrap('_shared_pruning_scores', 'cross_dataset_scoring_seconds', 'scoring_calls',
             count_scored_normals)
        wrap('_select_question', 'question_selection_seconds', 'question_selection_calls')
        wrap('_clip_hdpi_partition_cache', 'incremental_clip_seconds', 'incremental_clip_calls')

        original_cache = algorithm._state_region_cache

        def measured_cache(state, utility_range):
            expected = algorithm._state_cache_version(state, utility_range)
            phase['cache_was_stale'] = (
                state.local_region_cache_version != expected
                or len(state.local_regions) != len(state.candidates)
            )
            start = time.perf_counter()
            value = original_cache(state, utility_range)
            sq_stats['partition_cache_seconds'] += time.perf_counter() - start
            sq_counts['partition_cache_calls'] += 1
            cache_built(value, (state, utility_range), {})
            phase['cache_was_stale'] = False
            return value
        algorithm._state_region_cache = measured_cache

        start = time.perf_counter()
        sq_result = algorithm.search(data, utility)
        sq_wall = time.perf_counter() - start
        sq_accuracy = []
        for item in sq_result.dataset_results:
            scores = data[item.dataset_id] @ utility
            sq_accuracy.append(bool(scores[item.point_index] >= scores.max() - 1e-9))
        sq_counts['partition_constraints_initial_estimate'] = sum(
            int(sq_counts[f'hull_candidates_dataset_{i}'])
            * max(0, int(sq_counts[f'hull_candidates_dataset_{i}']) - 1)
            for i in range(10)
        )
        result['sharedq'] = {
            'status': 'ok' if all(sq_accuracy) and all(
                item.candidate_count == 1 for item in sq_result.dataset_results
            ) else 'incomplete',
            'questions': sq_result.questions,
            'search_seconds': sq_result.elapsed,
            'wall_seconds': sq_wall,
            'accuracy': float(np.mean(sq_accuracy)),
            'resolved_datasets': sum(item.candidate_count == 1 for item in sq_result.dataset_results),
            'datasets': len(sq_result.dataset_results),
            'candidate_counts': [item.candidate_count for item in sq_result.dataset_results],
            'metrics': dict(sq_stats),
            'counts': dict(sq_counts),
        }
        save(output / 'diagnostic.json', result)
    finally:
        FastRegion.refine_vertices = original_refine
        sharedq_module.filter_points_by_shared_range = original_filter

    # HDPI is intrinsically single-dataset. Run it once per dataset for the
    # same utility and aggregate its elapsed times, questions, and exact region
    # construction work. The outer one-hour process limit covers this whole u.
    hdpi_total_questions = 0
    hdpi_total_elapsed = 0.0
    for dataset_id, points in enumerate(data):
        stats = defaultdict(float)
        counts = defaultdict(int)
        algorithm = HDPIFast(
            beta=float(params['beta']),
            max_questions=int(params['max_questions']),
            candidate_mode='accurate',
            random_state=seed,
        )
        phase['name'] = f'hdpi_{dataset_id}'

        def wrap_hdpi(name, timer_key, count_key=None, counter=None):
            original = getattr(algorithm, name)

            def measured(*args, **kwargs):
                start = time.perf_counter()
                value = original(*args, **kwargs)
                stats[timer_key] += time.perf_counter() - start
                if count_key:
                    counts[count_key] += 1
                if counter:
                    counter(value, args, kwargs)
                return value
            setattr(algorithm, name, measured)

        def candidate_counter(value, args, kwargs):
            counts['candidate_count'] = len(value)

        def region_counter(value, args, kwargs):
            counts['partition_regions_built'] = len(value[0])
            counts['partition_constraints_initial_estimate'] = sum(
                max(0, len(value[1]) - 1) for _ in value[0]
            )

        def item_counter(value, args, kwargs):
            counts['choose_items_built'] = len(value)

        def modify_counter(value, args, kwargs):
            items, regions, item_idx = args[:3]
            if 0 <= item_idx < len(items):
                counts['regions_refined_after_answer'] += len(items[item_idx].intersect_case)

        def select_counter(value, args, kwargs):
            counts['questions_selected'] += 1

        wrap_hdpi('_candidate_points', 'candidate_generation_seconds', 'candidate_generation_calls', candidate_counter)
        wrap_hdpi('_construct_regions', 'partition_construction_seconds', 'partition_construction_calls', region_counter)
        wrap_hdpi('_build_choose_items', 'choose_table_build_seconds', 'choose_table_build_calls', item_counter)
        wrap_hdpi('_select_item', 'choose_item_selection_seconds', 'choose_item_selection_calls', select_counter)
        wrap_hdpi('_modify_choose_table', 'incremental_update_seconds', 'incremental_update_calls', modify_counter)

        before_refine = defaultdict(float)
        original_hdpi_refine = FastRegion.refine_vertices

        def timed_hdpi_refine(region, *args, **kwargs):
            start = time.perf_counter()
            answer = original_hdpi_refine(region, *args, **kwargs)
            stats['all_region_refine_seconds'] += time.perf_counter() - start
            counts['all_region_refine_calls'] += 1
            return answer
        FastRegion.refine_vertices = timed_hdpi_refine
        try:
            start = time.perf_counter()
            hdpi_result = algorithm.search(points, utility)
            wall = time.perf_counter() - start
        finally:
            FastRegion.refine_vertices = original_hdpi_refine

        scores = points @ utility
        correct = bool(scores[hdpi_result.point_index] >= scores.max() - 1e-9)
        row = {
            'dataset_id': dataset_id,
            'dataset_name': paths[dataset_id].name,
            'status': 'ok' if correct else 'incorrect',
            'questions': hdpi_result.questions,
            'search_seconds': hdpi_result.elapsed,
            'wall_seconds': wall,
            'accuracy': float(correct),
            'metrics': dict(stats),
            'counts': dict(counts),
        }
        result['hdpi_per_dataset'].append(row)
        hdpi_total_questions += hdpi_result.questions
        hdpi_total_elapsed += hdpi_result.elapsed
        result['hdpi_summary'] = {
            'completed_datasets': len(result['hdpi_per_dataset']),
            'total_questions': hdpi_total_questions,
            'sum_search_seconds': hdpi_total_elapsed,
            'all_correct_so_far': all(r['accuracy'] == 1.0 for r in result['hdpi_per_dataset']),
        }
        save(output / 'diagnostic.json', result)
        print('HDPI_DATASET ' + json.dumps(row), flush=True)

    print('DIAGNOSTIC_COMPLETE ' + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
