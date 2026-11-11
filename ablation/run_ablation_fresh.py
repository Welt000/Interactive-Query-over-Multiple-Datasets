"""Run a fresh 2x2 SharedQ ablation with a supplied manifest.

The existing run_ablation.py imports the historical full results. This runner
reuses its data loading, result serialization, and summary code while running
all four configurations on the new utility vectors.
"""
from __future__ import annotations

import json
import csv
import os
import subprocess
import sys
from pathlib import Path
import time

import run_ablation as base

ROOT = base.ROOT
OUT = Path(os.environ.get(
    'SHAREDQ_ABLATION_OUT',
    str(ROOT / 'results' / 'ablation_sharedq_10u_20261024'),
)).resolve()
base.OUT = OUT
GROUPS = base.GROUPS


class ExhaustivePartitionSharedQuestions(base.SharedQuestions):
    """Keep hull top-1 candidates, but disable neighbor-walk region discovery."""

    def _convex_top1_candidates_and_adjacency(self, points):
        candidates, _ = super()._convex_top1_candidates_and_adjacency(points)
        return candidates, {}


def worker(group: str, u: int) -> None:
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    pca, local = GROUPS[group]
    data = [base.load_dataset(ROOT / 'original_datasets' / '4d_100k_10' / entry['name'])
            for entry in manifest['datasets']]
    params = manifest['parameters']
    kwargs = dict(
        alpha=params['alpha'], beta=params['beta'],
        max_questions=params['max_questions'],
        max_pairs_per_dataset=params['max_pairs_per_dataset'],
        random_state=manifest['seed'] + u * 1009,
        hyperplane_reduction=pca,
        detect_workers=1,

    )
    if local:
        kwargs.update(
            radius=base.radius_from_rho(0.1, 4),
            radius_growth=params['radius_growth'],
            detect_round_limit=params['detect_round_limit'],
            divide_round_limit=params['divide_round_limit'],
            random_initial_center=True,
            boundary_center_strategy='ray_midpoint',
            detect_point_timeout_seconds=None,
        )
        algorithm = base.DetectDivideSharedQuestions(**kwargs)
    elif group == 'neither':
        # Match the raw no-PCA/no-detect run: full per-candidate partition
        # scans, without convex-hull neighbor walking.
        algorithm = ExhaustivePartitionSharedQuestions(**kwargs)
    else:
        kwargs['partition_construction'] = 'hdpi' if group == 'pca_only' else 'neighbor'
        algorithm = base.SharedQuestions(**kwargs)

    utility = base.np.asarray(manifest['utilities'][u])
    result = algorithm.search(data, utility)
    details = []
    for item in result.dataset_results:
        scores = data[item.dataset_id] @ utility
        details.append({
            'dataset_id': item.dataset_id,
            'dataset_name': manifest['datasets'][item.dataset_id]['name'],
            'point_index': item.point_index,
            'truth_index': int(base.np.argmax(scores)),
            'correct': bool(scores[item.point_index] >= scores.max() - params['tolerance']),
            'candidate_count': item.candidate_count,
        })
    resolved = len(details) == 10 and all(d['candidate_count'] == 1 for d in details)
    accuracy = float(base.np.mean([d['correct'] for d in details]))
    base.save(OUT / group / f'u{u}.json', {
        'group': group, 'utility_id': u,
        'status': 'ok' if resolved and accuracy == 1 else 'incomplete',
        'reused': False, 'questions': result.questions, 'elapsed': result.elapsed,
        'accuracy': accuracy, 'resolved': resolved, 'datasets': len(details),
        'detect_rounds': len(result.detect_centers), 'timings': result.timings,
        'details': details,
    })


def summarize(group: str) -> dict:
    records = [json.loads(path.read_text(encoding='utf-8'))
               for path in sorted((OUT / group).glob('u*.json'))]
    ok = [record for record in records if record.get('status') == 'ok']
    row = {
        'group': group,
        'pca_reduction': GROUPS[group][0],
        'detect_divide': GROUPS[group][1],
        'finished_utilities': len(records),
        'successful_utilities': len(ok),
        'timeout_utilities': sum(record.get('status') == 'timeout' for record in records),
        'error_utilities': sum(record.get('status') not in ('ok', 'timeout') for record in records),
        'datasets_per_utility': 10,
        'reused': False,
        'status': 'ok' if len(ok) == 10 else ('partial' if len(records) == 10 else 'running'),
        'avg_questions_successful': float(base.np.mean([r['questions'] for r in ok])) if ok else None,
        'avg_seconds_successful': float(base.np.mean([r['elapsed'] for r in ok])) if ok else None,
        'accuracy_successful': float(base.np.mean([r['accuracy'] for r in ok])) if ok else None,
    }
    base.save(OUT / group / 'summary.json', row)
    all_rows = [json.loads(path.read_text(encoding='utf-8'))
                for path in sorted(OUT.glob('*/summary.json'))]
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(all_rows)
    print('GROUP_RESULT ' + json.dumps(row), flush=True)
    return row


def run(groups: list[str]) -> None:
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['datasets']:
        path = ROOT / 'original_datasets' / '4d_100k_10' / entry['name']
        assert base.dataset_matches(path, entry), f'Dataset changed: {path}'
    base.save(OUT / 'runtime.json', {
        'python': sys.version, 'numpy': base.np.__version__, 'platform': sys.platform,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'threads': {key: os.environ.get(key) for key in
                    ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
    })
    for group in groups:
        for u in range(10):
            path = OUT / group / f'u{u}.json'
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            print(f'RUN group={group} utility={u} datasets=10 timeout=3600', flush=True)
            with (path.parent / f'u{u}.log').open('w', encoding='utf-8') as log:
                try:
                    child = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()),
                         '--worker', group, '--utility', str(u)],
                        stdout=log, stderr=subprocess.STDOUT,
                        timeout=manifest['timeout_seconds'],
                    )
                    if child.returncode != 0 or not path.exists():
                        base.save(path, {
                            'group': group, 'utility_id': u, 'status': 'error',
                            'returncode': child.returncode,
                            'wall_seconds': time.monotonic() - started,
                        })
                except subprocess.TimeoutExpired:
                    base.save(path, {
                        'group': group, 'utility_id': u, 'status': 'timeout',
                        'wall_seconds': time.monotonic() - started,
                    })
            print('UTILITY_RESULT ' + path.read_text(encoding='utf-8').replace('\n', ' '), flush=True)
            summarize(group)
        print('GROUP_COMPLETE ' + json.dumps(summarize(group)), flush=True)
    base.save(OUT / 'completed.json', {
        'groups': groups, 'completed': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    })


base.worker = worker
base.summarize = summarize
base.run = run

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', nargs='+', choices=list(GROUPS),
                        default=['neither', 'pca_only', 'detect_divide_only', 'full'])
    parser.add_argument('--worker', choices=list(GROUPS))
    parser.add_argument('--utility', type=int)
    args = parser.parse_args()
    if args.worker:
        if args.utility is None or not 0 <= args.utility < 10:
            parser.error('--worker requires --utility in [0, 9]')
        worker(args.worker, args.utility)
    else:
        run(args.groups)
