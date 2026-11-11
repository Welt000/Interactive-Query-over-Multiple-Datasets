"""SharedQ 2x2 ablation. All paths and outputs stay inside code_1.

Each utility is one IMD run over ALL ten datasets, with a hard wall timeout.
The full variant is imported from audited existing results, never rerun.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from SharedQ.algorithm import SharedQuestions, DetectDivideSharedQuestions
from baseline.experiment_utils import load_dataset
from baseline.experiment_final import radius_from_rho
from experiment_config import dataset_default_order_key

OUT = Path(os.environ.get(
    'SHAREDQ_ABLATION_OUT', str(ROOT / 'results' / 'ablation_sharedq')
)).resolve()
GROUPS = {'neither': (False, False), 'pca_only': (True, False),
          'detect_divide_only': (False, True), 'full': (True, True)}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dataset_matches(path, entry):
    if digest(path) == entry["sha256"]:
        return True
    expected_lf = entry.get("sha256_lf")
    return bool(expected_lf and hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected_lf)

def prepare():
    source = ROOT / 'results' / '4d_100k_10_result'
    config = json.loads((source / 'config.json').read_text(encoding='utf-8-sig'))
    paths = sorted((ROOT / 'original_datasets' / '4d_100k_10').glob('*.txt'),
                   key=dataset_default_order_key)
    assert len(paths) == 10
    counts = {'corr': 0, 'anti': 0, 'ind': 0}
    for p in paths:
        for kind in counts:
            if '_' + kind + '_' in p.stem:
                counts[kind] += 1
    assert counts == {'corr': 3, 'anti': 3, 'ind': 4}, counts
    assert [p.name for p in paths] == [Path(p).name for p in config['dataset_paths']]
    utilities = config['utilities']
    assert len(utilities) == 10 and np.asarray(utilities).shape == (10, 4)
    assert np.allclose(np.sum(utilities, axis=1), 1)
    entries = []
    for p in paths:
        data = load_dataset(p)
        assert data.shape == (100000, 4), (p, data.shape)
        assert np.isfinite(data).all()
        entries.append({'name': p.name, 'sha256': digest(p), 'shape': list(data.shape)})
    manifest = {'datasets': entries, 'distribution': counts, 'utilities': utilities,
                'seed': config['seed'], 'timeout_seconds': 3600,
                'timeout_unit': 'one utility over all ten datasets',
                'hyperplane_reduction_off': 'exhaustive cross-dataset scoring, no pair cap',
                'partition_construction_by_group': {
                    'pca_only': 'HDPI all-candidate partition construction with incremental clipping',
                    'other_groups': 'SharedQ convex-hull neighbor partition walk',
                },
                'parameters': {key: config[key] for key in (
                    'rhos', 'alpha', 'beta', 'radius_growth', 'detect_round_limit',
                    'divide_round_limit', 'max_questions', 'max_pairs_per_dataset', 'tolerance')},
                'source_config_sha256': digest(source / 'config.json')}
    save(OUT / 'manifest.json', manifest)
    reuse(manifest, source)
    return manifest


def reuse(manifest, source):
    summary_path = source / 'experiment_SharedQ_summary.csv'
    detail_path = source / 'experiment_SharedQ_detail.csv'
    with summary_path.open(encoding='utf-8-sig', newline='') as f:
        summaries = list(csv.DictReader(f))
    summary = next(r for r in summaries if r['method'].lower() == 'sharedq'
                   and r['status'] == 'ok' and float(r['rho']) == 0.1
                   and float(r['alpha']) == 1 and float(r['beta']) == 1)
    assert (int(summary['utilities']), int(summary['datasets']), int(summary['runs'])) == (10, 10, 100)
    with detail_path.open(encoding='utf-8-sig', newline='') as f:
        details = list(csv.DictReader(f))
    details = [r for r in details if r['method'].lower() == 'sharedq'
               and float(r['rho']) == 0.1 and float(r['alpha']) == 1 and float(r['beta']) == 1]
    assert len(details) == 100
    assert {(int(r['utility_id']), int(r['dataset_id'])) for r in details} == {
        (u, d) for u in range(10) for d in range(10)}
    # Verify the recorded ground truth and returned point against current raw
    # local data and the exact saved utility vectors, not just file names.
    for d, entry in enumerate(manifest['datasets']):
        data = load_dataset(ROOT / 'original_datasets' / '4d_100k_10' / entry['name'])
        for r in (r for r in details if int(r['dataset_id']) == d):
            assert Path(r['dataset_path']).name == entry['name']
            scores = data @ np.asarray(manifest['utilities'][int(r['utility_id'])])
            assert int(r['truth_index']) == int(np.argmax(scores))
            assert abs(float(scores[int(r['result_index'])]) - float(scores.max())) <= 1e-9
            assert abs(float(r['truth_score']) - float(scores.max())) <= 1e-9
            assert r['correct'].lower() in ('true', '1')
    for u in range(10):
        rows = [r for r in details if int(r['utility_id']) == u]
        assert len({r['questions'] for r in rows}) == 1
        assert len({r['elapsed'] for r in rows}) == 1
        save(OUT / 'full' / f'u{u}.json', {
            'group': 'full', 'utility_id': u, 'status': 'ok', 'reused': True,
            'questions': int(rows[0]['questions']), 'elapsed': float(rows[0]['elapsed']),
            'accuracy': 1.0, 'resolved': True, 'datasets': 10, 'details': rows})
    assert np.isclose(np.mean([int(next(r for r in details if int(r['utility_id']) == u)['questions'])
                              for u in range(10)]), float(summary['avg_total_questions_per_utility']))
    assert np.isclose(np.mean([float(next(r for r in details if int(r['utility_id']) == u)['elapsed'])
                              for u in range(10)]), float(summary['avg_total_elapsed_per_utility']))
    save(OUT / 'reuse_audit.json', {'source_summary': str(summary_path.relative_to(ROOT)),
        'source_detail': str(detail_path.relative_to(ROOT)),
        'summary_sha256': digest(summary_path), 'detail_sha256': digest(detail_path),
        'validation': '100 ground truths and returned scores match saved utilities and current raw data',
        'limitation': 'historical source has no original dataset hashes or runtime library versions',
        'summary': summary})
    summarize('full')


def worker(group, u):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    pca, local = GROUPS[group]
    data = [load_dataset(ROOT / 'original_datasets' / '4d_100k_10' / e['name'])
            for e in manifest['datasets']]
    params = manifest['parameters']
    kwargs = dict(alpha=params['alpha'], beta=params['beta'],
                  max_questions=params['max_questions'],
                  max_pairs_per_dataset=params['max_pairs_per_dataset'],
                  random_state=manifest['seed'] + u * 1009,
                  hyperplane_reduction=pca, detect_workers=1)
    if not local:
        kwargs['partition_construction'] = 'hdpi' if group == 'pca_only' else 'neighbor'
    if local:
        kwargs.update(radius=radius_from_rho(0.1, 4), radius_growth=params['radius_growth'],
                      detect_round_limit=params['detect_round_limit'],
                      divide_round_limit=params['divide_round_limit'], random_initial_center=True,
                      boundary_center_strategy='ray_midpoint',
                      detect_point_timeout_seconds=None)
    algorithm = (DetectDivideSharedQuestions if local else SharedQuestions)(**kwargs)
    utility = np.asarray(manifest['utilities'][u])
    result = algorithm.search(data, utility)
    details = []
    for item in result.dataset_results:
        scores = data[item.dataset_id] @ utility
        correct = bool(scores[item.point_index] >= scores.max() - params['tolerance'])
        details.append({'dataset_id': item.dataset_id,
                        'dataset_name': manifest['datasets'][item.dataset_id]['name'],
                        'point_index': item.point_index, 'truth_index': int(np.argmax(scores)),
                        'correct': correct, 'candidate_count': item.candidate_count})
    resolved = all(d['candidate_count'] == 1 for d in details) and len(details) == 10
    accuracy = float(np.mean([d['correct'] for d in details]))
    save(OUT / group / f'u{u}.json', {'group': group, 'utility_id': u,
        'status': 'ok' if resolved and accuracy == 1 else 'incomplete',
        'reused': False, 'questions': result.questions, 'elapsed': result.elapsed,
        'accuracy': accuracy, 'resolved': resolved, 'datasets': len(details),
        'detect_rounds': len(result.detect_centers), 'timings': result.timings, 'details': details})


def summarize(group):
    records = [json.loads(p.read_text()) for p in sorted((OUT / group).glob('u*.json'))]
    ok = [r for r in records if r['status'] == 'ok']
    row = {'group': group, 'pca_reduction': GROUPS[group][0], 'detect_divide': GROUPS[group][1],
           'finished_utilities': len(records), 'successful_utilities': len(ok),
           'timeout_utilities': sum(r['status'] == 'timeout' for r in records),
           'error_utilities': sum(r['status'] not in ('ok', 'timeout') for r in records),
           'datasets_per_utility': 10, 'reused': group == 'full',
           'status': 'ok' if len(ok) == 10 else ('partial' if len(records) == 10 else 'running'),
           'avg_questions_successful': float(np.mean([r['questions'] for r in ok])) if ok else None,
           'avg_seconds_successful': float(np.mean([r['elapsed'] for r in ok])) if ok else None,
           'accuracy_successful': float(np.mean([r['accuracy'] for r in ok])) if ok else None}
    save(OUT / group / 'summary.json', row)
    all_rows = [json.loads(p.read_text()) for p in OUT.glob('*/summary.json')]
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(all_rows)
    print('GROUP_RESULT ' + json.dumps(row), flush=True)
    return row


def run(groups):
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for entry in manifest['datasets']:
        path = ROOT / 'original_datasets' / '4d_100k_10' / entry['name']
        assert dataset_matches(path, entry), f'Dataset changed: {path}'
    save(OUT / 'runtime.json', {'python': sys.version, 'numpy': np.__version__,
        'platform': sys.platform, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}})
    for group in groups:
        if group == 'full':
            summarize(group)
            continue
        for u in range(10):
            path = OUT / group / f'u{u}.json'
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            print(f'RUN group={group} utility={u} datasets=10 timeout=3600', flush=True)
            with (path.parent / f'u{u}.log').open('w') as log:
                try:
                    child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--worker', group, '--utility', str(u)], stdout=log,
                        stderr=subprocess.STDOUT, timeout=manifest['timeout_seconds'])
                    if child.returncode != 0 or not path.exists():
                        save(path, {'group': group, 'utility_id': u, 'status': 'error',
                                    'returncode': child.returncode, 'wall_seconds': time.monotonic() - started})
                except subprocess.TimeoutExpired:
                    save(path, {'group': group, 'utility_id': u, 'status': 'timeout',
                                'wall_seconds': time.monotonic() - started})
            print('UTILITY_RESULT ' + path.read_text().replace('\n', ' '), flush=True)
            summarize(group)
        print('GROUP_COMPLETE ' + json.dumps(summarize(group)), flush=True)
    save(OUT / 'completed.json', {'groups': groups, 'completed': time.strftime('%Y-%m-%dT%H:%M:%S%z')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', choices=list(GROUPS))
    parser.add_argument('--utility', type=int)
    parser.add_argument('--groups', nargs='+', choices=list(GROUPS),
                        default=['neither', 'pca_only', 'detect_divide_only', 'full'])
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker(args.worker, args.utility)
    else:
        run(args.groups)
