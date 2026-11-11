"""Create a reproducible fresh utility batch for the SharedQ 2x2 ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True,
                        help='Experiment output directory, relative to code_1')
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Output must stay inside code_1')
    source = json.loads((ROOT / 'results' / 'ablation_sharedq' / 'manifest.json')
                        .read_text(encoding='utf-8'))
    rng = np.random.default_rng(args.seed)
    utilities = rng.dirichlet(np.ones(4), size=10).tolist()
    previous = np.asarray(source['utilities'], dtype=float)
    fresh = np.asarray(utilities, dtype=float)
    if any(np.allclose(row, previous, rtol=0, atol=1e-14) for row in fresh):
        raise RuntimeError('Fresh utility batch unexpectedly duplicates a previous vector')
    manifest = {
        'datasets': source['datasets'],
        'distribution': source['distribution'],
        'utilities': utilities,
        'seed': args.seed,
        'utility_generation': 'Dirichlet(1,1,1,1), NumPy default_rng',
        'utility_generation_seed': args.seed,
        'timeout_seconds': 3600,
        'timeout_unit': 'one utility over all ten datasets',
        'hyperplane_reduction_off': 'exhaustive cross-dataset scoring, no pair cap',
        'partition_construction_by_group': {
            'neither': 'original all-candidate top-1 partition scans; no neighbor walk',
            'pca_only': 'HDPI all-candidate partition construction with incremental clipping',
            'detect_divide_only': 'SharedQ convex-hull neighbor partition walk',
            'full': 'SharedQ convex-hull neighbor partition walk',
        },
        'parameters': source['parameters'],
        'source_manifest': 'results/ablation_sharedq/manifest.json',
        'notes': 'Fresh utility batch; all four configurations use these same ten vectors. '
                 'Full is rerun, not imported from historical results.',
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'utilities.json').write_text(json.dumps({
        'seed': args.seed, 'utilities': utilities,
    }, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'seed': args.seed, 'utility_count': len(utilities)}, indent=2))


if __name__ == '__main__':
    main()
