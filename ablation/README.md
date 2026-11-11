# SharedQ ablation

Run the commands from the repository root. The ablations use ten 100,000 x 4
datasets (correlated : anti-correlated : independent = 3 : 3 : 4). Each utility
queries the ten datasets jointly and has a one-hour wall-clock limit.

| Group | PCA reduction | Detect-and-divide | Partition construction |
| --- | --- | --- | --- |
| neither | off | off | original per-candidate scan |
| pca_original | on | off | original per-candidate scan |
| pca_hdpi | on | off | HDPI-style initialization and incremental clipping |
| detect_divide_only | off | on | Detect neighborhoods |
| full | on | on | Detect neighborhoods |

With PCA off, every unasked, nondegenerate pair that cuts both sides of the
utility range is scored using the shared pruning score. Batching limits memory
without sampling or capping the candidate pairs.

## Replay the four-group run

Run `python ablation/test_ablation.py` for small-data tests. To replay the fixed
utility batch in a separate directory (Bash):

```bash
export SHAREDQ_ABLATION_OUT=results/replay_original_pca
mkdir -p "$SHAREDQ_ABLATION_OUT"
cp results/ablation_sharedq_original_pca_10u_20261025_r2/manifest.json "$SHAREDQ_ABLATION_OUT/"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u ablation/run_ablation_dual_pca.py --groups full detect_divide_only pca_original neither
```

In PowerShell, set the output with `$env:SHAREDQ_ABLATION_OUT = 'results/replay_original_pca'`
and use `Copy-Item` to copy the manifest. Add `pca_hdpi` to the group list only
for the five-variant comparison. For fresh vectors, use
`python ablation/create_fresh_manifest.py --out results/replay_fresh --seed 20261112`,
then set the output environment variable to that directory. Use a fresh
directory for every changed setting; runners skip existing per-utility
records. Dataset checks keep the original byte hashes and allow an explicit
LF-normalized hash for identical files checked out with different line endings.

The historical `run_ablation.py --prepare` and diagnostic scripts may require
separately generated experiment records. The replay above runs every group
from scratch and does not depend on those records.

## Published results

- [Four groups, PCA-only using original scans](../results/ablation_sharedq_original_pca_10u_20261025_r2/REPORT.md)
- [Five groups, including both PCA partition implementations](../results/ablation_sharedq_dual_pca_10u_20261025/REPORT.md)
- [Earlier four-group comparison, PCA-only using HDPI partitions](../results/ablation_sharedq_10u_20261024/REPORT.md)

The reports include per-utility outcomes and comparisons. Recorded algorithm
time excludes process startup and dataset loading. Server settings and raw
logs are not required to replay these experiments.
