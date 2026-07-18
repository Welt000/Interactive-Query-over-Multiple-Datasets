# SharedQ Experiment Package

This folder is a compact runnable package for the Baseline, SUR, and SharedQ experiments.

## Contents

- `baseline/`: experiment wrappers and shared utilities.
- `HD_PI/`: HD-PI implementation.
- `RH/`: RH implementation.
- `UH_Simplex/`: UH-Simplex implementation.
- `UH_Random/`: UH-Random implementation.
- `SUR/`: Shared Utility Range implementation.
- `SharedQ/`: SharedQ implementation.
- `structure/`: common data structures, utility range, skyline helpers.
- `experiment_config.py`: unified experiment configuration and runner for Baseline, SUR, and SharedQ.
- `run_experiment.py`: experiment entry point.
- `experiment_datasets.py`: dataset folder list used by the experiment runner.
- `preprocess_skyline.py`: skyline preprocessing used by Baseline and SUR.
- `original_datasets/4d_100k_10/`: included original dataset batch.
- `after_skyline_datasets/`: generated automatically by preprocessing.
- `results/`: experiment outputs.

## Dependencies

Use Python 3.10+.

Required packages:

```bash
pip install numpy scipy matplotlib
```

Optional but recommended for UH frame acceleration:

```bash
pip install swiglpk
```

## Run The Included Experiment

From this folder:

```bash
python run_experiment.py
```

Default behavior:

- reads `experiment_datasets.py`;
- uses `original_datasets/4d_100k_10`;
- preprocesses skyline into `after_skyline_datasets/4d_100k_10`;
- runs methods configured in `ExperimentConfig.methods`;
- runs algorithms configured in `ExperimentConfig.algorithms`;
- writes results into `results/4d_100k_10_result`.

By default, `ExperimentConfig` runs:

```python
methods = ("Baseline", "SUR", "SharedQ")
algorithms = ("HD-PI", "RH", "UH-Simplex", "UH-Random")
random_utility_count = 10
rhos = (0.1,)
alpha_beta_ratios = (1,)
```

You can edit these values directly near the top of `experiment_config.py`.

## Output Files

For the included dataset, results are written to:

```text
results/4d_100k_10_result/
```

Important files:

- `experiment_summary.csv`: average result rows.
- `experiment_detail.csv`: per-utility and per-dataset details.
- `sharedq_param_metrics.csv`: compact SharedQ parameter metrics.
- `config.json`: resolved experiment configuration.
- `_tmp_detail/`: intermediate detail rows from each utility batch.


## Notes

- SharedQ is implemented by `SharedQ/algorithm.py`; the internal folder name `SharedQ` is preserved to keep imports stable.
- Baseline and SUR use preprocessed skyline datasets.
- SharedQ uses original datasets directly.

