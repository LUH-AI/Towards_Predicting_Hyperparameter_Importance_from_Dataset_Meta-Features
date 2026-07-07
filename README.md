# Predicting Hyperparameter Importance from Dataset Meta-Features

This repository contains the code for the paper **"Predicting Hyperparameter Importance from Dataset Meta-Features"** (submitted to AutoML 2026).

We study whether dataset meta-features can predict which hyperparameters matter for a given dataset *before* running any HPO trials, using fANOVA-based importance estimates for [RealMLP](https://github.com/dholzmueller/pytabkit) on the [TabArena](https://github.com/autogluon/tabarena) and UCI benchmark datasets.

## Requirements

Python 3.10+ is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Purpose |
|---|---|---|
| `pytabkit` | 1.7.3 | RealMLP model |
| `deepcave` | 1.4.1 | fANOVA HPI estimation |
| `pymfe` | 0.4.4 | Meta-feature extraction |
| `openml` | ≥0.14 | TabArena dataset download |
| `smac` | 2.4.0 | Hyperparameter tuning of meta-learners |
| `scikit-learn` | 1.7.2 | Random forest models |
| `torch` | 2.11.0 | Required by pytabkit |
| `pandas`, `numpy` | 2.x, 1.26.4 | Data handling |


## Repository Structure

```
paper_AutoML/
├── scripts/                         # End-to-end pipeline (run in order)
│   ├── 01_download_prepare_data.py  # Download TabArena datasets via OpenML
│   ├── 02_run_realmlp_configs.py    # Evaluate 100 sampled RealMLP configs per dataset
│   ├── 03_compute_fanova_hpi.py     # Compute and normalize fANOVA HPI estimates
│   ├── 04_extract_metafeatures.py   # Extract PyMFE meta-features, build train/holdout CSVs
│   ├── 05_run_classification.py     # Train binary HPI classifiers (per HP)
│   ├── 06_run_ranking.py            # Train pairwise HPI rankers
│   ├── 07_run_regression.py         # Train HPI regressors (per HP)
│   ├── 08_evaluate_topk_f1.py       # Compute Top-K set F1 across approaches
│   ├── 09_run_downstream_hpo_validation.py  # HPO validation on held-out datasets
│   └── 10_make_all_figures.py       # Reproduce all paper figures
└── src/
    ├── data/                        # Dataset loading and preprocessing
    ├── hpi/                         # fANOVA computation and normalization
    ├── metafeatures/                # PyMFE extraction and table building
    ├── meta_learning/               # Classification, ranking, regression models
    ├── evaluation/                  # Top-K F1 and metric evaluation
    ├── downstream/                  # Downstream HPO validation experiments
    ├── plotting/                    # Figure generation utilities
    └── realmlp/                     # RealMLP config sampling and evaluation
```

## Reproducing the Results

Run the numbered scripts in order from the repository root. Each script is self-contained with its paths defined at the top.
Benchmarks can be enabled or disabled by commenting/uncommenting entries in the corresponding list.
### Step 1 — Download datasets

**TabArena**:

```bash
python scripts/01_download_prepare_data
```

Output: `data/raw/tabarena_51/`

**UCI** (91 classification datasets from [Holzmüller et al., 2024](https://github.com/dholzmueller/pytabkit)):

Download using the official pytabkit script from [`pytabkit/scripts/download_data.py`](https://github.com/dholzmueller/pytabkit/blob/main/scripts/download_data.py) and place the output under:

`data/raw/uci/tab_bench_data/`.

### Step 2 — Evaluate RealMLP configurations

Samples and evaluates 100 RealMLP configurations per dataset (30 epochs each).
For TabArena, the shared 8-hyperparameter setup excludes `ls_eps`; UCI experiments use the full 9-hyperparameter configuration space.

> **Note:** This is the most compute-intensive step. The paper used a single NVIDIA A100 GPU with 20 GB RAM. The script defaults to `DEVICE = "cuda:0"`; change it to `"cpu"` to run on CPU.
If running UCI, first export the TabBench data path:

```bash
export TAB_BENCH_DATA_DIR=data/raw/uci/tab_bench_data
```
then run:
```bash
python scripts/02_run_realmlp_configs
```

Output: `results/realmlp/tabarena_classification_8HPs/` and `results/realmlp/tabarena_regression_8HPs/`


### Step 3 — Compute fANOVA HPI

Fits fANOVA via DeepCAVE and normalizes importance scores to sum to one:

```bash
python scripts/03_compute_fanova_hpi
```

Output: `hpi.json` and `hpi_normalized.json` per dataset folder.

### Step 4 — Extract meta-features

Extracts 405 PyMFE meta-features per dataset and joins them with HPI scores:

```bash
python scripts/04_extract_metafeatures
```
The holdout datasets used for downstream HPO validation are automatically separated into dedicated holdout CSV files:
Output: `data/processed/features_and_importances_tabarena_train.csv` and `data/processed/features_and_importances_tabarena_holdout.csv`

### Steps 5–7 — Train meta-learners

Train the three meta-learning approaches (classification, ranking, regression) using nested 5×5 cross-validation with SMAC-tuned random forests:

```bash
python scripts/05_run_classification
python scripts/06_run_ranking
python scripts/07_run_regression
```

### Step 8 — Evaluate Top-K set F1

Computes the unified Top-4 set F1 metric across all approaches and benchmarks (TabArena and UCI):

```bash
python scripts/08_evaluate_topk_f1
```

### Step 9 — HPO validation experiment

Runs SMAC-based HPO on three held-out datasets, comparing full-space search vs. predicted Top-4:

```bash
python scripts/09_run_downstream_hpo_validation
```

### Step 10 — Generate figures

Reproduces all figures from the paper:

```bash
python scripts/10_make_all_figures
```

## Results

### Top-4 Set F1 (mean ± 95% CI)

| Approach | TabArena | UCI |
|---|---|---|
| Classification | 0.58 ± 0.02 | 0.56 ± 0.02 |
| Ranking | 0.85 ± 0.01 | 0.79 ± 0.02 |
| **Regression** | **0.86 ± 0.01** | **0.82 ± 0.02** |

### Ranking aggregation strategies (TabArena)

| Strategy | Top-3 Accuracy | Spearman Rank Correlation |
|---|---|---|
| Hard Wins | **0.94 ± 0.03** | 0.80 ± 0.03 |
| Soft Wins | 0.90 ± 0.03 | **0.81 ± 0.01** |

### HPO Validation

Tuning only the predicted Top-4 hyperparameters achieves lower or equal regret compared to full-space search on all three held-out datasets within the same 50-trial budget.

### License

The code is licensed under the BSD 3-Clause License.