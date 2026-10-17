# pro-gradu

Reproducible machine-learning analyses of the [UCI Heart Failure Clinical Records dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records). The published thesis is available [from the University of Vaasa repository](https://osuva.uwasa.fi/items/4e35e122-c299-4722-89b1-0081f9274bec). The thesis and draft report files are not included here.

## Scope

The dataset contains 299 records, 11 baseline predictors, follow-up time in days, and a binary death-event indicator. The classification task predicts whether a death event was observed during the recorded follow-up. It uses baseline predictors and excludes `time`. It does not predict lifetime mortality.

The code provides:

- Validated dataset loading and shared scikit-learn workflows in `src/pro_gradu/`.
- Exploratory cohort summaries and marginal associations, clearly labeled as descriptive.
- A broad nested-CV benchmark for 14 classifier families, each compared with and without mutual-information feature selection. Scaling, selection, and parameter searches are fitted inside training folds. The outer folds are paired and held out from tuning.
- A PyMC Bayesian logistic classifier with fold-local standardization and held-out posterior probabilities.
- A separate PyMC Weibull accelerated-failure-time model that handles observed events and right-censored follow-up correctly.
- A callable penalized Cox prediction engine plus nested Cox-versus-Weibull AFT validation with censoring-aware horizon scores.

This is exploratory research code for a small, observational cohort. Cross-validation is internal validation, not independent replication. Broad algorithm comparisons can favor a model due to sampling noise. Results are not a clinical decision tool.

## Repository contents

| Path | Purpose |
| --- | --- |
| `feature_selection.ipynb` | Descriptive cohort summaries and marginal Spearman associations; no full-data feature selection |
| `initial_models.ipynb` | Data checks and classifier roster |
| `hyperparameter_tuning_and_final_models.ipynb` | Paired nested cross-validation benchmark and held-out predictions |
| `notebooks/bayesian_classification_pymc.ipynb` | Bayesian logistic classification with PyMC |
| `notebooks/bayesian_survival_analysis.ipynb` | Weibull AFT survival model with right-censoring |
| `src/pro_gradu/survival.py` | Censoring-aware Cox PH prediction API and nested validation |
| `src/pro_gradu/survival_validation.py` | Nested Cox PH and Weibull AFT evaluation with IPCW metrics |
| `src/pro_gradu/cli.py` | Fit, predict, and validate command-line interface |
| `src/pro_gradu/` | Data validation, feature construction, workflows, model registry, and evaluation |
| `tests/` | Dataset contract, leakage-sensitive workflow, nested evaluation, and model registry tests |

## Setup

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required. The committed `uv.lock` pins a tested dependency resolution.

Install the full environment, including optional Bayesian packages:

```powershell
uv sync --all-extras
```

For the scikit-learn analyses and tests without PyMC:

```powershell
uv sync --extra notebooks --extra dev
```

Install the PyMC notebook dependencies separately from the fast survival engine:

```powershell
uv sync --extra notebooks --extra bayesian
```

The original dataset CSV is included so analyses can run offline. To install the package in editable mode as part of the lockfile environment, `uv sync` handles it automatically.

The survival engine is included in the lightweight `survival` extra and is installed by `uv sync --all-extras`. PyMC is kept in a separate `bayesian` extra so using the engine does not require MCMC dependencies. The engine uses the baseline clinical features as predictors and treats `time` and `DEATH_EVENT` as duration and event outcomes. Prediction input CSVs must contain only the 11 baseline features; follow-up and outcome columns are rejected to prevent leakage.

To install only the survival engine and its tests, use `uv sync --extra survival --extra dev`.

## Run and reproduce

Run the unit tests:

```powershell
uv run pytest
```

Execute the scikit-learn notebooks in a clean kernel:

```powershell
uv run jupyter nbconvert --to notebook --execute --output-dir executed initial_models.ipynb feature_selection.ipynb hyperparameter_tuning_and_final_models.ipynb
```

Execute the PyMC notebooks with the full extras installed:

```powershell
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 --output-dir executed notebooks/bayesian_classification_pymc.ipynb notebooks/bayesian_survival_analysis.ipynb
```

The nested benchmark writes fold metrics and out-of-fold predictions under `results/`; generated outputs are ignored by Git. Seeds and split definitions are explicit in each notebook. MCMC sampling is stochastic but uses fixed seeds; check the reported convergence diagnostics before interpreting posterior summaries. The PyMC notebooks can take an hour or more on Windows without a PyTensor C++ compiler, especially the four-chain survival fit. A shortened sampling run can check that notebook code executes, but it is not a convergence check.

## Survival prediction engine

Fit and save a ridge-penalized Cox proportional-hazards model from the included cohort:

```powershell
uv run pro-gradu fit-survival heart_failure_clinical_records_dataset.csv --model-out models/cox.joblib
```

Create `patients.csv` with one row per person and the 11 baseline feature columns only, then request estimates at supported horizons:

```powershell
uv run pro-gradu predict-survival patients.csv --model models/cox.joblib --horizons 30,90,180 --output results/predictions.csv
```

The prediction output is long-form: each input row has a `survival_probability` and `event_risk` (`1 - survival_probability`) for each requested day horizon. Horizons must be positive and cannot exceed the maximum follow-up observed in the training data. The default L2 penalty is 1.0; the nested validation selected it in four of five outer folds, but this small-sample result is not claimed to be globally optimal.

Run nested, event-stratified validation and write fold metrics, cross-fitted horizon predictions, calibration bins, a proportional-hazards diagnostic, and a JSON summary:

```powershell
uv run pro-gradu validate-survival heart_failure_clinical_records_dataset.csv --output-dir results/survival-validation
```

The validator compares ridge-penalized Cox PH and Weibull AFT models on the same event-stratified folds. Inner folds select each model's penalty using Harrell's concordance. Outer-fold outputs include concordance and IPCW Brier scores at 90 and 180 days by default; `--horizons 60,120` changes the prespecified horizons. Censoring weights are estimated from each outer training fold. The calibration CSV groups the cross-fitted predictions into up to five risk bins and reports weighted predicted and observed event rates. These internal estimates are exploratory, especially with 299 patients, and are not external validation or clinical guidance. Only load model artifacts that you trust because joblib artifacts use Python pickle internally.

Validation writes `model_comparison_fold_metrics.csv`, `model_comparison_predictions.csv`, `calibration.csv`, `proportional_hazards.csv`, and `summary.json` under the output directory. `fold_metrics.csv` and `out_of_fold_predictions.csv` retain their previous Cox-only format for compatibility. The proportional-hazards diagnostic is computed for the full-data Cox fit only. The prediction CLI still uses the saved Cox model; Weibull AFT is currently included in validation rather than artifact-based prediction.

## Evaluation details

- The primary classification target is `DEATH_EVENT = 1` (an observed death event); the negative class is 0.
- The nested model search optimizes average precision for class 1. Outer-fold metrics include ROC AUC, average precision, balanced accuracy, sensitivity, specificity, and accuracy.
- Sensitivity, specificity, balanced accuracy, and accuracy use a fixed probability threshold of 0.5. It is a comparison convention, not a clinical decision threshold.
- Five stratified outer folds provide held-out predictions. Five stratified inner folds tune hyperparameters and (when enabled) the number of selected features. Both feature-selection variants share the same outer folds.
- The Bayesian classifier uses regularizing priors and the same baseline prediction target. The survival analysis instead uses `time` and treats `DEATH_EVENT = 0` as right-censored.

## Data attribution

The included CSV is the UCI Heart Failure Clinical Records dataset. Please cite the dataset record and its associated paper, and retain attribution under the dataset's [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/):

> Chicco, D. and Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20, 16. https://doi.org/10.1186/s12911-020-1023-5

The dataset license applies to the data. It does not set a software license for this repository.

