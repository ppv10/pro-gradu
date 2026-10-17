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

This is exploratory research code for a small, observational cohort. Cross-validation is internal validation, not independent replication. Broad algorithm comparisons can favor a model due to sampling noise. Results are not a clinical decision tool.

## Repository contents

| Path | Purpose |
| --- | --- |
| `feature_selection.ipynb` | Descriptive cohort summaries and marginal Spearman associations; no full-data feature selection |
| `initial_models.ipynb` | Data checks and classifier roster |
| `hyperparameter_tuning_and_final_models.ipynb` | Paired nested cross-validation benchmark and held-out predictions |
| `notebooks/bayesian_classification_pymc.ipynb` | Bayesian logistic classification with PyMC |
| `notebooks/bayesian_survival_analysis.ipynb` | Weibull AFT survival model with right-censoring |
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

The original dataset CSV is included so analyses can run offline. To install the package in editable mode as part of the lockfile environment, `uv sync` handles it automatically.

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
