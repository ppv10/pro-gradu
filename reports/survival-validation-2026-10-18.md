# Survival model validation results (2026-10-18)

This report records the bundled cohort evaluation run with the nested Cox PH versus Weibull AFT validator in [commit e4bb22c](https://github.com/ppv10/pro-gradu/commit/e4bb22c457f2dbe28e9259a8ab187b79126e54a2).

## Cohort and method

The included heart-failure dataset has 299 records: 96 observed death events and 203 right-censored follow-ups. The run used seed 42, five stratified outer folds, four inner folds, ridge penalties 0.01, 0.1, and 1.0, and prediction horizons of 90 and 180 days. Inner folds selected each model's penalty by Harrell concordance. Scores below come from held-out outer folds; censoring weights were estimated using the corresponding outer training fold.

The execution used Python 3.12.14, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.9.1, and lifelines 0.30.3.

## Results

| Model | Mean outer Harrell C-index (fold SD) | 90-day pooled IPCW Brier | 180-day pooled IPCW Brier | Most common penalty |
| --- | ---: | ---: | ---: | ---: |
| Cox PH | 0.7113 (0.0671) | 0.1702 | 0.2061 | 1.0 |
| Weibull AFT | 0.7108 (0.0659) | 0.1709 | 0.2069 | 1.0 |

Harrell's C-index measures how well the model ranks patients by event timing; higher is better. The IPCW Brier score measures horizon-specific risk prediction error while accounting for censoring; lower is better. At each horizon, 254 records had known 90-day outcomes and 192 had known 180-day outcomes.

The C-index difference is 0.0004, while fold-to-fold standard deviations are about 0.066. The models therefore performed almost identically on discrimination in this run. Cox had slightly lower pooled Brier scores at both horizons, by less than 0.001. This is a small numerical difference, not evidence of a meaningful improvement.

The cross-fitted calibration table uses up to five risk groups. In the highest-risk group, both models underpredicted the IPCW observed event rate: at 90 days, predicted versus observed was 0.388 versus 0.484 for Cox and 0.352 versus 0.485 for Weibull AFT; at 180 days it was 0.561 versus 0.629 and 0.584 versus 0.629, respectively. These grouped estimates are descriptive and noisy at this cohort size.

## Interpretation and limits

This run compares Cox PH with Weibull AFT. It does not establish improvement over an earlier baseline because no directly comparable historical baseline was evaluated. The outer folds are internal validation on one small dataset, not an independent replication or clinical validation.

## Reproduce

With the repository environment synchronized, run:

```sh
uv run pro-gradu validate-survival heart_failure_clinical_records_dataset.csv --output-dir results/survival-validation
```

The command writes fold metrics, cross-fitted predictions, calibration, the Cox proportional-hazards diagnostic, and a JSON summary under `results/survival-validation/`.
