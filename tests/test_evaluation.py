import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from pro_gradu.evaluation import ModelSpec, evaluate_nested_cv, threshold_metrics
from pro_gradu.data import BASELINE_FEATURES


def synthetic_frame(n=60):
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(n, len(BASELINE_FEATURES))), columns=BASELINE_FEATURES)
    X["age"] += np.tile([0, 1], n // 2) * 1.5
    y = np.tile([0, 1], n // 2)
    X["time"] = 30
    X["DEATH_EVENT"] = y
    X.index = np.arange(n) + 100
    return X


def test_nested_cv_returns_one_oof_prediction_per_row():
    frame = synthetic_frame()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
    spec = ModelSpec("logistic", LogisticRegression(max_iter=1000), {"model__C": [0.1, 1.0]})
    metrics, predictions = evaluate_nested_cv(frame, [spec], cv, cv, random_state=12)
    assert len(metrics) == 5
    assert predictions.index.is_unique
    assert set(predictions.index) == set(frame.index)
    assert predictions["fold"].nunique() == 5


def test_feature_selection_and_baseline_share_outer_fold_ids():
    frame = synthetic_frame()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)
    base = ModelSpec("logistic", LogisticRegression(max_iter=1000), {"model__C": [1.0]}, False)
    selected = ModelSpec("logistic", LogisticRegression(max_iter=1000), {"model__C": [1.0]}, True)
    metrics, _ = evaluate_nested_cv(frame, [base, selected], cv, cv, random_state=12)
    fold_ids = metrics.pivot(index="fold", columns="feature_selection", values="fold")
    assert fold_ids[False].equals(fold_ids[True])


def test_metrics_use_death_event_as_positive_class_and_threshold_05():
    metrics = threshold_metrics(np.array([1, 1, 0, 0]), np.array([0.8, 0.4, 0.7, 0.1]))
    assert metrics["threshold"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5


def test_curated_benchmark_covers_major_classifier_families_with_paired_selection():
    from pro_gradu.evaluation import default_model_specs

    specs = default_model_specs()
    names = {spec.name for spec in specs}
    assert len(names) >= 12
    assert {False, True} == {spec.select_features for spec in specs}
    assert all(hasattr(spec.estimator, "predict_proba") for spec in specs)
