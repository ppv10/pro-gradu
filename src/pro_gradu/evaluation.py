"""Nested cross-validation for mortality classification."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

from .classification import classification_features, make_classification_workflow


@dataclass(frozen=True)
class ModelSpec:
    """A classifier and its inner-CV search space."""

    name: str
    estimator: Any
    param_distributions: dict[str, Any]
    select_features: bool = False
    n_iter: int = 10


def default_model_specs(
    include_feature_selection: bool = True,
    random_state: int = 42,
) -> list[ModelSpec]:
    """Curated, probability-capable sklearn models for the small cohort.

    Search spaces are intentionally compact. Returning paired specifications
    lets every family use the same outer folds with and without fold-local
    mutual-information selection.
    """
    families = [
        ("Logistic regression", LogisticRegression(max_iter=3000, random_state=random_state), {
            "model__C": [0.01, 0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]
        }),
        ("K-nearest neighbors", KNeighborsClassifier(), {
            "model__n_neighbors": [3, 5, 9, 15, 21, 31], "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }),
        ("RBF support vector machine", CalibratedClassifierCV(
            estimator=SVC(probability=False, random_state=random_state),
            method="sigmoid", cv=3, ensemble=False,
        ), {
            "model__estimator__C": [0.1, 1.0, 10.0],
            "model__estimator__gamma": ["scale", 0.01, 0.1],
        }),
        ("Calibrated linear support vector machine", CalibratedClassifierCV(
            estimator=LinearSVC(max_iter=5000, random_state=random_state), cv=3
        ), {"model__estimator__C": [0.01, 0.1, 1.0, 10.0]}),
        ("Decision tree", DecisionTreeClassifier(random_state=random_state), {
            "model__max_depth": [None, 2, 3, 5], "model__min_samples_leaf": [2, 5, 10]
        }),
        ("Random forest", RandomForestClassifier(random_state=random_state, n_jobs=1), {
            "model__n_estimators": [100, 250], "model__max_features": ["sqrt", 0.75],
            "model__min_samples_leaf": [1, 3, 7]
        }),
        ("Extra trees", ExtraTreesClassifier(random_state=random_state, n_jobs=1), {
            "model__n_estimators": [100, 250], "model__max_features": ["sqrt", 0.75],
            "model__min_samples_leaf": [1, 3, 7]
        }),
        ("Gradient boosting", GradientBoostingClassifier(random_state=random_state), {
            "model__n_estimators": [50, 100], "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [1, 2]
        }),
        ("Histogram gradient boosting", HistGradientBoostingClassifier(random_state=random_state), {
            "model__max_iter": [75, 150], "model__learning_rate": [0.03, 0.1],
            "model__max_leaf_nodes": [7, 15]
        }),
        ("AdaBoost", AdaBoostClassifier(random_state=random_state), {
            "model__n_estimators": [25, 50, 100], "model__learning_rate": [0.03, 0.1, 0.5]
        }),
        ("Gaussian naive Bayes", GaussianNB(), {"model__var_smoothing": [1e-11, 1e-9, 1e-7]}),
        ("Linear discriminant analysis", LinearDiscriminantAnalysis(solver="lsqr"), {
            "model__shrinkage": [None, "auto", 0.1, 0.5]
        }),
        ("Quadratic discriminant analysis", QuadraticDiscriminantAnalysis(), {
            "model__reg_param": [0.0, 0.1, 0.5, 0.9]
        }),
        ("Multilayer perceptron", MLPClassifier(
            max_iter=1200, early_stopping=True, random_state=random_state
        ), {"model__hidden_layer_sizes": [(8,), (16,), (8, 4)], "model__alpha": [0.001, 0.01, 0.1]}),
    ]
    variants = (False, True) if include_feature_selection else (False,)
    return [
        ModelSpec(name, estimator, parameters, select_features=selected, n_iter=3)
        for name, estimator, parameters in families
        for selected in variants
    ]


def threshold_metrics(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute class-1 sensitivity/specificity and other threshold metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _positive_class_probability(estimator, X: pd.DataFrame) -> np.ndarray:
    """Get probabilities for label 1, regardless of estimator class order."""
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Nested evaluation requires estimators with predict_proba")
    classes = np.asarray(estimator.classes_)
    positive_column = np.flatnonzero(classes == 1)
    if positive_column.size != 1:
        raise ValueError("Fitted estimator must expose class label 1")
    return estimator.predict_proba(X)[:, positive_column[0]]


def evaluate_nested_cv(
    frame: pd.DataFrame,
    model_specs: list[ModelSpec],
    outer_cv,
    inner_cv,
    scoring: str = "average_precision",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models using paired outer folds and fold-local inner searches.

    Scaling, feature selection, and hyperparameter search are fit using only
    the current outer-training partition. The returned prediction frame has
    one held-out prediction per input index for each model specification.
    """
    X, y = classification_features(frame)
    if y.nunique() != 2 or set(y.unique()) != {0, 1}:
        raise ValueError("Nested classification requires both target classes 0 and 1")
    if not model_specs:
        raise ValueError("At least one model specification is required")

    outer_splits = list(outer_cv.split(X, y))
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold_id, (train_positions, test_positions) in enumerate(outer_splits, start=1):
        X_train, X_test = X.iloc[train_positions], X.iloc[test_positions]
        y_train, y_test = y.iloc[train_positions], y.iloc[test_positions]

        for spec in model_specs:
            pipeline = make_classification_workflow(
                spec.estimator, select_features=spec.select_features
            )
            search_space = dict(spec.param_distributions)
            if spec.select_features:
                search_space["select__k"] = list(range(1, X.shape[1] + 1))
            if not search_space:
                search_space = {"model": [spec.estimator]}

            finite_choices = all(
                isinstance(values, (list, tuple, np.ndarray))
                for values in search_space.values()
            )
            n_iter = min(spec.n_iter, len(ParameterGrid(search_space))) if finite_choices else spec.n_iter

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=search_space,
                n_iter=n_iter,
                scoring=scoring,
                cv=inner_cv,
                refit=True,
                random_state=random_state + fold_id,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            probability = _positive_class_probability(best, X_test)
            threshold_results = threshold_metrics(y_test, probability, threshold=0.5)
            metric_rows.append(
                {
                    "model": spec.name,
                    "feature_selection": spec.select_features,
                    "fold": fold_id,
                    "roc_auc": float(roc_auc_score(y_test, probability)),
                    "average_precision": float(average_precision_score(y_test, probability)),
                    **threshold_results,
                    "best_params": search.best_params_,
                    "selected_features": (
                        list(X.columns[best.named_steps["select"].get_support()])
                        if spec.select_features
                        else list(X.columns)
                    ),
                }
            )
            for row_index, y_observed, p in zip(X_test.index, y_test, probability):
                prediction_rows.append(
                    {
                        "index": row_index,
                        "model": spec.name,
                        "feature_selection": spec.select_features,
                        "fold": fold_id,
                        "y_true": int(y_observed),
                        "probability_class_1": float(p),
                        "prediction_at_0_5": int(p >= 0.5),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows).set_index("index")
    return metrics, predictions
