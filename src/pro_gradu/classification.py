"""Leakage-safe classification features and sklearn pipelines."""

from functools import partial

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import BASELINE_FEATURES


def classification_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return baseline predictors and the observed-event target, excluding time."""
    missing = sorted(set((*BASELINE_FEATURES, "DEATH_EVENT")).difference(frame.columns))
    if missing:
        raise ValueError(f"Frame is missing classification columns: {missing}")
    X = frame.loc[:, BASELINE_FEATURES].copy()
    y = frame.loc[:, "DEATH_EVENT"].copy()
    return X, y


def make_classification_workflow(estimator, select_features: bool = False) -> Pipeline:
    """Build preprocessing and estimator steps for fitting inside CV folds.

    When selection is enabled, callers tune ``select__k`` using only inner-CV
    training folds. The score function has a fixed random seed for ties/noise.
    """
    selector = (
        SelectKBest(score_func=partial(mutual_info_classif, random_state=42))
        if select_features
        else "passthrough"
    )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", selector),
            ("model", estimator),
        ]
    )
