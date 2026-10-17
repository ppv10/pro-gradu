"""Censoring-aware survival prediction for the heart-failure cohort."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import BASELINE_FEATURES, BINARY_FEATURES

_ARTIFACT_VERSION = 1
_OUTCOME_COLUMNS = ("time", "DEATH_EVENT")
_PREDICTION_COLUMNS = (
    "patient_index",
    "horizon_days",
    "survival_probability",
    "event_risk",
)


def _lifelines_cox_class():
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ImportError(
            "Survival modeling requires the optional dependencies. "
            "Install them with `uv sync --extra survival`."
        ) from exc
    return CoxPHFitter


def _validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Training data must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Training column names must be unique")
    expected = set((*BASELINE_FEATURES, *_OUTCOME_COLUMNS))
    missing = sorted(expected.difference(frame.columns))
    unexpected = sorted(set(frame.columns).difference(expected))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected training columns: {unexpected}")
    if frame.empty:
        raise ValueError("Training data must contain at least one row")

    data = frame.loc[:, [*BASELINE_FEATURES, *_OUTCOME_COLUMNS]].copy()
    for column in data.columns:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {column!r} must contain numeric values") from exc
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training data must contain only finite numeric values")
    for column in BINARY_FEATURES:
        if not data[column].isin((0, 1)).all():
            raise ValueError(f"Binary feature {column!r} must contain only 0 or 1")
    if (data["time"] <= 0).any():
        raise ValueError("Follow-up time must be positive")
    if not data["DEATH_EVENT"].isin((0, 1)).all():
        raise ValueError("DEATH_EVENT must contain only binary values 0 and 1")
    data["DEATH_EVENT"] = data["DEATH_EVENT"].astype("int8")
    if int(data["DEATH_EVENT"].sum()) < 2:
        raise ValueError("At least two observed death events are required to fit")
    return data


def _validate_prediction_frame(
    patients: pd.DataFrame, feature_names: tuple[str, ...]
) -> pd.DataFrame:
    if not isinstance(patients, pd.DataFrame):
        raise TypeError("Prediction inputs must be a pandas DataFrame")
    if not patients.columns.is_unique:
        raise ValueError("Prediction column names must be unique")
    missing = sorted(set(feature_names).difference(patients.columns))
    unexpected = sorted(set(patients.columns).difference(feature_names))
    if missing:
        raise ValueError(f"Prediction data is missing required features: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected prediction columns: {unexpected}")
    if patients.empty:
        raise ValueError("Prediction data must contain at least one patient")

    data = patients.loc[:, feature_names].copy()
    for column in feature_names:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {column!r} must contain numeric values") from exc
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Prediction features must contain only finite numeric values")
    for column in BINARY_FEATURES:
        if not data[column].isin((0, 1)).all():
            raise ValueError(f"Binary feature {column!r} must contain only 0 or 1")
    return data


def _validate_horizons(
    horizons_days: Sequence[float], max_followup_days: float
) -> tuple[float, ...]:
    try:
        horizons = tuple(float(value) for value in horizons_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("horizons_days must be a sequence of positive numbers") from exc
    if not horizons:
        raise ValueError("At least one prediction horizon is required")
    if not np.isfinite(horizons).all() or any(value <= 0 for value in horizons):
        raise ValueError("Prediction horizons must be finite and positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Prediction horizons must be unique")
    if any(value > max_followup_days for value in horizons):
        raise ValueError(
            "Prediction horizon exceeds maximum training follow-up "
            f"({max_followup_days:g} days)"
        )
    return tuple(sorted(horizons))


@dataclass
class SurvivalPredictionEngine:
    """Fitted penalized Cox model and the schema needed to use it safely."""

    model: object
    feature_names: tuple[str, ...]
    max_followup_days: float
    penalizer: float

    def predict(
        self, patients: pd.DataFrame, horizons_days: Sequence[float]
    ) -> pd.DataFrame:
        """Return model survival and event-risk estimates at requested days.

        The event risk is ``1 - S(t)``. These are model estimates at fixed
        horizons, not clinical categories or individualized confidence bounds.
        """
        inputs = _validate_prediction_frame(patients, self.feature_names)
        horizons = _validate_horizons(horizons_days, self.max_followup_days)
        survival = self.model.predict_survival_function(
            inputs, times=list(horizons)
        ).to_numpy(dtype=float).T
        rows = []
        for row_index, probabilities in zip(inputs.index, survival):
            for horizon, probability in zip(horizons, probabilities):
                probability = float(np.clip(probability, 0.0, 1.0))
                rows.append(
                    {
                        "patient_index": row_index,
                        "horizon_days": horizon,
                        "survival_probability": probability,
                        "event_risk": 1.0 - probability,
                    }
                )
        return pd.DataFrame(rows, columns=_PREDICTION_COLUMNS)

    def save(self, path: str | Path) -> None:
        """Save a versioned model artifact. Only load trusted artifacts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "artifact_version": _ARTIFACT_VERSION,
            "feature_names": self.feature_names,
            "max_followup_days": self.max_followup_days,
            "penalizer": self.penalizer,
            "model": self.model,
        }
        joblib.dump(artifact, path)


def fit_survival_engine(
    frame: pd.DataFrame, penalizer: float = 1.0
) -> SurvivalPredictionEngine:
    """Fit a ridge-penalized Cox model using right-censored durations."""
    try:
        penalizer = float(penalizer)
    except (TypeError, ValueError) as exc:
        raise ValueError("penalizer must be finite and greater than zero") from exc
    if not np.isfinite(penalizer) or penalizer <= 0:
        raise ValueError("penalizer must be finite and greater than zero")
    data = _validate_training_frame(frame)
    cox_class = _lifelines_cox_class()
    model = cox_class(penalizer=float(penalizer), l1_ratio=0.0)
    model.fit(data, duration_col="time", event_col="DEATH_EVENT")
    return SurvivalPredictionEngine(
        model=model,
        feature_names=tuple(BASELINE_FEATURES),
        max_followup_days=float(data["time"].max()),
        penalizer=penalizer,
    )


def load_survival_engine(path: str | Path) -> SurvivalPredictionEngine:
    """Load a versioned model artifact created by this package.

    Joblib uses pickle internally; never load artifacts from untrusted sources.
    """
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != _ARTIFACT_VERSION:
        raise ValueError("Unsupported or invalid survival model artifact")
    expected_features = tuple(BASELINE_FEATURES)
    if tuple(artifact.get("feature_names", ())) != expected_features:
        raise ValueError("Model artifact feature schema does not match this package")
    model = artifact.get("model")
    if not hasattr(model, "predict_survival_function"):
        raise ValueError("Model artifact does not contain a Cox survival estimator")
    return SurvivalPredictionEngine(
        model=model,
        feature_names=expected_features,
        max_followup_days=float(artifact["max_followup_days"]),
        penalizer=float(artifact["penalizer"]),
    )


def _harrell_concordance(model, frame: pd.DataFrame) -> float:
    from lifelines.utils import concordance_index

    risk_score = model.predict_partial_hazard(frame.loc[:, BASELINE_FEATURES])
    try:
        return float(
            concordance_index(
                frame["time"],
                -np.asarray(risk_score, dtype=float),
                frame["DEATH_EVENT"],
            )
        )
    except ZeroDivisionError as exc:
        raise ValueError("Survival fold has no comparable event pairs") from exc


def evaluate_survival_engine(
    frame: pd.DataFrame,
    penalizers: Sequence[float] = (0.01, 0.1, 1.0),
    outer_folds: int = 5,
    inner_folds: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune ridge penalty inside event-stratified folds and return held-out scores.

    Harrell's concordance is used for ranking risk. Each censored observation
    contributes its follow-up duration and censoring indicator to fitting and
    scoring; folds never use the outer test partition to tune the penalty.
    """
    data = _validate_training_frame(frame)
    if not data.index.is_unique:
        raise ValueError("Nested validation requires a unique row index")
    if data["DEATH_EVENT"].nunique() != 2:
        raise ValueError(
            "Nested survival validation requires both observed events and censored cases"
        )
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must both be at least two")
    grid = tuple(float(value) for value in penalizers)
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("penalizers must contain finite positive values")
    if len(set(grid)) != len(grid):
        raise ValueError("penalizers must be unique")

    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=random_state
    )
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    events = data["DEATH_EVENT"]
    for fold, (train_positions, test_positions) in enumerate(
        outer_cv.split(data, events), start=1
    ):
        train = data.iloc[train_positions]
        test = data.iloc[test_positions]
        inner_cv = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=random_state + fold
        )
        inner_scores: dict[float, list[float]] = {penalty: [] for penalty in grid}
        for inner_train_pos, inner_valid_pos in inner_cv.split(
            train, train["DEATH_EVENT"]
        ):
            inner_train = train.iloc[inner_train_pos]
            inner_valid = train.iloc[inner_valid_pos]
            for penalty in grid:
                model = fit_survival_engine(inner_train, penalizer=penalty).model
                inner_scores[penalty].append(_harrell_concordance(model, inner_valid))

        mean_scores = {
            penalty: float(np.mean(scores)) for penalty, scores in inner_scores.items()
        }
        # On a numerical tie, prefer stronger shrinkage for this small cohort.
        best_penalty = max(grid, key=lambda penalty: (mean_scores[penalty], penalty))
        engine = fit_survival_engine(train, penalizer=best_penalty)
        concordance = _harrell_concordance(engine.model, test)
        fold_rows.append(
            {
                "fold": fold,
                "concordance_index": concordance,
                "selected_penalizer": best_penalty,
                "inner_mean_concordance": mean_scores[best_penalty],
                "n_train": len(train),
                "n_test": len(test),
                "n_test_events": int(test["DEATH_EVENT"].sum()),
            }
        )
        risk = engine.model.predict_partial_hazard(test.loc[:, BASELINE_FEATURES])
        for row_index, row_risk in zip(test.index, risk):
            prediction_rows.append(
                {
                    "index": row_index,
                    "fold": fold,
                    "time": float(test.loc[row_index, "time"]),
                    "DEATH_EVENT": int(test.loc[row_index, "DEATH_EVENT"]),
                    "risk_score": float(row_risk),
                    "selected_penalizer": best_penalty,
                }
            )
    metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows).set_index("index")
    return metrics, predictions


def proportional_hazards_diagnostic(
    engine: SurvivalPredictionEngine, frame: pd.DataFrame
) -> pd.DataFrame:
    """Return Schoenfeld-residual proportional-hazards tests per feature.

    Small-sample p-values are diagnostic only and should not be treated as a
    reliable automated test of model validity.
    """
    from lifelines.statistics import proportional_hazard_test

    data = _validate_training_frame(frame)
    result = proportional_hazard_test(
        engine.model, data, time_transform="rank"
    ).summary
    return (
        result.rename_axis("feature")
        .reset_index()
        .rename(columns={"test_statistic": "test_statistic", "p": "p_value"})
    )
