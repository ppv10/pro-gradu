"""Nested validation and censoring-aware metrics for survival models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import BASELINE_FEATURES
from .survival import _validate_training_frame


def _lifelines_model_classes():
    try:
        from lifelines import CoxPHFitter, WeibullAFTFitter
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Survival validation requires lifelines. "
            "Install it with `uv sync --extra survival`."
        ) from exc
    return {"cox_ph": CoxPHFitter, "weibull_aft": WeibullAFTFitter}


def _fit_model(model_name: str, frame: pd.DataFrame, penalizer: float):
    classes = _lifelines_model_classes()
    try:
        model_class = classes[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported survival model: {model_name!r}") from exc
    model = model_class(penalizer=penalizer, l1_ratio=0.0)
    model.fit(frame, duration_col="time", event_col="DEATH_EVENT")
    return model


def _risk_score(model_name: str, model, patients: pd.DataFrame) -> np.ndarray:
    if model_name == "cox_ph":
        # lifelines' concordance_index treats larger scores as longer survival.
        return -np.asarray(model.predict_partial_hazard(patients), dtype=float).reshape(-1)
    # For AFT, a larger median lifetime means a lower event risk.
    return np.asarray(model.predict_median(patients), dtype=float).reshape(-1)


def _concordance(model_name: str, model, frame: pd.DataFrame) -> float:
    from lifelines.utils import concordance_index

    score = _risk_score(model_name, model, frame.loc[:, BASELINE_FEATURES])
    try:
        return float(concordance_index(frame["time"], score, frame["DEATH_EVENT"]))
    except ZeroDivisionError as exc:
        raise ValueError("Survival fold has no comparable event pairs") from exc


def _censoring_survival_at(
    training: pd.DataFrame, query_times: np.ndarray, *, left_limit: bool = False
) -> np.ndarray:
    """Estimate G(t)=P(C>t) by reverse Kaplan-Meier on training data."""
    durations = training["time"].to_numpy(dtype=float)
    censoring_events = (training["DEATH_EVENT"].to_numpy(dtype=int) == 0)
    event_times = np.unique(durations[censoring_events])
    survival_after_time = []
    survival = 1.0
    at_risk = len(training)
    for time in np.unique(durations):
        at_time = durations == time
        censorings = int(np.sum(censoring_events & at_time))
        if censorings:
            survival *= 1.0 - censorings / at_risk
        if time in event_times:
            survival_after_time.append((time, survival))
        at_risk -= int(at_time.sum())

    if not survival_after_time:
        return np.ones_like(query_times, dtype=float)
    censor_times = np.asarray([item[0] for item in survival_after_time])
    censor_survival = np.asarray([item[1] for item in survival_after_time])
    side = "left" if left_limit else "right"
    positions = np.searchsorted(censor_times, query_times, side=side) - 1
    result = np.ones_like(query_times, dtype=float)
    observed = positions >= 0
    result[observed] = censor_survival[positions[observed]]
    return result


def _ipcw_outcomes(
    training: pd.DataFrame, validation: pd.DataFrame, horizon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return known event-by-horizon labels and training-fold IPCW weights."""
    durations = validation["time"].to_numpy(dtype=float)
    events = validation["DEATH_EVENT"].to_numpy(dtype=int).astype(bool)
    observed_by_horizon = events & (durations <= horizon)
    known_negative = durations > horizon
    known = observed_by_horizon | known_negative
    if not known.any():
        raise ValueError(
            f"No validation outcomes are observable at {horizon:g} days"
        )
    labels = np.full(len(validation), np.nan, dtype=float)
    labels[observed_by_horizon] = 1.0
    labels[known_negative] = 0.0

    weights = np.zeros(len(validation), dtype=float)
    event_positions = np.flatnonzero(observed_by_horizon)
    negative_positions = np.flatnonzero(known_negative)
    if event_positions.size:
        g_event = _censoring_survival_at(
            training, durations[event_positions], left_limit=True
        )
        if np.any(g_event <= 0):
            raise ValueError(
                f"Cannot estimate IPCW at {horizon:g} days: censoring survival is zero"
            )
        weights[event_positions] = 1.0 / g_event
    if negative_positions.size:
        g_horizon = float(
            _censoring_survival_at(
                training, np.asarray([horizon], dtype=float)
            )[0]
        )
        if g_horizon <= 0:
            raise ValueError(
                f"Cannot estimate IPCW at {horizon:g} days: censoring survival is zero"
            )
        weights[negative_positions] = 1.0 / g_horizon
    return labels, weights, known


def _calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model_name, horizon), group in predictions.groupby(
        ["model", "horizon_days"], sort=True
    ):
        known = group["event_by_horizon"].notna()
        scored = group.loc[known].copy()
        if scored.empty:
            continue
        unique_risks = scored["event_risk"].nunique()
        n_bins = min(5, unique_risks)
        if n_bins <= 1:
            scored["risk_bin"] = 1
        else:
            scored["risk_bin"] = (
                pd.qcut(scored["event_risk"], q=n_bins, labels=False, duplicates="drop")
                .astype(int)
                + 1
            )
        for bin_number, bin_rows in scored.groupby("risk_bin", sort=True):
            weights = bin_rows["ipcw_weight"].to_numpy(dtype=float)
            total_weight = float(weights.sum())
            if total_weight <= 0:
                continue
            rows.append(
                {
                    "model": model_name,
                    "horizon_days": float(horizon),
                    "risk_bin": int(bin_number),
                    "n_known": int(len(bin_rows)),
                    "effective_n": float(total_weight**2 / np.square(weights).sum()),
                    "mean_predicted_risk": float(
                        np.average(bin_rows["event_risk"], weights=weights)
                    ),
                    "ipcw_observed_event_rate": float(
                        np.average(bin_rows["event_by_horizon"], weights=weights)
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "model",
            "horizon_days",
            "risk_bin",
            "n_known",
            "effective_n",
            "mean_predicted_risk",
            "ipcw_observed_event_rate",
        ),
    )


def evaluate_survival_models(
    frame: pd.DataFrame,
    horizons_days: Sequence[float] = (90.0, 180.0),
    penalizers: Sequence[float] = (0.01, 0.1, 1.0),
    outer_folds: int = 5,
    inner_folds: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested CV for Cox PH and Weibull AFT with horizon-specific metrics.

    Harrell concordance selects each model's penalty within the training data.
    IPCW Brier scores and calibration use censoring distributions fitted on the
    corresponding outer-training fold. Calibration rows aggregate the
    cross-fitted predictions into up to five risk bins.
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

    try:
        horizons = tuple(float(value) for value in horizons_days)
        grid = tuple(float(value) for value in penalizers)
    except (TypeError, ValueError) as exc:
        raise ValueError("horizons and penalizers must be numeric sequences") from exc
    if (
        not horizons
        or any(not np.isfinite(value) or value <= 0 for value in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("horizons must be unique, finite, positive numbers")
    if any(value > float(data["time"].max()) for value in horizons):
        raise ValueError("A prediction horizon exceeds maximum observed follow-up")
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("penalizers must contain finite positive values")
    if len(set(grid)) != len(grid):
        raise ValueError("penalizers must be unique")

    outer_cv = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=random_state
    )
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    models = ("cox_ph", "weibull_aft")

    for fold, (train_positions, test_positions) in enumerate(
        outer_cv.split(data, data["DEATH_EVENT"]), start=1
    ):
        train = data.iloc[train_positions]
        test = data.iloc[test_positions]
        if any(horizon > float(train["time"].max()) for horizon in horizons):
            raise ValueError(
                "A prediction horizon exceeds an outer training fold's follow-up"
            )
        inner_cv = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=random_state + fold,
        )

        for model_name in models:
            inner_scores: dict[float, list[float]] = {penalty: [] for penalty in grid}
            for inner_train_positions, inner_valid_positions in inner_cv.split(
                train, train["DEATH_EVENT"]
            ):
                inner_train = train.iloc[inner_train_positions]
                inner_valid = train.iloc[inner_valid_positions]
                for penalty in grid:
                    fitted = _fit_model(model_name, inner_train, penalty)
                    inner_scores[penalty].append(
                        _concordance(model_name, fitted, inner_valid)
                    )

            mean_scores = {
                penalty: float(np.mean(scores))
                for penalty, scores in inner_scores.items()
            }
            best_penalty = max(grid, key=lambda value: (mean_scores[value], value))
            fitted = _fit_model(model_name, train, best_penalty)
            outer_concordance = _concordance(model_name, fitted, test)
            fold_risk_score = _risk_score(
                model_name, fitted, test.loc[:, BASELINE_FEATURES]
            )
            survival = fitted.predict_survival_function(
                test.loc[:, BASELINE_FEATURES], times=list(horizons)
            ).to_numpy(dtype=float).T

            for horizon_index, horizon in enumerate(horizons):
                labels, weights, known = _ipcw_outcomes(train, test, horizon)
                risk = np.clip(1.0 - survival[:, horizon_index], 0.0, 1.0)
                brier = float(
                    np.sum(weights[known] * np.square(risk[known] - labels[known]))
                    / len(test)
                )
                effective_n = (
                    float(weights[known].sum() ** 2 / np.square(weights[known]).sum())
                    if np.any(known) and np.square(weights[known]).sum() > 0
                    else 0.0
                )
                metric_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "horizon_days": horizon,
                        "concordance_index": outer_concordance,
                        "ipcw_brier_score": brier,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_test_events": int(test["DEATH_EVENT"].sum()),
                        "n_known_at_horizon": int(known.sum()),
                        "effective_n_at_horizon": effective_n,
                        "selected_penalizer": best_penalty,
                        "inner_mean_concordance": mean_scores[best_penalty],
                    }
                )
                for position, (row_index, row) in enumerate(test.iterrows()):
                    prediction_rows.append(
                        {
                            "row_index": row_index,
                            "fold": fold,
                            "model": model_name,
                            "horizon_days": horizon,
                            "followup_days": float(row["time"]),
                            "observed_event": int(row["DEATH_EVENT"]),
                            "risk_score": float(fold_risk_score[position]),
                            "survival_probability": float(1.0 - risk[position]),
                            "event_risk": float(risk[position]),
                            "event_by_horizon": (
                                float(labels[position]) if known[position] else np.nan
                            ),
                            "ipcw_weight": float(weights[position]),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    calibration = _calibration_table(predictions)
    return metrics, predictions, calibration
