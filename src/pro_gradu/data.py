"""Dataset loading and schema validation."""

from pathlib import Path

import pandas as pd

BASELINE_FEATURES = (
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
)
BINARY_FEATURES = (
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
)
REQUIRED_COLUMNS = (*BASELINE_FEATURES, "time", "DEATH_EVENT")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read and validate the supplied clinical-record CSV.

    Identical rows are not assumed to be duplicate patients: the public source
    does not provide patient identifiers, so repeated measurements are retained.
    """
    frame = pd.read_csv(path)

    if not frame.columns.is_unique:
        duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Dataset has duplicate column names: {duplicate_columns}")

    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if frame.isna().any().any():
        missing_values = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Dataset contains missing values in columns: {missing_values}")

    try:
        target = pd.to_numeric(frame["DEATH_EVENT"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("DEATH_EVENT must contain only binary values 0 and 1") from exc
    if not target.isin((0, 1)).all():
        raise ValueError("DEATH_EVENT must contain only binary values 0 and 1")
    frame["DEATH_EVENT"] = target.astype("int8")

    try:
        followup = pd.to_numeric(frame["time"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("time must contain positive numeric follow-up days") from exc
    if not (followup > 0).all():
        raise ValueError("time must contain positive follow-up days")
    frame["time"] = followup

    return frame
