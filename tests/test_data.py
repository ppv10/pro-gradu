import pandas as pd
import pytest

from pro_gradu.data import load_dataset


def valid_frame():
    return pd.DataFrame({"age": [50], "anaemia": [0], "creatinine_phosphokinase": [100], "diabetes": [0], "ejection_fraction": [40], "high_blood_pressure": [0], "platelets": [200000.0], "serum_creatinine": [1.0], "serum_sodium": [140], "sex": [1], "smoking": [0], "time": [10], "DEATH_EVENT": [0]})


def test_load_dataset_rejects_missing_required_column(tmp_path):
    path = tmp_path / "missing.csv"
    valid_frame().drop(columns="age").to_csv(path, index=False)
    with pytest.raises(ValueError, match="age"):
        load_dataset(path)


def test_load_dataset_rejects_nonbinary_death_event(tmp_path):
    path = tmp_path / "bad_target.csv"
    frame = valid_frame()
    frame.loc[0, "DEATH_EVENT"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="DEATH_EVENT"):
        load_dataset(path)


def test_load_dataset_rejects_missing_values(tmp_path):
    path = tmp_path / "missing_value.csv"
    frame = valid_frame()
    frame.loc[0, "age"] = float("nan")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_dataset(path)


def test_load_dataset_requires_positive_followup_time(tmp_path):
    path = tmp_path / "bad_time.csv"
    frame = valid_frame()
    frame.loc[0, "time"] = 0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="time"):
        load_dataset(path)
