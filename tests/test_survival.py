import numpy as np
import pandas as pd
import pytest

from pro_gradu.cli import main as cli_main
from pro_gradu.data import BASELINE_FEATURES
from pro_gradu.survival import (
    evaluate_survival_engine,
    fit_survival_engine,
    load_survival_engine,
    proportional_hazards_diagnostic,
)


def survival_frame(n=80, seed=17):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(n, len(BASELINE_FEATURES))), columns=BASELINE_FEATURES
    )
    for column in ("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"):
        frame[column] = rng.integers(0, 2, size=n)
    frame["time"] = rng.integers(5, 286, size=n)
    frame["DEATH_EVENT"] = np.arange(n) % 2
    return frame


def test_fit_uses_duration_and_keeps_censored_observations():
    frame = survival_frame()

    engine = fit_survival_engine(frame, penalizer=0.1)

    assert engine.feature_names == BASELINE_FEATURES
    assert engine.max_followup_days == frame["time"].max()
    assert engine.model.event_observed.sum() == frame["DEATH_EVENT"].sum()
    assert len(engine.model.event_observed) == len(frame)


def test_prediction_returns_horizon_survival_and_risk_probabilities():
    engine = fit_survival_engine(survival_frame(), penalizer=0.1)
    patients = survival_frame(3).drop(columns=["time", "DEATH_EVENT"])
    patients.index = [101, 205, 309]

    predictions = engine.predict(patients, horizons_days=[30, 90, 180])

    assert list(predictions.columns) == [
        "patient_index",
        "horizon_days",
        "survival_probability",
        "event_risk",
    ]
    assert set(predictions["patient_index"]) == {101, 205, 309}
    assert predictions.groupby("patient_index").size().eq(3).all()
    assert predictions["event_risk"].between(0, 1).all()
    for _, patient_predictions in predictions.groupby("patient_index"):
        assert patient_predictions["event_risk"].is_monotonic_increasing
        assert np.allclose(
            patient_predictions["survival_probability"],
            1 - patient_predictions["event_risk"],
        )


def test_prediction_rejects_followup_and_event_as_input_features():
    engine = fit_survival_engine(survival_frame(), penalizer=0.1)
    patients = survival_frame(2)

    with pytest.raises(ValueError, match="Unexpected prediction columns"):
        engine.predict(patients, horizons_days=[30])


def test_prediction_rejects_nonfinite_features():
    engine = fit_survival_engine(survival_frame(), penalizer=0.1)
    patients = survival_frame(2).loc[:, BASELINE_FEATURES]
    patients.iloc[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite numeric values"):
        engine.predict(patients, horizons_days=[30])


def test_prediction_rejects_nonbinary_clinical_flags():
    engine = fit_survival_engine(survival_frame(), penalizer=0.1)
    patients = survival_frame(2).loc[:, BASELINE_FEATURES]
    patients.loc[patients.index[0], "smoking"] = 2

    with pytest.raises(ValueError, match="Binary feature 'smoking'"):
        engine.predict(patients, horizons_days=[30])


def test_fit_rejects_nonbinary_clinical_flags():
    frame = survival_frame()
    frame.loc[frame.index[0], "diabetes"] = -1

    with pytest.raises(ValueError, match="Binary feature 'diabetes'"):
        fit_survival_engine(frame)


def test_fit_rejects_duplicate_feature_columns():
    frame = survival_frame()
    frame.insert(len(frame.columns), "age", frame["age"], allow_duplicates=True)

    with pytest.raises(ValueError, match="column names must be unique"):
        fit_survival_engine(frame)


def test_prediction_rejects_horizons_outside_observed_followup():
    engine = fit_survival_engine(survival_frame(), penalizer=0.1)
    patients = survival_frame(2).loc[:, BASELINE_FEATURES]

    with pytest.raises(ValueError, match="exceeds maximum training follow-up"):
        engine.predict(patients, horizons_days=[engine.max_followup_days + 1])


def test_saved_engine_preserves_horizon_predictions(tmp_path):
    engine = fit_survival_engine(survival_frame(), penalizer=0.1)
    patients = survival_frame(3).loc[:, BASELINE_FEATURES]
    expected = engine.predict(patients, horizons_days=[30, 90])
    artifact_path = tmp_path / "cox-model.joblib"

    engine.save(artifact_path)
    restored = load_survival_engine(artifact_path)
    actual = restored.predict(patients, horizons_days=[30, 90])

    pd.testing.assert_frame_equal(expected, actual)
    assert restored.penalizer == 0.1


@pytest.mark.parametrize("bad_penalizer", [0, -1, np.inf, np.nan, "invalid"])
def test_fit_rejects_invalid_penalizer(bad_penalizer):
    with pytest.raises(ValueError, match="penalizer must be finite"):
        fit_survival_engine(survival_frame(), penalizer=bad_penalizer)


def test_nested_survival_validation_returns_one_held_out_risk_per_patient():
    frame = survival_frame(80)

    fold_metrics, predictions = evaluate_survival_engine(
        frame,
        penalizers=(0.1, 1.0),
        outer_folds=3,
        inner_folds=2,
        random_state=21,
    )

    assert len(fold_metrics) == 3
    assert predictions.index.is_unique
    assert set(predictions.index) == set(frame.index)
    assert set(predictions["fold"]) == {1, 2, 3}
    assert predictions["risk_score"].map(np.isfinite).all()
    assert fold_metrics["concordance_index"].between(0, 1).all()
    assert set(fold_metrics["selected_penalizer"]).issubset({0.1, 1.0})


def test_nested_validation_requires_events_and_censored_observations():
    frame = survival_frame()
    frame["DEATH_EVENT"] = 1

    with pytest.raises(ValueError, match="both observed events and censored cases"):
        evaluate_survival_engine(frame, outer_folds=3, inner_folds=2)


def test_ph_diagnostic_returns_one_row_per_feature():
    frame = survival_frame()
    engine = fit_survival_engine(frame, penalizer=0.1)

    diagnostic = proportional_hazards_diagnostic(engine, frame)

    assert set(diagnostic["feature"]) == set(BASELINE_FEATURES)
    assert diagnostic["p_value"].between(0, 1).all()


def test_fit_and_predict_cli_write_model_and_horizon_results(tmp_path):
    source = survival_frame(80)
    source_path = tmp_path / "training.csv"
    model_path = tmp_path / "engine.joblib"
    patients_path = tmp_path / "patients.csv"
    output_path = tmp_path / "predictions.csv"
    source.to_csv(source_path, index=False)
    source.loc[:2, BASELINE_FEATURES].to_csv(patients_path, index=False)

    assert cli_main(
        ["fit-survival", str(source_path), "--model-out", str(model_path)]
    ) == 0
    assert cli_main(
        [
            "predict-survival",
            str(patients_path),
            "--model",
            str(model_path),
            "--horizons",
            "30,90",
            "--output",
            str(output_path),
        ]
    ) == 0

    predictions = pd.read_csv(output_path)
    assert len(predictions) == 6
    assert set(predictions["horizon_days"]) == {30, 90}
    assert predictions["event_risk"].between(0, 1).all()


def test_validate_cli_writes_nested_metrics_and_oof_predictions(tmp_path):
    source_path = tmp_path / "training.csv"
    output_dir = tmp_path / "validation"
    survival_frame().to_csv(source_path, index=False)

    assert cli_main(
        [
            "validate-survival",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--outer-folds",
            "3",
            "--inner-folds",
            "2",
            "--penalizers",
            "0.1,1.0",
        ]
    ) == 0

    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    predictions = pd.read_csv(output_dir / "out_of_fold_predictions.csv")
    diagnostic = pd.read_csv(output_dir / "proportional_hazards.csv")
    assert len(fold_metrics) == 3
    assert len(predictions) == 80
    assert len(diagnostic) == len(BASELINE_FEATURES)
