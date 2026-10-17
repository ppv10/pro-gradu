"""Command-line interface for training and using the survival engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .data import BASELINE_FEATURES, load_dataset
from .survival import fit_survival_engine, load_survival_engine, proportional_hazards_diagnostic
from .survival_validation import evaluate_survival_models


def _parse_float_list(raw: str) -> tuple[float, ...]:
    try:
        return tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def _write_csv(frame: pd.DataFrame, path: str | Path, include_index: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=include_index)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pro-gradu",
        description="Censoring-aware survival modeling for heart-failure records.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser(
        "fit-survival", help="fit a penalized Cox model and save its artifact"
    )
    fit.add_argument("data", help="CSV with baseline features, time, and DEATH_EVENT")
    fit.add_argument("--model-out", required=True, help="output joblib model path")
    fit.add_argument("--penalizer", type=float, default=1.0, help="positive L2 penalty")

    predict = commands.add_parser(
        "predict-survival", help="predict survival probabilities at day horizons"
    )
    predict.add_argument("data", help="CSV with baseline features only")
    predict.add_argument("--model", required=True, help="trusted joblib model path")
    predict.add_argument("--horizons", type=_parse_float_list, default=(30.0, 90.0, 180.0))
    predict.add_argument("--output", required=True, help="output CSV path")

    validate = commands.add_parser(
        "validate-survival", help="nested event-stratified survival validation"
    )
    validate.add_argument("data", help="CSV with baseline features, time, and DEATH_EVENT")
    validate.add_argument("--output-dir", required=True, help="directory for validation CSVs")
    validate.add_argument("--outer-folds", type=int, default=5)
    validate.add_argument("--inner-folds", type=int, default=4)
    validate.add_argument(
        "--horizons",
        type=_parse_float_list,
        default=(90.0, 180.0),
        help="comma-separated days for IPCW Brier scores and calibration",
    )
    validate.add_argument(
        "--penalizers", type=_parse_float_list, default=(0.01, 0.1, 1.0)
    )
    validate.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "fit-survival":
        data = load_dataset(args.data)
        engine = fit_survival_engine(data, penalizer=args.penalizer)
        engine.save(args.model_out)
        print(
            json.dumps(
                {
                    "model": str(args.model_out),
                    "model_type": "penalized_cox_ph",
                    "n_patients": len(data),
                    "n_events": int(data["DEATH_EVENT"].sum()),
                    "n_censored": int((data["DEATH_EVENT"] == 0).sum()),
                    "max_followup_days": engine.max_followup_days,
                    "penalizer": engine.penalizer,
                    "features": list(BASELINE_FEATURES),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "predict-survival":
        patients = pd.read_csv(args.data)
        engine = load_survival_engine(args.model)
        predictions = engine.predict(patients, horizons_days=args.horizons)
        _write_csv(predictions, args.output)
        print(
            json.dumps(
                {
                    "predictions": str(args.output),
                    "n_patients": len(patients),
                    "horizons_days": sorted(args.horizons),
                    "note": "Model estimates are exploratory, not clinical guidance.",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate-survival":
        data = load_dataset(args.data)
        fold_metrics, predictions, calibration = evaluate_survival_models(
            data,
            horizons_days=args.horizons,
            penalizers=args.penalizers,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            random_state=args.random_state,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(fold_metrics, output_dir / "model_comparison_fold_metrics.csv")
        _write_csv(predictions, output_dir / "model_comparison_predictions.csv")
        _write_csv(calibration, output_dir / "calibration.csv")

        cox_metrics = (
            fold_metrics.loc[fold_metrics["model"] == "cox_ph"]
            .drop_duplicates("fold")
            .loc[
                :,
                [
                    "fold",
                    "concordance_index",
                    "selected_penalizer",
                    "inner_mean_concordance",
                    "n_train",
                    "n_test",
                    "n_test_events",
                ],
            ]
        )
        _write_csv(cox_metrics, output_dir / "fold_metrics.csv")
        cox_oof = predictions.loc[
            (predictions["model"] == "cox_ph")
            & (predictions["horizon_days"] == float(args.horizons[0]))
        ].copy()
        cox_oof = cox_oof.rename(
            columns={"followup_days": "time", "observed_event": "DEATH_EVENT"}
        ).set_index("row_index")
        selected_penalty_by_fold = cox_metrics.set_index("fold")["selected_penalizer"]
        cox_oof["selected_penalizer"] = cox_oof["fold"].map(
            selected_penalty_by_fold
        )
        cox_oof = cox_oof.loc[
            :, ["fold", "time", "DEATH_EVENT", "risk_score", "selected_penalizer"]
        ]
        _write_csv(cox_oof, output_dir / "out_of_fold_predictions.csv", include_index=True)

        cox_penalties = fold_metrics.loc[
            fold_metrics["model"] == "cox_ph", "selected_penalizer"
        ]
        chosen_penalty = float(cox_penalties.mode().iloc[0])
        full_cox_engine = fit_survival_engine(data, penalizer=chosen_penalty)
        diagnostic = proportional_hazards_diagnostic(full_cox_engine, data)
        _write_csv(diagnostic, output_dir / "proportional_hazards.csv")

        model_summaries = {}
        for model_name, model_rows in fold_metrics.groupby("model", sort=True):
            horizon_scores = {}
            model_predictions = predictions.loc[predictions["model"] == model_name]
            fold_concordance = model_rows.groupby("fold")["concordance_index"].first()
            for horizon in args.horizons:
                horizon_rows = model_predictions.loc[
                    model_predictions["horizon_days"] == float(horizon)
                ]
                known = horizon_rows["event_by_horizon"].notna()
                pooled_brier = float(
                    (
                        horizon_rows.loc[known, "ipcw_weight"]
                        * (
                            horizon_rows.loc[known, "event_risk"]
                            - horizon_rows.loc[known, "event_by_horizon"]
                        ) ** 2
                    ).sum()
                    / len(horizon_rows)
                )
                horizon_scores[str(float(horizon))] = {
                    "pooled_ipcw_brier_score": pooled_brier,
                    "n_known_at_horizon": int(known.sum()),
                }
            selected = model_rows["selected_penalizer"]
            model_summaries[model_name] = {
                "mean_outer_concordance": float(fold_concordance.mean()),
                "sd_outer_concordance": float(fold_concordance.std(ddof=1)),
                "selected_penalizer_mode": float(selected.mode().iloc[0]),
                "horizons": horizon_scores,
            }
        summary = {
            "models": model_summaries,
            "n_patients": len(data),
            "n_events": int(data["DEATH_EVENT"].sum()),
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "horizons_days": [float(value) for value in args.horizons],
            "legacy_cox_outputs_note": (
                "fold_metrics.csv and out_of_fold_predictions.csv retain the original Cox-only format."
            ),
            "ipcw_note": "Censoring distributions are estimated separately in each outer training fold.",
            "calibration_note": "Cross-fitted IPCW calibration bins are descriptive and may be noisy in this cohort.",
            "ph_diagnostic_note": "The Schoenfeld-residual diagnostic applies to the full-data Cox fit only.",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")

