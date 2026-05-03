from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STANDARD_MODEL = SCRIPT_DIR.parent / "output" / "pu_model" / "pu_model.joblib"
DEFAULT_HARDENED_MODEL = SCRIPT_DIR.parent / "output" / "pu_model_hardened" / "pu_model.joblib"
DEFAULT_TEST_FEATURES = SCRIPT_DIR.parent / "output" / "test_results" / "enriched_features_audit.csv"
DEFAULT_TEST_OUTPUT = SCRIPT_DIR.parent / "output" / "test_results" / "model_classification_results.csv"

POSITIVE_STATUSES = {
    "phishing",
    "phishing_campaign",
    "source_verified_phishing",
}

TRUTH_COLUMN_CANDIDATES = (
    "ground_truth",
    "ground_truth_label",
    "actual_label",
    "true_label",
    "manual_label",
    "test_label",
    "expected_label",
    "Phishing/Suspected Domains (i.e. Class Label)",
    "Class Label",
    "source_label",
)

EMPTY_FEATURE_COLUMNS = [
    "url",
    "target_brand_domain",
    "validation_status",
]

MODEL_OUTPUT_COLUMNS = [
    "standard_score",
    "standard_threshold",
    "standard_prediction",
    "hardened_score",
    "hardened_threshold",
    "hardened_prediction",
]

CLASSIFICATION_OUTPUT_COLUMNS = [
    "final_classification",
    "classification_confidence",
    "recommended_action",
    "classification_reason",
]


def model_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for column in feature_columns:
        if column in df.columns:
            series = df[column]
            if series.dtype == bool:
                converted = series.astype(int)
            else:
                normalized = series.replace(
                    {
                        True: 1,
                        False: 0,
                        "True": 1,
                        "False": 0,
                        "true": 1,
                        "false": 0,
                    }
                )
                converted = pd.to_numeric(normalized, errors="coerce")
            values[column] = converted.replace([np.inf, -np.inf], np.nan).fillna(-1)
        else:
            values[column] = pd.Series([-1] * len(df))
    return pd.DataFrame(values)


def score_model(df: pd.DataFrame, model_path: Path, model_name: str) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact["threshold"])
    X = model_feature_frame(df, feature_columns)
    scores = artifact["model"].predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "model": model_name,
            "score": scores,
            "threshold": threshold,
            "prediction": (scores >= threshold).astype(int),
            "feature_count": len(feature_columns),
        }
    )


def _status_text(row: pd.Series) -> str:
    value = row.get("validation_status", "")
    return "" if pd.isna(value) else str(value).strip().lower()


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _truth_from_value(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return int(float(value) > 0)

    text = str(value).strip().lower()
    if text in {"", "nan", "nat", "none", "null", "unknown", "unlabeled", "suspected", "review", "pending"}:
        return None
    if text in {"1", "true", "yes", "y", "positive"} or "phish" in text or "malicious" in text or "fraud" in text:
        return 1
    if text in {"0", "false", "no", "n", "negative"} or "legit" in text or "benign" in text or "safe" in text or "clean" in text:
        return 0
    return None


def _truth_column(df: pd.DataFrame, requested_column: str = "") -> str:
    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(f"Ground-truth column not found: {requested_column}")
        return requested_column
    for column in TRUTH_COLUMN_CANDIDATES:
        if column in df.columns:
            truth = df[column].map(_truth_from_value)
            if truth.notna().any():
                return column
    return ""


def _confusion_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    truth = y_true.astype(int).to_numpy()
    predicted = y_pred.astype(int).to_numpy()
    tp = int(((truth == 1) & (predicted == 1)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    recall = _safe_rate(tp, tp + fn)
    specificity = _safe_rate(tn, tn + fp)
    precision = _safe_rate(tp, tp + fp)
    return {
        "rows": int(len(truth)),
        "positives": int((truth == 1).sum()),
        "negatives": int((truth == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": _safe_rate(tp + tn, len(truth)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": _safe_rate(fp, fp + tn),
        "false_negative_rate": _safe_rate(fn, fn + tp),
        "f1": _safe_rate(2 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2,
    }


def _threshold_sweep(y_true: pd.Series, scores: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric_scores = pd.to_numeric(scores, errors="coerce").fillna(-1)
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = (numeric_scores >= threshold).astype(int)
        row = _confusion_summary(y_true, predicted)
        row["threshold"] = float(threshold)
        rows.append(row)
    return rows


def _first_numeric(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else default


def evaluation_metrics(
    classified: pd.DataFrame,
    truth_column: str = "",
) -> dict[str, Any]:
    resolved_truth_column = _truth_column(classified, truth_column)
    if not resolved_truth_column:
        return {
            "skipped": True,
            "reason": "No explicit ground-truth label column found.",
            "candidate_truth_columns": list(TRUTH_COLUMN_CANDIDATES),
        }

    truth = classified[resolved_truth_column].map(_truth_from_value)
    evaluated = classified.loc[truth.notna()].copy()
    evaluated_truth = truth.loc[truth.notna()].astype(int)
    if evaluated.empty:
        return {
            "skipped": True,
            "reason": f"Ground-truth column {resolved_truth_column} contains no usable binary labels.",
            "truth_column": resolved_truth_column,
        }

    final_confirmed = evaluated["final_classification"].eq("confirmed_phishing").astype(int)
    final_review_or_confirmed = evaluated["final_classification"].isin(
        {"confirmed_phishing", "suspicious_needs_review"}
    ).astype(int)

    standard_metrics = _confusion_summary(evaluated_truth, evaluated["standard_prediction"])
    hardened_metrics = _confusion_summary(evaluated_truth, evaluated["hardened_prediction"])
    standard_metrics["threshold"] = _first_numeric(evaluated["standard_threshold"])
    hardened_metrics["threshold"] = _first_numeric(evaluated["hardened_threshold"])

    metrics = {
        "skipped": False,
        "truth_column": resolved_truth_column,
        "input_rows": int(len(classified)),
        "evaluated_rows": int(len(evaluated)),
        "ignored_rows": int(len(classified) - len(evaluated)),
        "models": {
            "standard": standard_metrics,
            "hardened": hardened_metrics,
            "final_confirmed_only": _confusion_summary(evaluated_truth, final_confirmed),
            "final_review_or_confirmed": _confusion_summary(evaluated_truth, final_review_or_confirmed),
        },
        "threshold_sweeps": {
            "standard": _threshold_sweep(evaluated_truth, evaluated["standard_score"]),
            "hardened": _threshold_sweep(evaluated_truth, evaluated["hardened_score"]),
        },
    }
    return metrics


def write_evaluation_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _classify_row(row: pd.Series) -> pd.Series:
    standard_positive = int(row["standard_prediction"]) == 1
    hardened_positive = int(row["hardened_prediction"]) == 1
    status = _status_text(row)
    status_positive = status in POSITIVE_STATUSES
    status_suspected = status.startswith("suspected")

    if standard_positive:
        return pd.Series(
            {
                "final_classification": "confirmed_phishing",
                "classification_confidence": "high",
                "recommended_action": "block_or_escalate_as_phishing",
                "classification_reason": "standard model exceeds conservative high-confidence threshold",
            }
        )

    if status_positive and hardened_positive:
        return pd.Series(
            {
                "final_classification": "confirmed_phishing",
                "classification_confidence": "high",
                "recommended_action": "block_or_escalate_as_phishing",
                "classification_reason": "rule evidence is phishing and hardened model also exceeds threshold",
            }
        )

    if hardened_positive:
        return pd.Series(
            {
                "final_classification": "suspicious_needs_review",
                "classification_confidence": "medium",
                "recommended_action": "manual_review_or_monitor",
                "classification_reason": "hardened model flags suspicious infrastructure but standard model is below threshold",
            }
        )

    if status_suspected:
        return pd.Series(
            {
                "final_classification": "suspicious_needs_review",
                "classification_confidence": "low",
                "recommended_action": "manual_review_or_monitor",
                "classification_reason": f"pipeline status is {status} but both models are below threshold",
            }
        )

    return pd.Series(
        {
            "final_classification": "low_risk",
            "classification_confidence": "low",
            "recommended_action": "no_phishing_action",
            "classification_reason": "both models are below threshold and pipeline status is not suspicious",
        }
    )


def score_feature_rows(
    df: pd.DataFrame,
    standard_model_path: Path,
    hardened_model_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return empty_score_frames(df)

    standard_scores = score_model(df, standard_model_path, "standard")
    hardened_scores = score_model(df, hardened_model_path, "hardened")

    classified = df.reset_index(drop=True).copy()
    for model_name, scores in (("standard", standard_scores), ("hardened", hardened_scores)):
        classified[f"{model_name}_score"] = scores["score"].to_numpy()
        classified[f"{model_name}_threshold"] = scores["threshold"].to_numpy()
        classified[f"{model_name}_prediction"] = scores["prediction"].to_numpy()

    classification = classified.apply(_classify_row, axis=1)
    classified = pd.concat([classified, classification], axis=1)

    scored_long = pd.concat(
        [
            pd.concat([df.reset_index(drop=True), standard_scores], axis=1),
            pd.concat([df.reset_index(drop=True), hardened_scores], axis=1),
        ],
        ignore_index=True,
    )
    return classified, scored_long


def empty_score_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified = df.reset_index(drop=True).copy()
    for column in EMPTY_FEATURE_COLUMNS:
        if column not in classified.columns:
            classified[column] = pd.Series(dtype="object")
    for column in MODEL_OUTPUT_COLUMNS:
        if column.endswith("_score") or column.endswith("_threshold"):
            classified[column] = pd.Series(dtype="float64")
        else:
            classified[column] = pd.Series(dtype="int64")
    for column in CLASSIFICATION_OUTPUT_COLUMNS:
        classified[column] = pd.Series(dtype="object")

    feature_columns = [
        column
        for column in classified.columns
        if column not in MODEL_OUTPUT_COLUMNS and column not in CLASSIFICATION_OUTPUT_COLUMNS
    ]
    scored_long = pd.DataFrame(
        columns=[
            *feature_columns,
            "model",
            "score",
            "threshold",
            "prediction",
            "feature_count",
        ]
    )
    return classified, scored_long


def score_feature_file(
    feature_path: Path,
    output_path: Path,
    standard_model_path: Path = DEFAULT_STANDARD_MODEL,
    hardened_model_path: Path = DEFAULT_HARDENED_MODEL,
    long_output_path: Path | None = None,
) -> pd.DataFrame:
    try:
        df = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=EMPTY_FEATURE_COLUMNS)
    classified, scored_long = score_feature_rows(df, standard_model_path, hardened_model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(output_path, index=False)
    if long_output_path is not None:
        long_output_path.parent.mkdir(parents=True, exist_ok=True)
        scored_long.to_csv(long_output_path, index=False)
    return classified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify extracted feature rows with standard and hardened PU models.")
    parser.add_argument(
        "--features",
        default=str(DEFAULT_TEST_FEATURES),
        help="Extracted feature CSV, usually enriched_features_audit.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_TEST_OUTPUT),
        help="Output CSV for final production classification results.",
    )
    parser.add_argument("--long-output", default="", help="Optional long-format per-model score CSV.")
    parser.add_argument("--metrics-output", default="", help="Optional JSON metrics output when ground-truth labels are available.")
    parser.add_argument(
        "--truth-column",
        default="",
        help="Ground-truth label column to use for metrics. Defaults to auto-detecting explicit label columns.",
    )
    parser.add_argument("--standard-model", default=str(DEFAULT_STANDARD_MODEL))
    parser.add_argument("--hardened-model", default=str(DEFAULT_HARDENED_MODEL))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = Path(args.features)
    output_path = Path(args.output)
    long_output_path = Path(args.long_output) if args.long_output else None
    metrics_output_path = Path(args.metrics_output) if args.metrics_output else None
    classified = score_feature_file(
        feature_path=feature_path,
        output_path=output_path,
        standard_model_path=Path(args.standard_model),
        hardened_model_path=Path(args.hardened_model),
        long_output_path=long_output_path,
    )

    display_columns = [
        "url",
        "target_brand_domain",
        "validation_status",
        "standard_score",
        "standard_prediction",
        "hardened_score",
        "hardened_prediction",
        "final_classification",
        "classification_confidence",
        "recommended_action",
    ]
    print(classified[display_columns].to_string(index=False))
    print(f"Classification results: {output_path}")
    if long_output_path is not None:
        print(f"Long model scores: {long_output_path}")
    if metrics_output_path is not None:
        metrics = evaluation_metrics(classified, args.truth_column)
        write_evaluation_metrics(metrics_output_path, metrics)
        if metrics.get("skipped"):
            print(f"Metrics skipped: {metrics['reason']}")
        else:
            print(f"Metrics: {metrics_output_path}")
            print(f"Final confirmed-only metrics: {metrics['models']['final_confirmed_only']}")


if __name__ == "__main__":
    main()
