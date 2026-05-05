from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = SCRIPT_DIR.parent / "output" / "enriched_features_v2_audit.csv"
FALLBACK_FEATURES = SCRIPT_DIR.parent / "output" / "enriched_features_v2.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "pu_model"
DEFAULT_HARDENED_OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "pu_model_hardened"
DEFAULT_GROUP_HOLDOUT_COLUMNS = ("registered_domain", "target_brand_domain", "html_dom_hash")

POSITIVE_STATUSES = {
    "phishing",
    "phishing_campaign",
    "source_verified_phishing",
}

LEAK_OR_REVIEW_COLUMNS = {
    "url",
    "domain",
    "target_brand_domain",
    "critical_sector_entity_name",
    "source_label",
    "source_file",
    "source_row",
    "remarks",
    "matched_brand_domain",
    "registered_domain",
    "resolved_ips",
    "favicon_hash",
    "logo_hash",
    "html_dom_hash",
    "fetch_error",
    "dns_error",
    "tls_error",
    "dns_status",
    "http_status",
    "rdap_status",
    "whois_status",
    "tls_status",
    "page_status",
    "extraction_status",
    "validation_status",
    "validation_reason",
    "label",
    "label_conflict",
    "include_in_ml",
    "phishing",
    "phishing_campaign",
    "suspected_inactive",
    "suspected_active_infra",
    "suspected_active_no_phish_evidence",
}

HARDENED_EXCLUDED_FEATURE_COLUMNS = {
    # Inputs used directly or indirectly by classify_features() to create validation_status.
    "lexical_similarity_score",
    "high_lexical_similarity",
    "url_fetch_success",
    "page_fetch_success",
    "page_render_success",
    "fetch_status_code",
    "has_login_form",
    "has_password_input",
    "form_action_external",
    "no_login_form",
    "brand_token_or_logo_present",
    "logo_detected",
    "logo_brand_matches_target_brand",
    "logo_brand_domain_mismatch",
    "visual_brand_domain_mismatch",
    "no_brand_visual_claim",
    "same_html_hash_domain_count_7d",
}


def _normal_label(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().lower()
    if "phish" in text:
        return "phishing"
    if "suspect" in text:
        return "suspected"
    if "legit" in text:
        return "legitimate"
    if "unlabel" in text or text in {"unknown", ""}:
        return "unlabeled"
    return text


def load_feature_file(path: Path) -> pd.DataFrame:
    if not path.exists() and path == DEFAULT_FEATURES and FALLBACK_FEATURES.exists():
        path = FALLBACK_FEATURES
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_csv(path)


def build_pu_roles(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    source = df.get("source_label", pd.Series(["unknown"] * len(df))).map(_normal_label)
    status = df.get("validation_status", pd.Series([""] * len(df))).astype(str).str.lower()
    label = pd.to_numeric(df.get("label", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)

    positive = source.eq("phishing") | status.isin(POSITIVE_STATUSES)
    positive = positive | (label.eq(1) & status.isin(POSITIVE_STATUSES))

    explicit_legitimate = source.eq("legitimate")
    lexical_ok = pd.to_numeric(df.get("high_lexical_similarity", pd.Series([1] * len(df))), errors="coerce").fillna(1).eq(1)
    usable = (positive | lexical_ok) & ~explicit_legitimate
    unlabeled = usable & ~positive

    return positive.astype(bool), unlabeled.astype(bool)


def numeric_feature_frame(df: pd.DataFrame, extra_excluded_columns: set[str] | None = None) -> pd.DataFrame:
    numeric_columns: dict[str, pd.Series] = {}
    excluded_columns = LEAK_OR_REVIEW_COLUMNS | (extra_excluded_columns or set())
    for column in df.columns:
        if column in excluded_columns:
            continue
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
        if converted.notna().sum() == 0:
            continue
        numeric_columns[column] = converted

    if not numeric_columns:
        raise ValueError("No numeric training features found.")

    X = pd.DataFrame(numeric_columns).replace([np.inf, -np.inf], np.nan).fillna(-1)
    varying = [column for column in X.columns if X[column].nunique(dropna=False) > 1]
    if not varying:
        raise ValueError("All numeric training features are constant.")
    return X[varying]


def fit_random_forest(n_estimators: int, random_state: int, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
    )


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, target_recall: float) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = (probabilities >= threshold).astype(int)
        counts = _confusion_counts(y_true, predicted)
        rows.append(
            {
                "threshold": float(threshold),
                **counts,
                "accuracy": _safe_rate(counts["tp"] + counts["tn"], len(y_true)),
                "precision": float(precision_score(y_true, predicted, zero_division=0)),
                "recall": float(recall_score(y_true, predicted, zero_division=0)),
                "specificity": _safe_rate(counts["tn"], counts["tn"] + counts["fp"]),
                "false_positive_rate": _safe_rate(counts["fp"], counts["fp"] + counts["tn"]),
                "false_negative_rate": _safe_rate(counts["fn"], counts["fn"] + counts["tp"]),
                "f1": float(f1_score(y_true, predicted, zero_division=0)),
            }
        )

    viable = [row for row in rows if row["recall"] >= target_recall]
    if viable:
        selected = max(viable, key=lambda row: (row["precision"], row["f1"], row["threshold"]))
    else:
        selected = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return selected["threshold"], rows


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confusion_counts(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    true_values = np.asarray(y_true).astype(int)
    predicted_values = np.asarray(predicted).astype(int)
    return {
        "tp": int(((true_values == 1) & (predicted_values == 1)).sum()),
        "fp": int(((true_values == 0) & (predicted_values == 1)).sum()),
        "tn": int(((true_values == 0) & (predicted_values == 0)).sum()),
        "fn": int(((true_values == 1) & (predicted_values == 0)).sum()),
    }


def _prediction_metric_summary(
    y_true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    counts = _confusion_counts(y_true, predicted)
    summary: dict[str, Any] = {
        "rows": int(len(y_true)),
        "positives": positives,
        "reliable_negatives": negatives,
        **counts,
        "accuracy": _safe_rate(counts["tp"] + counts["tn"], len(y_true)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": _safe_rate(counts["tn"], counts["tn"] + counts["fp"]),
        "false_positive_rate": _safe_rate(counts["fp"], counts["fp"] + counts["tn"]),
        "false_negative_rate": _safe_rate(counts["fn"], counts["fn"] + counts["tp"]),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }
    summary["balanced_accuracy"] = (summary["recall"] + summary["specificity"]) / 2
    if probabilities is not None:
        summary["positive_score_min"] = float(probabilities[y_true == 1].min()) if positives else None
        summary["reliable_negative_score_max"] = float(probabilities[y_true == 0].max()) if negatives else None
    return summary


def _metric_summary(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (probabilities >= threshold).astype(int)
    return _prediction_metric_summary(y_true, predicted, probabilities)


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for metric in ("accuracy", "precision", "recall", "specificity", "balanced_accuracy", "f1"):
        values = np.array([row[metric] for row in rows], dtype=float)
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_std"] = float(values.std(ddof=0))
    return aggregate


def select_reliable_negative_indices(
    unlabeled_indices: np.ndarray,
    unlabeled_probabilities: np.ndarray,
    reliable_negative_quantile: float,
    reliable_negative_max_prob: float,
    min_reliable_negatives: int,
) -> tuple[np.ndarray, float]:
    quantile_cutoff = float(np.quantile(unlabeled_probabilities, reliable_negative_quantile))
    cutoff = min(quantile_cutoff, reliable_negative_max_prob)
    reliable_negative_indices = unlabeled_indices[unlabeled_probabilities <= cutoff]

    min_reliable_negatives = min(min_reliable_negatives, len(unlabeled_indices))
    if len(reliable_negative_indices) < min_reliable_negatives:
        order = np.argsort(unlabeled_probabilities)
        reliable_negative_indices = unlabeled_indices[order[:min_reliable_negatives]]
        cutoff = float(unlabeled_probabilities[order[min_reliable_negatives - 1]])

    return reliable_negative_indices, cutoff


def evaluate_group_holdouts(
    df: pd.DataFrame,
    X: pd.DataFrame,
    positive_indices: np.ndarray,
    reliable_negative_indices: np.ndarray,
    threshold: float,
    group_columns: list[str],
    group_holdout_splits: int,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    if not group_columns or group_holdout_splits <= 0:
        return {}

    base_indices = np.concatenate([positive_indices, reliable_negative_indices])
    base_y = np.concatenate(
        [np.ones(len(positive_indices), dtype=int), np.zeros(len(reliable_negative_indices), dtype=int)]
    )
    reports: dict[str, Any] = {}

    for group_column in group_columns:
        if group_column not in df.columns:
            reports[group_column] = {"skipped": f"Column not found: {group_column}"}
            continue

        groups = df.iloc[base_indices][group_column].fillna("").astype(str).to_numpy()
        if len(set(groups)) < 2:
            reports[group_column] = {"skipped": "Need at least two distinct groups."}
            continue

        splitter = GroupShuffleSplit(
            n_splits=group_holdout_splits,
            test_size=0.25,
            random_state=random_state,
        )
        split_rows: list[dict[str, Any]] = []

        for fold_number, (train_positions, test_positions) in enumerate(
            splitter.split(X.iloc[base_indices], base_y, groups),
            start=1,
        ):
            train_y = base_y[train_positions]
            test_y = base_y[test_positions]
            if len(set(train_y)) < 2 or len(set(test_y)) < 2:
                continue

            model = fit_random_forest(n_estimators, random_state + 1000 + fold_number, n_jobs)
            model.fit(X.iloc[base_indices[train_positions]], train_y)
            probabilities = model.predict_proba(X.iloc[base_indices[test_positions]])[:, 1]

            row = _metric_summary(test_y, probabilities, threshold)
            row["fold"] = int(fold_number)
            row["train_rows"] = int(len(train_positions))
            row["test_groups"] = int(len(set(groups[test_positions])))
            split_rows.append(row)

        if split_rows:
            reports[group_column] = {
                "splits": split_rows,
                "aggregate": _aggregate_metric_rows(split_rows),
            }
        else:
            reports[group_column] = {"skipped": "No fold contained both classes in train and test."}

    return reports


def _select_test_reliable_negatives(
    unlabeled_indices: np.ndarray,
    unlabeled_probabilities: np.ndarray,
    train_cutoff: float,
    reliable_negative_quantile: float,
) -> tuple[np.ndarray, float]:
    reliable_negative_indices = unlabeled_indices[unlabeled_probabilities <= train_cutoff]
    min_reliable_negatives = max(1, int(round(len(unlabeled_indices) * reliable_negative_quantile)))
    min_reliable_negatives = min(min_reliable_negatives, len(unlabeled_indices))
    cutoff = train_cutoff

    if len(reliable_negative_indices) < min_reliable_negatives:
        order = np.argsort(unlabeled_probabilities)
        reliable_negative_indices = unlabeled_indices[order[:min_reliable_negatives]]
        cutoff = float(unlabeled_probabilities[order[min_reliable_negatives - 1]])

    return reliable_negative_indices, cutoff


def _calibrate_fold_threshold(
    X: pd.DataFrame,
    train_indices: np.ndarray,
    train_y: np.ndarray,
    fallback_threshold: float,
    target_recall: float,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
) -> tuple[float, str]:
    class_counts = np.bincount(train_y, minlength=2)
    if class_counts.min() < 2:
        return fallback_threshold, "fallback_main_validation"

    positions = np.arange(len(train_indices))
    fit_positions, calibration_positions = train_test_split(
        positions,
        test_size=max(2, int(round(len(positions) * 0.25))),
        random_state=random_state,
        stratify=train_y,
    )
    calibration_y = train_y[calibration_positions]
    if len(set(calibration_y)) < 2:
        return fallback_threshold, "fallback_main_validation"

    calibration_model = fit_random_forest(n_estimators, random_state, n_jobs)
    calibration_model.fit(X.iloc[train_indices[fit_positions]], train_y[fit_positions])
    calibration_probabilities = calibration_model.predict_proba(X.iloc[train_indices[calibration_positions]])[:, 1]
    threshold, _ = choose_threshold(calibration_y, calibration_probabilities, target_recall)
    return threshold, "inner_calibration"


def evaluate_pu_kfold_cross_validation(
    X: pd.DataFrame,
    positive_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    min_unlabeled: int,
    reliable_negative_quantile: float,
    reliable_negative_max_prob: float,
    fallback_threshold: float,
    target_recall: float,
    cv_folds: int,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    if cv_folds < 2:
        return {}

    effective_folds = min(cv_folds, len(positive_indices), len(unlabeled_indices))
    if effective_folds < 2:
        return {
            "summary": {
                "requested_folds": int(cv_folds),
                "skipped": "Need at least two positive rows and two unlabeled rows for k-fold CV.",
            }
        }

    positive_splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    unlabeled_splitter = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state + 17)
    positive_splits = list(positive_splitter.split(positive_indices))
    unlabeled_splits = list(unlabeled_splitter.split(unlabeled_indices))

    out_of_fold_probabilities = np.full(X.shape[0], np.nan)
    out_of_fold_predictions = np.full(X.shape[0], -1, dtype=int)
    out_of_fold_numbers = np.full(X.shape[0], -1, dtype=int)
    out_of_fold_roles = np.array([""] * X.shape[0], dtype=object)
    split_rows: list[dict[str, Any]] = []

    for fold_number, ((pos_train_pos, pos_test_pos), (unl_train_pos, unl_test_pos)) in enumerate(
        zip(positive_splits, unlabeled_splits),
        start=1,
    ):
        pos_train = positive_indices[pos_train_pos]
        pos_test = positive_indices[pos_test_pos]
        unl_train = unlabeled_indices[unl_train_pos]
        unl_test = unlabeled_indices[unl_test_pos]

        prelim_indices = np.concatenate([pos_train, unl_train])
        prelim_y = np.concatenate([np.ones(len(pos_train), dtype=int), np.zeros(len(unl_train), dtype=int)])
        prelim_model = fit_random_forest(n_estimators, random_state + 2000 + fold_number, n_jobs)
        prelim_model.fit(X.iloc[prelim_indices], prelim_y)

        train_unlabeled_probabilities = prelim_model.predict_proba(X.iloc[unl_train])[:, 1]
        rn_train, train_rn_cutoff = select_reliable_negative_indices(
            unlabeled_indices=unl_train,
            unlabeled_probabilities=train_unlabeled_probabilities,
            reliable_negative_quantile=reliable_negative_quantile,
            reliable_negative_max_prob=reliable_negative_max_prob,
            min_reliable_negatives=min_unlabeled,
        )

        test_unlabeled_probabilities = prelim_model.predict_proba(X.iloc[unl_test])[:, 1]
        rn_test, test_rn_cutoff = _select_test_reliable_negatives(
            unlabeled_indices=unl_test,
            unlabeled_probabilities=test_unlabeled_probabilities,
            train_cutoff=train_rn_cutoff,
            reliable_negative_quantile=reliable_negative_quantile,
        )

        final_train_indices = np.concatenate([pos_train, rn_train])
        final_y = np.concatenate([np.ones(len(pos_train), dtype=int), np.zeros(len(rn_train), dtype=int)])
        fold_threshold, threshold_source = _calibrate_fold_threshold(
            X=X,
            train_indices=final_train_indices,
            train_y=final_y,
            fallback_threshold=fallback_threshold,
            target_recall=target_recall,
            n_estimators=n_estimators,
            random_state=random_state + 3000 + fold_number,
            n_jobs=n_jobs,
        )

        final_model = fit_random_forest(n_estimators, random_state + 4000 + fold_number, n_jobs)
        final_model.fit(X.iloc[final_train_indices], final_y)

        test_indices = np.concatenate([pos_test, rn_test])
        test_y = np.concatenate([np.ones(len(pos_test), dtype=int), np.zeros(len(rn_test), dtype=int)])
        test_probabilities = final_model.predict_proba(X.iloc[test_indices])[:, 1]
        test_predictions = (test_probabilities >= fold_threshold).astype(int)

        out_of_fold_probabilities[test_indices] = test_probabilities
        out_of_fold_predictions[test_indices] = test_predictions
        out_of_fold_numbers[test_indices] = fold_number
        out_of_fold_roles[pos_test] = "positive"
        out_of_fold_roles[rn_test] = "reliable_negative"

        row = _prediction_metric_summary(test_y, test_predictions, test_probabilities)
        row.update(
            {
                "fold": int(fold_number),
                "threshold": float(fold_threshold),
                "threshold_source": threshold_source,
                "train_positives": int(len(pos_train)),
                "train_reliable_negatives": int(len(rn_train)),
                "test_positives": int(len(pos_test)),
                "test_reliable_negatives": int(len(rn_test)),
                "train_reliable_negative_cutoff": float(train_rn_cutoff),
                "test_reliable_negative_cutoff": float(test_rn_cutoff),
            }
        )
        split_rows.append(row)

    tested_indices = np.flatnonzero(out_of_fold_numbers > 0)
    tested_y = np.array([1 if out_of_fold_roles[index] == "positive" else 0 for index in tested_indices], dtype=int)
    overall = _prediction_metric_summary(
        tested_y,
        out_of_fold_predictions[tested_indices],
        out_of_fold_probabilities[tested_indices],
    )

    return {
        "summary": {
            "strategy": "pu_kfold_fold_mined_reliable_negatives",
            "requested_folds": int(cv_folds),
            "folds": int(effective_folds),
            "evaluated_rows": int(len(tested_indices)),
            "positive_rows_evaluated": int((tested_y == 1).sum()),
            "reliable_negative_rows_evaluated": int((tested_y == 0).sum()),
            "overall": overall,
            "aggregate": _aggregate_metric_rows(split_rows),
            "splits": split_rows,
        },
        "probabilities": out_of_fold_probabilities,
        "predictions": out_of_fold_predictions,
        "folds": out_of_fold_numbers,
        "roles": out_of_fold_roles,
    }


def train_pu_model(
    df: pd.DataFrame,
    min_positives: int,
    min_unlabeled: int,
    reliable_negative_quantile: float,
    reliable_negative_max_prob: float,
    target_recall: float,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
    extra_excluded_columns: set[str] | None = None,
    model_variant: str = "standard",
) -> dict[str, Any]:
    positive_mask, unlabeled_mask = build_pu_roles(df)
    positive_indices = np.flatnonzero(positive_mask.to_numpy())
    unlabeled_indices = np.flatnonzero(unlabeled_mask.to_numpy())

    if len(positive_indices) < min_positives:
        raise ValueError(
            f"Need at least {min_positives} positive rows, found {len(positive_indices)}. "
            "Collect more CSE-matched phishing-feed rows before training."
        )
    if len(unlabeled_indices) < min_unlabeled:
        raise ValueError(
            f"Need at least {min_unlabeled} unlabeled rows, found {len(unlabeled_indices)}."
        )

    extra_excluded_columns = extra_excluded_columns or set()
    X = numeric_feature_frame(df, extra_excluded_columns)

    pos_train, pos_val = train_test_split(
        positive_indices,
        test_size=max(1, int(round(len(positive_indices) * 0.25))),
        random_state=random_state,
    )

    prelim_indices = np.concatenate([pos_train, unlabeled_indices])
    prelim_y = np.concatenate([np.ones(len(pos_train), dtype=int), np.zeros(len(unlabeled_indices), dtype=int)])
    prelim_model = fit_random_forest(n_estimators, random_state, n_jobs)
    
    # Use cross_val_predict for out-of-fold, unbiased probabilities for the preliminary model
    n_splits = max(2, min(5, np.bincount(prelim_y).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_prelim_probs = cross_val_predict(
        prelim_model, 
        X.iloc[prelim_indices], 
        prelim_y, 
        cv=cv, 
        method='predict_proba', 
        n_jobs=n_jobs
    )[:, 1]
    
    unlabeled_probabilities = all_prelim_probs[len(pos_train):]

    reliable_negative_indices, cutoff = select_reliable_negative_indices(
        unlabeled_indices=unlabeled_indices,
        unlabeled_probabilities=unlabeled_probabilities,
        reliable_negative_quantile=reliable_negative_quantile,
        reliable_negative_max_prob=reliable_negative_max_prob,
        min_reliable_negatives=min_unlabeled,
    )

    rn_train, rn_val = train_test_split(
        reliable_negative_indices,
        test_size=max(1, int(round(len(reliable_negative_indices) * 0.25))),
        random_state=random_state,
    )

    final_train_indices = np.concatenate([pos_train, rn_train])
    final_y = np.concatenate([np.ones(len(pos_train), dtype=int), np.zeros(len(rn_train), dtype=int)])
    
    # Hyperparameter tuning for the final Random Forest
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced_subsample", "balanced"]
    }
    
    cv_folds = max(2, min(3, np.bincount(final_y).min()))
    base_rf = RandomForestClassifier(random_state=random_state + 11, n_jobs=n_jobs)
    search = RandomizedSearchCV(
        base_rf, 
        param_distributions=param_dist, 
        n_iter=10, 
        cv=cv_folds, 
        scoring="f1", 
        random_state=random_state + 12, 
        n_jobs=n_jobs
    )
    search.fit(X.iloc[final_train_indices], final_y)
    final_model = search.best_estimator_

    validation_indices = np.concatenate([pos_val, rn_val])
    validation_y = np.concatenate([np.ones(len(pos_val), dtype=int), np.zeros(len(rn_val), dtype=int)])
    validation_probabilities = final_model.predict_proba(X.iloc[validation_indices])[:, 1]
    threshold, threshold_table = choose_threshold(validation_y, validation_probabilities, target_recall)
    validation_predicted = (validation_probabilities >= threshold).astype(int)

    all_probabilities = final_model.predict_proba(X)[:, 1]
    roles = np.array(["ignored"] * len(df), dtype=object)
    roles[unlabeled_indices] = "unlabeled"
    roles[positive_indices] = "positive"
    roles[reliable_negative_indices] = "reliable_negative"
    roles[pos_val] = "positive_validation"
    roles[rn_val] = "reliable_negative_validation"

    summary = {
        "rows": int(len(df)),
        "model_variant": model_variant,
        "positive_rows": int(len(positive_indices)),
        "unlabeled_rows": int(len(unlabeled_indices)),
        "reliable_negative_rows": int(len(reliable_negative_indices)),
        "reliable_negative_cutoff": float(cutoff),
        "feature_count": int(X.shape[1]),
        "feature_columns": X.columns.tolist(),
        "excluded_feature_columns": sorted(extra_excluded_columns),
        "threshold": float(threshold),
        "validation": {
            **_prediction_metric_summary(validation_y, validation_predicted, validation_probabilities),
            "positives": int(len(pos_val)),
            "reliable_negatives": int(len(rn_val)),
        },
        "threshold_table": threshold_table,
        "best_params": search.best_params_,
    }

    return {
        "model": final_model,
        "features": X,
        "feature_columns": X.columns.tolist(),
        "probabilities": all_probabilities,
        "predictions": (all_probabilities >= threshold).astype(int),
        "roles": roles,
        "positive_indices": positive_indices,
        "unlabeled_indices": unlabeled_indices,
        "reliable_negative_indices": reliable_negative_indices,
        "summary": summary,
    }


def _parse_column_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_repeated_column_args(values: list[str] | None) -> set[str]:
    columns: set[str] = set()
    for value in values or []:
        columns.update(_parse_column_list(value))
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-step PU phishing model from extracted CSE features.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES), help="Feature CSV, preferably enriched_features_v2_audit.csv.")
    parser.add_argument("--output-dir", default=None, help="Directory for model artifacts.")
    parser.add_argument(
        "--hardened",
        action="store_true",
        help="Drop rule-evidence features and write to output/pu_model_hardened unless --output-dir is set.",
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        help="Additional feature column to exclude. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--group-holdout-columns",
        default="",
        help="Comma-separated grouping columns for holdout diagnostics. Defaults to registered_domain,target_brand_domain,html_dom_hash in hardened mode.",
    )
    parser.add_argument("--group-holdout-splits", type=int, default=5)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Run PU-aware k-fold cross-validation. Use 5 for a small ~1000 URL dataset.",
    )
    parser.add_argument("--min-positives", type=int, default=10)
    parser.add_argument("--min-unlabeled", type=int, default=50)
    parser.add_argument("--reliable-negative-quantile", type=float, default=0.30)
    parser.add_argument("--reliable-negative-max-prob", type=float, default=0.30)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = Path(args.features)
    df = load_feature_file(feature_path)
    extra_excluded_columns = _parse_repeated_column_args(args.exclude_feature)
    model_variant = "hardened" if args.hardened else "standard"
    if args.hardened:
        extra_excluded_columns.update(HARDENED_EXCLUDED_FEATURE_COLUMNS)

    result = train_pu_model(
        df=df,
        min_positives=args.min_positives,
        min_unlabeled=args.min_unlabeled,
        reliable_negative_quantile=args.reliable_negative_quantile,
        reliable_negative_max_prob=args.reliable_negative_max_prob,
        target_recall=args.target_recall,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        extra_excluded_columns=extra_excluded_columns,
        model_variant=model_variant,
    )

    if args.group_holdout_columns:
        group_holdout_columns = _parse_column_list(args.group_holdout_columns)
    elif args.hardened:
        group_holdout_columns = list(DEFAULT_GROUP_HOLDOUT_COLUMNS)
    else:
        group_holdout_columns = []

    if group_holdout_columns:
        result["summary"]["group_holdout"] = evaluate_group_holdouts(
            df=df,
            X=result["features"],
            positive_indices=result["positive_indices"],
            reliable_negative_indices=result["reliable_negative_indices"],
            threshold=result["summary"]["threshold"],
            group_columns=group_holdout_columns,
            group_holdout_splits=args.group_holdout_splits,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )

    cv_result: dict[str, Any] = {}
    if args.cv_folds:
        cv_result = evaluate_pu_kfold_cross_validation(
            X=result["features"],
            positive_indices=result["positive_indices"],
            unlabeled_indices=result["unlabeled_indices"],
            min_unlabeled=args.min_unlabeled,
            reliable_negative_quantile=args.reliable_negative_quantile,
            reliable_negative_max_prob=args.reliable_negative_max_prob,
            fallback_threshold=result["summary"]["threshold"],
            target_recall=args.target_recall,
            cv_folds=args.cv_folds,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        if cv_result.get("summary"):
            result["summary"]["kfold_cross_validation"] = cv_result["summary"]

    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_HARDENED_OUTPUT_DIR if args.hardened else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = df.copy()
    scored["pu_training_role"] = result["roles"]
    scored["pu_score"] = result["probabilities"]
    scored["pu_prediction"] = result["predictions"]
    scored_path = output_dir / "pu_scored_rows.csv"
    scored.to_csv(scored_path, index=False)

    summary = result["summary"]
    if cv_result.get("probabilities") is not None:
        cv_scored = df.copy()
        cv_scored["pu_training_role"] = result["roles"]
        cv_scored["pu_cv_role"] = cv_result["roles"]
        cv_scored["pu_cv_fold"] = [
            "" if fold_number < 0 else int(fold_number)
            for fold_number in cv_result["folds"]
        ]
        cv_scored["pu_cv_score"] = [
            "" if np.isnan(probability) else float(probability)
            for probability in cv_result["probabilities"]
        ]
        cv_scored["pu_cv_prediction"] = [
            "" if prediction < 0 else int(prediction)
            for prediction in cv_result["predictions"]
        ]
        cv_scored_path = output_dir / "pu_kfold_cv_scored_rows.csv"
        cv_scored.to_csv(cv_scored_path, index=False)
        summary["cv_scored_rows_path"] = str(cv_scored_path)

    summary["feature_path"] = str(feature_path)
    summary["model_path"] = str(output_dir / "pu_model.joblib")
    summary["scored_rows_path"] = str(scored_path)

    model_path = output_dir / "pu_model.joblib"
    joblib.dump(
        {
            "model": result["model"],
            "threshold": summary["threshold"],
            "feature_columns": result["feature_columns"],
            "summary": summary,
        },
        model_path,
    )

    summary_path = output_dir / "pu_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Extract and save feature importances
    importances = result["model"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": result["feature_columns"],
        "importance": importances
    }).sort_values("importance", ascending=False)
    importance_path = output_dir / "pu_feature_importances.csv"
    importance_df.to_csv(importance_path, index=False)

    print("\n" + "="*55)
    print(" 🚀 PU MODEL TRAINING COMPLETE")
    print("="*55)
    print(f"{'Total Rows:':<25} {summary['rows']}")
    print(f"{'Model Variant:':<25} {summary['model_variant']}")
    print(f"{'Positive Rows:':<25} {summary['positive_rows']}")
    print(f"{'Unlabeled Rows:':<25} {summary['unlabeled_rows']}")
    print(f"{'Reliable Negatives:':<25} {summary['reliable_negative_rows']}")
    print(f"{'Feature Count:':<25} {summary['feature_count']}")
    
    print("\n--- BEST HYPERPARAMETERS ---")
    for key, val in summary['best_params'].items():
        print(f" {key:<23}: {val}")

    print(f"\n{'Selected Threshold:':<25} {summary['threshold']:.3f}")
    
    print("\n--- VALIDATION METRICS ---")
    val_metrics = summary["validation"]
    print(f"{'Accuracy:':<20} {val_metrics.get('accuracy', 0.0):.4f}")
    print(f"{'Precision:':<20} {val_metrics.get('precision', 0.0):.4f}")
    print(f"{'Recall:':<20} {val_metrics.get('recall', 0.0):.4f}")
    print(f"{'Specificity:':<20} {val_metrics.get('specificity', 0.0):.4f}")
    print(f"{'F1 Score:':<20} {val_metrics.get('f1', 0.0):.4f}")
    print(f"{'Balanced Accuracy:':<20} {val_metrics.get('balanced_accuracy', 0.0):.4f}")

    if summary.get("kfold_cross_validation"):
        cv_summary = summary["kfold_cross_validation"]
        if "overall" in cv_summary:
            cv_overall = cv_summary['overall']
            print("\n--- K-FOLD CV OVERALL METRICS ---")
            print(f"{'Accuracy:':<20} {cv_overall.get('accuracy', 0.0):.4f}")
            print(f"{'Precision:':<20} {cv_overall.get('precision', 0.0):.4f}")
            print(f"{'Recall:':<20} {cv_overall.get('recall', 0.0):.4f}")
            print(f"{'F1 Score:':<20} {cv_overall.get('f1', 0.0):.4f}")

    if summary.get("group_holdout"):
        print(f"\nGroup holdout columns: {', '.join(summary['group_holdout'])}")

    print("\n--- TOP 10 IMPORTANT FEATURES ---")
    top_features = importance_df.head(10)
    for idx, row in top_features.iterrows():
        print(f" - {row['feature']:<30} : {row['importance']:.4f}")

    print("\n--- ARTIFACTS SAVED ---")
    print(f"Model:                {model_path.resolve()}")
    print(f"Feature Importances:  {importance_path.resolve()}")
    print(f"Scores:               {scored_path.resolve()}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
