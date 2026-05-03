from __future__ import annotations

import argparse
import os
from pathlib import Path

from score_pu_models import (
    DEFAULT_HARDENED_MODEL,
    DEFAULT_STANDARD_MODEL,
    evaluation_metrics,
    score_feature_file,
    write_evaluation_metrics,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR.parent / "input" / "test_dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "test_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features for input/test_dataset and classify with standard + hardened PU models."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Folder containing test CSV/XLSX files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Folder for extracted features and results.")
    parser.add_argument("--standard-model", default=str(DEFAULT_STANDARD_MODEL))
    parser.add_argument("--hardened-model", default=str(DEFAULT_HARDENED_MODEL))
    parser.add_argument("--metrics-output", default="", help="Optional JSON metrics output path.")
    parser.add_argument(
        "--truth-column",
        default="",
        help="Ground-truth label column to use for metrics. Defaults to auto-detecting explicit label columns.",
    )
    parser.add_argument("--skip-extraction", action="store_true", help="Reuse existing extracted feature CSVs.")
    return parser.parse_args()


def configure_pipeline_env(input_dir: Path, output_dir: Path) -> dict[str, Path]:
    paths = {
        "features": output_dir / "enriched_features.csv",
        "audit": output_dir / "enriched_features_audit.csv",
        "dns_failed": output_dir / "dns_failed_urls.csv",
    }
    os.environ["PIPELINE_INPUT_DIR"] = str(input_dir)
    os.environ["PIPELINE_OUTPUT_DIR"] = str(output_dir)
    os.environ["PIPELINE_OUTPUT_FILE"] = str(paths["features"])
    os.environ["PIPELINE_OUTPUT_AUDIT_FILE"] = str(paths["audit"])
    os.environ["PIPELINE_OUTPUT_DNS_FAILED_FILE"] = str(paths["dns_failed"])
    return paths


def run_feature_extraction(input_dir: Path, output_dir: Path) -> dict[str, Path]:
    paths = configure_pipeline_env(input_dir, output_dir)
    import sys
    sys.path.append(str(SCRIPT_DIR.parent))
    import run_pipeline

    run_pipeline.main()
    return paths


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = configure_pipeline_env(input_dir, output_dir)
    if not args.skip_extraction:
        paths = run_feature_extraction(input_dir, output_dir)

    classification_path = output_dir / "model_classification_results.csv"
    long_scores_path = output_dir / "model_scores_long.csv"
    metrics_path = Path(args.metrics_output) if args.metrics_output else output_dir / "model_evaluation_metrics.json"
    classified = score_feature_file(
        feature_path=paths["audit"],
        output_path=classification_path,
        standard_model_path=Path(args.standard_model),
        hardened_model_path=Path(args.hardened_model),
        long_output_path=long_scores_path,
    )
    metrics = evaluation_metrics(classified, args.truth_column)
    write_evaluation_metrics(metrics_path, metrics)

    display_columns = [
        "url",
        "target_brand_domain",
        "validation_status",
        "standard_score",
        "hardened_score",
        "final_classification",
        "classification_confidence",
    ]
    print(classified[display_columns].to_string(index=False))
    print(f"Classification results: {classification_path}")
    print(f"Long model scores: {long_scores_path}")
    if metrics.get("skipped"):
        print(f"Metrics skipped: {metrics['reason']}")
    else:
        print(f"Metrics: {metrics_path}")
        print(f"Final confirmed-only metrics: {metrics['models']['final_confirmed_only']}")


if __name__ == "__main__":
    main()
