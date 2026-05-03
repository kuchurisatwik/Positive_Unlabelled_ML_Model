from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from model_scripts.score_pu_models import score_feature_file


REPO_ROOT = Path(__file__).resolve().parents[1]


class ScorePuModelsTests(unittest.TestCase):
    def test_score_feature_file_writes_empty_outputs_for_blank_features(self) -> None:
        tmp_path = REPO_ROOT / "output" / "_test_score_pu_models"
        tmp_path.mkdir(parents=True, exist_ok=True)

        def cleanup() -> None:
            for path in tmp_path.glob("*"):
                path.unlink()
            tmp_path.rmdir()

        self.addCleanup(cleanup)

        feature_path = tmp_path / "blank_features.csv"
        output_path = tmp_path / "classified.csv"
        long_output_path = tmp_path / "scores_long.csv"
        feature_path.write_text("\n", encoding="utf-8")

        classified = score_feature_file(
            feature_path=feature_path,
            output_path=output_path,
            standard_model_path=tmp_path / "missing_standard.joblib",
            hardened_model_path=tmp_path / "missing_hardened.joblib",
            long_output_path=long_output_path,
        )

        self.assertTrue(classified.empty)
        self.assertIn("url", classified.columns)
        self.assertIn("target_brand_domain", classified.columns)
        self.assertIn("validation_status", classified.columns)
        self.assertIn("standard_score", classified.columns)
        self.assertIn("hardened_score", classified.columns)
        self.assertIn("final_classification", classified.columns)

        written = pd.read_csv(output_path)
        self.assertTrue(written.empty)
        self.assertIn("final_classification", written.columns)

        long_written = pd.read_csv(long_output_path)
        self.assertTrue(long_written.empty)
        self.assertEqual(
            list(long_written.columns),
            [
                "url",
                "target_brand_domain",
                "validation_status",
                "model",
                "score",
                "threshold",
                "prediction",
                "feature_count",
            ],
        )


if __name__ == "__main__":
    unittest.main()
