"""
Unit Tests for Reproducibility Artifacts, Subsample Verification, and Cache Operation (Step 13).
"""

import os
import sys
import unittest
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    GOLDEN_SET_DIR,
    GOLDEN_DEV_CSV_PATH,
    GOLDEN_TEST_CSV_PATH,
    CACHE_DIR,
    PROCESSED_DATA_DIR,
    THREADS_PARQUET_PATH,
    REPRO_THREADS_PARQUET_PATH
)
from src.llm import call_llm, call_llm_json

class TestReproducibility(unittest.TestCase):

    def test_01_reproducibility_subsample_exists_and_valid(self):
        """Committed processed dataset must exist, have 10k-20k rows, and be < 5MB."""
        self.assertTrue(
            REPRO_THREADS_PARQUET_PATH.exists(),
            f"Missing reproducibility dataset at {REPRO_THREADS_PARQUET_PATH}"
        )
        file_size_mb = REPRO_THREADS_PARQUET_PATH.stat().st_size / (1024 * 1024)
        self.assertLess(file_size_mb, 5.0, f"Repro dataset is too large: {file_size_mb:.2f} MB")

        df = pd.read_parquet(REPRO_THREADS_PARQUET_PATH)
        self.assertGreaterEqual(len(df), 10000, f"Expected >= 10,000 rows, got {len(df)}")
        self.assertLessEqual(len(df), 20000, f"Expected <= 20,000 rows, got {len(df)}")

        # Check required columns
        expected_cols = [
            "customer_tweet_id",
            "customer_text",
            "brand_tweet_id",
            "brand_reply_text",
            "has_followup"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column {col} in repro dataset")
            self.assertEqual(df[col].isnull().sum(), 0, f"Column {col} contains null values")

    def test_02_golden_set_artifacts_present(self):
        """Full golden evaluation set must be committed and intact."""
        required_golden_files = [
            "golden_set.csv",
            "golden_dev.csv",
            "golden_test.csv",
            "test_split_manifest.json",
            "sample600_labelled.csv",
            "intent_taxonomy.md",
            "judge_human_scores.csv"
        ]
        for fname in required_golden_files:
            fpath = GOLDEN_SET_DIR / fname
            self.assertTrue(fpath.exists(), f"Missing required golden artifact: {fpath}")

        # Check locked test split count
        test_df = pd.read_csv(GOLDEN_TEST_CSV_PATH)
        self.assertEqual(len(test_df), 80, f"Expected 80 locked test rows, got {len(test_df)}")

    def test_03_cache_files_present_and_readable(self):
        """results/cache/ must contain valid cached LLM evaluation JSON outputs."""
        self.assertTrue(CACHE_DIR.exists(), f"Missing cache dir: {CACHE_DIR}")
        cache_files = list(CACHE_DIR.glob("*.json"))
        self.assertGreaterEqual(len(cache_files), 100, f"Expected >= 100 cached responses, got {len(cache_files)}")

        # Verify a sample cache file structure
        with open(cache_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("response", data)
            self.assertIn("model", data)

    def test_04_offline_llm_cache_fallback(self):
        """LLM calls must return responses without API key (via cache or local fallback)."""
        # Temporarily unset OPENAI_API_KEY
        orig_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            resp = call_llm("Classify support intent: My iPhone screen is frozen", temperature=0.0)
            self.assertIsInstance(resp, str)
            self.assertGreater(len(resp), 0)
        finally:
            if orig_key:
                os.environ["OPENAI_API_KEY"] = orig_key

    def test_05_config_resolves_repro_dataset(self):
        """src/config.py must resolve THREADS_PARQUET_PATH to reproducibility dataset."""
        self.assertEqual(
            THREADS_PARQUET_PATH.resolve(),
            REPRO_THREADS_PARQUET_PATH.resolve(),
            "THREADS_PARQUET_PATH should default to reproducibility dataset"
        )

    def test_06_no_huge_raw_datasets_tracked(self):
        """twcs.csv and raw parquet must not be committed."""
        raw_twcs = REPO_ROOT / "data" / "raw" / "twcs.csv"
        # Even if present locally, ensure .gitignore prevents tracking
        gitignore_path = REPO_ROOT / ".gitignore"
        self.assertTrue(gitignore_path.exists())
        with open(gitignore_path, "r", encoding="utf-8") as f:
            gi_content = f.read()
        self.assertIn("data/raw/*", gi_content)
        self.assertIn("archive/", gi_content)

if __name__ == "__main__":
    unittest.main()
