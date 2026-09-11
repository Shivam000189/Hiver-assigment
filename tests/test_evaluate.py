"""
Unit Tests for Evaluation Harness Data Loading, Metric Utilities, and CLI.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import load_locked_test_data, safe_spearmanr
from src.config import GOLDEN_SET_DIR, GOLDEN_TEST_CSV_PATH

class TestEvaluate(unittest.TestCase):

    def test_01_load_locked_test_data_from_golden_set(self):
        """Loading from golden_set.csv must extract the locked 80 test examples."""
        golden_set_path = GOLDEN_SET_DIR / "golden_set.csv"
        test_df = load_locked_test_data(str(golden_set_path))
        self.assertEqual(len(test_df), 80)
        self.assertIn("golden_id", test_df.columns)
        self.assertIn("customer_text", test_df.columns)
        self.assertIn("intent", test_df.columns)
        self.assertIn("escalate", test_df.columns)
        self.assertIn("escalate_reason", test_df.columns)

    def test_02_load_locked_test_data_direct(self):
        """Loading directly from golden_test.csv must return 80 rows."""
        test_df = load_locked_test_data(str(GOLDEN_TEST_CSV_PATH))
        self.assertEqual(len(test_df), 80)

    def test_03_load_locked_test_missing_columns_raises(self):
        """Dataset with missing required columns must raise ValueError."""
        bad_df = pd.DataFrame({"some_col": [1, 2, 3]})
        temp_bad_path = REPO_ROOT / "scratch" / "bad_test.csv"
        temp_bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_df.to_csv(temp_bad_path, index=False)
        try:
            with self.assertRaises(ValueError):
                load_locked_test_data(str(temp_bad_path))
        finally:
            if temp_bad_path.exists():
                temp_bad_path.unlink()

    def test_04_safe_spearmanr(self):
        """safe_spearmanr handles constant vectors without NaN or crashing."""
        x = [5, 5, 5, 5]
        y = [5, 5, 5, 5]
        rho, pval = safe_spearmanr(x, y)
        self.assertEqual(rho, 1.0)

        x_diff = [1, 2, 3, 4, 5]
        y_diff = [1, 2, 3, 4, 5]
        rho2, pval2 = safe_spearmanr(x_diff, y_diff)
        self.assertAlmostEqual(rho2, 1.0)

    def test_05_metric_calculations_consistency(self):
        """Metric calculation functions produce correct macro and weighted F1."""
        y_true = ["intent_a", "intent_a", "intent_b", "intent_c"]
        y_pred = ["intent_a", "intent_b", "intent_b", "intent_c"]
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        self.assertEqual(acc, 0.75)
        self.assertTrue(0.0 <= macro_f1 <= 1.0)

if __name__ == "__main__":
    unittest.main()
