"""
Unit Tests for Step 10 Judge vs Human Agreement Measurement & Calibration Pipeline.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.judge_agreement import (
    validate_human_scores,
    safe_spearmanr,
    compute_agreement,
    categorize_disagreements,
    HUMAN_SCORE_COLUMNS
)

class TestJudgeAgreement(unittest.TestCase):

    def setUp(self):
        # Create valid mock 60-example human dataset
        records = []
        for i in range(1, 61):
            records.append({
                "example_id": i,
                "customer_text": f"Customer issue {i}",
                "agent_draft": f"Agent reply {i}",
                "human_correctness": 4 if i % 2 == 0 else 5,
                "human_groundedness": 5,
                "human_completeness": 4,
                "human_brand_voice": 5,
                "human_tone": 4 if i % 3 == 0 else 5,
                "human_notes": "Valid note"
            })
        self.valid_df = pd.DataFrame(records)

    def test_01_validation_valid_dataset(self):
        """Valid 60-example dataset must pass validation."""
        is_valid, msg = validate_human_scores(self.valid_df)
        self.assertTrue(is_valid, f"Validation failed on valid data: {msg}")

    def test_02_validation_invalid_row_count(self):
        """Dataset with <50 or >80 rows must fail validation."""
        short_df = self.valid_df.head(40).copy()
        is_valid, msg = validate_human_scores(short_df)
        self.assertFalse(is_valid)
        self.assertIn("50–80", msg)

        long_df = pd.concat([self.valid_df, self.valid_df.head(30)], ignore_index=True)
        is_valid, msg = validate_human_scores(long_df)
        self.assertFalse(is_valid)

    def test_03_validation_duplicate_ids(self):
        """Dataset with duplicate example_ids must fail validation."""
        dup_df = self.valid_df.copy()
        dup_df.loc[1, "example_id"] = dup_df.loc[0, "example_id"]
        is_valid, msg = validate_human_scores(dup_df)
        self.assertFalse(is_valid)
        self.assertIn("duplicate", msg.lower())

    def test_04_validation_invalid_scores(self):
        """Scores outside [1, 5] or non-integers/NaNs must be rejected."""
        # Score out of bounds (6)
        bad_score_df = self.valid_df.copy()
        bad_score_df.loc[0, "human_correctness"] = 6
        is_valid, msg = validate_human_scores(bad_score_df)
        self.assertFalse(is_valid)

        # Score out of bounds (0)
        bad_score_df2 = self.valid_df.copy()
        bad_score_df2.loc[0, "human_tone"] = 0
        is_valid, msg = validate_human_scores(bad_score_df2)
        self.assertFalse(is_valid)

        # NaN score
        nan_df = self.valid_df.copy()
        nan_df.loc[0, "human_groundedness"] = np.nan
        is_valid, msg = validate_human_scores(nan_df)
        self.assertFalse(is_valid)

    def test_05_safe_spearmanr_zero_variance(self):
        """Constant scores must return (None, 1.0) without crashing."""
        x = [5, 5, 5, 5]
        y = [4, 4, 4, 4]
        rho, pval = safe_spearmanr(x, y)
        self.assertIsNone(rho)

        # Variable scores calculate genuine correlation
        x_var = [1, 2, 3, 4, 5]
        y_var = [1, 2, 3, 4, 5]
        rho_var, pval_var = safe_spearmanr(x_var, y_var)
        self.assertAlmostEqual(rho_var, 1.0)

    def test_06_large_disagreement_detection(self):
        """Differences >= 2 must be flagged and categorized."""
        comp_records = []
        for i in range(1, 61):
            # Example 1 has a large disagreement on completeness: human=5, judge=3
            h_comp = 5 if i == 1 else 4
            j_comp = 3 if i == 1 else 4
            comp_records.append({
                "example_id": i,
                "customer_text": f"Query {i}",
                "agent_draft": f"Draft reply {i}",
                "human_correctness": 5,
                "judge_v1_correctness": 5,
                "human_groundedness": 5,
                "judge_v1_groundedness": 5,
                "human_completeness": h_comp,
                "judge_v1_completeness": j_comp,
                "human_brand_voice": 5,
                "judge_v1_brand_voice": 5,
                "human_tone": 5,
                "judge_v1_tone": 5,
                "judge_v1_correctness_reason": "ok",
                "judge_v1_groundedness_reason": "ok",
                "judge_v1_completeness_reason": "ok",
                "judge_v1_brand_voice_reason": "ok",
                "judge_v1_tone_reason": "ok"
            })
        comp_df = pd.DataFrame(comp_records)

        disagreements, cat_counts = categorize_disagreements(comp_df, judge_prefix="judge_v1")
        self.assertEqual(len(disagreements), 1)
        self.assertEqual(disagreements[0]["example_id"], 1)
        self.assertEqual(disagreements[0]["criterion"], "completeness")
        self.assertEqual(disagreements[0]["difference"], 2)

    def test_07_join_integrity(self):
        """Verify human and judge scores are joined by example_id without duplication or data loss."""
        h_df = self.valid_df.copy()
        j_records = [{
            "example_id": i,
            "judge_v1_correctness": 5,
            "judge_v1_groundedness": 5,
            "judge_v1_completeness": 5,
            "judge_v1_brand_voice": 5,
            "judge_v1_tone": 5,
            "judge_v1_correctness_reason": "ok",
            "judge_v1_groundedness_reason": "ok",
            "judge_v1_completeness_reason": "ok",
            "judge_v1_brand_voice_reason": "ok",
            "judge_v1_tone_reason": "ok"
        } for i in range(1, 61)]
        j_df = pd.DataFrame(j_records)

        joined = pd.merge(h_df, j_df, on="example_id")
        self.assertEqual(len(joined), 60)
        self.assertEqual(set(joined["example_id"]), set(range(1, 61)))
        self.assertIn("human_groundedness", joined.columns)
        self.assertIn("judge_v1_groundedness", joined.columns)

    def test_08_known_spearman_values(self):
        """Verify Spearman calculation on known synthetic sequences with known monotonic behavior."""
        # Exact inverse rank correlation should be -1.0
        x_inv = [1, 2, 3, 4, 5]
        y_inv = [5, 4, 3, 2, 1]
        rho, _ = safe_spearmanr(x_inv, y_inv)
        self.assertAlmostEqual(rho, -1.0)

        # Monotonically increasing should be +1.0
        x_inc = [2, 4, 6, 8, 10]
        y_inc = [10, 20, 30, 40, 50]
        rho_inc, _ = safe_spearmanr(x_inc, y_inc)
        self.assertAlmostEqual(rho_inc, 1.0)

if __name__ == "__main__":
    unittest.main()

