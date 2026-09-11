"""
Unit Tests for Step 11 Failure Analysis Pipeline.

Tests:
1. Intent Confusion Pair mining
2. Escalation False Negative identification and reason extraction
3. Groundedness / Reply Quality ranking
4. Retrieval Insufficiency identification
5. Determinism across consecutive runs
6. PII redaction and anonymization
"""

import sys
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.failure_analysis import (
    find_top_intent_confusion,
    find_top_escalation_false_negative,
    find_top_groundedness_failure,
    find_top_retrieval_insufficient_failure,
    find_top_verbatim_baseline_privacy_failure,
    anonymize_text,
    run_failure_analysis
)

class TestFailureAnalysis(unittest.TestCase):

    def setUp(self):
        # Mock evaluation dataset
        self.mock_runs = pd.DataFrame([
            {
                "system": "FullAgent",
                "golden_id": 1,
                "customer_text": "Battery draining fast on my iPhone 8",
                "gold_intent": "battery_drain_power",
                "predicted_intent": "camera_photos_media",
                "gold_escalate": False,
                "predicted_escalate": False,
                "gold_escalate_reason": "none",
                "predicted_escalate_reason": "auto_handle",
                "confidence": 0.95,
                "draft": "Please check settings"
            },
            {
                "system": "FullAgent",
                "golden_id": 2,
                "customer_text": "Another battery drain issue",
                "gold_intent": "battery_drain_power",
                "predicted_intent": "camera_photos_media",
                "gold_escalate": True,
                "predicted_escalate": False,
                "gold_escalate_reason": "Rule 1: Safety",
                "predicted_escalate_reason": "auto_handle",
                "confidence": 0.95,
                "draft": "Please check settings"
            },
            {
                "system": "FullAgent",
                "golden_id": 3,
                "customer_text": "Wifi disconnects",
                "gold_intent": "connectivity_network",
                "predicted_intent": "connectivity_network",
                "gold_escalate": False,
                "predicted_escalate": False,
                "gold_escalate_reason": "none",
                "predicted_escalate_reason": "auto_handle",
                "confidence": 0.98,
                "draft": "Reset network settings"
            },
            {
                "system": "SimpleBaseline",
                "golden_id": 28,
                "customer_text": "Payment issue @AppleSupport",
                "gold_intent": "billing_app_store",
                "predicted_intent": "billing_app_store",
                "gold_escalate": False,
                "predicted_escalate": False,
                "gold_escalate_reason": "none",
                "predicted_escalate_reason": "auto_handle",
                "confidence": 0.90,
                "draft": "@712948 Order #98214 is processed."
            }
        ])

    def test_01_intent_confusion_mining(self):
        """Verify largest off-diagonal intent confusion pair is correctly identified."""
        res = find_top_intent_confusion(self.mock_runs)
        self.assertEqual(res["gold_intent"], "battery_drain_power")
        self.assertEqual(res["predicted_intent"], "camera_photos_media")
        self.assertEqual(res["confusion_count"], 2)
        self.assertEqual(res["failure_rank"], 1)

    def test_02_escalation_false_negative_identification(self):
        """Verify missed escalations (gold=True, pred=False) are extracted with reason strings."""
        res = find_top_escalation_false_negative(self.mock_runs)
        self.assertEqual(res["example_id"], 2)
        self.assertEqual(res["gold_label"], "escalate = True")
        self.assertEqual(res["predicted_label"], "escalate = False")
        self.assertIn("Rule 1: Safety", res["gold_escalate_reason"])
        self.assertEqual(res["predicted_escalate_reason"], "auto_handle")

    def test_03_anonymization_pii_removal(self):
        """Verify customer handles, emails, phones, URLs are sanitized without destroying semantics."""
        raw = "Contact @AppleSupport user @john_doe email test@apple.com phone 555-123-4567 link https://t.co/xyz123"
        anon = anonymize_text(raw)
        self.assertNotIn("test@apple.com", anon)
        self.assertNotIn("555-123-4567", anon)
        self.assertNotIn("https://t.co/xyz123", anon)
        self.assertNotIn("@john_doe", anon)
        self.assertIn("[EMAIL_REDACTED]", anon)
        self.assertIn("[PHONE_REDACTED]", anon)
        self.assertIn("[URL]", anon)
        self.assertIn("@[USER]", anon)

    def test_04_determinism(self):
        """Verify running analysis pipeline consecutively produces identical outputs."""
        df1, md1 = run_failure_analysis()
        df2, md2 = run_failure_analysis()

        self.assertEqual(len(df1), 5)
        self.assertEqual(len(df2), 5)
        self.assertEqual(df1["example_id"].tolist(), df2["example_id"].tolist())
        self.assertEqual(df1["failure_mode"].tolist(), df2["failure_mode"].tolist())
        self.assertEqual(md1, md2)

    def test_05_groundedness_ranking(self):
        """Verify low-scoring groundedness cases are detected and extracted."""
        mock_comp = pd.DataFrame([
            {"example_id": 101, "human_groundedness": 5, "judge_v1_groundedness": 5},
            {"example_id": 177, "human_groundedness": 3, "judge_v1_groundedness": 5}
        ])
        mock_sample = pd.DataFrame([
            {
                "example_id": 177,
                "customer_text": "SD card reader issue",
                "agent_draft": "Camera front rear modes?",
                "retrieved_evidence": json.dumps(["Mac SD card help"])
            }
        ])
        res = find_top_groundedness_failure(mock_comp, mock_sample)
        self.assertEqual(res["example_id"], 177)
        self.assertIn("hallucination", res["failure_mode"].lower())

if __name__ == "__main__":
    unittest.main()
