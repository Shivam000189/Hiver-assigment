"""
Unit Tests for Baseline Systems (TrivialBaseline, SimpleBaseline, FullAgentSystem).
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import TrivialBaseline, SimpleBaseline, FullAgentSystem
from src.config import GOLDEN_DEV_CSV_PATH
import pandas as pd

class TestBaselines(unittest.TestCase):

    def setUp(self):
        self.trivial = TrivialBaseline()
        self.simple = SimpleBaseline()
        self.full_agent = FullAgentSystem()

    def test_01_trivial_baseline_dev_majority(self):
        """Trivial baseline must predict the DEV majority intent and never escalate."""
        dev_df = pd.read_csv(GOLDEN_DEV_CSV_PATH)
        expected_majority = str(dev_df['intent'].mode()[0])
        self.assertEqual(self.trivial.majority_intent, expected_majority)

        res = self.trivial.reply("Help with my battery drain.")
        self.assertEqual(res["intent"], expected_majority)
        self.assertFalse(res["escalate"])
        self.assertEqual(res["escalate_reason"], "auto_handle")
        self.assertIn("canned", res["model_name"])
        self.assertTrue(len(res["draft"]) > 0)

    def test_02_simple_baseline_tfidf_and_rules_only(self):
        """Simple baseline uses TF-IDF for intent and rule layer only for escalation."""
        # 1. Normal inquiry: Rule does not fire -> auto_handle, returns verbatim top-1 reply
        res = self.simple.reply("My battery is draining fast on my iPhone.")
        self.assertEqual(res["intent"], "battery_drain_power")
        self.assertFalse(res["escalate"])
        self.assertEqual(res["escalate_reason"], "auto_handle")
        self.assertTrue(len(res["draft"]) > 0)

        # 2. Legal threat: Rule fires -> escalated via rule
        res_legal = self.simple.reply("I will sue your company in court.")
        self.assertTrue(res_legal["escalate"])
        self.assertEqual(res_legal["escalate_reason"], "rule:legal_threat")
        self.assertIn("[ESCALATED VIA RULE: rule:legal_threat]", res_legal["draft"])

        # 3. Escalation-prone intent without rule keyword (e.g. routine Apple ID query):
        # Full agent escalates, but SimpleBaseline (rules only) must NOT escalate
        res_account = self.simple.reply("How do I update my profile picture in iCloud settings?")
        self.assertFalse(res_account["escalate"])

    def test_03_full_agent_integration(self):
        """FullAgentSystem adapts agent_reply into common interface."""
        res = self.full_agent.reply("My battery is draining fast.")
        self.assertEqual(res["intent"], "battery_drain_power")
        self.assertIn("draft", res)
        self.assertIn("draft_reply", res)
        self.assertIn("confidence", res)
        self.assertIn("intent_confidence", res)
        self.assertIn("escalate", res)
        self.assertIn("escalate_reason", res)
        self.assertIn("retrieved_ids", res)

if __name__ == "__main__":
    unittest.main()
