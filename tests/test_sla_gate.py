"""
Unit Tests for SLA-Aware Confidence Gate and 3-Way Routing Engine (src/sla_gate.py).
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sla_gate import (
    compute_intent_sla_risk,
    compute_sentiment_urgency_risk,
    compute_sla_risk_score,
    route_message,
    MIN_AUTO_SEND_SIMILARITY,
    MAX_AUTO_SEND_SLA_RISK,
    ROUTING_TIER_AUTO_SEND,
    ROUTING_TIER_DRAFT_FOR_REVIEW,
    ROUTING_TIER_ESCALATE,
    HISTORICAL_INTENT_MEDIAN_MINUTES,
    GLOBAL_FALLBACK_MEDIAN_MINUTES
)

class TestSLAGate(unittest.TestCase):

    def test_01_intent_sla_risk_monotonicity(self):
        """SLA risk score must be strictly monotonic with historical response times."""
        # battery_drain_power (101.7 min) > keyboard (82.3 min) > freeze (64.5 min) > camera (38.2 min)
        risk_battery = compute_intent_sla_risk("battery_drain_power")
        risk_keyboard = compute_intent_sla_risk("keyboard_autocorrect_bug")
        risk_freeze = compute_intent_sla_risk("system_performance_freeze")
        risk_camera = compute_intent_sla_risk("camera_photos_media")

        self.assertGreater(risk_battery, risk_keyboard)
        self.assertGreater(risk_keyboard, risk_freeze)
        self.assertGreater(risk_freeze, risk_camera)

        # Check all registered intents are monotonic with their table values
        sorted_intents = sorted(
            HISTORICAL_INTENT_MEDIAN_MINUTES.items(),
            key=lambda x: x[1]
        )
        for i in range(len(sorted_intents) - 1):
            intent_low, time_low = sorted_intents[i]
            intent_high, time_high = sorted_intents[i + 1]
            risk_low = compute_intent_sla_risk(intent_low)
            risk_high = compute_intent_sla_risk(intent_high)
            self.assertLessEqual(
                risk_low,
                risk_high,
                f"Monotonicity violated between {intent_low} ({risk_low}) and {intent_high} ({risk_high})"
            )

    def test_02_three_way_routing_all_branches(self):
        """Routing engine must accurately hit AUTO_SEND, DRAFT_FOR_REVIEW, and ESCALATE."""
        # 1. ESCALATE branch: escalation gate is True
        res_esc = route_message(
            customer_text="My phone battery is swollen and smoking!",
            intent="battery_drain_power",
            is_escalated=True,
            escalate_reason="rule:safety",
            retrieval_similarity=0.90
        )
        self.assertEqual(res_esc["routing_tier"], ROUTING_TIER_ESCALATE)
        self.assertIn("Escalated via gate", res_esc["routing_reason"])

        # 2. AUTO_SEND branch: not escalated, high similarity (>= 0.40), low SLA risk (<= 0.45)
        # camera_photos_media has very low intent SLA risk (38.2 min -> ~0.046)
        res_auto = route_message(
            customer_text="How do I view recent photos in camera roll?",
            intent="camera_photos_media",
            is_escalated=False,
            escalate_reason="auto_handle",
            retrieval_similarity=0.85
        )
        self.assertEqual(res_auto["routing_tier"], ROUTING_TIER_AUTO_SEND)
        self.assertLessEqual(res_auto["sla_risk_score"], MAX_AUTO_SEND_SLA_RISK)
        self.assertGreaterEqual(res_auto["retrieval_similarity"], MIN_AUTO_SEND_SIMILARITY)

        # 3. DRAFT_FOR_REVIEW branch A: low retrieval similarity (< 0.40)
        res_review_sim = route_message(
            customer_text="How do I view recent photos?",
            intent="camera_photos_media",
            is_escalated=False,
            escalate_reason="auto_handle",
            retrieval_similarity=0.25  # Below MIN_AUTO_SEND_SIMILARITY
        )
        self.assertEqual(res_review_sim["routing_tier"], ROUTING_TIER_DRAFT_FOR_REVIEW)
        self.assertIn("similarity", res_review_sim["routing_reason"].lower())

        # 4. DRAFT_FOR_REVIEW branch B: high SLA risk (> 0.45) e.g. battery drain + frustration
        res_review_sla = route_message(
            customer_text="This battery drain is terrible and draining in 10 minutes!",
            intent="battery_drain_power",
            is_escalated=False,
            escalate_reason="auto_handle",
            retrieval_similarity=0.75
        )
        self.assertEqual(res_review_sla["routing_tier"], ROUTING_TIER_DRAFT_FOR_REVIEW)
        self.assertGreater(res_review_sla["sla_risk_score"], MAX_AUTO_SEND_SLA_RISK)

    def test_03_unseen_and_none_intent_graceful_fallback(self):
        """Unseen, rare, or None intents must fall back gracefully to global median without crashing."""
        fallback_expected = compute_intent_sla_risk(None)
        self.assertTrue(0.0 <= fallback_expected <= 1.0)

        # Unseen intent string
        risk_unseen = compute_intent_sla_risk("completely_unknown_quantum_bug")
        self.assertEqual(risk_unseen, fallback_expected)

        # Empty string
        risk_empty = compute_intent_sla_risk("")
        self.assertEqual(risk_empty, fallback_expected)

        # None
        risk_none = compute_intent_sla_risk(None)
        self.assertEqual(risk_none, fallback_expected)

        # End-to-end route_message call with None intent
        res = route_message(
            customer_text="Routine question",
            intent=None,
            is_escalated=False,
            escalate_reason="auto_handle",
            retrieval_similarity=0.50
        )
        self.assertIn(res["routing_tier"], [ROUTING_TIER_AUTO_SEND, ROUTING_TIER_DRAFT_FOR_REVIEW])
        self.assertIsInstance(res["sla_risk_score"], float)

    def test_04_threshold_bounds_and_constants(self):
        """Named threshold constants must be valid floats in [0, 1]."""
        self.assertTrue(0.0 <= MIN_AUTO_SEND_SIMILARITY <= 1.0)
        self.assertTrue(0.0 <= MAX_AUTO_SEND_SLA_RISK <= 1.0)
        self.assertGreater(MIN_AUTO_SEND_SIMILARITY, 0.20)
        self.assertLess(MAX_AUTO_SEND_SLA_RISK, 0.80)

if __name__ == "__main__":
    unittest.main()
