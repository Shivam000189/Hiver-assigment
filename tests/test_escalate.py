"""
Unit and Integration Tests for Step 8 Escalation Decision Gate.

Tests:
1. Rule Layer Categories (Legal, Safety, Security, Media, Abuse, PII Needed).
2. Model Layer Conditions (Low Confidence, Escalation-Prone Intent, Severe Sentiment).
3. Rule Layer Priority (Rule layer MUST preempt model layer even with low confidence).
4. Negative Cases (Benign messages containing sensitive keywords must NOT escalate).
5. Robustness / Edge-Case Handling (None, NaN, out-of-range confidence).
"""

import sys
import math
import unittest
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.escalate import gate, check_rules, check_model_layer
from src.config import CONFIDENCE_ESCALATION_THRESHOLD, ESCALATION_PRONE_INTENTS

class TestEscalationGate(unittest.TestCase):

    # -------------------------------------------------------------
    # 1. Rule Layer Tests (Positive Cases)
    # -------------------------------------------------------------
    def test_01_rule_legal_threat(self):
        """Legal threats must trigger rule:legal_threat."""
        cases = [
            "I will sue your company if this is not resolved immediately.",
            "I have contacted my lawyer to take legal action against Apple.",
            "I am filing a complaint with the consumer court and FTC regarding fraud.",
            "I will initiate a chargeback with my bank tomorrow."
        ]
        for text in cases:
            esc, reason = gate(text, intent="system_performance_freeze", confidence=0.95)
            self.assertTrue(esc, f"Failed on: {text}")
            self.assertEqual(reason, "rule:legal_threat", f"Expected rule:legal_threat, got {reason}")

    def test_02_rule_safety_and_harm(self):
        """Physical safety hazards and self-harm signals must trigger rule:safety."""
        cases = [
            "I want to kill myself over this stress.",
            "My iPhone battery is swollen and smoking after charging.",
            "The charger sparked and melted my power outlet, it's a hazardous fire risk.",
            "My macbook exploded and caused an electric shock."
        ]
        for text in cases:
            esc, reason = gate(text, intent="battery_drain_power", confidence=0.95)
            self.assertTrue(esc, f"Failed on: {text}")
            self.assertEqual(reason, "rule:safety", f"Expected rule:safety, got {reason}")

    def test_03_rule_account_security(self):
        """Account takeover and compromised credentials must trigger rule:security."""
        cases = [
            "My Apple ID has been hacked and someone is accessing my files.",
            "Someone logged into my account from an unauthorized device.",
            "My password was stolen and my iCloud account is compromised.",
            "We were hit by ransomware and our IDs are compromised."
        ]
        for text in cases:
            esc, reason = gate(text, intent="account_icloud_login", confidence=0.95)
            self.assertTrue(esc, f"Failed on: {text}")
            self.assertEqual(reason, "rule:security", f"Expected rule:security, got {reason}")

    def test_04_rule_media_and_press(self):
        """Journalist and media outreach must trigger rule:media."""
        cases = [
            "I'm a journalist writing an article about Apple customer service for the newspaper.",
            "This is a press inquiry from a reporter at TechNews regarding iOS bugs.",
            "I am publishing a story about this defect for a news outlet."
        ]
        for text in cases:
            esc, reason = gate(text, intent="other", confidence=0.90)
            self.assertTrue(esc, f"Failed on: {text}")
            self.assertEqual(reason, "rule:media", f"Expected rule:media, got {reason}")

    def test_05_rule_abuse_and_harassment(self):
        """Severe profanity, abusive harassment, or direct threats must trigger rule:abuse."""
        cases = [
            "Fuck you and your pieces of shit support team.",
            "Go to hell you useless bastards.",
            "I'm going to find you and kill you."
        ]
        for text in cases:
            esc, reason = gate(text, intent="system_performance_freeze", confidence=0.90)
            self.assertTrue(esc, f"Failed on: {text}")
            self.assertEqual(reason, "rule:abuse", f"Expected rule:abuse, got {reason}")

    def test_06_rule_pii_needed(self):
        """Requests requiring sensitive PII exchange must trigger rule:pii_needed."""
        cases = [
            "Please send me your full card number and expiration date to verify.",
            "Can you give me my account number and full credit card number?",
            "Here is my phone number and social security number to verify my ID."
        ]
        for text in cases:
            esc, reason = gate(text, intent="billing_app_store", confidence=0.90)
            self.assertTrue(esc, f"Failed on: {text}")
            self.assertEqual(reason, "rule:pii_needed", f"Expected rule:pii_needed, got {reason}")

    # -------------------------------------------------------------
    # 2. Model Layer Tests
    # -------------------------------------------------------------
    def test_07_model_low_confidence(self):
        """Inquiries with intent confidence below threshold must trigger model:low_confidence."""
        text = "I need help with some strange device behavior."
        low_conf = 0.25
        self.assertLess(low_conf, CONFIDENCE_ESCALATION_THRESHOLD)
        esc, reason = gate(text, intent="other", confidence=low_conf)
        self.assertTrue(esc)
        self.assertEqual(reason, f"model:low_confidence(intent=other, conf={low_conf:.2f})")

    def test_08_model_escalation_prone_intent(self):
        """Escalation-prone intents (e.g. account_icloud_login) must trigger model:escalation_prone_intent."""
        text = "How do I update my payment method in App Store settings?"
        for prone_intent in ESCALATION_PRONE_INTENTS:
            esc, reason = gate(text, intent=prone_intent, confidence=0.90)
            self.assertTrue(esc, f"Expected escalation for prone intent {prone_intent}")
            self.assertEqual(reason, f"model:escalation_prone_intent(intent={prone_intent})")

    def test_09_model_severe_sentiment(self):
        """Inquiries with severe destructive distress must trigger model:severe_sentiment."""
        text = "I am absolutely furious, your software update is destroying my work and I am beyond angry."
        # No rule pattern matches 'destroying' or 'furious' directly, but LLM severity classifier detects severe sentiment
        esc, reason = gate(text, intent="system_performance_freeze", confidence=0.85)
        self.assertTrue(esc)
        self.assertEqual(reason, "model:severe_sentiment")

    def test_10_normal_auto_handle(self):
        """Routine, resolvable customer requests must NOT escalate and return auto_handle."""
        routine_cases = [
            ("How do I check battery health in Settings?", "battery_drain_power", 0.94),
            ("My Wi-Fi keeps dropping when I walk into another room.", "connectivity_network", 0.88),
            ("How do I turn on predictive text on my keyboard?", "keyboard_autocorrect_bug", 0.91),
            ("Where can I find my downloaded music in Apple Music?", "audio_music_playback", 0.89),
            ("Can you help me take a screenshot on iPhone 8?", "system_performance_freeze", 0.95)
        ]
        for text, intent, conf in routine_cases:
            esc, reason = gate(text, intent=intent, confidence=conf)
            self.assertFalse(esc, f"False positive escalation on: '{text}' (reason: {reason})")
            self.assertEqual(reason, "auto_handle")

    # -------------------------------------------------------------
    # 3. Negative Cases (Precision / False-Positive Resistance)
    # -------------------------------------------------------------
    def test_11_negative_cases_benign_keywords(self):
        """Benign sentences containing words like 'media', 'legal', 'account', 'frustrating' must not escalate."""
        benign_cases = [
            ("I love your media coverage on the new iPhone features.", "other", 0.85),
            ("My account is working perfectly, just wondering about iOS 11 tips.", "system_performance_freeze", 0.85),
            ("This is frustrating but I still need help with my camera app.", "camera_photos_media", 0.85),
            ("Can you tell me your legal business name for my receipt records?", "other", 0.85),
            ("I want to know if security updates are included in this download.", "system_performance_freeze", 0.85)
        ]
        for text, intent, conf in benign_cases:
            esc, reason = gate(text, intent=intent, confidence=conf)
            self.assertFalse(esc, f"False positive escalation on benign sentence: '{text}' (reason: {reason})")
            self.assertEqual(reason, "auto_handle")

    # -------------------------------------------------------------
    # 4. Rule Layer Priority (Rule Layer MUST Preempt Model Layer)
    # -------------------------------------------------------------
    def test_12_rule_priority_over_model_layer(self):
        """Rule escalation MUST take precedence even when confidence is low or intent is prone."""
        # Low confidence + Legal threat -> Rule layer MUST win
        esc, reason = gate("I will sue your company.", intent="unknown", confidence=0.10)
        self.assertTrue(esc)
        self.assertEqual(reason, "rule:legal_threat", f"Rule layer must win over low_confidence. Got: {reason}")

        # Escalation-prone intent + Security rule -> Security rule MUST be stated
        esc, reason = gate("My account has been hacked.", intent="account_icloud_login", confidence=0.95)
        self.assertTrue(esc)
        self.assertEqual(reason, "rule:security", f"Security rule must be stated. Got: {reason}")

    # -------------------------------------------------------------
    # 5. Robustness & Input Validation
    # -------------------------------------------------------------
    def test_13_robustness_and_invalid_inputs(self):
        """Invalid or missing confidence/text must default safely to escalation without crashing."""
        # None or empty text
        esc, reason = gate("", intent="other", confidence=0.85)
        self.assertFalse(esc)
        self.assertEqual(reason, "auto_handle")

        # None confidence -> low confidence escalation
        esc, reason = gate("Need help with Wi-Fi", intent="connectivity_network", confidence=None)
        self.assertTrue(esc)
        self.assertIn("model:low_confidence", reason)

        # NaN confidence -> low confidence escalation
        esc, reason = gate("Need help with Wi-Fi", intent="connectivity_network", confidence=float('nan'))
        self.assertTrue(esc)
        self.assertIn("model:low_confidence", reason)

        # Out-of-bounds confidence (<0 or >1) -> low confidence escalation
        esc, reason = gate("Need help with Wi-Fi", intent="connectivity_network", confidence=-0.5)
        self.assertTrue(esc)
        self.assertIn("model:low_confidence", reason)

if __name__ == "__main__":
    unittest.main()
