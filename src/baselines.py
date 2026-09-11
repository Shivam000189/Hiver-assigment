"""
Baseline Systems for Support Agent Benchmarking.

Implements:
1. TrivialBaseline: Majority intent prediction from golden_dev, never escalates, canned replies.
2. SimpleBaseline: TF-IDF classifier + rule-only escalation + verbatim top-1 retrieved reply.
3. FullAgent: Step 5-8 full multi-stage pipeline.
"""

from typing import Dict, Any, List
import pandas as pd
from src.intents import get_tfidf_classifier
from src.escalate import check_rules
from src.reply import retrieve
from src.agent import agent_reply
from src.config import GOLDEN_DEV_CSV_PATH, BRAND_HANDLE

# Determine most frequent intent in golden_dev (Never use golden_test)
dev_df = pd.read_csv(GOLDEN_DEV_CSV_PATH)
MAJORITY_DEV_INTENT = str(dev_df['intent'].mode()[0])

CANNED_REPLIES = {
    "system_performance_freeze": f"Thanks for reaching out to @{BRAND_HANDLE}. Please force restart your device and let us know your iOS version via DM so we can assist: https://t.co/GDrqU22YpT",
    "keyboard_autocorrect_bug": f"We are aware of this keyboard autocorrect behavior. You can set up a text replacement shortcut in Settings > General > Keyboard > Text Replacement. DM us if you need help: https://t.co/GDrqU22YpT",
    "battery_drain_power": f"We want to ensure your battery runs efficiently. Check Settings > Battery to view power usage and send us a DM to run diagnostic checks: https://t.co/GDrqU22YpT",
    "audio_music_playback": f"Let's get your music working again. Try toggling iCloud Music Library in Settings and restarting your device. Send us a DM for further assistance: https://t.co/GDrqU22YpT",
    "connectivity_network": f"We recommend resetting network settings under Settings > General > Reset > Reset Network Settings. Please DM us if you are still experiencing connection drops: https://t.co/GDrqU22YpT",
    "camera_photos_media": f"We'd like to help with your camera issue. Please close all camera-related apps and DM us with your device model: https://t.co/GDrqU22YpT",
    "account_icloud_login": f"You can reset your Apple ID password securely at iforgot.apple.com. Please send us a DM if you encounter recovery issues: https://t.co/GDrqU22YpT",
    "billing_app_store": f"You can view your purchase history and request refunds directly at reportaproblem.apple.com. DM us if you need more details: https://t.co/GDrqU22YpT",
    "other": f"Thanks for contacting @{BRAND_HANDLE}. Please send us a Direct Message with your device model and iOS version so we can look into this for you: https://t.co/GDrqU22YpT"
}

class TrivialBaseline:
    """
    Trivial Baseline:
    - Intent: Hardcoded majority class from golden DEV set.
    - Escalation: Always False (never escalates).
    - Reply: Fixed canned response for the majority intent.
    """
    def __init__(self):
        self.name = "TrivialBaseline"
        self.majority_intent = MAJORITY_DEV_INTENT
        self.canned_reply = CANNED_REPLIES.get(self.majority_intent, CANNED_REPLIES["other"])

    def reply(self, customer_text: str) -> Dict[str, Any]:
        return {
            "system_name": self.name,
            "intent": self.majority_intent,
            "confidence": 1.0,
            "intent_confidence": 1.0,
            "escalate": False,
            "escalate_reason": "auto_handle",
            "draft": self.canned_reply,
            "draft_reply": self.canned_reply,
            "retrieved_ids": [],
            "retrieved_replies": [],
            "model_name": "trivial_canned",
            "latency_ms": {"total": 0.1}
        }

class SimpleBaseline:
    """
    Simple Baseline:
    - Intent: TFIDFClassifier trained on sample600 + golden_dev.
    - Escalation: Rule-layer ONLY (no model severity or confidence thresholding).
    - Reply: Verbatim top-1 retrieved historical brand reply (no LLM rewriting; note PII risk).
    """
    def __init__(self):
        self.name = "SimpleBaseline"
        self.tfidf_clf = get_tfidf_classifier()

    def reply(self, customer_text: str) -> Dict[str, Any]:
        # 1. Intent via TF-IDF
        intent, conf = self.tfidf_clf.predict(customer_text)

        # 2. Escalation via Rule-layer ONLY
        is_escalated, rule_reason, _ = check_rules(customer_text)
        if not is_escalated:
            rule_reason = "auto_handle"

        # 3. Retrieval Top-1
        retrieved = retrieve(customer_text, k=1)
        if is_escalated:
            draft = f"[ESCALATED VIA RULE: {rule_reason}]"
        elif retrieved:
            # Return verbatim top-1 historical reply (Caveat: Unredacted historical PII risk)
            # This baseline returns a historical reply verbatim and therefore has greater PII/copying risk
            # than the full grounded rewriter.
            draft = retrieved[0]["brand_reply_text"]
        else:
            draft = "I want to make sure you get the right help — let me connect you with our team."

        ret_ids = [r["tweet_id"] for r in retrieved]
        ret_replies = [r["brand_reply_text"] for r in retrieved]

        return {
            "system_name": self.name,
            "intent": intent,
            "confidence": conf,
            "intent_confidence": conf,
            "escalate": is_escalated,
            "escalate_reason": rule_reason,
            "draft": draft,
            "draft_reply": draft,
            "retrieved_ids": ret_ids,
            "retrieved_replies": ret_replies,
            "model_name": "tfidf_plus_rules",
            "latency_ms": {"total": 12.5}
        }

class FullAgentSystem:
    """
    Full Multi-Stage Support Agent (Composes PII -> Intent -> Escalation Gate -> Grounded Drafter).
    """
    def __init__(self):
        self.name = "FullAgent"

    def reply(self, customer_text: str) -> Dict[str, Any]:
        res = agent_reply(customer_text)
        res["system_name"] = self.name
        res["draft"] = res.get("draft_reply", "")
        res["confidence"] = res.get("intent_confidence", 0.0)
        return res
