"""
SLA-Aware Confidence Gate and 3-Way Message Router.

Why this exists:
----------------
In actual support workflows, making a strict binary choice between full auto-sending and
human escalation leaves a gap. A lot of incoming messages get clear, grounded drafts from the
retrieval step, but either touch slow-to-resolve topics or come from frustrated users.
Instead of sending those blindly or kicking them to tier-2 supervisors, we route them to
'DRAFT_FOR_REVIEW'. An agent can glance at the draft in a few seconds, edit if needed, and send.

Signals used:
-------------
1. Intent SLA Risk (intent_risk in [0.0, 1.0]):
   Computed from the median response time (in minutes) for that intent in the training set
   (sample600_labelled.csv joined with applesupport_threads_repro.parquet). Slower historical
   resolution means higher SLA breach risk.
   Formula:
       intent_risk = clip((median_minutes - 35.0) / (105.0 - 35.0), 0.0, 1.0)
   If an intent is unknown or missing, it falls back to the corpus median (~73.0 min -> 0.54).

2. Sentiment & Urgency Risk (sentiment_risk in [0.0, 1.0]):
   Uses the sentiment severity check from src.escalate:
       - 'routine'    -> 0.10
       - 'frustrated' -> 0.60
       - 'severe'     -> 1.00
   Adds +0.15 if urgency keywords ('asap', 'urgent', 'immediately') are present.

3. Composite SLA Risk Score (sla_risk_score in [0.0, 1.0]):
       sla_risk_score = clip(0.50 * intent_risk + 0.50 * sentiment_risk, 0.0, 1.0)

Routing tiers:
--------------
- ESCALATE:
    If the escalation gate fired (safety hazard, legal threat, account breach, etc.).
- AUTO_SEND:
    If not escalated, retrieval similarity >= 0.40, and sla_risk_score <= 0.45.
- DRAFT_FOR_REVIEW:
    Everything else — drafted and grounded, but held for human review.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd

from src.config import (
    THREADS_PARQUET_PATH,
    SAMPLE_600_CSV_PATH,
    PROCESSED_DATA_DIR,
    RETRIEVAL_MIN_SIMILARITY
)
from src.escalate import get_sentiment_severity

logger = logging.getLogger("hiver.sla_gate")

# Threshold constants for 3-way routing.
# 0.40 similarity and 0.45 SLA risk were chosen based on manual inspection of DEV split edge cases.
MIN_AUTO_SEND_SIMILARITY: float = 0.40
MAX_AUTO_SEND_SLA_RISK: float = 0.45

ROUTING_TIER_AUTO_SEND: str = "AUTO_SEND"
ROUTING_TIER_DRAFT_FOR_REVIEW: str = "DRAFT_FOR_REVIEW"
ROUTING_TIER_ESCALATE: str = "ESCALATE"

ROUTING_TIERS = (
    ROUTING_TIER_AUTO_SEND,
    ROUTING_TIER_DRAFT_FOR_REVIEW,
    ROUTING_TIER_ESCALATE
)

SLA_LOOKUP_PATH = PROCESSED_DATA_DIR / "intent_response_times.json"

# Normalization bounds for mapping minutes to [0.0, 1.0]
MIN_RESPONSE_TIME_MINUTES: float = 35.0
MAX_RESPONSE_TIME_MINUTES: float = 105.0
GLOBAL_FALLBACK_MEDIAN_MINUTES: float = 72.98

# Regex for common customer urgency phrasing
URGENCY_REGEX = re.compile(
    r'\b(?:urgent|urgently|immediately|asap|right now|critical|emergency|hurry|today|cannot wait)\b',
    re.IGNORECASE
)


def load_or_build_intent_response_times(force_recompute: bool = False) -> Dict[str, Any]:
    """
    Loads historical median response times per intent from intent_response_times.json.
    If the file is missing or force_recompute is True, computes medians by joining
    sample600_labelled.csv with the threads parquet and writes the JSON.
    """
    if not force_recompute and SLA_LOOKUP_PATH.exists():
        try:
            with open(SLA_LOOKUP_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "intent_median_minutes" in data:
                return data
        except Exception as e:
            logger.warning(f"Could not read {SLA_LOOKUP_PATH}: {e}. Recomputing...")

    # Compute from dataset directly
    intent_medians: Dict[str, float] = {}
    global_median = GLOBAL_FALLBACK_MEDIAN_MINUTES

    if THREADS_PARQUET_PATH.exists() and SAMPLE_600_CSV_PATH.exists():
        try:
            df_threads = pd.read_parquet(THREADS_PARQUET_PATH)
            df_s600 = pd.read_csv(SAMPLE_600_CSV_PATH)
            merged = pd.merge(
                df_s600,
                df_threads[["customer_tweet_id", "response_time_minutes"]],
                on="customer_tweet_id",
                how="inner"
            )
            stats = merged.groupby("intent")["response_time_minutes"].median().to_dict()
            intent_medians = {k: round(float(v), 2) for k, v in stats.items()}
            global_median = round(float(df_threads["response_time_minutes"].median()), 2)
        except Exception as e:
            logger.warning(f"Failed computing response time medians: {e}")

    # Fallback dictionary if parquet/csv are unavailable
    if not intent_medians:
        intent_medians = {
            "battery_drain_power": 101.7,
            "billing_app_store": 100.0,
            "keyboard_autocorrect_bug": 82.33,
            "account_icloud_login": 75.43,
            "system_performance_freeze": 64.48,
            "audio_music_playback": 62.53,
            "other": 41.79,
            "connectivity_network": 41.58,
            "camera_photos_media": 38.15
        }

    lookup_data = {
        "intent_median_minutes": intent_medians,
        "global_median_minutes": global_median,
        "min_response_time_minutes": MIN_RESPONSE_TIME_MINUTES,
        "max_response_time_minutes": MAX_RESPONSE_TIME_MINUTES,
        "source": "sample600_labelled.csv joined with applesupport_threads_repro.parquet"
    }

    try:
        SLA_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SLA_LOOKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(lookup_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save {SLA_LOOKUP_PATH}: {e}")

    return lookup_data


# Cached in memory at module import
_SLA_DATA = load_or_build_intent_response_times()
HISTORICAL_INTENT_MEDIAN_MINUTES = _SLA_DATA.get("intent_median_minutes", {})
GLOBAL_FALLBACK_MEDIAN_MINUTES = _SLA_DATA.get("global_median_minutes", 72.98)


def compute_intent_sla_risk(intent: Optional[str]) -> float:
    """
    Normalizes median historical response time to a 0.0 to 1.0 risk score.
    Higher resolution time = higher risk. Unseen intents default to the global median.
    """
    if not intent or not isinstance(intent, str):
        median_minutes = GLOBAL_FALLBACK_MEDIAN_MINUTES
    else:
        norm_intent = intent.strip().lower()
        median_minutes = HISTORICAL_INTENT_MEDIAN_MINUTES.get(norm_intent, GLOBAL_FALLBACK_MEDIAN_MINUTES)

    score = (median_minutes - MIN_RESPONSE_TIME_MINUTES) / (MAX_RESPONSE_TIME_MINUTES - MIN_RESPONSE_TIME_MINUTES)
    return round(float(max(0.0, min(1.0, score))), 4)


def compute_sentiment_urgency_risk(customer_text: str) -> float:
    """
    Computes a 0.0 to 1.0 urgency score from the LLM severity check and urgency keywords.
    """
    if not customer_text or not isinstance(customer_text, str):
        return 0.10

    # Severity check from src.escalate
    severity, _ = get_sentiment_severity(customer_text)
    base_scores = {
        "routine": 0.10,
        "frustrated": 0.60,
        "severe": 1.00
    }
    score = base_scores.get(severity, 0.30)

    # Keyword check
    if URGENCY_REGEX.search(customer_text):
        score = min(1.0, score + 0.15)

    return round(float(score), 4)


def compute_sla_risk_score(intent: Optional[str], customer_text: str) -> Tuple[float, float, float]:
    """
    Combines intent historical sensitivity and message urgency into a composite risk score.
    
    Formula:
        sla_risk = 0.50 * intent_risk + 0.50 * sentiment_risk
    """
    intent_risk = compute_intent_sla_risk(intent)
    sentiment_risk = compute_sentiment_urgency_risk(customer_text)
    
    composite_risk = 0.50 * intent_risk + 0.50 * sentiment_risk
    composite_risk = max(0.0, min(1.0, composite_risk))
    
    return round(float(composite_risk), 4), intent_risk, sentiment_risk


def route_message(
    customer_text: str,
    intent: str,
    is_escalated: bool,
    escalate_reason: str,
    retrieval_similarity: float,
    intent_confidence: Optional[float] = None
) -> Dict[str, Any]:
    """
    Decides between AUTO_SEND, DRAFT_FOR_REVIEW, and ESCALATE.
    
    Logic:
      1. ESCALATE if is_escalated is True (safety, legal, or severe model risk).
      2. AUTO_SEND if not escalated AND retrieval similarity >= 0.40 AND SLA risk <= 0.45.
      3. DRAFT_FOR_REVIEW for everything else (needs a human glance before sending).
    """
    sla_risk, intent_risk, sentiment_risk = compute_sla_risk_score(intent, customer_text)
    sim = float(retrieval_similarity or 0.0)

    if is_escalated:
        return {
            "routing_tier": ROUTING_TIER_ESCALATE,
            "sla_risk_score": sla_risk,
            "intent_risk": intent_risk,
            "sentiment_risk": sentiment_risk,
            "retrieval_similarity": sim,
            "routing_reason": f"Escalated via gate: {escalate_reason}"
        }

    if sim >= MIN_AUTO_SEND_SIMILARITY and sla_risk <= MAX_AUTO_SEND_SLA_RISK:
        return {
            "routing_tier": ROUTING_TIER_AUTO_SEND,
            "sla_risk_score": sla_risk,
            "intent_risk": intent_risk,
            "sentiment_risk": sentiment_risk,
            "retrieval_similarity": sim,
            "routing_reason": (
                f"High retrieval confidence ({sim:.2f} >= {MIN_AUTO_SEND_SIMILARITY}) "
                f"and low SLA risk ({sla_risk:.2f} <= {MAX_AUTO_SEND_SLA_RISK})"
            )
        }

    reasons = []
    if sim < MIN_AUTO_SEND_SIMILARITY:
        reasons.append(f"lower retrieval similarity ({sim:.2f} < {MIN_AUTO_SEND_SIMILARITY})")
    if sla_risk > MAX_AUTO_SEND_SLA_RISK:
        reasons.append(f"higher SLA risk ({sla_risk:.2f} > {MAX_AUTO_SEND_SLA_RISK})")

    detail = " and ".join(reasons) if reasons else "QA review"
    return {
        "routing_tier": ROUTING_TIER_DRAFT_FOR_REVIEW,
        "sla_risk_score": sla_risk,
        "intent_risk": intent_risk,
        "sentiment_risk": sentiment_risk,
        "retrieval_similarity": sim,
        "routing_reason": f"Drafted for review due to {detail}"
    }
