"""
SLA-Aware Confidence Gate and 3-Way Routing Engine.

This module implements a product-grade SLA management and AI Quality Assurance (AI QA)
routing layer for customer support automation.

Product Context & Purpose:
--------------------------
In enterprise support platforms (such as Hiver), binary auto-handle vs. escalation decisions
are insufficient. Many inquiries can be accurately drafted by AI but carry elevated SLA
breach risk, subtle ambiguity, or moderate customer frustration. Routing these to
'DRAFT_FOR_REVIEW' allows human agents to glance at, approve, or edit the AI draft in <10 seconds,
preventing bad automated replies while saving 80%+ of agent authoring time.

Mathematical Formulation & Scoring Rules:
-----------------------------------------
1. Intent SLA Risk (R_intent in [0.0, 1.0]):
   Derived from the empirical median brand response time per intent category in historical data:
       R_intent = clip((median_minutes - 35.0) / (105.0 - 35.0), 0.0, 1.0)
   Slower historical resolution times correlate with higher complexity and stricter SLA exposure.
   Unseen or missing intents default monotonically to the global corpus median (73.0 min -> ~0.54).

2. Sentiment & Urgency Risk (R_sentiment in [0.0, 1.0]):
   Extracted from the escalation gate's sentiment severity classifier and urgency regex cues:
       - 'routine'    -> 0.10
       - 'frustrated' -> 0.60
       - 'severe'     -> 1.00
       - Urgency boost (+0.15 if customer explicitly mentions ASAP / urgent / critical)

3. Combined Composite SLA Risk Score (S_sla in [0.0, 1.0]):
       S_sla = clip(0.50 * R_intent + 0.50 * R_sentiment, 0.0, 1.0)

4. Three-Way Routing Decision:
   - ESCALATE:
       Triggered if the primary escalation gate fired (is_escalated == True).
   - AUTO_SEND:
       Triggered if NOT escalated AND retrieval_similarity >= MIN_AUTO_SEND_SIMILARITY
       AND S_sla <= MAX_AUTO_SEND_SLA_RISK.
   - DRAFT_FOR_REVIEW:
       All other inquiries (e.g. valid technical reply drafted, but retrieval similarity
       or SLA sensitivity indicates human review is prudent prior to public posting).
"""

import os
import re
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd

from src.config import (
    THREADS_PARQUET_PATH,
    SAMPLE_600_CSV_PATH,
    RETRIEVAL_MIN_SIMILARITY
)
from src.escalate import get_sentiment_severity

logger = logging.getLogger("hiver.sla_gate")

# =====================================================================
# NAMED THRESHOLD CONSTANTS (Explicitly tuned for high precision review)
# =====================================================================
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

# Historical response time lookup (in minutes) derived from training data
HISTORICAL_INTENT_MEDIAN_MINUTES: Dict[str, float] = {
    "battery_drain_power": 101.7,
    "billing_app_store": 100.0,
    "keyboard_autocorrect_bug": 82.3,
    "account_icloud_login": 75.4,
    "system_performance_freeze": 64.5,
    "audio_music_playback": 62.5,
    "other": 41.8,
    "connectivity_network": 41.6,
    "camera_photos_media": 38.2,
}

GLOBAL_FALLBACK_MEDIAN_MINUTES: float = 73.0
MIN_RESPONSE_TIME_MINUTES: float = 35.0
MAX_RESPONSE_TIME_MINUTES: float = 105.0

# Explicit urgency keyword patterns
URGENCY_REGEX = re.compile(
    r'\b(?:urgent|urgently|immediately|asap|right now|critical|emergency|hurry|today|cannot wait)\b',
    re.IGNORECASE
)


def compute_intent_sla_risk(intent: Optional[str]) -> float:
    """
    Computes a normalized [0.0, 1.0] SLA risk score from historical brand response times.
    Monotonically maps higher median resolution minutes to higher risk.
    
    Parameters:
        intent: Classified customer support intent.
        
    Returns:
        Normalized float in [0.0, 1.0].
    """
    if not intent or not isinstance(intent, str):
        median_minutes = GLOBAL_FALLBACK_MEDIAN_MINUTES
    else:
        norm_intent = intent.strip().lower()
        median_minutes = HISTORICAL_INTENT_MEDIAN_MINUTES.get(norm_intent, GLOBAL_FALLBACK_MEDIAN_MINUTES)

    # Min-max normalization clamped to [0.0, 1.0]
    score = (median_minutes - MIN_RESPONSE_TIME_MINUTES) / (MAX_RESPONSE_TIME_MINUTES - MIN_RESPONSE_TIME_MINUTES)
    return round(float(max(0.0, min(1.0, score))), 4)


def compute_sentiment_urgency_risk(customer_text: str) -> float:
    """
    Reuses the escalation gate sentiment severity classifier and urgency cues
    to calculate a normalized [0.0, 1.0] urgency score.
    
    Parameters:
        customer_text: Inbound customer message text.
        
    Returns:
        Normalized float in [0.0, 1.0].
    """
    if not customer_text or not isinstance(customer_text, str):
        return 0.10

    # 1. Call reused sentiment severity classifier from src.escalate
    severity, _ = get_sentiment_severity(customer_text)
    base_scores = {
        "routine": 0.10,
        "frustrated": 0.60,
        "severe": 1.00
    }
    score = base_scores.get(severity, 0.30)

    # 2. Check for explicit urgency phrases
    if URGENCY_REGEX.search(customer_text):
        score = min(1.0, score + 0.15)

    return round(float(score), 4)


def compute_sla_risk_score(intent: Optional[str], customer_text: str) -> Tuple[float, float, float]:
    """
    Computes the composite SLA risk score from intent response time and sentiment urgency.
    
    Formula:
        sla_risk = 0.50 * intent_risk + 0.50 * sentiment_risk
        
    Returns:
        (sla_risk_score, intent_risk, sentiment_risk) all in [0.0, 1.0].
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
    Determines 3-way routing tier (AUTO_SEND, DRAFT_FOR_REVIEW, ESCALATE).
    
    Decision Rules:
      1. Tier 1: ESCALATE
         Triggered whenever is_escalated is True.
      2. Tier 2: AUTO_SEND
         Triggered when NOT escalated AND retrieval similarity >= MIN_AUTO_SEND_SIMILARITY
         AND SLA risk <= MAX_AUTO_SEND_SLA_RISK.
      3. Tier 3: DRAFT_FOR_REVIEW
         Triggered for all remaining non-escalated cases where reply is drafted but
         human QA is required before sending.
         
    Parameters:
        customer_text: Raw or sanitized customer tweet.
        intent: Classified intent category.
        is_escalated: Boolean escalation flag from src.escalate.
        escalate_reason: Escalation reason string.
        retrieval_similarity: Top-1 cosine similarity score of retrieved historical examples.
        intent_confidence: Optional intent classification confidence.
        
    Returns:
        dict containing:
            - routing_tier: 'AUTO_SEND' | 'DRAFT_FOR_REVIEW' | 'ESCALATE'
            - sla_risk_score: float [0.0, 1.0]
            - intent_sla_risk: float [0.0, 1.0]
            - sentiment_urgency_score: float [0.0, 1.0]
            - routing_reason: str explaining the routing justification
    """
    sla_risk, intent_risk, sentiment_risk = compute_sla_risk_score(intent, customer_text)
    sim = float(retrieval_similarity or 0.0)

    # 1. Primary Escalation Check
    if is_escalated:
        return {
            "routing_tier": ROUTING_TIER_ESCALATE,
            "sla_risk_score": sla_risk,
            "intent_risk": intent_risk,
            "sentiment_risk": sentiment_risk,
            "retrieval_similarity": sim,
            "routing_reason": f"Escalated via gate: {escalate_reason}"
        }

    # 2. Automated Dispatch Check
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

    # 3. AI QA Draft-For-Review Check
    reasons = []
    if sim < MIN_AUTO_SEND_SIMILARITY:
        reasons.append(f"moderate/low retrieval similarity ({sim:.2f} < {MIN_AUTO_SEND_SIMILARITY})")
    if sla_risk > MAX_AUTO_SEND_SLA_RISK:
        reasons.append(f"elevated SLA risk ({sla_risk:.2f} > {MAX_AUTO_SEND_SLA_RISK})")

    detail = " and ".join(reasons) if reasons else "quality assurance check"
    return {
        "routing_tier": ROUTING_TIER_DRAFT_FOR_REVIEW,
        "sla_risk_score": sla_risk,
        "intent_risk": intent_risk,
        "sentiment_risk": sentiment_risk,
        "retrieval_similarity": sim,
        "routing_reason": f"Drafted for human QA due to {detail}"
    }
