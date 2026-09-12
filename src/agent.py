"""
Single Unified Entry Point for AppleSupport AI Agent.

Composes:
PII Redaction -> Intent Classification -> Escalation Decision Gate -> Grounded Retrieval & Drafting.
"""

import time
import logging
from typing import Dict, Any

from src.config import LLM_MODEL, RETRIEVAL_TOP_K
from src.pii import redact_pii
from src.intents import classify
from src.escalate import gate
from src.reply import retrieve, draft_reply
from src.sla_gate import route_message

logger = logging.getLogger("hiver.agent")

def agent_reply(customer_text: str) -> Dict[str, Any]:
    """
    Unified end-to-end agent pipeline.
    
    Parameters:
        customer_text: Raw incoming tweet text from a customer.
        
    Returns:
        dict containing:
            - intent (str): Predicted support intent.
            - intent_confidence (float): Confidence score (0.0 to 1.0).
            - escalate (bool): True if escalated to human supervisor/tier-2.
            - escalate_reason (str): Justification for escalation decision.
            - routing_tier (str): 'AUTO_SEND' | 'DRAFT_FOR_REVIEW' | 'ESCALATE'.
            - sla_risk_score (float): Composite SLA risk score [0.0, 1.0].
            - sla_routing (dict): Detailed SLA gate metrics and reasons.
            - draft_reply (str): Grounded draft response or escalation hand-off.
            - retrieved_ids (list): IDs of historical resolved tweets used for grounding.
            - retrieved_replies (list): Text of historical brand replies.
            - model_name (str): Active model identifier.
            - latency_ms (dict): Latency breakdown per stage in milliseconds.
    """
    total_start = time.perf_counter()
    latencies = {}

    # Stage 1: PII Redaction
    t0 = time.perf_counter()
    sanitized_text, pii_count, pii_types = redact_pii(customer_text)
    latencies["pii"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 2: Intent Classification
    t0 = time.perf_counter()
    intent, confidence, clf_source = classify(sanitized_text)
    latencies["classify"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 3: Escalation Decision Gate
    t0 = time.perf_counter()
    is_escalated, escalate_reason = gate(sanitized_text, intent, confidence)
    latencies["escalate"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 4: Grounded Historical Retrieval (Always performed for citation/audit)
    t0 = time.perf_counter()
    retrieved_items = retrieve(sanitized_text, k=RETRIEVAL_TOP_K)
    retrieved_ids = [item["tweet_id"] for item in retrieved_items]
    retrieved_replies = [item["brand_reply_text"] for item in retrieved_items]
    top_sim = retrieved_items[0]["similarity"] if retrieved_items else 0.0
    latencies["retrieve"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 5: SLA-Aware Confidence Gate (3-Way Routing)
    t0 = time.perf_counter()
    sla_info = route_message(
        customer_text=sanitized_text,
        intent=intent,
        is_escalated=is_escalated,
        escalate_reason=escalate_reason,
        retrieval_similarity=top_sim,
        intent_confidence=confidence
    )
    latencies["sla_gate"] = round((time.perf_counter() - t0) * 1000, 2)

    # Stage 6: Reply Drafting
    t0 = time.perf_counter()
    if is_escalated:
        draft = f"[ESCALATED TO HUMAN SPECIALIST: {escalate_reason}] We want to ensure you get dedicated support on this urgent matter. Our senior team has been notified and will assist you directly."
    else:
        draft = draft_reply(sanitized_text, retrieved_items)
    latencies["draft"] = round((time.perf_counter() - t0) * 1000, 2)

    latencies["total"] = round((time.perf_counter() - total_start) * 1000, 2)

    return {
        "intent": intent,
        "intent_confidence": round(confidence, 4),
        "escalate": is_escalated,
        "escalate_reason": escalate_reason,
        "routing_tier": sla_info["routing_tier"],
        "sla_risk_score": sla_info["sla_risk_score"],
        "sla_routing": sla_info,
        "retrieval_similarity": top_sim,
        "draft_reply": draft,
        "retrieved_ids": retrieved_ids,
        "retrieved_replies": retrieved_replies,
        "model_name": LLM_MODEL,
        "classifier_source": clf_source,
        "pii_redacted": pii_count > 0,
        "pii_count": pii_count,
        "latency_ms": latencies
    }
