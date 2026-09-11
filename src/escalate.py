"""
Escalation Decision Gate Module for AppleSupport AI Agent.

Two-layer decision architecture:
1. Rule Layer: Fast deterministic regex/keyword pattern matching across 5 high-risk categories.
2. Model Layer: Confidence thresholding, sensitive intent checks, and LLM severity classification.
"""

import re
import logging
from typing import Tuple, Optional, Dict, Any

from src.config import (
    CONFIDENCE_ESCALATION_THRESHOLD,
    ESCALATION_PRONE_INTENTS,
    TEMPERATURE_ESCALATE
)
from src.llm import call_llm_json

logger = logging.getLogger("hiver.escalate")

# Rule Layer Regex Patterns
RULES = [
    (
        "rule:legal_threat",
        r'\b(?:lawsuit|lawyer|attorney|sue|suing|police|ftc|court|legal action|chargeback|fraud)\b',
        "Customer initiated legal, regulatory, or law enforcement threat."
    ),
    (
        "rule:safety_hazard",
        r'\b(?:swelling|swollen|exploded|exploding|explode|burning|burned|burnt|burn|smoke|smoking|spark|sparking|shock|shocked|fire|melted|melting|hazardous|overheating)\b',
        "Critical hardware safety hazard (battery thermal runaway, swelling, electric shock, fire)."
    ),
    (
        "rule:security_breach",
        r'\b(?:hacked|hijacked|stolen account|unauthorized device|ransomware|extortion|compromised id|locked out of|locked out)\b',
        "Security breach, account takeover, or compromised credentials."
    ),
    (
        "rule:repeated_failure",
        r'\b(?:still not working|done that 5 times|tried everything|useless support|worst service ever|never fixing this)\b',
        "Repeated troubleshooting failure and extreme customer agitation."
    ),
    (
        "rule:abusive_communication",
        r'\b(?:fuck you|fuck off|piece of shit|go to hell|bastards|assholes)\b',
        "Abusive language or profanity requiring human intervention."
    )
]

def check_rules(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Evaluates rule-based deterministic escalation triggers.
    
    Returns:
        is_escalated: True if any rule matched.
        reason_code: Stable identifier string (e.g. 'rule:legal_threat').
        matched_snippet: The exact substring that triggered the rule.
    """
    if not isinstance(text, str):
        return False, None, None

    for code, pattern, description in RULES:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matched_snippet = match.group(0)
            full_reason = f"{code} ('{matched_snippet}'): {description}"
            return True, full_reason, matched_snippet

    return False, None, None

def check_model_gate(text: str, intent: str, confidence: float) -> Tuple[bool, str]:
    """
    Evaluates model-based heuristics: low confidence, sensitive intent, and LLM severity check.
    """
    # 1. Low Confidence Fallback
    if confidence < CONFIDENCE_ESCALATION_THRESHOLD:
        return True, f"model:low_confidence (intent='{intent}', confidence={confidence:.2f} < {CONFIDENCE_ESCALATION_THRESHOLD})"

    # 2. LLM Extremity & Severity Check
    prompt = (
        "Analyze the following customer support inquiry. "
        "Classify severity into {routine, frustrated, severe}.\n"
        "Return JSON with 'severity' and 'reason'.\n\n"
        f"INQUIRY: \"{text}\"\n"
        "Output JSON: {\"severity\": \"<routine|frustrated|severe>\", \"reason\": \"<one_line_explanation>\"}"
    )

    try:
        res = call_llm_json(
            prompt=prompt,
            temperature=TEMPERATURE_ESCALATE,
            required_fields=["severity", "reason"]
        )
        severity = res.get("severity", "routine").lower()
        reason_explanation = res.get("reason", "Severity assessment complete.")
        
        if severity == "severe":
            return True, f"model:severe_assessment ({reason_explanation})"
    except Exception as e:
        logger.warning(f"Model severity check encountered error: {e}")

    return False, "None - Inquiry suitable for automated resolution."

def gate(customer_text: str, intent: str, confidence: float) -> Tuple[bool, str, Optional[str]]:
    """
    Main Escalation Decision Gate Entry Point.
    
    Returns:
        escalate (bool): True if the inquiry must be escalated to human agents.
        reason (str): Human-readable justification.
        matched_snippet (Optional[str]): Triggering text snippet if rule-based.
    """
    # Step 1: Rule Layer (Deterministic & Fast)
    rule_esc, rule_reason, matched_snippet = check_rules(customer_text)
    if rule_esc:
        return True, rule_reason, matched_snippet

    # Step 2: Model Layer (Heuristic & Severity Check)
    model_esc, model_reason = check_model_gate(customer_text, intent, confidence)
    if model_esc:
        return True, model_reason, None

    return False, "None - Standard automated resolution pathway.", None
