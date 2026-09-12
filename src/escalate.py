"""
Escalation Decision Gate Module for AppleSupport AI Agent.

Architecture:
  Inbound Customer Message
             │
             ▼
   ┌───────────────────┐
   │    RULE LAYER     │  (Priority 1: Fast deterministic safety/legal/security regex)
   └───────────────────┘
             │
     Rule triggered?
       /          \
     YES           NO
      │             │
      ▼             ▼
  ESCALATE    ┌───────────────────┐
              │    MODEL LAYER    │  (Priority 2: Low confidence, sensitive intent, severe sentiment)
              └───────────────────┘
                        │
                Model condition?
                  /          \
                YES           NO
                 │             │
                 ▼             ▼
             ESCALATE     AUTO-HANDLE

Public Interface:
  gate(customer_text: str, intent: str, confidence: float) -> Tuple[bool, str]
"""

import re
import math
import logging
from typing import Tuple, Optional, Dict, Any, List

from src.config import (
    CONFIDENCE_ESCALATION_THRESHOLD,
    ESCALATION_PRONE_INTENTS,
    TEMPERATURE_ESCALATE
)
from src.pii import redact_pii
from src.llm import call_llm_json

logger = logging.getLogger("hiver.escalate")

# Rule Layer Regex Patterns (Maintainable & Configurable Dictionary)
RULE_PATTERNS: Dict[str, List[str]] = {
    "rule:legal_threat": [
        r'\b(?:lawsuit|lawyer|attorney|legal action|legal complaint|legal notice|sue|suing|take you to court|consumer court|complaint to regulator|file a complaint with|chargeback|fraud)\b',
        r'\b(?:contact(?:ed|ing)? (?:my|a) lawyer|involve (?:my|a) lawyer|legal counsel)\b'
    ],
    "rule:safety": [
        r'\b(?:kill myself|suicide|self harm|hurt myself|end my life|want to die|commit suicide)\b',
        r'\b(?:swelling|swollen battery|exploded|exploding|explode|burning|burned|burnt|smoke|smoking|spark|sparking|shock|shocked|fire|melted|melting|hazardous|overheating)\b'
    ],
    "rule:security": [
        r'\b(?:(?:account|apple id|iphone|device) (?:has been |was )?hacked|someone hacked|account (?:is )?compromised|compromised (?:account|id))\b',
        r'\b(?:unauthorized (?:login|device|access|charge)|someone (?:logged into|accessed) my account|password (?:was )?stolen|stolen account|ransomware|extortion)\b'
    ],
    "rule:media": [
        r'\b(?:i(?:\'m| am)? (?:a )?(?:journalist|reporter)|writing an article|for the (?:press|newspaper|magazine|news)|press inquiry|media inquiry|news outlet)\b',
        r'\b(?:report(?:er)? from|interview for|publishing a story)\b'
    ],
    "rule:abuse": [
        r'\b(?:fuck you|fuck off|piece of shit|go to hell|bastards|assholes|die in a fire|bitch)\b',
        r'\b(?:kill you|murder you|threaten you)\b'
    ],
    "rule:pii_needed": [
        r'\b(?:send (?:me )?(?:your|my) full card number|give me my account number|here is my phone number|send my personal details)\b',
        r'\b(?:full credit card number|full debit card number|send your password|verify credit card number|full ssn|social security number|my full ssn)\b'
    ]
}

# Rule Descriptions for Human Readability & Audit Logs
RULE_DESCRIPTIONS = {
    "rule:legal_threat": "Customer initiated legal, regulatory, or law enforcement threat.",
    "rule:safety": "Critical hardware safety hazard, battery failure, or physical harm signal.",
    "rule:security": "Account compromise, unauthorized access, or stolen credentials.",
    "rule:media": "Press, journalist, or public media inquiry requiring PR/communications routing.",
    "rule:abuse": "Severe profanity, abusive harassment, or direct threats requiring agent protection.",
    "rule:pii_needed": "Resolution requires exchanging full card/SSN/passwords prohibited on public channels."
}

def check_rules(text: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Evaluates rule-based deterministic escalation triggers.
    
    Parameters:
        text: Inbound customer message.
        
    Returns:
        is_escalated (bool): True if any deterministic rule matched.
        reason_code (Optional[str]): Stable reason identifier (e.g. 'rule:legal_threat').
        matched_snippet (Optional[str]): Triggering substring.
    """
    if not isinstance(text, str) or not text.strip():
        return False, None, None

    for code, patterns in RULE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                matched_snippet = match.group(0)
                logger.info(f"Rule Layer triggered: {code} on snippet '{matched_snippet}'")
                return True, code, matched_snippet

    return False, None, None

def get_sentiment_severity(text: str) -> Tuple[str, str]:
    """
    Classifies customer inquiry severity into routine, frustrated, or severe.
    Reuses the cached LLM prompt for deterministic offline evaluation.
    
    Returns:
        (severity: str, reason: str) where severity in {'routine', 'frustrated', 'severe'}.
    """
    sanitized_text, _, _ = redact_pii(text)
    prompt = (
        "Analyze the following customer support inquiry. "
        "Classify severity into {routine, frustrated, severe}.\n"
        "Return JSON with 'severity' and 'reason'.\n\n"
        f"INQUIRY: \"{sanitized_text}\"\n"
        "Output JSON: {\"severity\": \"<routine|frustrated|severe>\", \"reason\": \"<one_line_explanation>\"}"
    )

    try:
        res = call_llm_json(
            prompt=prompt,
            temperature=TEMPERATURE_ESCALATE,
            required_fields=["severity"]
        )
        severity = str(res.get("severity", "routine")).strip().lower()
        reason = str(res.get("reason", ""))
        return severity, reason
    except Exception as e:
        logger.warning(f"Model severity check encountered error: {e}. Defaulting safely.")
        return "routine", "error_fallback"

def check_model_layer(text: str, intent: str, confidence: float) -> Tuple[bool, str]:
    """
    Evaluates model-based heuristics: low confidence, sensitive intent, and LLM severity check.
    
    Order of Evaluation:
      1. Low Confidence Fallback (< CONFIDENCE_ESCALATION_THRESHOLD or invalid)
      2. Escalation-Prone Intent (e.g. account_icloud_login, billing_app_store)
      3. Extreme Negative Sentiment / Severity (LLM classification: routine, frustrated, severe)
      4. Auto-Handle (False, 'auto_handle')
    """
    # 1. Validate confidence score
    is_invalid_conf = (
        confidence is None or
        not isinstance(confidence, (int, float)) or
        math.isnan(confidence) or
        confidence < 0.0 or
        confidence > 1.0
    )
    conf_val = 0.0 if is_invalid_conf else float(confidence)

    # Condition A: Low Intent Confidence
    if is_invalid_conf or conf_val < CONFIDENCE_ESCALATION_THRESHOLD:
        reason = f"model:low_confidence(intent={intent or 'unknown'}, conf={conf_val:.2f})"
        logger.info(f"Model Layer triggered: {reason}")
        return True, reason

    # Condition B: Escalation-Prone Intent
    if intent in ESCALATION_PRONE_INTENTS:
        reason = f"model:escalation_prone_intent(intent={intent})"
        logger.info(f"Model Layer triggered: {reason}")
        return True, reason

    # Condition C: Extreme Negative Sentiment / Severity Check
    severity, _ = get_sentiment_severity(text)
    if severity == "severe":
        reason = "model:severe_sentiment"
        logger.info(f"Model Layer triggered: {reason}")
        return True, reason

    return False, "auto_handle"

def gate(
    customer_text: str,
    intent: str,
    confidence: float
) -> Tuple[bool, str]:
    """
    Decides whether a customer message should be escalated to human agents.

    Rule-based escalation is evaluated before model-based escalation.
    
    Parameters:
        customer_text: Inbound customer message text.
        intent: Predicted intent category.
        confidence: Confidence score of intent classification [0.0, 1.0].
        
    Returns:
        (escalate: bool, reason: str)
    """
    if not isinstance(customer_text, str):
        customer_text = str(customer_text or "")

    # Priority 1: Rule Layer (Deterministic & Fast)
    rule_esc, rule_reason, _ = check_rules(customer_text)
    if rule_esc:
        return True, rule_reason

    # Priority 2: Model Layer (Runs ONLY if no rule fires)
    model_esc, model_reason = check_model_layer(customer_text, intent, confidence)
    if model_esc:
        return True, model_reason

    return False, "auto_handle"

