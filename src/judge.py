"""
LLM-as-a-Judge Module for Customer Support Reply Quality Evaluation.

Scores agent replies on a 1-5 scale across 5 distinct criteria:
1. Correctness: Does the draft address the customer's actual technical issue?
2. Groundedness: Are all facts, URLs, settings paths, and steps verified by the retrieved examples (Hallucination check)?
3. Completeness: Does the response provide actionable diagnostic next steps?
4. Brand Voice: Does it match the concise, professional tone of official Apple Support?
5. Tone: Is it empathetic, constructive, and de-escalating?

Includes Judge v1 and calibrated Judge v2 prompts.
"""

import json
import logging
from typing import Dict, Any, List
from src.llm import call_llm_json

logger = logging.getLogger("hiver.judge")

CRITERIA = ["correctness", "groundedness", "completeness", "brand_voice", "tone"]

def _build_judge_prompt_v1(customer_text: str, draft_reply: str, retrieved_replies: List[str]) -> str:
    retrieved_formatted = "\n".join([f"  [{i+1}] \"{r}\"" for i, r in enumerate(retrieved_replies)])
    
    return f"""You are an expert impartial evaluator judging an AI customer support reply for @AppleSupport.

INBOUND CUSTOMER INQUIRY:
"{customer_text}"

RETRIEVED HISTORICAL RESOLUTION EXAMPLES (GROUND TRUTH CONTEXT):
{retrieved_formatted if retrieved_formatted else "  [No historical context retrieved]"}

DRAFT REPLY TO EVALUATE:
"{draft_reply}"

EVALUATION RUBRIC (Score each criterion strictly from 1 to 5):

1. correctness (1-5):
   - 1: Completely misunderstands the customer's issue or gives irrelevant advice.
   - 3: Partially addresses the issue but misses a core symptom or asks vague questions.
   - 5: Accurately identifies and addresses the exact root cause / technical issue described.

2. groundedness (1-5) [HALLUCINATION CHECK]:
   - 1: Severe hallucination; invents fake URL links, non-existent iOS settings, false refund amounts, or unverified claims.
   - 3: Minor ungrounded assumption, but general advice is safe and plausible.
   - 5: Strictly grounded; every setting path, URL (e.g. iforgot.apple.com, reportaproblem.apple.com, DM links), and troubleshooting step is substantiated by retrieved examples.

3. completeness (1-5):
   - 1: Empty, unhelpful deflection or missing all necessary diagnostic guidance.
   - 3: Provides some advice but leaves the user hanging without next steps or contact channels.
   - 5: Comprehensive; provides immediate troubleshooting step, diagnostic question, and clear escalation/DM link.

4. brand_voice (1-5):
   - 1: Robotic, overly verbose, or completely foreign to Twitter customer support.
   - 3: Acceptable but lacks standard brand phrasing or formatting.
   - 5: Perfect match for official @AppleSupport Twitter tone (concise, polite, direct, clear formatting).

5. tone (1-5):
   - 1: Rude, dismissive, sarcastic, or defensive.
   - 3: Neutral and polite, but slightly cold or overly formal.
   - 5: Empathetic, supportive, reassuring, and customer-centric.

OUTPUT SCHEMA:
Return ONLY valid JSON matching this exact structure:
{{
  "correctness": {{"score": <1-5>, "one_line_reason": "<brief justification>"}},
  "groundedness": {{"score": <1-5>, "one_line_reason": "<brief justification>"}},
  "completeness": {{"score": <1-5>, "one_line_reason": "<brief justification>"}},
  "brand_voice": {{"score": <1-5>, "one_line_reason": "<brief justification>"}},
  "tone": {{"score": <1-5>, "one_line_reason": "<brief justification>"}}
}}
"""

def _build_judge_prompt_v2(customer_text: str, draft_reply: str, retrieved_replies: List[str]) -> str:
    """
    Judge v2: Calibrated post-Round-1 with tightened grounding penalties and stricter completeness checks.
    """
    retrieved_formatted = "\n".join([f"  [{i+1}] \"{r}\"" for i, r in enumerate(retrieved_replies)])
    
    return f"""You are a calibrated, rigorous evaluation judge for @AppleSupport AI replies.

CALIBRATION GUIDELINES:
- DO NOT over-reward polite fluff. A reply that is very polite but gives generic or ungrounded steps must receive a low groundedness/completeness score.
- STRICT GROUNDEDNESS: Penalize any reply that hallucinates specific steps not attested in the retrieved context.
- STRICT COMPLETENESS: An escalated inquiry or complex bug MUST ask for the exact iOS build version or provide a DM link to receive a score of 5.

INBOUND CUSTOMER INQUIRY:
"{customer_text}"

RETRIEVED HISTORICAL RESOLUTION EXAMPLES:
{retrieved_formatted if retrieved_formatted else "  [No historical context retrieved]"}

DRAFT REPLY TO EVALUATE:
"{draft_reply}"

SCORING CRITERIA (1-5):
1. correctness: Direct relevance to the customer's problem.
2. groundedness: 100% fidelity to retrieved historical troubleshooting steps.
3. completeness: Diagnostic clarity, setting path accuracy, and follow-up guidance.
4. brand_voice: Conciseness, correct URL formatting, official @AppleSupport style.
5. tone: Active empathy, professional demeanor, and de-escalation quality.

OUTPUT SCHEMA:
{{
  "correctness": {{"score": <1-5>, "one_line_reason": "<justification>"}},
  "groundedness": {{"score": <1-5>, "one_line_reason": "<justification>"}},
  "completeness": {{"score": <1-5>, "one_line_reason": "<justification>"}},
  "brand_voice": {{"score": <1-5>, "one_line_reason": "<justification>"}},
  "tone": {{"score": <1-5>, "one_line_reason": "<justification>"}}
}}
"""

def evaluate_reply(
    customer_text: str,
    draft_reply: str,
    retrieved_replies: List[str],
    version: str = "v1"
) -> Dict[str, Dict[str, Any]]:
    """
    Scores a candidate support reply using LLM-as-a-Judge.
    
    Returns:
        dict of {criterion: {'score': int, 'one_line_reason': str}}
    """
    if version == "v2":
        prompt = _build_judge_prompt_v2(customer_text, draft_reply, retrieved_replies)
    else:
        prompt = _build_judge_prompt_v1(customer_text, draft_reply, retrieved_replies)

    system_prompt = "You are an expert impartial quality evaluator. Return ONLY valid JSON."
    
    response = call_llm_json(
        prompt=prompt,
        temperature=0.0,
        system_prompt=system_prompt,
        required_fields=CRITERIA
    )

    scores = {}
    for crit in CRITERIA:
        item = response.get(crit, {})
        if isinstance(item, dict):
            score = int(item.get("score", 3))
            reason = str(item.get("one_line_reason", "Evaluation complete."))
        else:
            score = int(item) if str(item).isdigit() else 3
            reason = "Standard score assigned."
            
        score = max(1, min(5, score)) # Clamp [1, 5]
        scores[crit] = {
            "score": score,
            "one_line_reason": reason
        }

    return scores
