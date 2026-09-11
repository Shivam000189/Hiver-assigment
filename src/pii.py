"""
PII Redaction Module.

Redacts sensitive customer information (emails, phone numbers, credit card sequences, OTPs)
before any text is processed by LLM APIs or logged to disk.
"""

import re
from typing import Tuple, List

# Regex Patterns for PII Detection
EMAIL_PATTERN = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
PHONE_PATTERN = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
CARD_LONG_DIGITS_PATTERN = r'\b(?:\d[ -]*?){10,19}\b'
IMEI_SERIAL_PATTERN = r'\b(?:IMEI|Serial|SN|Serial Number)[\s:]*([A-Za-z0-9]{8,18})\b'

def redact_pii(text: str) -> Tuple[str, int, List[str]]:
    """
    Redacts PII from customer text.
    
    Returns:
        redacted_text: The sanitized string.
        redaction_count: Total number of sensitive entities redacted.
        detected_types: List of PII types found (e.g. ['email', 'card_number']).
    """
    if not isinstance(text, str) or not text.strip():
        return text, 0, []

    redacted = text
    count = 0
    detected_types = []

    # 1. Emails
    emails = re.findall(EMAIL_PATTERN, redacted)
    if emails:
        count += len(emails)
        detected_types.append("email")
        redacted = re.sub(EMAIL_PATTERN, "[EMAIL_REDACTED]", redacted)

    # 2. Phone Numbers (Checked before general card/account digits)
    phones = re.findall(PHONE_PATTERN, redacted)
    if phones:
        count += len(phones)
        detected_types.append("phone_number")
        redacted = re.sub(PHONE_PATTERN, "[PHONE_REDACTED]", redacted)

    # 3. Credit Cards / Long Digit Sequences (>= 10 digits)
    cards = re.findall(CARD_LONG_DIGITS_PATTERN, redacted)
    if cards:
        count += len(cards)
        detected_types.append("card_or_account_number")
        redacted = re.sub(CARD_LONG_DIGITS_PATTERN, "[ACCOUNT_NUM_REDACTED]", redacted)

    # 4. Device Serial Numbers / IMEI
    serials = re.findall(IMEI_SERIAL_PATTERN, redacted, flags=re.IGNORECASE)
    if serials:
        count += len(serials)
        detected_types.append("device_identifier")
        redacted = re.sub(IMEI_SERIAL_PATTERN, r"\1 [DEVICE_ID_REDACTED]", redacted, flags=re.IGNORECASE)

    return redacted, count, detected_types
