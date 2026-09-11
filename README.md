# Hiver Support Agent

A customer-support AI agent built from real customer-support conversation data.

## Project Status

Currently setting up the project environment (Step 0).

## Project Structure

```text
hiver-support-agent/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden_set/
├── notebooks/
├── src/
│   ├── prepare_data.py
│   ├── intents.py
│   ├── reply.py
│   ├── escalate.py
│   └── evaluate.py
├── results/
└── report/
```

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development

Source modules in `src/`:
- `prepare_data.py`: Dataset preparation and inspection.
- `intents.py`: Intent taxonomy mining and intent classification.
- `reply.py`: Historical retrieval and grounded reply drafting.
- `escalate.py`: Escalation decision gate.
- `evaluate.py`: Evaluation harness and metrics.

## Roadmap

1. Setup environment [Done]
2. Get and inspect the dataset [Done]
3. Reconstruct threads and select one brand [Done - @AppleSupport]
4. Mine the intent taxonomy [Done - 9 Intents]
5. Build the golden evaluation set [Done - 200 Examples: 120 Dev / 80 Locked Test]
6. Build the intent classifier [Done]
7. Build grounded reply generation [Done - Step 7]
8. Build the escalation gate [Done]
9. Build the evaluation harness [Done]
10. Measure judge-vs-human agreement [Done]
11. Perform failure analysis [Upcoming]
12. Write the report [Upcoming]
13. Make the project reproducible [Done]
14. Final submission [Upcoming]

---

## Grounded Reply Generation (Step 7)

### Architecture: `retrieve → rewrite → cite`
The grounded reply drafter guarantees that generated responses are strictly anchored in historical brand handling patterns rather than permitting free-form LLM hallucination:

```text
NEW CUSTOMER MESSAGE
        ↓
   Redact PII (Email, Phone, Account/Card digits)
        ↓
   Retrieve similar resolved examples (Cosine similarity >= 0.15)
        ↓
   Top-k historical customer → brand replies
        ↓
   LLM Grounded Rewrite (Strict grounding rules, no invented policies)
        ↓
   Draft + Complete Citation Trail (customer_tweet_id, brand_tweet_id)
```

### 1. Resolved-Thread Filtering
The retrieval corpus is filtered from all 106,646 `@AppleSupport` conversation threads down to **103,771 usable threads (97.30%)**.
- **Usable Definition**: Requires a non-empty historical brand reply and excludes threads where `has_followup == True` AND the customer returned with transparent frustration/dissatisfaction signals (matching `\b(?:still|again|not working|useless|terrible|worst|refund|scam|lawsuit|sue|lawyer|worse|hate|broken|broke)\b`).
- **Exclusion Breakdown**: 2,875 threads (2.70%) excluded due to persistent customer frustration follow-ups; 0 threads lacked brand replies.

### 2. Embeddings & In-Memory Retrieval Index
- **Index Target**: Historical **customer messages** (`customer_text`) are embedded rather than brand replies because inbound customer inquiries reflect the customer's problem statement.
- **Model**: Sublinear TF-IDF n-gram vectorizer (ngram_range=(1,2), min_df=2, max_features=25,000, English stop-words removed) stored as a fast compressed CSR matrix in `data/processed/reply_index.npz` with metadata in `data/processed/reply_index_metadata.parquet`.
- **Latency**: Vector search over 103,771 threads takes $<5$ ms per query.

### 3. Relevance Filtering & Similarity Threshold
- **Cosine Similarity**: Explicitly computed against all 103,771 indexed historical queries.
- **Configurable Threshold (`RETRIEVAL_MIN_SIMILARITY = 0.15`)**: If the top similarity score is below threshold, retrieval returns an empty list, immediately triggering the safe fallback.

### 4. Strict Grounding Rules (Prompt Constraint)
1. **Rule 1 — Historical Grounding**: Retrieved historical examples serve as the sole source of truth for support policies, diagnostic questions, and resolutions.
2. **Rule 2 — No Invented Policies**: The model is forbidden from hallucinating refund amounts, timelines, replacement fees, or escalation guarantees.
3. **Rule 3 — Adapt to Current Message**: Synthesizes the response to address the new customer's specific technical issue rather than copying historical text verbatim.
4. **Rule 4 — Preserve Facts**: Retains verified settings paths (e.g. `Settings > General > Keyboard`), support links (`reportaproblem.apple.com`, `iforgot.apple.com`), and DM links.
5. **Rule 5 — Never Combine Unrelated Policies**: Does not blend conflicting guidance from disparate examples.
6. **Rule 6 — Fallback on Insufficient Evidence**: If evidence is missing, outputs the standard fallback: `"I want to make sure you get the right help — let me connect you with our team."`

### 5. Citation Trail & Output Schema
Every drafted response returns the full citation trail:
```python
{
    "draft": "We'd like to help get your device running smoothly again. What version of iOS are you currently using? Please send us a DM with more details so we can assist: https://t.co/GDrqU22YpT",
    "retrieved_ids": [283601, 898634, 2321346],
    "retrieved_customer_ids": [283600, 898635, 2321345],
    "retrieved_brand_ids": [283601, 898634, 2321346],
    "retrieved_replies": [
        "@183558 Which version of iOS is installed? When did the issue begin?...",
        "@333383 Great job going through those steps. This time, let's sign out...",
        "@672691 Now we want to test the Live Photos with your friend again..."
    ],
    "grounded": True
}
```

### 6. PII Handling
All incoming customer text and retrieved historical contexts pass through `src/pii.py` before any external LLM prompt generation or disk logging, redacting emails (`[EMAIL_REDACTED]`), phone numbers (`[PHONE_REDACTED]`), credit cards/long account digits (`[ACCOUNT_NUM_REDACTED]`), and device serials/IMEI while preserving non-sensitive troubleshooting context.

---

## Step 7 Decision Log

1. **Why historical customer messages are embedded rather than brand replies**: A new inbound tweet expresses a symptom/question; matching customer-to-customer semantics finds identical problem situations, allowing the agent to fetch the attached brand resolution.
2. **Why sublinear TF-IDF with n-grams was selected**: Provides microsecond CPU search latency over 103,771 threads while capturing exact technical n-grams (e.g., `iOS 11.0.3`, `Settings > General`, `Apple ID`, `iforgot.apple.com`) with zero out-of-vocabulary drift.
3. **Why $k=3$ is the default retrieval value**: Provides sufficient diversity of historical resolutions without diluting the prompt with lower-ranked examples.
4. **Why unresolved/frustrated threads are excluded**: Historical multi-turn interactions where customers express anger or persistent failure represent failed resolutions; filtering them prevents the LLM from imitating unhelpful troubleshooting cycles.
5. **How "resolved" is approximated**: A proxy filter requiring non-null brand replies and the absence of frustration keywords (`still`, `again`, `useless`, `refund`, `broken`, etc.) in subsequent customer turns.
6. **Why cosine similarity is used**: Normalizes for query length differences across short and long tweets.
7. **Why a similarity threshold exists**: Prevents the LLM from attempting to answer out-of-domain or gibberish inquiries with spurious matches.
8. **Why the LLM is forbidden from inventing policies**: Guarantees brand safety and prevents false promises regarding financial compensation or unsupported repair warranties.
9. **Why tweet IDs are returned**: Enables end-to-end provenance verification, auditability, and offline judge verification.
10. **Why PII is redacted before external LLM calls**: Enforces customer data privacy and prevents logging sensitive credentials to disk cache.

---

## Escalation Gate (Step 8)

### Hybrid Architecture: Rule Layer + Model Layer
The escalation gate decides whether an incoming customer message can be safely auto-handled or must be routed to a human specialist. It combines deterministic safety boundaries with adaptive model heuristics:

```text
                 Inbound Customer Message
                            |
                            v
                  +-------------------+
                  |    RULE LAYER     |  (Deterministic safety / legal / security regex)
                  +-------------------+
                            |
                    Rule triggered?
                      /          \
                    YES           NO
                     |             |
                     v             v
                 ESCALATE    +-------------------+
                             |    MODEL LAYER    |  (Confidence, sensitive intent, severe sentiment)
                             +-------------------+
                                       |
                               Model condition?
                                  /          \
                                YES           NO
                                 |             |
                                 v             v
                             ESCALATE     AUTO-HANDLE
```

### 1. Rule Layer (Evaluated First)
Deterministic rules run prior to any model inference to guarantee zero-latency, 100% reliable containment of high-risk inquiries:
- **`rule:legal_threat`**: Explicit legal action, attorneys, lawsuits, consumer court, regulatory complaints, or chargeback threats.
- **`rule:safety`**: Critical hardware hazards (battery swelling, smoking, fire, explosion, electrical shock) or self-harm signals.
- **`rule:security`**: Account compromise, hacked Apple ID, unauthorized access/logins, or stolen passwords.
- **`rule:media`**: Press inquiries, journalists, reporters, or publication notices.
- **`rule:abuse`**: Severe profanity, targeted harassment, or direct threats.
- **`rule:pii_needed`**: Inquiries requiring transmission of sensitive credentials (full card numbers, SSNs, passwords).

### 2. Model Layer (Evaluated Only if No Rule Fires)
If no deterministic rule matches, the message is evaluated across three sequential model criteria:
1. **Low Intent Confidence**: If `confidence < CONFIDENCE_ESCALATION_THRESHOLD (0.40)` or if the confidence score is missing/invalid/NaN, returns `model:low_confidence(intent=<intent>, conf=<confidence>)`.
2. **Escalation-Prone Intent**: If the predicted intent is in `ESCALATION_PRONE_INTENTS` (`account_icloud_login`, `billing_app_store`), returns `model:escalation_prone_intent(intent=<intent>)`.
3. **Severe Sentiment Detection**: An LLM severity classifier (temperature = 0.0) classifies sentiment into `{routine, frustrated, severe}`. Standard customer frustration is auto-handled; only catastrophic distress or extreme hostility triggers `model:severe_sentiment`.
4. **Auto-Handle**: If all checks pass, returns `(False, "auto_handle")`.

### 3. Public Interface
```python
def gate(
    customer_text: str,
    intent: str,
    confidence: float
) -> tuple[bool, str]:
    """
    Decide whether a customer message should be escalated.
    Rule-based escalation is evaluated before model-based escalation.
    Returns (escalate: bool, reason: str).
    """
```

### 4. PII Protection & LLM Caching
Customer text passes through `src/pii.py` to mask sensitive entities prior to external LLM calls for sentiment classification. All LLM calls leverage disk caching in `results/cache/` to ensure offline reproducibility.

---

## Step 8 Decision Log

1. **Why a hybrid Rule + Model architecture was chosen**: Rule-based regex provides instantaneous, deterministic guarantees for mission-critical legal, safety, and security policies that must never fail, while model-based checks dynamically catch ambiguous inquiries and extreme emotional distress.
2. **Why the Rule layer executes before the Model layer**: To guarantee that severe compliance violations (e.g. legal threats or battery explosions) escalate immediately regardless of whether intent classification confidence is 0.99 or 0.10.
3. **Why `CONFIDENCE_ESCALATION_THRESHOLD = 0.40` was selected**: Prevents the agent from hallucinating or generating irrelevant troubleshooting steps for ambiguous, unclassifiable inquiries while allowing typical confidence scores (0.70–0.96) to proceed to auto-handling.
4. **Why `account_icloud_login` and `billing_app_store` are escalation-prone**: Empirical analysis of the `@AppleSupport` taxonomy shows that authentication lockouts and financial disputes carry high regulatory and security risks that often require Tier-2 human intervention.
5. **Why ordinary frustration is auto-handled while severe sentiment is escalated**: Real-world customer support messages frequently express mild frustration (e.g. "This is frustrating but I need help"). Auto-handling routine frustration is critical to maintaining operational automation volume, whereas severe destructive distress requires immediate human empathy.
6. **Why stable reason codes are enforced**: Standardized strings (e.g. `rule:legal_threat`, `model:low_confidence(...)`) enable automated telemetry grouping, precise failure analysis, and structured reporting.
7. **How regex patterns were tuned against benign negative cases**: Regex patterns use boundary markers and contextual phrases to avoid false positives on benign messages (e.g. "I love your media coverage", "Can you tell me your legal business name?").
8. **Why input validation safely catches invalid/NaN confidence**: Defensively wraps inputs to prevent unhandled runtime crashes, ensuring malformed inputs safely default to low-confidence escalation.
9. **Why locked test set isolation is strictly preserved**: Evaluation rules and thresholds were calibrated using only `data/golden_set/golden_dev.csv` and synthetic unit cases; `golden_test.csv` remains strictly untouched for subsequent unbiased benchmark grading.

