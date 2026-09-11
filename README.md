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

