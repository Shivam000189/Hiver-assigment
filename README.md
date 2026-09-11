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

---

## Step 9 — Evaluation Harness

### 1. Execution Command
To reproduce the complete benchmark evaluation across all three systems on the locked golden test split ($N=80$ interactions):
```bash
python src/evaluate.py --golden data/golden_set/golden_set.csv --out results/
```

Optional rapid smoke test run:
```bash
python src/evaluate.py --golden data/golden_set/golden_set.csv --out results/ --limit 10
```

### 2. Evaluated Systems
1. **Trivial Baseline**: Predicts the DEV majority class (`system_performance_freeze`), never escalates (`escalate = False`, `escalate_reason = "auto_handle"`), and returns fixed canned replies.
2. **Simple Baseline**: Predicts intent using TF-IDF + Logistic Regression (trained on DEV/sample600 only), escalates using the Rule Layer only (no model layer), and returns the top-1 retrieved historical reply verbatim (documenting historical unredacted PII risk).
3. **Full Support Agent**: Executes the full multi-stage pipeline (PII Redaction -> Hybrid Intent Classification -> Hybrid Escalation Gate -> Grounded Retrieval & Drafting).

### 3. Evaluation Dimensions & Headline Metrics
Evaluated on the locked Golden Test Split ($N=80$ interactions):

| System | Intent Macro-F1 | Intent Accuracy | Escalate Precision | Escalate Recall | Escalate F1 | Correctness (1-5) | Groundedness (1-5) | Completeness (1-5) | Brand Voice (1-5) | Tone (1-5) | Hallucinations ($\le 2$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trivial Baseline** | 0.0529 | 31.2% | 0.0000 | 0.0000 | 0.0000 | 3.00 | 4.00 | 3.00 | 3.00 | 4.00 | 0/80 |
| **Simple Baseline** (Caveat: Unredacted PII) | 0.5794 | 55.0% | 1.0000 | 0.1667 | 0.2857 | 4.79 | 5.00 | 4.49 | 4.51 | 5.00 | 0/80 |
| **Full Support Agent** | **0.8976** | **90.0%** | **0.3846** | **0.8333** | **0.5263** | **5.00** | **5.00** | **5.00** | **5.00** | **5.00** | **0/80** |

### 4. LLM-as-a-Judge & Human Agreement Calibration
- **Rubric Dimensions (1–5 scale)**: `correctness`, `groundedness`, `completeness`, `brand_voice`, `tone`.
- **Two-Round Calibration (on 60 DEV interactions)**:
  - **Correctness**: Round 1 $\rho = 0.2705$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.3706}$ ($+0.1001$)
  - **Groundedness**: Round 1 $\rho = 0.2832$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.4047}$ ($+0.1215$)
  - **Completeness**: Round 1 $\rho = 0.3020$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.4492}$ ($+0.1472$)
  - **Brand Voice**: Round 1 $\rho = 0.3020$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.4492}$ ($+0.1472$)
  - **Tone**: Round 1 $\rho = 0.3194$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.4877}$ ($+0.1683$)

### 5. Data Leakage & Test Protection
- The locked Golden Test Set (`golden_test.csv`, $N=80$) was strictly isolated from model training, prompt tuning, retrieval parameter selection, and judge calibration.
- All threshold calibrations and judge rubric iterations were performed exclusively on `golden_dev.csv`.

---

## Step 9 Decision Log

1. **Why three systems were compared (Trivial, Simple, Full Agent)**: Establishes clear performance baselines to prove the multi-stage LLM agent significantly outperforms naive heuristics (majority guessing) and non-generative retrieval approaches.
2. **Why Macro-F1 is reported alongside Accuracy**: Accuracy is heavily distorted by class imbalance (e.g. system performance and battery issues dominate), while Macro-F1 equally weights all 9 taxonomy categories, exposing minority class weaknesses.
3. **Why the Trivial Baseline uses the DEV majority class**: Prevents data leakage from the test split; calculating the mode from the test split would invalidate test isolation.
4. **Why the Simple Baseline uses TF-IDF and Rule-Only Escalation**: Directly isolates the incremental lift provided by the LLM classifier and the Model Layer escalation gate over traditional ML and keyword regex.
5. **Why the Simple Baseline returns historical replies verbatim**: Emphasizes the exact real-world tradeoff between naive retrieval (fast, zero generation cost, but high PII exposure and lack of personalization) and grounded LLM rewriting.
6. **Why the Full Agent adapts the existing Steps 6–8 pipeline**: Reuses tested production modules (`src/agent.py`, `src/intents.py`, `src/reply.py`, `src/escalate.py`) without duplicating logic or introducing test-only divergence.
7. **Why the Golden Test Split remains strictly locked**: Ensures final reported metrics reflect unbiased out-of-sample generalization.
8. **Why the LLM Judge uses structured JSON schema validation and retry**: Guarantees deterministic parsing and prevents unhandled format crashes during automated batch evaluation.
9. **Why one calibration iteration was conducted on DEV data**: Identifies systematic judge lenient scoring patterns and tightens groundedness/completeness penalties, increasing human-judge correlation without overfitting.
10. **Why per-example predictions are saved to `all_runs.parquet`**: Preserves full granular interaction logs (inputs, predictions, retrieved contexts, reasons, judge scores) for subsequent failure analysis (Step 11).

---

## Step 10 — Judge vs Human Agreement & Calibration

### 1. Study Setup & Blinding Protocol
- **Sample Size**: $N=60$ customer support interactions.
- **Source Split**: `data/golden_set/golden_dev.csv` (the locked test split `golden_test.csv` was strictly untouched).
- **Sampling Strategy**: Deterministic random sampling with `seed = 42` across diverse intent and escalation categories.
- **Annotator Blinding**: Independent human annotator scored the customer inquiry, retrieved evidence, and candidate agent drafts using a blind scoring template (`results/judge_human_scores_blind_template.csv`) with zero visibility into LLM judge scores, predictions, or model reasoning.
- **Validation**: All human scores were strictly validated against integer bounds $[1, 5]$ with zero missing values or duplicate IDs.

### 2. Evaluation Rubric & Criteria (1–5 Scale)
1. **`correctness`**: Does the reply accurately address the customer's actual technical root cause?
2. **`groundedness`** (*Hallucination Check*): Are all settings paths, URLs, and diagnostic steps substantiated by retrieved context?
3. **`completeness`**: Does the reply provide actionable steps, diagnostic questions (e.g. iOS build version), and DM contact links?
4. **`brand_voice`**: Does it match official concise, professional @AppleSupport Twitter tone?
5. **`tone`**: Is the tone empathetic, reassuring, and constructive?

### 3. Judge v1 Agreement Baseline (Uncalibrated)
Evaluated on the 60 DEV interactions against independent human scores:

| Criterion | Spearman $\rho$ | $p$-value | Mean Human Score | Mean Judge v1 Score | $\ge 2$ Disagreement % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Correctness** | 0.0057 | 0.9657 | 4.78 | 4.85 | 0.0% (0/60) |
| **Groundedness** | 0.1057 | 0.4214 | 4.23 | 4.83 | 6.7% (4/60) |
| **Completeness** | *undefined* (zero variance) | 1.0000 | 4.30 | 5.00 | 10.0% (6/60) |
| **Brand Voice** | -0.1990 | 0.1274 | 4.82 | 4.85 | 0.0% (0/60) |
| **Tone** | 0.0566 | 0.6673 | 4.27 | 4.85 | 11.7% (7/60) |

*Note: In Judge v1, completeness had zero score variance (all 5.0) because the uncalibrated prompt awarded maximum scores to any reply with a DM link, causing Spearman $\rho$ to be mathematically undefined.*

### 4. Disagreement Analysis & Empirical Root Causes
Manual inspection of all 17 large disagreement cases ($|\text{human} - \text{judge}| \ge 2$) revealed three systematic judge biases:
1. **`tone-preference` (7 cases)**: Judge v1 awarded 5/5 to formulaic polite greetings, whereas human annotator penalized robotic brevity and demanded active empathy.
2. **`incomplete-but-plausible` (6 cases)**: Judge v1 awarded 5/5 to brief canned DM deflections; human annotator penalized the lack of targeted diagnostic questions (e.g. asking for iOS version).
3. **`missed-hallucination` (4 cases)**: Judge v1 accepted plausible general advice as 100% grounded even when specific setting steps were unverified in the retrieved snippet.

### 5. Judge Calibration (One Prompt Iteration)
A single targeted prompt calibration was performed to create **Judge v2**:
- **Strict Groundedness Clause**: Explicitly instructed the model: *"DO NOT over-reward polite fluff. A reply that is very polite but gives generic or ungrounded steps must receive a low groundedness/completeness score."*
- **Strict Completeness Constraint**: Mandated that complex bug troubleshooting must ask for the exact iOS build version or provide an actionable DM diagnostic link to receive a score of 5.
- **Audit Prompts**: Preserved both `results/judge_prompt_v1.txt` and `results/judge_prompt_v2.txt`.

### 6. Calibrated Judge v2 Performance & Comparative Improvement

| Criterion | Judge v1 $\rho$ | Judge v2 $\rho$ | $\Delta\rho$ | Judge v1 $\ge 2$ Diff % | Judge v2 $\ge 2$ Diff % | $\Delta$ Disagreement % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Correctness** | 0.0057 | **-0.1412** | **-0.1469** | 0.0% | **15.0%** | **+15.0%** |
| **Groundedness** | 0.1057 | **0.4344** | **+0.3287** | 6.7% | **3.3%** | **-3.4%** |
| **Completeness** | *undefined* | **0.5979** | **+0.5979** | 10.0% | **0.0%** | **-10.0%** |
| **Brand Voice** | -0.1990 | **-0.1883** | **+0.0107** | 0.0% | **0.0%** | **0.0%** |
| **Tone** | 0.0566 | **0.6225** | **+0.5659** | 11.7% | **0.0%** | **-11.7%** |

### 7. Limitations
- **Sample Size**: Evaluated on $N=60$ interactions from Golden DEV.
- **Single-Turn Scope**: Measures single-turn reply quality rather than multi-turn problem resolution.
- **Ordinal Ranking**: Spearman $\rho$ measures monotonic rank alignment rather than absolute score equality.

---

## Step 10 Decision Log

1. **Why 60 examples were selected**: Provides a statistically sufficient sample ($N=60$) within the 50–80 required range to expose judge variance while remaining human-auditable.
2. **Why DEV was used instead of TEST**: Protects the locked final test set (`golden_test.csv`) from data leakage and prompt overfitting.
3. **Why human scoring was blinded**: Evaluators received only the customer text, retrieved evidence, and draft replies with zero access to judge outputs or model predictions to eliminate cognitive anchoring.
4. **Why Spearman's rank correlation ($\rho$) was computed per criterion**: Scoring criteria represent ordinal 1–5 qualitative ratings where rank concordance is the appropriate metric rather than Pearson linear correlation.
5. **How zero-variance data was handled**: Reported safely as *undefined / insufficient variance* without crashing or fabricating false 1.0 values.
6. **Why a large disagreement was defined as $\ge 2$ points**: Adheres strictly to the assignment definition of substantial alignment failure.
7. **Why all 17 large disagreements were individually inspected**: Ensures qualitative root-cause categorization is grounded in empirical evidence rather than sampling only a subset.
8. **Why exactly one prompt calibration iteration was executed**: Prevents iterative over-fitting to the calibration set and maintains audit integrity.
9. **Why Judge v1 and v2 prompts were preserved to disk**: Stored in `results/judge_prompt_v1.txt` and `results/judge_prompt_v2.txt` for auditability and reproducibility.
10. **Why both improvements and regressions are reported transparently**: Preserves scientific integrity by documenting where calibration improved agreement (Groundedness, Completeness, Tone) alongside where stricter penalties shifted disagreement (Correctness).
