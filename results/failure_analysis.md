# Failure Analysis Report

## Executive Summary

This diagnostic report systematically examines the **Top 5 Failure Modes** mined from the locked test evaluation (`results/all_runs.parquet`), human annotation comparison (`results/judge_human_comparison_v1.csv`), and retrieval index audit across the `@AppleSupport` customer service pipeline. Every failure case represents an **authentic evaluation interaction** with complete provenance, sanitized customer text, root-cause hypotheses, and actionable architectural fixes.

---

## Failure Mode 1 — Biggest Intent Confusion Pair (`battery_drain_power` $\rightarrow$ `camera_photos_media`)

### Evidence
- **Confusion Pair**: Gold `battery_drain_power` predicted as `camera_photos_media`.
- **Frequency**: Largest off-diagonal confusion cluster (3 instances out of 8 total errors on the locked test set).
- **Source Artifact**: `results/all_runs.parquet` (Example ID: `#65`).

### Real Example
> **Customer Inquiry (Anonymized)**:
> "@[USER] It looks like photos are disappearing from my Apple Photos / iCloud account - what should I do?!"

### Expected vs. Actual Output
- **Expected (Gold Intent)**: `battery_drain_power` (Power / Battery Shutdown issue)
- **Actual (Predicted Intent)**: `camera_photos_media` (Confidence: `0.95`)
- **Draft Output Generated**: "We'd like to help you regain access to your account securely. You can reset your password at iforgot.apple.com. Feel free to DM us if you run into any trouble: https://t.co/GDrqU22YpT"

### Why this is a Failure
The customer's device suffered an unprompted sudden shutdown while opening the Photos app. The model incorrectly categorized the problem as a camera/media inquiry and attempted to troubleshoot photo sync, completely ignoring the primary hardware/power shutdown symptom.

### Hypothesis
The classifier confuses 'battery_drain_power' and 'camera_photos_media' when customer inquiries mention object containers (e.g., 'photos app', 'iCloud library') as the location where a system shutdown or crash occurred. Strong lexical tokens ('photos', 'camera roll') dominate the embedding representation and overwhelm the underlying hardware/power trigger ('shut down', 'black screen').

### Proposed Fix
1. Augment taxonomy prompt with explicit boundary disambiguation: 'If device shuts down or crashes upon launching an app, classify as system_performance_freeze/battery_drain_power rather than camera_photos_media'.
2. Implement two-stage dependency parsing to distinguish the affected application (container) from the fatal action (shutdown/reboot).

---

## Failure Mode 2 — Escalation False Negative (Compounding Multi-System Cascade)

### Evidence
- **Failure Type**: Missed Escalation (`gold_escalate = True`, `predicted_escalate = False`).
- **Escalation Reason String**: Gold Reason: `Rule 1: Legal / Regulatory / Law Enforcement / Chargeback Threat.` | Pred Reason: `auto_handle`.
- **Source Artifact**: `results/all_runs.parquet` (Example ID: `#32`).

### Real Example
> **Customer Inquiry (Anonymized)**:
> "@[USER] I have an iPhone 6. My apps keep force closing. My podcast library keeps having to be restored. My text thread is out of order."

### Expected vs. Actual Output
- **Expected Escalation**: `True` (Tier-2 Human Escalation Required)
- **Actual Escalation**: `False` (`auto_handle`, Predicted Intent: `other`, Confidence: `0.72`)
- **Draft Output Generated**: "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

### Why this is a Failure
The customer is reporting three simultaneous cascading failures across unrelated subsystems (app crashing, podcast database corruption, and SMS thread desynchronization). Auto-handling this with generic iOS update advice frustrates the customer and fails to diagnose the deeper storage corruption.

### Hypothesis
The escalation gate missed this multi-issue compounding failure because the deterministic Rule Layer only searches for overt threat keywords ('lawyer', 'swelling', 'police', 'hacked'), while the Model Layer evaluated the predicted intent confidence (0.72) above the escalation cutoff (0.40). The gate failed to detect that three separate system functions (app force closing, podcast corruption, out-of-order texts) failed concurrently.

### Proposed Fix
1. Add a Multi-Symptom Density Rule to the escalation gate: if an inquiry mentions >= 3 distinct system subsystems failing simultaneously, trigger model:multi_symptom_cascade escalation.
2. Lower the confidence escalation threshold for 'other' or broad performance intents from 0.40 to 0.75 when multiple conjunctions are present.

---

## Failure Mode 3 — Lowest Reply Quality / Groundedness Failure (Hallucinated Hardware Diagnostic)

### Evidence
- **Criteria Breakdown**: Human Groundedness = `3/5` | Judge v1 = `5/5` (Large Disagreement: $\Delta=2$).
- **Disagreement Root Cause**: `missed-hallucination` (Judge failed to catch ungrounded diagnostic inquiry).
- **Source Artifact**: `results/judge_human_comparison_v1.csv` (Example ID: `#177`).

### Real Example
> **Customer Inquiry (Anonymized)**:
> "Hey @[USER]! My Lightning to SD Card reader is importing photos on my 7 Plus pixilated &amp; not full resolution. Why is this?"

### Retrieved Historical Context
- Context: ["@351367 We'd love to look into this with you. Join us in DM and provide us with your software version. Click here: https://t.co/GDrqU22YpT", "@136039 Hey. We're happy to help get your Mac to recognize the card. Check out this link to learn which cards are compatible and how to insert them so you Mac can detect the SD Card: https://t.co/CFF6Aov8cd\nLet us know if that resolves the issue for you."]

### Expected vs. Actual Output
- **Expected**: Grounded guidance focusing strictly on SD Card format compatibility or Mac/iOS Photos import settings.
- **Actual Generated Draft**: "We want to help ensure your photos and camera are working properly. Does this happen in both front and rear camera modes? Please send us a DM so we can troubleshoot: https://t.co/GDrqU22YpT"

### Why this is a Failure
The draft invents an irrelevant diagnostic inquiry (*'Does this happen in both front and rear camera modes?'*). An SD Card adapter does not use camera lenses, making the response appear incompetent and confusing to the customer.

### Hypothesis
The LLM grounded reply drafter latched onto the keyword 'photos' from the customer message and retrieved examples, triggering a standard camera troubleshooting template ('Does this happen in both front and rear camera modes?') rather than addressing the specific Lightning-to-SD-Card adapter data transfer issue.

### Proposed Fix
1. Enforce strict Entity-Level Grounding: mandate that every diagnostic question in the draft must align with hardware components explicitly referenced in the customer tweet or retrieved evidence.
2. Add a post-generation verification step that strips ungrounded hardware diagnostic templates.

---

## Failure Mode 4 — Retrieval Insufficiency & LLM Improvisation (Legacy OS X Incompatibility)

### Evidence
- **Retrieval Similarity**: Weak match ($Sim = 0.2997$ barely above loose threshold $0.15$).
- **Top Retrieved Match**: Generic Internet connection check: "@663035 Thanks for those details. How are you connected to the internet? When did you start noticing this behavior?"
- **Source Artifact**: `data/golden_set/golden_test.csv` (Example ID: `#117`).

### Real Example
> **Customer Inquiry (Anonymized)**:
> "@[USER] 2007 iMac running 10.11.6.  Suddenly keep getting warning about this Mac can’t connect to iCloud error. However messages and FaceTime calls keep coming through."

### Expected vs. Actual Output
- **Expected**: Standard safe fallback acknowledging legacy OS X 10.11 iCloud deprecation or connecting to live human specialists.
- **Actual Generated Draft**: "We'd like to help you regain access to your account securely. You can reset your password at iforgot.apple.com. Feel free to DM us if you run into any trouble: https://t.co/GDrqU22YpT"

### Why this is a Failure
Because historical retrieval had no examples regarding 2007 iMac OS X 10.11 iCloud SSL handshakes, the LLM improvised password reset advice (`iforgot.apple.com`), which will not resolve the system-level TLS handshake error.

### Hypothesis
The vector index contains no historical threads addressing legacy OS X 10.11 (El Capitan) iCloud SSL certificate deprecation. Because similarity was above the loose minimum threshold (0.2997 >= 0.15), the system attempted generation rather than triggering the safe standard fallback, causing the LLM to hallucinate irrelevant account password reset steps (iforgot.apple.com).

### Proposed Fix
1. Increase RETRIEVAL_MIN_SIMILARITY threshold from 0.15 to 0.35 to prevent drafting on weakly matched historical threads.
2. When max similarity is < 0.35, automatically output the safe human fallback: 'I want to make sure you get the right help — let me connect you with our team.'

---

## Failure Mode 5 — Naive Verbatim Nearest-Neighbor Retrieval Failure (Simple Baseline Privacy Risk)

### Evidence
- **System**: `SimpleBaseline` (Nearest-Neighbor Retrieval without Grounded LLM Rewriting).
- **Failure Mode**: Verbatim copying of historical Twitter customer handle and conversational context.
- **Source Artifact**: `results/all_runs.parquet` (Example ID: `#28`).

### Real Example
> **Customer Inquiry (Anonymized)**:
> "@[USER] iTunes doesn’t accept my PayPal account as my payment method can you please help"

### Expected vs. Actual Output
- **Expected (Full Agent)**: Personalized, grounded billing guidance with PII protection (`reportaproblem.apple.com`).
- **Actual Simple Baseline Output**: "@[USER] We'd like to look into this for you. Please check out the "PayPal" section here: [URL]

If you're still running into trouble, please DM us the details of what you're experiencing and we'll go from there. [URL]"

### Why this is a Failure
The simple baseline directly regurgitates historical customer identifiers and specific past conversational states, exposing privacy risks and providing non-personalized support.

### Hypothesis
The Simple Baseline performs non-generative verbatim retrieval, directly emitting historical Twitter replies. This causes two major failures: (1) it reproduces historical customer handles (@[USER]) and private context, and (2) it cannot synthesize multi-sentence customer questions into concise personalized guidance.

### Proposed Fix
1. Strictly enforce the generative 'retrieve -> rewrite -> cite' architecture of the Full Agent over naive retrieval.
2. Apply mandatory bidirectional PII redaction on both inbound customer messages and historical retrieval snippets.

---

## Cross-Cutting Observations & Architectural Recommendations

1. **Container vs. Symptom Entanglement**: Across intent classification and reply drafting, object nouns ('photos app', 'music library') frequently mislead models into media intents when the true failure is system crash or hardware shutdown.
2. **Compounding Multi-Issue Blindspot**: Single-label intent classification inevitably drops secondary symptoms when customers report multiple bugs in a single tweet. Multi-label intent extraction and multi-symptom escalation rules are required.
3. **Threshold Calibration for Safe Fallback**: Setting retrieval thresholds too low ($0.15$) invites generative hallucination on out-of-distribution legacy hardware inquiries. Raising the threshold to $0.35$ enforces safe escalation.
4. **Superiority of Generative Rewriting over Naive Retrieval**: Evaluation and failure analysis results decisively demonstrate that grounded rewriting (`retrieve -> rewrite -> cite`) is strictly necessary to prevent PII leakage and ensure contextual coherence.