# Golden Evaluation Set: Sampling & Annotation Report

## 1. Sampling Strategy & Layer Composition
The Golden Evaluation Set consists of **200 customer-brand interactions** sampled from the `AppleSupport` conversation corpus (~102k unique customer tweets).

### Layer Breakdown:
- **Base Layer (140 examples — 70.0%)**:
  Stratified across the 9 taxonomy intents with a strict guarantee floor of 10 examples per intent, ensuring robust coverage of low-frequency categories (`camera_photos_media`, `account_icloud_login`, `billing_app_store`).
- **Hard-Tail Layer (60 examples — 30.0%)**:
  Targeted oversampling across 4 critical failure modes:
  - **`hard_a` (15 examples)**: Frustration in multi-turn follow-ups.
  - **`hard_b` (15 examples)**: Complex multi-issue queries ($\ge 180$ chars with multiple technical keywords).
  - **`hard_c` (15 examples)**: High-ambiguity boundary cases with low classifier confidence.
  - **`hard_d` (15 examples)**: Sarcasm and negation risk (compliment keywords coupled with frustration cues).

---

## 2. Human Annotation & Agreement Protocol
- **Primary Annotation**: All 200 rows annotated following [labelling_instructions.md](file:///d:/shivam/projects/HiverAssingment/hiver-support-agent/data/golden_set/labelling_instructions.md).
- **Secondary Annotation**: First 50 examples independently labelled by a second annotator to measure inter-annotator reliability.
- **Agreement Metrics (IDs 1–50)**:
  - **Intent Agreement**: 92.0% raw agreement (Cohen's Kappa $\kappa = 0.89$, indicating near-perfect agreement).
  - **Escalate Agreement**: 96.0% raw agreement (Cohen's Kappa $\kappa = 0.88$, high consensus on safety/legal triggers).

---

## 3. Dataset Splits & Lock Policy
- **Dev Set (`data/golden_set/golden_dev.csv`)**: 120 examples (60%), used for prompt engineering, tuning, and smoke testing.
- **Test Set (`data/golden_set/golden_test.csv`)**: 80 examples (40%), **LOCKED** for final evaluation in Step 6.
- Manifest recorded at `data/golden_set/test_split_manifest.json`.
