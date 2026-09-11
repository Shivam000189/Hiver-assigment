# Benchmark Summary Table: AppleSupport AI Agent Evaluation

**Evaluation Target**: Locked Golden Test Split (`data/golden_set/golden_test.csv`, $N=80$ interactions).

| System | Intent Macro-F1 | Intent Accuracy | Escalate Precision | Escalate Recall | Escalate F1 | Correctness (1-5) | Groundedness (1-5) | Completeness (1-5) | Brand Voice (1-5) | Tone (1-5) | Hallucinations ($\le 2$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trivial Baseline** | 0.0529 | 31.2% | 0.0000 | 0.0000 | 0.0000 | 3.00 | 4.00 | 3.00 | 3.00 | 4.00 | 0/80 |
| **Simple Baseline** (Caveat: Unredacted PII) | 0.5794 | 55.0% | 1.0000 | 0.1667 | 0.2857 | 4.79 | 5.00 | 4.49 | 4.51 | 5.00 | 0/80 |
| **Full Support Agent** | **0.8976** | **90.0%** | **0.3846** | **0.8333** | **0.5263** | **5.00** | **5.00** | **5.00** | **5.00** | **5.00** | **0/80** |

---

### Judge-vs-Human Calibration Agreement (Spearman Rank Correlation $\rho$):
- **Correctness**: Round 1 $\rho = 0.2705$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.3706}$
- **Groundedness**: Round 1 $\rho = 0.2832$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.4047}$
- **Completeness**: Round 1 $\rho = 0.3020$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.4492}$
- **Brand Voice**: Round 1 $\rho = 0.3020$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.4492}$
- **Tone**: Round 1 $\rho = 0.3194$ $\rightarrow$ Round 2 (Calibrated) $\mathbf{\rho = 0.4877}$

---

### Calibrated Constants & Dev-Based Justifications:
- **Intent Classifier Routing Threshold (`CONFIDENCE_ROUTING_THRESHOLD = 0.50`)**:
  - *Dev Justification*: On `golden_dev.csv` ($N=120$), routing LLM predictions with confidence $<0.50$ to TF-IDF cross-validation increased classification accuracy from 88.3% to 93.3% while eliminating out-of-taxonomy hallucinated labels.
- **Escalation Low-Confidence Threshold (`CONFIDENCE_ESCALATION_THRESHOLD = 0.40`)**:
  - *Dev Justification*: On `golden_dev.csv`, queries where classifier confidence remained $<0.40$ were predominantly high-ambiguity edge cases. Defaulting to escalation captured 100% of ambiguous customer complaints without generating false positives on routine queries.

---

### Methodological Notes & Known Biases:
1. **Sampling Bias**: The Golden Test Set incorporates a 30% oversampled hard-tail layer (frustration signals, low-confidence boundaries, multi-issue queries). This intentionally inflates the test escalation prevalence relative to real-world ambient traffic.
2. **Simple Baseline PII Vulnerability**: The Simple Baseline directly copies unredacted historical replies, creating customer privacy risks when historical customer names/handles are present.
3. **Escalation Source Split**: In the Full Agent, **7.69%** of escalations were triggered deterministically by the Rule Layer, while **92.31%** were triggered by the Model Layer.
4. **Judge Agreement Uncertainty**: Judge correlation $\rho \approx 0.37 - 0.49$ implies $\pm 0.3$ noise on mean reply quality scores.
