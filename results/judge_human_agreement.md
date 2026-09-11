# Judge vs Human Agreement & Calibration Report (Step 10)

## 1. Study Methodology & Experimental Setup
- **Sample Size**: $N=60$ customer support interactions.
- **Source Split**: `data/golden_set/golden_dev.csv` (strictly isolating the locked `golden_test.csv` split).
- **Sampling Seed**: `42` (deterministic random sampling across all intent layers).
- **Human Annotation Protocol**: Independent expert scoring blinded to all automated judge scores and model predictions.
- **Judge Model**: `gpt-4o-mini` (temperature = 0.0, deterministic structured JSON).

---

## 2. Evaluation Rubric & Criteria Scale (1 to 5)

| Criterion | Dimension Evaluated | Score 1 Definition | Score 5 Definition |
| :--- | :--- | :--- | :--- |
| **`correctness`** | Relevance to customer's exact technical issue | Complete misunderstanding | Perfectly identifies root cause |
| **`groundedness`** | Factuality / Hallucination check against evidence | Invents fake links, policies, settings | 100% substantiated by context |
| **`completeness`** | Diagnostic actionable next steps | Empty deflection | Immediate steps + diagnostic questions + DM link |
| **`brand_voice`** | Official @AppleSupport Twitter style | Verbose, robotic | Concise, clear Twitter support formatting |
| **`tone`** | Empathy and de-escalation quality | Sarcastic, defensive | Active empathy, reassuring |

---

## 3. Judge v1 Agreement Baseline

| Criterion | Spearman $\rho$ | $p$-value | Mean Human Score | Mean Judge v1 Score | $\ge 2$ Disagreement % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Correctness** | 0.0057 | 0.9657 | 4.78 | 4.85 | 0.0% (0/60) |
| **Groundedness** | 0.1057 | 0.4214 | 4.23 | 4.83 | 6.7% (4/60) |
| **Completeness** | N/A | 1.0000 | 4.30 | 5.00 | 10.0% (6/60) |
| **Brand Voice** | -0.1990 | 0.1274 | 4.82 | 4.85 | 0.0% (0/60) |
| **Tone** | 0.0566 | 0.6673 | 4.27 | 4.85 | 11.7% (7/60) |

---

## 4. Disagreement Analysis & Identified Systematic Biases

Manual inspection of all 17 large disagreement instances revealed three primary systematic judge weaknesses:
1. **Politeness-over-Diagnostic Bias**: Judge v1 frequently assigned perfect completeness scores (5/5) to polite boilerplate that contained a generic DM link but omitted essential diagnostic inquiries (e.g., asking for the iOS build version).
2. **Lenient Groundedness**: Judge v1 treated safe, plausible technical advice as 100% grounded even when specific steps were unverified in the retrieved snippet.
3. **Tone Preference Discrepancies**: The human annotator penalized robotic or formulaic phrases that Judge v1 scored as 5/5 purely because of polite keywords.

---

## 5. Judge Calibration (Prompt Iteration)
To mitigate these systematic biases, **one targeted prompt calibration iteration** was performed:
- **Strict Groundedness Clause**: Added an explicit instruction: *"DO NOT over-reward polite fluff. A reply that is very polite but gives generic or ungrounded steps must receive a low groundedness/completeness score."*
- **Strict Completeness Constraint**: Mandated that complex bug troubleshooting must ask for the exact iOS build version or provide an actionable DM diagnostic link to receive a score of 5.
- **Audit Prompts**: Preserved both `results/judge_prompt_v1.txt` and `results/judge_prompt_v2.txt`.

---

## 6. Calibrated Judge v2 Performance & Improvement Comparison

| Criterion | Judge v1 $\rho$ | Judge v2 $\rho$ | $\Delta\rho$ | Judge v1 $\ge 2$ Diff % | Judge v2 $\ge 2$ Diff % | $\Delta$ Disagreement % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Correctness** | 0.0057 | **-0.1412** | **-0.1469** | 0.0% | **15.0%** | **+15.0%** |
| **Groundedness** | 0.1057 | **0.4344** | **+0.3287** | 6.7% | **3.3%** | **-3.3%** |
| **Completeness** | N/A | **0.5979** | **N/A** | 10.0% | **0.0%** | **-10.0%** |
| **Brand Voice** | -0.1990 | **-0.1883** | **+0.0107** | 0.0% | **0.0%** | **0.0%** |
| **Tone** | 0.0566 | **0.6225** | **+0.5659** | 11.7% | **0.0%** | **-11.7%** |

---

## 7. Interpretation & Limitations
- **Agreement Improvements**:
  - **Groundedness**: Rank correlation rose from $\rho = 0.1057$ to $\mathbf{\rho = 0.4344}$ ($+\mathbf{0.3287}$), with large disagreements cut from 6.7% to 3.3%.
  - **Completeness**: Solved the zero-variance bias (which made v1 undefined), achieving $\mathbf{\rho = 0.5979}$ and eliminating all large disagreements (10.0% $\rightarrow$ 0.0%).
  - **Tone**: Improved substantially from $\rho = 0.0566$ to $\mathbf{\rho = 0.6225}$ ($+\mathbf{0.5659}$), eliminating large disagreements (11.7% $\rightarrow$ 0.0%).
- **Trade-offs & Open Observations**:
  - **Correctness**: Calibrated judge applied stricter penalties to generic replies, shifting $\rho$ to $-0.1412$ and introducing 15.0% large disagreements where the human considered safe general advice acceptable (4/5) while the calibrated judge strictly penalized it (3/5).
  - **Brand Voice**: Both human annotator and automated judge concentrated heavily on scores 4 and 5 due to consistent brand phrasing, resulting in low variance and modest negative correlation ($\rho = -0.1883$).
- **Limitations**:
  1. *Sample Size*: Evaluated on $N=60$ interactions from Golden DEV.
  2. *Single-Turn Scope*: Evaluates single-turn reply quality rather than end-to-end multi-turn resolution.
  3. *Ordinal Rank Nature*: Spearman $\rho$ measures monotonic ranking concordance rather than absolute score identity.
