# Human Judge Scoring Instructions (Reply Quality)

## 1. Evaluation Goal
Score whether the candidate AI agent draft is a high-quality, helpful, and safe customer support reply for @AppleSupport.

## 2. Independence & Blinding
- Evaluate each draft independently.
- Do NOT attempt to guess what the automated LLM judge would score.
- Base your score strictly on the provided customer message, retrieved historical evidence, and official Apple Support standards.

## 3. Scoring Scale (1 to 5)
Every criterion must be scored as an integer from **1 to 5**:
- **1 (Very Poor)**: Unacceptable; severe errors, hallucinations, rudeness, or total irrelevance.
- **2 (Poor)**: Substantial flaws; partially ungrounded, unhelpful, or robotic.
- **3 (Acceptable)**: Safe and passable, but lacks diagnostic precision or active empathy.
- **4 (Good)**: High quality; addresses the core issue accurately with clear guidance.
- **5 (Excellent)**: Exemplary; perfectly grounded, comprehensive, empathetic, and actionable.

---

## 4. The 5 Evaluation Criteria

### A. Correctness & Relevance (1–5)
Does the reply address the customer's actual technical problem?
- **1**: Completely misunderstands the issue or provides irrelevant advice.
- **3**: Partially relevant but misses a secondary symptom or gives generic advice.
- **5**: Accurately targets the exact root cause / technical issue described.

### B. Groundedness / Hallucination Check (1–5)
**CRITICAL**: Are all facts, URLs, settings paths, and diagnostic steps substantiated by the retrieved historical examples?
- **1**: Severe hallucination (invents fake refund amounts, fabricated URLs, fake iOS settings, or policy guarantees).
- **3**: Plausible general technical advice, but contains minor unverified assumptions.
- **5**: 100% grounded; every setting path and link is verified in retrieved context.
*Note*: A polite reply that contains unverified claims MUST receive a low score (1 or 2).

### C. Completeness & Diagnostic Clarity (1–5)
Does the reply provide actionable next steps, diagnostic questions, and appropriate contact channels (DM links)?
- **1**: Empty deflection without troubleshooting steps.
- **3**: Provides an initial step but leaves the customer stranded without next steps.
- **5**: Comprehensive; provides immediate troubleshooting step, diagnostic question (e.g. iOS build version), and DM link.

### D. Brand Voice (1–5)
Does the reply match the concise, direct, and professional style of official @AppleSupport Twitter replies?
- **1**: Overly verbose, robotic, or unnatural for social support.
- **3**: Acceptable phrasing but lacks standard Apple Support formatting.
- **5**: Perfect match for official concise brand communication.

### E. Tone & Empathy (1–5)
Is the tone empathetic, reassuring, professional, and de-escalating?
- **1**: Sarcastic, cold, dismissive, or defensive.
- **3**: Neutral and polite, but formulaic.
- **5**: Genuinely empathetic, supportive, and customer-focused.

---

## 5. Output Format
Record all scores in `judge_human_scores.csv`. All 5 score columns must contain integer values from `1` to `5`.
