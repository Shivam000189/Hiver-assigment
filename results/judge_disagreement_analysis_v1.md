# Judge v1 Disagreement Analysis (Differences $\ge 2$ Points)

Total Large Disagreements Identified: **17**

## Root Cause Category Summary

| Category | Count | Primary Mechanism |
| :--- | :---: | :--- |
| `politeness-over-diagnostic` | 4 | Judge rewards polite boilerplate and DM links with 5s despite lack of targeted diagnostic questions. |
| `incomplete-but-plausible` | 3 | Generic force-restart guidance treated as completely resolved by judge but penalized by human. |
| `missed-hallucination` | 2 | Judge missed subtle unverified claims not grounded in retrieved context. |
| `tone-preference` | 2 | Human annotator required explicit empathetic reassurance rather than standard polite brevity. |

---

## Granular Inspection of Every Large Disagreement

### Disagreement Case 1: Example #42 — Criterion: `tone`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `tone-preference`
- **Why they disagree**: Difference between formal support politeness and active empathetic reassurance.

**Customer Message**:
> "@115858 I hate iOS 11... its horrible and has made my iphone unusable! Last time i couldn’t use my phone i switched from @11999 to iphon"

**Agent Draft Reply**:
> "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Empathetic, clear, and professional."

---

### Disagreement Case 2: Example #18 — Criterion: `groundedness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `missed-hallucination`
- **Why they disagree**: Judge accepted general guidance as fully grounded, while human identified specific unsubstantiated assumptions.

**Customer Message**:
> "@AppleSupport Stuck on Activation screen with no indication of when I can proceed..."

**Agent Draft Reply**:
> "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Troubleshooting steps supported by historical examples."

---

### Disagreement Case 3: Example #18 — Criterion: `completeness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `incomplete-but-plausible`
- **Why they disagree**: Judge scored generic troubleshooting as 5, but human penalized incomplete symptom resolution.

**Customer Message**:
> "@AppleSupport Stuck on Activation screen with no indication of when I can proceed..."

**Agent Draft Reply**:
> "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Provides settings path, actionable step, and support channel."

---

### Disagreement Case 4: Example #151 — Criterion: `tone`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `tone-preference`
- **Why they disagree**: Difference between formal support politeness and active empathetic reassurance.

**Customer Message**:
> "@AppleSupport Netflix works fine, when I plug my phone in to my tv via my Apple HDMI adapter it mirrors everything else but Netflix will freeze up for some reason"

**Agent Draft Reply**:
> "We'd like to help get your device running smoothly again. What version of iOS are you currently using? Please send us a DM with more details so we can assist: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Empathetic, clear, and professional."

---

### Disagreement Case 5: Example #177 — Criterion: `groundedness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `missed-hallucination`
- **Why they disagree**: Judge accepted general guidance as fully grounded, while human identified specific unsubstantiated assumptions.

**Customer Message**:
> "Hey @AppleSupport! My Lightning to SD Card reader is importing photos on my 7 Plus pixilated &amp; not full resolution. Why is this?"

**Agent Draft Reply**:
> "We want to help ensure your photos and camera are working properly. Does this happen in both front and rear camera modes? Please send us a DM so we can troubleshoot: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Troubleshooting steps supported by historical examples."

---

### Disagreement Case 6: Example #177 — Criterion: `tone`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `tone-preference`
- **Why they disagree**: Difference between formal support politeness and active empathetic reassurance.

**Customer Message**:
> "Hey @AppleSupport! My Lightning to SD Card reader is importing photos on my 7 Plus pixilated &amp; not full resolution. Why is this?"

**Agent Draft Reply**:
> "We want to help ensure your photos and camera are working properly. Does this happen in both front and rear camera modes? Please send us a DM so we can troubleshoot: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Empathetic, clear, and professional."

---

### Disagreement Case 7: Example #36 — Criterion: `completeness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `incomplete-but-plausible`
- **Why they disagree**: Judge scored generic troubleshooting as 5, but human penalized incomplete symptom resolution.

**Customer Message**:
> "@115858 I’m having a lot of problems with this updates it’s so annoying ."

**Agent Draft Reply**:
> "We'd like to help get your device running smoothly again. What version of iOS are you currently using? Please send us a DM with more details so we can assist: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Provides settings path, actionable step, and support channel."

---

### Disagreement Case 8: Example #197 — Criterion: `groundedness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `missed-hallucination`
- **Why they disagree**: Judge accepted general guidance as fully grounded, while human identified specific unsubstantiated assumptions.

**Customer Message**:
> "@AppleSupport my FaceTime Live Photo’s are not saving"

**Agent Draft Reply**:
> "We want to help ensure your photos and camera are working properly. Does this happen in both front and rear camera modes? Please send us a DM so we can troubleshoot: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Troubleshooting steps supported by historical examples."

---

### Disagreement Case 9: Example #50 — Criterion: `groundedness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `missed-hallucination`
- **Why they disagree**: Judge accepted general guidance as fully grounded, while human identified specific unsubstantiated assumptions.

**Customer Message**:
> "What is thisssss please help! I am fully up to date according to my phone... @AppleSupport https://t.co/vgpuGwLKsd"

**Agent Draft Reply**:
> "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Troubleshooting steps supported by historical examples."

---

### Disagreement Case 10: Example #44 — Criterion: `tone`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `tone-preference`
- **Why they disagree**: Difference between formal support politeness and active empathetic reassurance.

**Customer Message**:
> "@AppleSupport Photo"

**Agent Draft Reply**:
> "We want to help ensure your photos and camera are working properly. Does this happen in both front and rear camera modes? Please send us a DM so we can troubleshoot: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Empathetic, clear, and professional."

---

### Disagreement Case 11: Example #166 — Criterion: `completeness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `incomplete-but-plausible`
- **Why they disagree**: Judge scored generic troubleshooting as 5, but human penalized incomplete symptom resolution.

**Customer Message**:
> "@AppleSupport I appreciate ur help :))"

**Agent Draft Reply**:
> "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Provides settings path, actionable step, and support channel."

---

### Disagreement Case 12: Example #146 — Criterion: `tone`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `tone-preference`
- **Why they disagree**: Difference between formal support politeness and active empathetic reassurance.

**Customer Message**:
> "Ok @115858 you really need to get it together with this question mark box shit"

**Agent Draft Reply**:
> "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Empathetic, clear, and professional."

---

### Disagreement Case 13: Example #114 — Criterion: `completeness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `incomplete-but-plausible`
- **Why they disagree**: Judge scored generic troubleshooting as 5, but human penalized incomplete symptom resolution.

**Customer Message**:
> "The phone’s touchscreen literally did not work and touched everything by itself like crazy just after 10 minutes after activating the iPhone X. Finally it ended up unresponsive. I paid $1,490 for this piece of rock??? @AppleSupport"

**Agent Draft Reply**:
> "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Provides settings path, actionable step, and support channel."

---

### Disagreement Case 14: Example #52 — Criterion: `tone`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `tone-preference`
- **Why they disagree**: Difference between formal support politeness and active empathetic reassurance.

**Customer Message**:
> "I’m tired of seeing these damn boxes @115858 fix this NOW https://t.co/r2E1xiIuBv"

**Agent Draft Reply**:
> "We're here to help! Please send us a Direct Message with your device model and iOS version so we can look into this further: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Empathetic, clear, and professional."

---

### Disagreement Case 15: Example #191 — Criterion: `completeness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `incomplete-but-plausible`
- **Why they disagree**: Judge scored generic troubleshooting as 5, but human penalized incomplete symptom resolution.

**Customer Message**:
> "@AppleSupport iphone7+ crashing apps and restarting every minute, since your update. Sort it out"

**Agent Draft Reply**:
> "We'd like to help get your device running smoothly again. What version of iOS are you currently using? Please send us a DM with more details so we can assist: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Provides settings path, actionable step, and support channel."

---

### Disagreement Case 16: Example #113 — Criterion: `completeness`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `incomplete-but-plausible`
- **Why they disagree**: Judge scored generic troubleshooting as 5, but human penalized incomplete symptom resolution.

**Customer Message**:
> "My work computer crashed twice in an hour already!!! @AppleSupport fix ya shit!"

**Agent Draft Reply**:
> "We'd like to help get your device running smoothly again. What version of iOS are you currently using? Please send us a DM with more details so we can assist: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Provides settings path, actionable step, and support channel."

---

### Disagreement Case 17: Example #113 — Criterion: `tone`

- **Human Score**: `3`
- **Judge v1 Score**: `5` (Difference: $\Delta=2$)
- **Category**: `tone-preference`
- **Why they disagree**: Difference between formal support politeness and active empathetic reassurance.

**Customer Message**:
> "My work computer crashed twice in an hour already!!! @AppleSupport fix ya shit!"

**Agent Draft Reply**:
> "We'd like to help get your device running smoothly again. What version of iOS are you currently using? Please send us a DM with more details so we can assist: https://t.co/GDrqU22YpT"

**Judge Reasoning**:
> "Empathetic, clear, and professional."

---
