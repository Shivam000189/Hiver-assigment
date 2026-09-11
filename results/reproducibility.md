# Reproducibility Test Report (Step 13)

## Environment
- **Operating System**: Windows 10 (AMD64)
- **Python Version**: 3.11.9
- **Repository**: `hiver-support-agent`
- **Execution Mode**: Offline Evaluation (No API Key Required, Cached LLM Artifacts)
- **Target Threshold**: < 15.00 minutes (900 seconds)

---

## Dataset Provenance & Artifacts
- **Reproducibility Dataset**: `data/processed/applesupport_threads_repro.parquet`
  - **Row Count**: 15,000 threads (Stratified subsample across 9 intents + all historical evaluation citations)
  - **File Size**: 2.75 MB (Committed directly into git)
  - **Full 3M Raw Kaggle Dataset (`twcs.csv`)**: **NOT Required** for evaluation or reproduction.
- **Golden Evaluation Set**: `data/golden_set/`
  - `golden_set.csv`: $N=200$ hand-labeled interactions
  - `golden_dev.csv`: $N=120$ dev interactions
  - `golden_test.csv`: $N=80$ locked test interactions
  - `sample600_labelled.csv`: $N=600$ human training annotations
  - `judge_human_scores.csv`: $N=60$ calibration interactions
- **Committed LLM Cache**: `results/cache/` (527 JSON responses, 1.23 MB)

---

## Reproduction Commands
```bash
# 1. Clone repository
git clone <repo-url>
cd hiver-support-agent

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run end-to-end evaluation
python src/evaluate.py

# 5. (Optional) Recompile evaluation report
python src/report.py
```

---

## Benchmark Timing Results

| Reproduction Stage | Measured Execution Time | Percentage of Total |
| :--- | :---: | :---: |
| **Clone & Directory Setup** | 1.83s | 0.3% |
| **Virtual Environment Creation (`.venv`)** | 19.45s | 3.4% |
| **Dependency Installation (`pip install`)** | 506.29s | 87.5% |
| **End-to-End Evaluation (`evaluate.py`)** | 47.70s | 8.2% |
| **Report Compilation (`report.py`)** | 3.54s | 0.6% |
| **TOTAL REPRODUCTION TIME** | **578.82s (9.65 min)** | **100.0%** |

**Status**: **PASS** (Completed in **9.65 minutes**, well under the 15-minute hard limit).

---

## Result Verification (Canonical vs. Fresh Clone)

| Metric Evaluated | Canonical Step 9 Metric | Fresh Clean-Clone Metric | Verification Status |
| :--- | :---: | :---: | :---: |
| **Trivial Baseline Intent Macro-F1** | `0.0529` | `0.0529` | **PASS (Exact Match)** |
| **Simple Baseline Intent Macro-F1** | `0.5794` | `0.5794` | **PASS (Exact Match)** |
| **Full Support Agent Intent Macro-F1** | `0.8976` | `0.8976` | **PASS (Exact Match)** |
| **Full Support Agent Intent Accuracy** | `90.0%` | `90.0%` | **PASS (Exact Match)** |
| **Full Support Agent Escalation Recall** | `83.33%` | `83.33%` | **PASS (Exact Match)** |
| **Full Support Agent Escalation Precision** | `38.46%` | `38.46%` | **PASS (Exact Match)** |
| **Full Support Agent Escalation F1** | `0.5263` | `0.5263` | **PASS (Exact Match)** |
| **Full Support Agent Reply Correctness** | `4.72` | `4.72` | **PASS (Exact Match)** |
| **Full Support Agent Reply Groundedness** | `4.79` | `4.79` | **PASS (Exact Match)** |
| **Full Support Agent Reply Completeness** | `5.00` | `5.00` | **PASS (Exact Match)** |

---

## Reproducibility Guarantees & Safeguards
1. **Zero External API Requirement**: Grading executes completely offline using deterministic cached LLM responses in `results/cache/`.
2. **Deterministic Retrieval**: In-memory TF-IDF index initializes from the committed 15k-thread dataset in $<0.5$ seconds.
3. **No 3M-Row Data Dependency**: Raw `twcs.csv` is completely bypassed.
4. **Locked Split Isolation**: Test split (`golden_test.csv`, $N=80$) is evaluated strictly out-of-sample with zero training leakage.
