# Benchmark Evaluation Reproduction Guide (Step 6)

## 1. Quick Reproduction Command
To reproduce every metric, table, figure, and judge score end-to-end on the locked golden test split:

```bash
python src/evaluate.py --out results/
```

### Optional Smoke Run:
```bash
python src/evaluate.py --out results/ --limit 10
```

---

## 2. Determinism, Seed & Cache Behavior
- **Global Random Seed**: `42` is fixed across all sampling, baseline initializations, and train/test splits.
- **LLM Temperatures**:
  - Classification: `0.0`
  - Escalation Severity Gate: `0.0`
  - Quality Judge: `0.0`
  - Reply Drafting: `0.3`
- **Disk Caching**: All LLM requests are hashed and persisted in `results/cache/`. The first full run populates the cache; subsequent runs execute instantly offline without API cost.
- **Expected Full-Run Runtime**:
  - Cold run (without cache): ~15–20 seconds
  - Warm run (cached): < 3 seconds

---

## 3. Generated Artifacts Inventory
- `results/intent_metrics.json`: Detailed classification reports for all three systems.
- `results/escalation_metrics.json`: Escalation precision, recall, F1, and error breakdown.
- `results/reply_metrics.json`: Per-criterion judge score means, standard deviations, and hallucination counts.
- `results/confusion_intent.png`: Intent classification confusion matrix plot.
- `results/confusion_escalation.png`: Escalation gate confusion matrix plot.
- `results/all_runs.parquet`: Complete per-example predictions across all systems ($N=80$).
- `results/judge_scores.parquet`: Detailed per-criterion judge scores and justification reasons.
- `results/summary_table.md`: Comprehensive markdown benchmark comparison table.
