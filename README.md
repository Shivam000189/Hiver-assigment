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

1. Setup environment
2. Get and inspect the dataset
3. Reconstruct threads and select one brand
4. Mine the intent taxonomy
5. Build the golden evaluation set
6. Build the intent classifier
7. Build grounded reply generation
8. Build the escalation gate
9. Build the evaluation harness
10. Measure judge-vs-human agreement
11. Perform failure analysis
12. Write the report
13. Make the project reproducible
14. Final submission
