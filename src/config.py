"""
Global Configuration for Hiver Support Agent (AppleSupport).

Centralizes all model parameters, paths, thresholds, and operational constants.
"""

import os
from pathlib import Path

# Base Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GOLDEN_SET_DIR = DATA_DIR / "golden_set"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Brand Configuration
BRAND_HANDLE = "AppleSupport"
BRAND_HANDLE_LOWER = "applesupport"

# File Paths
THREADS_PARQUET_PATH = PROCESSED_DATA_DIR / f"{BRAND_HANDLE_LOWER}_threads.parquet"
REPLY_INDEX_NPZ_PATH = PROCESSED_DATA_DIR / "reply_index.npz"
REPLY_METADATA_PARQUET_PATH = PROCESSED_DATA_DIR / "reply_index_metadata.parquet"
SAMPLE_600_CSV_PATH = GOLDEN_SET_DIR / "sample600_labelled.csv"
GOLDEN_DEV_CSV_PATH = GOLDEN_SET_DIR / "golden_dev.csv"
GOLDEN_TEST_CSV_PATH = GOLDEN_SET_DIR / "golden_test.csv"
INTENT_TAXONOMY_MD_PATH = GOLDEN_SET_DIR / "intent_taxonomy.md"

# LLM Parameters
# Temperature 0.0 for classification and escalation ensures deterministic, reproducible decision boundaries.
# Temperature 0.3 for drafting allows subtle linguistic fluency while strictly adhering to retrieved grounded facts.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TEMPERATURE_CLASSIFY = 0.0
TEMPERATURE_ESCALATE = 0.0
TEMPERATURE_DRAFT = 0.3
LLM_MAX_RETRIES = 3
LLM_BACKOFF_FACTOR = 1.5

# Embedding & Retrieval Parameters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SIMILARITY = 0.15

# Classification & Routing Thresholds
# If LLM confidence is below 0.50, cross-check with calibrated TF-IDF model.
CONFIDENCE_ROUTING_THRESHOLD = 0.50
# If intent confidence is below 0.40, trigger model-level escalation gate.
CONFIDENCE_ESCALATION_THRESHOLD = 0.40

# Escalation-Prone Intents (High severity / sensitive financial or security domains)
ESCALATION_PRONE_INTENTS = {
    "account_icloud_login",
    "billing_app_store"
}

# Standard Support Fallback Message (Strictly required when retrieved examples do not cover inquiry)
STANDARD_FALLBACK_REPLY = "I want to make sure you get the right help — let me connect you with our team."
