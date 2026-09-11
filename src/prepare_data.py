"""
Dataset Preparation and Thread Reconstruction Module for Hiver Support Agent.

Source Dataset:
  Customer Support on Twitter (Kaggle: thoughtvector/customer-support-on-twitter)
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Reconstructs conversational threads between customers and brand support handles,
identifying inbound inquiries, brand resolutions, and customer follow-up satisfaction signals.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import (
    PROCESSED_DATA_DIR,
    BRAND_HANDLE,
    BRAND_HANDLE_LOWER,
    THREADS_PARQUET_PATH,
    REPRO_THREADS_PARQUET_PATH
)

logger = logging.getLogger("hiver.prepare_data")

def load_processed_threads(use_repro: bool = True) -> pd.DataFrame:
    """
    Loads processed threads dataset. Defaults to committed reproducibility dataset.
    """
    target_path = REPRO_THREADS_PARQUET_PATH if use_repro and REPRO_THREADS_PARQUET_PATH.exists() else THREADS_PARQUET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Processed threads parquet not found at {target_path}")
    logger.info(f"Loading processed threads from {target_path}")
    return pd.read_parquet(target_path)

def filter_usable_threads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for usable, non-frustrated resolved threads suitable for retrieval grounding.
    Criteria:
      1. Non-empty brand reply.
      2. If customer followed up (has_followup == True), excludes threads where customer
         expressed explicit frustration/dissatisfaction.
    """
    frustration_regex = r'\b(?:still|again|not working|useless|terrible|worst|refund|scam|lawsuit|sue|lawyer|worse|hate|broken|broke)\b'
    has_reply = df['brand_reply_text'].notna() & (df['brand_reply_text'].str.strip() != "")
    is_frustrated = (
        (df['has_followup'] == True) &
        (df['followup_text'].notna()) &
        (df['followup_text'].str.contains(frustration_regex, case=False, regex=True))
    )
    usable_mask = has_reply & (~is_frustrated)
    return df[usable_mask].copy()

def get_dataset_summary(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Generates summary statistics of the processed thread dataset.
    """
    if df is None:
        df = load_processed_threads()
    usable_df = filter_usable_threads(df)
    return {
        "total_threads": len(df),
        "usable_threads": len(usable_df),
        "usable_percentage": round(len(usable_df) / max(len(df), 1) * 100, 2),
        "columns": df.columns.tolist(),
        "has_followup_count": int(df['has_followup'].sum()) if 'has_followup' in df.columns else 0
    }

if __name__ == "__main__":
    summary = get_dataset_summary()
    print("Dataset Summary:")
    print(json.dumps(summary, indent=2))
