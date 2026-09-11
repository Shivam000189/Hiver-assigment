"""
Retrieval Index and Grounded Reply Drafting Module for AppleSupport AI Agent.

Builds and queries an in-memory retrieval index over historically resolved support threads,
and drafts grounded replies using LLM prompt constraints.
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    THREADS_PARQUET_PATH,
    REPLY_INDEX_NPZ_PATH,
    REPLY_METADATA_PARQUET_PATH,
    RETRIEVAL_TOP_K,
    TEMPERATURE_DRAFT,
    STANDARD_FALLBACK_REPLY,
    BRAND_HANDLE
)
from src.llm import call_llm

logger = logging.getLogger("hiver.reply")

FRUSTRATION_PATTERN = r'\b(?:still|again|not working|useless|terrible|worst|refund|scam|lawsuit|sue|lawyer|worse|hate|broken|broke)\b'

class RetrievalIndex:
    """
    Vector search index over historically resolved, usable AppleSupport conversation threads.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True)
        self.index_matrix = None
        self.metadata_df = None
        self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(REPLY_INDEX_NPZ_PATH) and os.path.exists(REPLY_METADATA_PARQUET_PATH):
            try:
                npz = np.load(REPLY_INDEX_NPZ_PATH, allow_pickle=True)
                # Load metadata
                self.metadata_df = pd.read_parquet(REPLY_METADATA_PARQUET_PATH)
                self.vectorizer.vocabulary_ = npz['vocab'].item()
                self.vectorizer.idf_ = npz['idf']
                # Reconstruct transform
                self.index_matrix = self.vectorizer.transform(self.metadata_df['customer_text'])
                logger.info(f"Loaded existing retrieval index ({len(self.metadata_df):,} usable threads).")
                return
            except Exception as e:
                logger.warning(f"Error loading cached index ({e}). Rebuilding...")

        self.build_index()

    def build_index(self) -> Dict[str, int]:
        """
        Builds the retrieval index from threads parquet.
        Filters for usable resolved threads (dropping frustrated multi-turn escalations).
        """
        if not os.path.exists(THREADS_PARQUET_PATH):
            raise FileNotFoundError(f"Threads file not found at {THREADS_PARQUET_PATH}")

        df = pd.read_parquet(THREADS_PARQUET_PATH)
        total_threads = len(df)

        # Usable filter
        has_reply = df['brand_reply_text'].notna() & (df['brand_reply_text'].str.strip() != "")
        is_frustrated_followup = (
            (df['has_followup'] == True) &
            (df['followup_text'].str.contains(FRUSTRATION_PATTERN, case=False, na=False, regex=True))
        )
        
        usable_mask = has_reply & (~is_frustrated_followup)
        usable_df = df[usable_mask].copy().reset_index(drop=True)
        excluded_count = total_threads - len(usable_df)

        logger.info(f"Index Construction: {total_threads:,} total | {len(usable_df):,} usable ({len(usable_df)/total_threads*100:.1f}%) | {excluded_count:,} excluded.")

        # Fit TF-IDF on usable customer queries
        self.index_matrix = self.vectorizer.fit_transform(usable_df['customer_text'])
        self.metadata_df = usable_df[[
            'customer_tweet_id', 'customer_text', 'brand_tweet_id', 'brand_reply_text', 'response_time_minutes'
        ]].copy()

        # Save artifacts
        os.makedirs(os.path.dirname(REPLY_INDEX_NPZ_PATH), exist_ok=True)
        np.savez_compressed(
            REPLY_INDEX_NPZ_PATH,
            vocab=self.vectorizer.vocabulary_,
            idf=self.vectorizer.idf_
        )
        self.metadata_df.to_parquet(REPLY_METADATA_PARQUET_PATH, index=False)
        logger.info("Saved retrieval index and metadata successfully.")

        return {
            "total_threads": total_threads,
            "usable_threads": len(usable_df),
            "excluded_threads": excluded_count
        }

    def query(self, text: str, k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        if self.index_matrix is None or self.metadata_df is None:
            self._load_or_build()

        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.index_matrix).flatten()
        top_indices = np.argsort(sims)[-k:][::-1]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            row = self.metadata_df.iloc[idx]
            results.append({
                "tweet_id": int(row['brand_tweet_id']),
                "customer_tweet_id": int(row['customer_tweet_id']),
                "customer_text": str(row['customer_text']),
                "brand_reply_text": str(row['brand_reply_text']),
                "similarity": round(score, 4)
            })
        return results

# Singleton index
_retrieval_index = None

def get_retrieval_index() -> RetrievalIndex:
    global _retrieval_index
    if _retrieval_index is None:
        _retrieval_index = RetrievalIndex()
    return _retrieval_index

def retrieve(text: str, k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieves top-k historical brand resolution examples for a given customer query.
    """
    index = get_retrieval_index()
    return index.query(text, k=k)

def draft_reply(customer_text: str, retrieved_examples: List[Dict[str, Any]]) -> str:
    """
    Drafts a grounded agent response constrained to facts and policies from retrieved examples.
    """
    if not retrieved_examples:
        return STANDARD_FALLBACK_REPLY

    system_prompt = (
        f"You are the official customer support AI agent for @{BRAND_HANDLE}. "
        "Your task is to draft a helpful, professional, and empathetic reply to an inbound customer tweet.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. You may ONLY use troubleshooting steps, settings paths, links, and policies present in the retrieved examples.\n"
        "2. Match the brand's concise, helpful tone (include diagnostic questions and DM links as done in historical examples).\n"
        "3. If the retrieved examples do not cover the customer's problem or are irrelevant, output EXACTLY:\n"
        f"   '{STANDARD_FALLBACK_REPLY}' and nothing else.\n"
        "4. NEVER request or include passwords, credit card numbers, or OTP codes.\n"
        "5. Output the draft reply text only."
    )

    examples_text = ""
    for i, ex in enumerate(retrieved_examples):
        examples_text += (
            f"Example {i+1} (Tweet ID {ex['tweet_id']}):\n"
            f"  Customer: \"{ex['customer_text']}\"\n"
            f"  {BRAND_HANDLE} Resolved: \"{ex['brand_reply_text']}\"\n\n"
        )

    user_prompt = (
        f"Here are {len(retrieved_examples)} past similar issues and how @{BRAND_HANDLE} resolved them:\n\n"
        f"{examples_text}"
        f"NEW INBOUND CUSTOMER TWEET:\n\"{customer_text}\"\n\n"
        f"Draft the official @{BRAND_HANDLE} reply adhering to all grounding rules:"
    )

    reply = call_llm(
        prompt=user_prompt,
        temperature=TEMPERATURE_DRAFT,
        system_prompt=system_prompt
    )
    return reply.strip()
