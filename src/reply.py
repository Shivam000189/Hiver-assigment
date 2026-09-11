"""
Retrieval Index and Grounded Reply Drafting Module for AppleSupport AI Agent.

Architecture:
  NEW CUSTOMER MESSAGE
          ↓
     Redact PII
          ↓
     Retrieve similar resolved examples (Cosine similarity >= MIN_RETRIEVAL_SIMILARITY)
          ↓
     Top-k historical customer → brand replies
          ↓
     LLM Grounded Rewrite (Strict grounding rules, no hallucinated policies)
          ↓
     Draft + Complete Citation Trail (customer_tweet_id, brand_tweet_id)
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    THREADS_PARQUET_PATH,
    REPLY_INDEX_NPZ_PATH,
    REPLY_METADATA_PARQUET_PATH,
    PROCESSED_DATA_DIR,
    RETRIEVAL_TOP_K,
    RETRIEVAL_MIN_SIMILARITY,
    TEMPERATURE_DRAFT,
    STANDARD_FALLBACK_REPLY,
    BRAND_HANDLE
)
from src.pii import redact_pii
from src.llm import call_llm, call_llm_json

logger = logging.getLogger("hiver.reply")

# Transparent regex proxy for identifying unresolved/frustrated customer multi-turn follow-ups
FRUSTRATION_PATTERN = r'\b(?:still|again|not working|useless|terrible|worst|refund|scam|lawsuit|sue|lawyer|worse|hate|broken|broke)\b'

METADATA_JSON_PATH = PROCESSED_DATA_DIR / "reply_index_metadata.json"

class RetrievalIndex:
    """
    Vector search index over historically resolved, usable AppleSupport conversation threads.
    Maps historical customer inquiries to official brand resolutions.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000, sublinear_tf=True, stop_words='english')
        self.index_matrix = None
        self.metadata_df = None
        self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(REPLY_INDEX_NPZ_PATH) and os.path.exists(REPLY_METADATA_PARQUET_PATH):
            try:
                npz = np.load(REPLY_INDEX_NPZ_PATH, allow_pickle=True)
                self.metadata_df = pd.read_parquet(REPLY_METADATA_PARQUET_PATH)
                self.vectorizer.vocabulary_ = npz['vocab'].item()
                self.vectorizer.idf_ = npz['idf']
                if 'matrix_data' in npz:
                    self.index_matrix = sp.csr_matrix(
                        (npz['matrix_data'], npz['matrix_indices'], npz['matrix_indptr']),
                        shape=npz['matrix_shape']
                    )
                else:
                    self.index_matrix = self.vectorizer.transform(self.metadata_df['customer_text'])
                logger.info(f"Loaded existing retrieval index ({len(self.metadata_df):,} usable threads).")
                return
            except Exception as e:
                logger.warning(f"Error loading cached index ({e}). Rebuilding...")

        self.build_index()

    def build_index(self) -> Dict[str, Any]:
        """
        Builds the retrieval index from threads parquet.
        Filters for usable resolved threads (dropping frustrated multi-turn escalations).
        """
        if not os.path.exists(THREADS_PARQUET_PATH):
            raise FileNotFoundError(f"Threads file not found at {THREADS_PARQUET_PATH}")

        df = pd.read_parquet(THREADS_PARQUET_PATH)
        total_threads = len(df)

        # Usability filter:
        # 1. Must have non-null brand reply
        # 2. Exclude threads where customer returned with clear frustration signals in follow-up
        has_reply = df['brand_reply_text'].notna() & (df['brand_reply_text'].str.strip() != "")
        no_reply_count = int((~has_reply).sum())

        is_frustrated_followup = (
            (df['has_followup'] == True) &
            (df['followup_text'].str.contains(FRUSTRATION_PATTERN, case=False, na=False, regex=True))
        )
        frustrated_count = int(is_frustrated_followup.sum())

        usable_mask = has_reply & (~is_frustrated_followup)
        usable_df = df[usable_mask].copy().reset_index(drop=True)
        excluded_count = total_threads - len(usable_df)

        logger.info(f"Index Construction: {total_threads:,} total | {len(usable_df):,} usable ({len(usable_df)/total_threads*100:.1f}%) | {excluded_count:,} excluded.")

        # Fit TF-IDF on usable customer queries
        self.index_matrix = self.vectorizer.fit_transform(usable_df['customer_text'])
        self.metadata_df = usable_df[[
            'customer_tweet_id', 'customer_text', 'brand_tweet_id', 'brand_reply_text', 'response_time_minutes'
        ]].copy()

        # Save binary artifacts
        os.makedirs(os.path.dirname(REPLY_INDEX_NPZ_PATH), exist_ok=True)
        np.savez_compressed(
            REPLY_INDEX_NPZ_PATH,
            vocab=self.vectorizer.vocabulary_,
            idf=self.vectorizer.idf_,
            matrix_data=self.index_matrix.data,
            matrix_indices=self.index_matrix.indices,
            matrix_indptr=self.index_matrix.indptr,
            matrix_shape=self.index_matrix.shape
        )
        self.metadata_df.to_parquet(REPLY_METADATA_PARQUET_PATH, index=False)

        # Save human-readable retrieval metadata JSON
        metadata_info = {
            "brand": BRAND_HANDLE,
            "source_dataset": str(THREADS_PARQUET_PATH),
            "total_threads": total_threads,
            "usable_threads": len(usable_df),
            "usable_percentage": round(len(usable_df) / total_threads * 100, 2),
            "excluded_threads": excluded_count,
            "exclusion_breakdown": {
                "no_brand_reply": no_reply_count,
                "frustrated_followup": frustrated_count
            },
            "usable_definition": "brand_reply_text is non-empty AND (has_followup is False OR followup does not contain frustration signals)",
            "embedding_method": "TfidfVectorizer (ngram_range=(1,2), min_df=2, max_features=25000, sublinear_tf=True)",
            "embedding_dimension": 25000,
            "similarity_metric": "Cosine similarity",
            "default_top_k": RETRIEVAL_TOP_K,
            "min_similarity_threshold": RETRIEVAL_MIN_SIMILARITY
        }
        with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata_info, f, indent=2)

        logger.info("Saved retrieval index, metadata parquet, and metadata JSON successfully.")

        return metadata_info

    def query(
        self,
        text: str,
        k: int = RETRIEVAL_TOP_K,
        min_similarity: float = RETRIEVAL_MIN_SIMILARITY
    ) -> List[Dict[str, Any]]:
        """
        Embeds customer message, computes cosine similarity against index,
        and returns top-k historical resolutions exceeding min_similarity.
        """
        if self.index_matrix is None or self.metadata_df is None:
            self._load_or_build()

        if not text or not text.strip():
            return []

        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.index_matrix).flatten()
        top_indices = np.argsort(sims)[-k:][::-1]

        # Check maximum similarity against threshold
        max_sim = float(sims[top_indices[0]]) if len(top_indices) > 0 else 0.0
        if max_sim < min_similarity:
            logger.info(f"No sufficiently similar historical resolution found (max similarity {max_sim:.4f} < {min_similarity}).")
            return []

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < min_similarity:
                continue
            row = self.metadata_df.iloc[idx]
            results.append({
                "tweet_id": int(row['brand_tweet_id']),
                "customer_tweet_id": int(row['customer_tweet_id']),
                "brand_tweet_id": int(row['brand_tweet_id']),
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

def retrieve(
    text: str,
    k: int = RETRIEVAL_TOP_K,
    min_similarity: float = RETRIEVAL_MIN_SIMILARITY
) -> List[Dict[str, Any]]:
    """
    Retrieves top-k historical brand resolution examples for a customer inquiry.
    
    Parameters:
        text: Inbound customer text.
        k: Number of examples to retrieve.
        min_similarity: Minimum cosine similarity threshold.
        
    Returns:
        List of dicts with customer_tweet_id, brand_tweet_id, customer_text, brand_reply_text, similarity.
    """
    index = get_retrieval_index()
    return index.query(text, k=k, min_similarity=min_similarity)

def draft(
    customer_text: str,
    retrieved_examples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Grounded LLM Rewrite function (Step 7 core contract).
    
    Parameters:
        customer_text: Raw or sanitized inbound customer message.
        retrieved_examples: List of top-k retrieved historical resolution dictionaries.
        
    Returns:
        Dict containing:
            - draft (str): The drafted reply text.
            - draft_reply (str): Alias for draft.
            - grounded (bool): True if grounded in retrieved historical evidence.
            - used_example_ids (list): IDs of historical examples cited.
            - retrieved_ids (list): List of historical brand tweet IDs retrieved.
            - retrieved_customer_ids (list): List of customer tweet IDs retrieved.
            - retrieved_brand_ids (list): List of brand tweet IDs retrieved.
            - retrieved_replies (list): Text of historical brand replies retrieved.
    """
    # 1. PII Redaction
    sanitized_text, _, _ = redact_pii(customer_text)

    # 2. Extract Citation IDs & Replies
    retrieved_brand_ids = [int(ex.get('brand_tweet_id', ex.get('tweet_id', 0))) for ex in retrieved_examples]
    retrieved_customer_ids = [int(ex.get('customer_tweet_id', 0)) for ex in retrieved_examples]
    retrieved_replies = [str(ex['brand_reply_text']) for ex in retrieved_examples]

    # 3. Check for Insufficient Evidence
    if not retrieved_examples:
        logger.info("Empty retrieval evidence — returning standard fallback.")
        return {
            "draft": STANDARD_FALLBACK_REPLY,
            "draft_reply": STANDARD_FALLBACK_REPLY,
            "grounded": False,
            "used_example_ids": [],
            "retrieved_ids": [],
            "retrieved_customer_ids": [],
            "retrieved_brand_ids": [],
            "retrieved_replies": []
        }

    # 4. Construct Grounded Prompt with Strict Grounding Rules
    system_prompt = (
        f"You are the official customer support AI agent for @{BRAND_HANDLE}.\n"
        "Your task is to draft a helpful, empathetic, and professional reply grounded in historical resolution examples.\n\n"
        "STRICT GROUNDING RULES:\n"
        "Rule 1 — Historical grounding: Use the retrieved examples as the sole source of truth for support policies, diagnostic steps, and resolutions.\n"
        "Rule 2 — No invented policies: Do NOT invent refund policies, fees, timelines, eligibility rules, account procedures, guarantees, or compensation.\n"
        "Rule 3 — Adapt to current message: Address the NEW customer's specific technical issue; do not simply copy a historical response verbatim.\n"
        "Rule 4 — Preserve facts: Preserve exact settings paths (e.g. Settings > General > Keyboard), URLs (e.g. reportaproblem.apple.com, iforgot.apple.com), and DM links.\n"
        "Rule 5 — Never combine unrelated policies: Do not merge facts from conflicting or unrelated examples.\n"
        "Rule 6 — If evidence is insufficient: If retrieved examples do not provide enough information, return the safe fallback.\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "draft_reply": "<your drafted response>",\n'
        '  "grounded": true,\n'
        '  "used_example_ids": ["<tweet_id_1>", "<tweet_id_2>"]\n'
        "}"
    )

    examples_text = ""
    for i, ex in enumerate(retrieved_examples):
        c_text_clean, _, _ = redact_pii(ex['customer_text'])
        b_text_clean, _, _ = redact_pii(ex['brand_reply_text'])
        examples_text += (
            f"Example {i+1} (Brand Tweet ID: {ex.get('brand_tweet_id', ex.get('tweet_id'))}, Customer Tweet ID: {ex.get('customer_tweet_id')}):\n"
            f"  Customer: \"{c_text_clean}\"\n"
            f"  {BRAND_HANDLE} Resolution: \"{b_text_clean}\"\n\n"
        )

    user_prompt = (
        f"HISTORICAL RESOLVED EXAMPLES:\n\n"
        f"{examples_text}"
        f"NEW INBOUND CUSTOMER TWEET:\n\"{sanitized_text}\"\n\n"
        f"Produce the grounded @{BRAND_HANDLE} reply adhering strictly to all 6 grounding rules:"
    )

    try:
        res = call_llm_json(
            prompt=user_prompt,
            temperature=TEMPERATURE_DRAFT,
            system_prompt=system_prompt,
            required_fields=["draft_reply"]
        )
        draft_text = str(res.get("draft_reply", STANDARD_FALLBACK_REPLY)).strip()
        is_grounded = bool(res.get("grounded", True))
        used_ids = res.get("used_example_ids", [str(tid) for tid in retrieved_brand_ids])
    except Exception as e:
        logger.warning(f"JSON parsing error during reply drafting: {e}. Falling back to plain text generation.")
        raw_text = call_llm(
            prompt=user_prompt,
            temperature=TEMPERATURE_DRAFT,
            system_prompt=system_prompt
        )
        draft_text = raw_text.strip()
        is_grounded = True
        used_ids = [str(tid) for tid in retrieved_brand_ids]

    return {
        "draft": draft_text,
        "draft_reply": draft_text,
        "grounded": is_grounded,
        "used_example_ids": used_ids,
        "retrieved_ids": retrieved_brand_ids,
        "retrieved_customer_ids": retrieved_customer_ids,
        "retrieved_brand_ids": retrieved_brand_ids,
        "retrieved_replies": retrieved_replies
    }

def draft_reply(customer_text: str, retrieved_examples: List[Dict[str, Any]]) -> str:
    """
    Backward-compatible string helper for draft().
    """
    res = draft(customer_text, retrieved_examples)
    return res["draft_reply"]
