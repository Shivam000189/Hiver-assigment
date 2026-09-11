"""
Golden Evaluation Set Sampling & Pipeline Module.

Implements reproducible stratified base + hard-tail sampling for the Golden Evaluation Set.
"""

import os
import re
import string
import pandas as pd
import numpy as np

FRUSTRATION_KEYWORDS = [
    'still', 'again', 'not working', 'useless', 'terrible', 'worst',
    'refund', 'scam', 'lawsuit', 'sue', 'lawyer', 'worse'
]

INTENT_KEYWORD_SETS = {
    'keyboard_autocorrect_bug': ['autocorrect', 'keyboard', 'typing', 'letter i', 'i glitch', 'predictive text', 'spacebar'],
    'system_performance_freeze': ['freeze', 'freezing', 'frozen', 'crashes', 'crashing', 'lag', 'slow', 'unresponsive', 'restart', 'reboot'],
    'battery_drain_power': ['battery', 'drain', 'draining', 'charge', 'charging', 'overheating', 'dies'],
    'audio_music_playback': ['music', 'apple music', 'playlist', 'headphone', 'earpods', 'airpods', 'song'],
    'connectivity_network': ['wifi', 'wi-fi', 'bluetooth', 'pair', 'pairing', 'airdrop', 'cellular', 'lte', 'service'],
    'camera_photos_media': ['camera', 'photos', 'flashlight', 'shutter', 'black screen', 'blurry'],
    'account_icloud_login': ['apple id', 'icloud', 'password', '2fa', 'verification code', 'locked out', 'login'],
    'billing_app_store': ['app store', 'charged', 'subscription', 'refund', 'receipt', 'purchase', 'billing']
}

def normalize_text_for_dedup(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = t.translate(str.maketrans('', '', string.punctuation))
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def sample_golden_set(threads_df: pd.DataFrame, pseudo_df: pd.DataFrame, n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Sample a 200-example Golden Evaluation Set using a 140 base layer + 60 hard-tail layer.
    
    Parameters:
        threads_df: DataFrame with thread metadata (must contain customer_tweet_id, customer_text, brand_reply_text, has_followup, followup_text).
        pseudo_df: DataFrame with pseudo_intent and pseudo_confidence.
        n: Total samples (default 200: 140 base + 60 hard-tail).
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame of 200 unlabelled golden examples with sample_layer and metadata.
    """
    np.random.seed(seed)
    
    # Merge thread metadata with pseudo labels
    merged = threads_df.merge(
        pseudo_df[['customer_tweet_id', 'pseudo_intent', 'pseudo_confidence']],
        on='customer_tweet_id',
        how='inner'
    )
    
    # Deduplicate normalized customer text
    merged['norm_text'] = merged['customer_text'].apply(normalize_text_for_dedup)
    merged = merged.drop_duplicates(subset=['norm_text']).reset_index(drop=True)
    
    selected_indices = set()
    layer_assignments = {}
    
    # --- Hard-tail Layer (60 samples: 15 per rule) ---
    n_hard_per_rule = 15
    
    # Rule A: Frustration in follow-up
    rule_a_pattern = r'\b(?:' + '|'.join(FRUSTRATION_KEYWORDS) + r')\b'
    cand_a = merged[
        (merged['has_followup'] == True) &
        (merged['followup_text'].str.contains(rule_a_pattern, case=False, na=False, regex=True))
    ]
    cand_a_avail = [idx for idx in cand_a.index if idx not in selected_indices]
    pick_a = np.random.choice(cand_a_avail, size=min(n_hard_per_rule, len(cand_a_avail)), replace=False)
    for idx in pick_a:
        selected_indices.add(idx)
        layer_assignments[idx] = 'hard_a'
        
    # Rule B: Multi-issue candidates (len >= 180 and 2+ intent keyword sets)
    def count_intent_keywords(text):
        t = str(text).lower()
        matched = 0
        for intent, kw_list in INTENT_KEYWORD_SETS.items():
            if any(kw in t for kw in kw_list):
                matched += 1
        return matched

    merged['kw_matches'] = merged['customer_text'].apply(count_intent_keywords)
    cand_b = merged[(merged['customer_text'].str.len() >= 180) & (merged['kw_matches'] >= 2)]
    cand_b_avail = [idx for idx in cand_b.index if idx not in selected_indices]
    pick_b = np.random.choice(cand_b_avail, size=min(n_hard_per_rule, len(cand_b_avail)), replace=False)
    for idx in pick_b:
        selected_indices.add(idx)
        layer_assignments[idx] = 'hard_b'
        
    # Rule C: High-ambiguity (classifier confidence <= 0.40)
    cand_c = merged[merged['pseudo_confidence'] <= 0.40]
    cand_c_avail = [idx for idx in cand_c.index if idx not in selected_indices]
    pick_c = np.random.choice(cand_c_avail, size=min(n_hard_per_rule, len(cand_c_avail)), replace=False)
    for idx in pick_c:
        selected_indices.add(idx)
        layer_assignments[idx] = 'hard_c'
        
    # Rule D: Sarcasm / Negation risk (compliment word + frustration signal)
    pos_words = ['thanks', 'great', 'love', 'amazing', 'good job', 'perfect']
    pos_pattern = r'\b(?:' + '|'.join(pos_words) + r')\b'
    cand_d = merged[
        (merged['customer_text'].str.contains(pos_pattern, case=False, na=False, regex=True)) &
        (merged['customer_text'].str.contains(rule_a_pattern, case=False, na=False, regex=True))
    ]
    cand_d_avail = [idx for idx in cand_d.index if idx not in selected_indices]
    pick_d = np.random.choice(cand_d_avail, size=min(n_hard_per_rule, len(cand_d_avail)), replace=False)
    for idx in pick_d:
        selected_indices.add(idx)
        layer_assignments[idx] = 'hard_d'
        
    total_hard = len(selected_indices)
    target_base = n - total_hard # 140
    
    # --- Base Layer (140 samples: Stratified with Floor of 10) ---
    remaining = merged.drop(index=list(selected_indices))
    intents = list(INTENT_KEYWORD_SETS.keys()) + ['other']
    
    # Base allocation: Floor of 10 per intent (90 samples), remaining 50 proportionally
    floor_per_intent = 10
    base_floor_total = floor_per_intent * len(intents) # 90
    proportional_pool = target_base - base_floor_total # 50
    
    intent_counts = remaining['pseudo_intent'].value_counts()
    intent_props = intent_counts / len(remaining)
    
    allocation = {}
    for intent in intents:
        prop_add = int(np.round(intent_props.get(intent, 0.0) * proportional_pool))
        allocation[intent] = floor_per_intent + prop_add
        
    # Adjust to exactly target_base
    diff = target_base - sum(allocation.values())
    if diff != 0:
        top_intent = intent_counts.index[0]
        allocation[top_intent] += diff
        
    for intent, count in allocation.items():
        sub_cand = remaining[remaining['pseudo_intent'] == intent]
        sub_avail = [idx for idx in sub_cand.index if idx not in selected_indices]
        pick_base = np.random.choice(sub_avail, size=min(count, len(sub_avail)), replace=False)
        for idx in pick_base:
            selected_indices.add(idx)
            layer_assignments[idx] = 'base'
            
    # Assemble final dataframe
    final_df = merged.loc[list(selected_indices)].copy().reset_index(drop=True)
    # Re-shuffle for unbiased presentation
    final_df = final_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    final_df['golden_id'] = range(1, len(final_df) + 1)
    final_df['sample_layer'] = [layer_assignments[merged[merged['customer_tweet_id'] == cid].index[0]] for cid in final_df['customer_tweet_id']]
    
    cols = [
        'golden_id', 'customer_tweet_id', 'customer_text', 'brand_reply_text',
        'has_followup', 'followup_text', 'sample_layer', 'pseudo_intent'
    ]
    return final_df[cols]
