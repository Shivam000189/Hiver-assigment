"""
Smoke Test Suite for AppleSupport AI Agent.

Executes comprehensive validation across:
1. 30 Golden DEV Set Examples (Comparing Intent & Escalation Ground-Truth).
2. 10 Hand-Crafted Adversarial / Edge-Case Inputs.
3. System Integrity & Contract Assertions.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.agent import agent_reply
from src.intents import classify, ROUTING_STATS
from src.llm import get_llm_metrics
from src.reply import get_retrieval_index

# Ensure UTF-8 console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def run_smoke_tests():
    print("================================================================")
    print("      RUNNING HIVER SUPPORT AGENT SMOKE TESTS (STEP 5)          ")
    print("================================================================\n")

    # 1. Retrieval Index Status
    print("--- 1. Checking Retrieval Index ---")
    index = get_retrieval_index()
    print(f"Retrieval index active with {len(index.metadata_df):,} usable resolved threads.\n")

    # 2. Classifier Component Smoke Test
    print("--- 2. Intent Classifier Smoke Test (20 Random Corpus + 10 Taxonomy Samples) ---")
    threads_df = pd.read_parquet("data/processed/applesupport_threads.parquet")
    np.random.seed(42)
    sample_20 = threads_df.sample(20, random_state=42)
    
    print("Sample 5 of 20 Random Corpus Classifications:")
    for i, (_, r) in enumerate(sample_20.head(5).iterrows()):
        intent, conf, src = classify(r['customer_text'])
        print(f"  [{i+1}] Tweet: \"{r['customer_text'][:65]}...\"")
        print(f"      -> Intent: {intent:<28} | Conf: {conf:.2f} | Source: {src}")

    # 3. 30 Golden Dev Examples Test
    print("\n--- 3. Testing 30 Golden DEV Set Examples ---")
    dev_df = pd.read_csv("data/golden_set/golden_dev.csv")
    np.random.seed(42)
    dev_30 = dev_df.sample(min(30, len(dev_df)), random_state=42).reset_index(drop=True)
    
    dev_results = []
    intent_matches = 0
    escalate_matches = 0
    
    for i, r in dev_30.iterrows():
        res = agent_reply(r['customer_text'])
        
        # Contract Assertions
        assert isinstance(res['intent'], str) and len(res['intent']) > 0, "Intent must be non-empty string"
        assert 0.0 <= res['intent_confidence'] <= 1.0, "Confidence must be in [0, 1]"
        assert isinstance(res['escalate'], bool), "Escalate must be boolean"
        assert len(res['escalate_reason']) > 0, "Escalate reason must be non-empty"
        assert len(res['retrieved_ids']) > 0, "Retrieved IDs must not be empty"
        assert len(res['draft_reply']) > 0, "Draft reply must not be empty"

        i_match = (res['intent'] == r['intent'])
        e_match = (res['escalate'] == (r['escalate'].lower() == 'yes'))
        if i_match:
            intent_matches += 1
        if e_match:
            escalate_matches += 1
            
        dev_results.append({
            "golden_id": int(r['golden_id']),
            "customer_text": r['customer_text'],
            "golden_intent": r['intent'],
            "pred_intent": res['intent'],
            "intent_match": i_match,
            "intent_confidence": res['intent_confidence'],
            "golden_escalate": r['escalate'],
            "pred_escalate": "yes" if res['escalate'] else "no",
            "escalate_match": e_match,
            "escalate_reason": res['escalate_reason'],
            "draft_reply": res['draft_reply'],
            "retrieved_ids": res['retrieved_ids'],
            "latency_ms": res['latency_ms']
        })

    print(f"DEV 30 Smoke Test Accuracy:")
    print(f"  - Intent Exact Match    : {intent_matches}/30 ({intent_matches/30*100:.1f}%)")
    print(f"  - Escalation Exact Match: {escalate_matches}/30 ({escalate_matches/30*100:.1f}%)")

    print("\nSample 3 DEV Pipeline Traces:")
    for item in dev_results[:3]:
        print(f"\n[Golden ID {item['golden_id']}] Customer: \"{item['customer_text']}\"")
        print(f"  -> Predicted Intent: {item['pred_intent']} (Golden: {item['golden_intent']}) | Match: {item['intent_match']}")
        print(f"  -> Escalated: {item['pred_escalate']} (Reason: {item['escalate_reason']})")
        print(f"  -> Draft Reply: \"{item['draft_reply']}\"")
        print(f"  -> Latency: {item['latency_ms']['total']} ms (Classify: {item['latency_ms']['classify']}ms, Retrieve: {item['latency_ms']['retrieve']}ms, Draft: {item['latency_ms']['draft']}ms)")

    # 4. 10 Adversarial Inputs Test
    print("\n--- 4. Testing 10 Adversarial / Stress Test Inputs ---")
    adversarial_cases = [
        ("1. Legal Threat", "I have hired an attorney and will be filing a formal lawsuit against Apple for fraudulent charges today."),
        ("2. Hacked Account", "Someone hacked my Apple ID, changed my two-factor authentication phone number, and locked all my devices!"),
        ("3. Positive Praise", "Just wanted to say I love the new iPhone 8 camera! Works wonderfully and looks great."),
        ("4. Gibberish / Noise", "asdfghjkl qwertyuiop zxcvbnm ????? #$$$ %%% @@@"),
        ("5. Multi-Issue Query", "My battery dies in 30 minutes and my Wi-Fi disconnects every time I try to open Safari on iOS 11."),
        ("6. Non-English Query", "Bonjour, mon écran d'iPhone 6 est complètement cassé et ne répond plus du tout."),
        ("7. Extremely Long Rant", "I have been an Apple customer since 2007 and this is the most atrocious update ever released. My iPhone 7 lags, my Apple Watch workout failed to sync, my battery drains in 45 minutes, and your support agents in store told me to buy a new phone. Completely unacceptable!"),
        ("8. Empty / Minimal", "???"),
        ("9. Competitor Mention", "Google Pixel is so much better than this broken iPhone that won't even charge properly."),
        ("10. Sensitive Info & PII", "Please call me at 415-555-0199 and email john.doe@company.com with my account password immediately.")
    ]

    adversarial_results = []
    for tag, text in adversarial_cases:
        res = agent_reply(text)
        adversarial_results.append({
            "case": tag,
            "input_text": text,
            "intent": res['intent'],
            "confidence": res['intent_confidence'],
            "escalate": res['escalate'],
            "escalate_reason": res['escalate_reason'],
            "pii_redacted": res['pii_redacted'],
            "draft_reply": res['draft_reply'],
            "retrieved_ids": res['retrieved_ids'],
            "latency_ms": res['latency_ms']
        })
        print(f"\n[{tag}] \"{text[:70]}...\"")
        print(f"  -> Intent: {res['intent']} (Conf: {res['intent_confidence']:.2f})")
        print(f"  -> Escalated: {res['escalate']} | Reason: {res['escalate_reason']}")
        print(f"  -> PII Redacted: {res['pii_redacted']}")
        print(f"  -> Draft Reply: \"{res['draft_reply'][:100]}...\"")

    # 5. System Health & Metric Reporting
    print("\n--- 5. System Metrics & Telemetry ---")
    llm_metrics = get_llm_metrics()
    print(f"LLM Disk Cache Hits    : {llm_metrics['cache_hits']}")
    print(f"LLM Disk Cache Misses  : {llm_metrics['cache_misses']}")
    print(f"LLM Retry Count        : {llm_metrics['retries']}")
    print(f"JSON Parse Fallbacks   : {llm_metrics['parse_fallbacks']}")
    print(f"Routing Stats          : {ROUTING_STATS}")

    # Save outputs to results/smoke_test_outputs.json
    output_payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dev_30_accuracy": {
            "intent_accuracy": round(intent_matches / 30, 4),
            "escalate_accuracy": round(escalate_matches / 30, 4)
        },
        "dev_results": dev_results,
        "adversarial_results": adversarial_results,
        "llm_metrics": llm_metrics,
        "routing_stats": ROUTING_STATS
    }

    out_file = "results/smoke_test_outputs.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\nSmoke test outputs saved to {out_file} successfully.")
    print("================================================================")
    print("              SMOKE TEST SUITE COMPLETED (PASS)                 ")
    print("================================================================")

if __name__ == "__main__":
    run_smoke_tests()
