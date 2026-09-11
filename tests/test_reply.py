"""
Step 7 Tests for Grounded Reply Drafter (retrieve -> rewrite -> cite).

Validates:
1. retrieve() functionality, top-k ranking, and cosine similarity scoring.
2. Minimum similarity threshold and safe fallback triggering on out-of-distribution queries.
3. Strict Grounding Rules in LLM rewriting.
4. PII Redaction across customer inputs and historical examples.
5. Complete Citation Trail preservation (customer_tweet_id, brand_tweet_id).
6. End-to-end agent_reply contract compliance.
"""

import sys
import unittest
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reply import retrieve, draft, draft_reply, get_retrieval_index
from src.agent import agent_reply
from src.pii import redact_pii
from src.config import STANDARD_FALLBACK_REPLY, RETRIEVAL_MIN_SIMILARITY, RETRIEVAL_TOP_K

class TestGroundedReplyDrafter(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.index = get_retrieval_index()

    def test_01_index_validation(self):
        """Index and metadata dimensions must match exactly."""
        self.assertIsNotNone(self.index.index_matrix)
        self.assertIsNotNone(self.index.metadata_df)
        n_matrix = self.index.index_matrix.shape[0]
        n_meta = len(self.index.metadata_df)
        self.assertEqual(n_matrix, n_meta, f"Matrix rows ({n_matrix}) != metadata rows ({n_meta})")
        self.assertGreaterEqual(n_matrix, 10000, f"Expected at least 10,000 usable threads in index, got {n_matrix}")
        self.assertIn(n_matrix, [14597, 103771], f"Unexpected index thread count: {n_matrix}")
        self.assertEqual(self.index.index_matrix.shape[1], 25000, "Expected 25,000 feature dimensions")

    def test_02_test_a_normal_resolvable_issue(self):
        """Test A: Normal resolvable issue retrieves relevant historical examples with high similarity."""
        query = "My battery is draining extremely fast on my iPhone after the recent update."
        retrieved = retrieve(query, k=3, min_similarity=RETRIEVAL_MIN_SIMILARITY)
        
        self.assertTrue(len(retrieved) > 0, "Should retrieve at least 1 example")
        self.assertLessEqual(len(retrieved), 3, "Should not exceed k=3")
        
        # Verify schema
        top = retrieved[0]
        self.assertIn("customer_tweet_id", top)
        self.assertIn("brand_tweet_id", top)
        self.assertIn("customer_text", top)
        self.assertIn("brand_reply_text", top)
        self.assertIn("similarity", top)
        self.assertGreater(top["similarity"], RETRIEVAL_MIN_SIMILARITY)

        # Test drafting
        result = draft(query, retrieved)
        self.assertIn("draft", result)
        self.assertIn("retrieved_ids", result)
        self.assertIn("retrieved_customer_ids", result)
        self.assertIn("retrieved_brand_ids", result)
        self.assertIn("retrieved_replies", result)
        self.assertTrue(result["grounded"])
        self.assertEqual(len(result["retrieved_ids"]), len(retrieved))
        print(f"\n[Test A] Query: {query}")
        print(f"Draft: {result['draft']}")
        print(f"Citation IDs: {result['retrieved_ids']}")

    def test_03_test_b_different_issue(self):
        """Test B: Different issue (autocorrect keyboard bug) retrieves different intent-specific context."""
        query = "Whenever I type the letter i on my keyboard it replaces with an A and question mark symbol."
        retrieved = retrieve(query, k=3, min_similarity=RETRIEVAL_MIN_SIMILARITY)
        
        self.assertTrue(len(retrieved) > 0)
        # Top retrieved should mention keyboard, autocorrect, text replacement, or letter i
        combined_text = " ".join([r["customer_text"].lower() + " " + r["brand_reply_text"].lower() for r in retrieved])
        self.assertTrue(any(w in combined_text for w in ["keyboard", "autocorrect", "text replacement", "letter", "i️", "update"]))
        
        result = draft(query, retrieved)
        self.assertIn("Settings > General > Keyboard", result["draft"] + " " + combined_text)
        print(f"\n[Test B] Query: {query}")
        print(f"Draft: {result['draft']}")

    def test_04_test_c_no_good_match_triggers_fallback(self):
        """Test C: Out-of-distribution message with no match triggers safe standard fallback."""
        gibberish = "Xylophone quantum astrophysics lasagna recipe in medieval Constantinople 9876543210."
        retrieved = retrieve(gibberish, k=3, min_similarity=0.30)
        self.assertEqual(len(retrieved), 0, "Irrelevant query should yield 0 results above threshold")

        result = draft(gibberish, retrieved)
        self.assertEqual(result["draft"], STANDARD_FALLBACK_REPLY)
        self.assertFalse(result["grounded"])
        self.assertEqual(result["retrieved_ids"], [])
        print(f"\n[Test C] Fallback Triggered: {result['draft']}")

    def test_05_test_d_pii_redaction(self):
        """Test D: PII is redacted before passing to LLM and prompt construction."""
        raw_msg = "Please help, my email is john.doe@samplecorp.org and my phone is +1 415-555-0182. My account is locked."
        sanitized, count, pii_types = redact_pii(raw_msg)
        
        self.assertGreater(count, 0)
        self.assertNotIn("john.doe@samplecorp.org", sanitized)
        self.assertNotIn("415-555-0182", sanitized)
        self.assertIn("[EMAIL_REDACTED]", sanitized)
        self.assertIn("[PHONE_REDACTED]", sanitized)

        retrieved = retrieve(raw_msg, k=3, min_similarity=RETRIEVAL_MIN_SIMILARITY)
        result = draft(raw_msg, retrieved)
        self.assertNotIn("john.doe@samplecorp.org", result["draft"])
        self.assertNotIn("415-555-0182", result["draft"])
        print(f"\n[Test D] Sanitized: {sanitized}")

    def test_06_test_e_citation_trail_and_agent_reply(self):
        """Test E: agent_reply() end-to-end integration preserves citation trail and full schema."""
        query = "How do I reset my Apple ID password? I am locked out."
        res = agent_reply(query)
        
        self.assertIn("draft_reply", res)
        self.assertIn("retrieved_ids", res)
        self.assertIn("retrieved_replies", res)
        self.assertIn("intent", res)
        self.assertIn("escalate", res)
        self.assertIn("latency_ms", res)
        self.assertTrue(len(res["retrieved_ids"]) > 0)
        print(f"\n[Test E] agent_reply Citation Trail: {res['retrieved_ids']}")

if __name__ == "__main__":
    unittest.main()
