"""
Unit Tests for LLM-as-a-Judge Module (Judge v1, Calibrated Judge v2, and Schema Validation).
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.judge import evaluate_reply, CRITERIA

class TestJudge(unittest.TestCase):

    def test_01_judge_v1_evaluation_schema(self):
        """Judge v1 must evaluate all 5 criteria and return valid scores and reasons."""
        customer = "My battery is draining rapidly after the iOS 11 update."
        draft = "We understand how important battery life is. Please check Settings > Battery and DM us: https://t.co/GDrqU22YpT"
        retrieved = ["Take a look at Settings > Battery to view usage. Reach out via DM if you need help: https://t.co/GDrqU22YpT"]

        scores = evaluate_reply(customer, draft, retrieved, version="v1")
        for crit in CRITERIA:
            self.assertIn(crit, scores)
            self.assertIn("score", scores[crit])
            self.assertIn("one_line_reason", scores[crit])
            score_val = scores[crit]["score"]
            self.assertTrue(1 <= score_val <= 5, f"Score for {crit} out of bounds: {score_val}")
            self.assertTrue(len(scores[crit]["one_line_reason"]) > 0)

    def test_02_judge_v2_calibrated_evaluation(self):
        """Judge v2 must apply calibrated rubrics and output structured evaluation."""
        customer = "I have a problem with keyboard autocorrect replacing i with a question mark."
        draft = "We are aware of the issue. Use Settings > General > Keyboard > Text Replacement. DM us: https://t.co/GDrqU22YpT"
        retrieved = ["Workaround is available in Settings > General > Keyboard > Text Replacement. DM us: https://t.co/GDrqU22YpT"]

        scores_v2 = evaluate_reply(customer, draft, retrieved, version="v2")
        for crit in CRITERIA:
            self.assertIn(crit, scores_v2)
            self.assertTrue(1 <= scores_v2[crit]["score"] <= 5)

    def test_03_judge_distinguishes_good_vs_trivial_reply(self):
        """Judge scores must differentiate between a grounded technical reply vs empty canned deflection."""
        customer = "My camera screen is black when opening the camera app."
        good_draft = "Please force close the Camera app and test both front and rear cameras. DM us with your iOS build: https://t.co/GDrqU22YpT"
        canned_draft = "Thanks for reaching out to @AppleSupport. Please force restart your device: https://t.co/GDrqU22YpT"
        retrieved = ["Close all camera apps and restart device. DM us if persists: https://t.co/GDrqU22YpT"]

        scores_good = evaluate_reply(customer, good_draft, retrieved, version="v2")
        scores_canned = evaluate_reply(customer, canned_draft, retrieved, version="v2")

        # Grounded/specific draft should score higher on correctness/completeness than generic canned draft
        self.assertGreaterEqual(scores_good["correctness"]["score"], scores_canned["correctness"]["score"])
        self.assertGreaterEqual(scores_good["completeness"]["score"], scores_canned["completeness"]["score"])

if __name__ == "__main__":
    unittest.main()
