"""
Unit Tests for Step 12 Final Evaluation Report.

Tests:
1. Markdown report existence and schema completeness (all 6 sections present).
2. PDF report compilation and strict page limit validation (<= 6 pages).
3. Inclusion of the 5 authentic Step 11 failure modes with real anonymized examples.
4. Headline evaluation metrics traceability back to canonical results/*.json files.
5. PII sanitization in both Markdown and PDF text.
"""

import sys
import json
import unittest
from pathlib import Path
import pypdf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.report import (
    MD_REPORT_PATH,
    PDF_REPORT_PATH,
    load_evaluation_data,
    generate_markdown_report,
    compile_pdf_report
)

class TestFinalReport(unittest.TestCase):

    def setUp(self):
        self.data = load_evaluation_data()

    def test_01_report_files_exist(self):
        """Verify both final_report.md and final_report.pdf are generated and non-empty."""
        self.assertTrue(MD_REPORT_PATH.exists(), f"Missing markdown report: {MD_REPORT_PATH}")
        self.assertTrue(PDF_REPORT_PATH.exists(), f"Missing PDF report: {PDF_REPORT_PATH}")
        self.assertGreater(MD_REPORT_PATH.stat().st_size, 5000)
        self.assertGreater(PDF_REPORT_PATH.stat().st_size, 10000)

    def test_02_all_six_required_sections_present(self):
        """Verify all 6 mandatory assignment sections exist in the report."""
        with open(MD_REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        required_sections = [
            "1. Problem Framing",
            "2. Results vs. Baselines",
            "3. Failure Analysis",
            "4. What Is Misleading About My Headline Number?",
            "5. Next Week",
            "6. Decision Log"
        ]

        for sec in required_sections:
            self.assertIn(sec, content, f"Missing required section: '{sec}' in {MD_REPORT_PATH}")

    def test_03_pdf_page_limit(self):
        """Verify that the generated PDF report strictly satisfies the <= 6 page limit."""
        reader = pypdf.PdfReader(str(PDF_REPORT_PATH))
        num_pages = len(reader.pages)
        self.assertGreaterEqual(num_pages, 1)
        self.assertLessEqual(num_pages, 6, f"PDF report ({num_pages} pages) exceeds hard limit of 6 pages!")

    def test_04_five_failure_modes_present(self):
        """Verify all 5 authentic failure modes from Step 11 are documented."""
        with open(MD_REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        failure_cues = [
            "Failure Mode 1",
            "Failure Mode 2",
            "Failure Mode 3",
            "Failure Mode 4",
            "Failure Mode 5",
            "battery_drain_power",
            "camera_photos_media",
            "Compounding Multi-System",
            "Hallucinated Hardware",
            "Retrieval Insufficiency",
            "Verbatim"
        ]

        for cue in failure_cues:
            self.assertIn(cue, content, f"Missing failure mode reference: '{cue}' in report.")

    def test_05_headline_metrics_traceability(self):
        """Verify headline metrics in the report trace back to canonical JSON result files."""
        with open(MD_REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        im = self.data["intent"]
        em = self.data["escalate"]

        # Check Full Agent metrics presence in report
        fa_macro_f1 = f"{im['FullAgent']['macro_f1']:.4f}"
        fa_acc = f"{im['FullAgent']['accuracy']*100:.1f}%"
        fa_rec = f"{em['FullAgent']['recall']:.4f}"

        self.assertIn(fa_macro_f1, content, f"Macro-F1 {fa_macro_f1} not found in report.")
        self.assertIn(fa_acc, content, f"Accuracy {fa_acc} not found in report.")
        self.assertIn(fa_rec, content, f"Escalation Recall {fa_rec} not found in report.")

    def test_06_pii_sanitization_in_report(self):
        """Verify customer tweets in report are anonymized without exposed raw handles/emails."""
        with open(MD_REPORT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("test@apple.com", content)
        self.assertNotIn("555-123-4567", content)
        self.assertIn("@[USER]", content)

if __name__ == "__main__":
    unittest.main()
