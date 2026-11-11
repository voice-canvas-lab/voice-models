import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from review_synthetic_asr import suspicion_reasons


class SyntheticReviewTests(unittest.TestCase):
    def row(self, text="open report", prediction="open report", deletions=0):
        return {"reference_text": text, "normalized_reference": text,
                "reference_words": len(text.split()), "prediction_text": prediction,
                "normalized_prediction": prediction, "word_deletions": deletions}

    def test_ordinary_command_not_flagged(self):
        row = self.row()
        self.assertEqual(suspicion_reasons(row, 2, [row, row]), [])

    def test_digit_and_shared_deletion_candidates(self):
        reference = self.row("open report 2023 10 06", "open report", 3)
        reasons = suspicion_reasons(reference, 2, [reference, reference])
        self.assertIn("reference_digits_but_neither_asr_returns_digits", reasons)
        self.assertIn("both_asr_delete_at_least_quarter_words_and_shorter_text", reasons)

    def test_number_words_are_only_a_candidate_not_an_exclusion(self):
        reference = self.row("version 2", "version two")
        self.assertIn("reference_digits_but_neither_asr_returns_digits",
                      suspicion_reasons(reference, 2, [reference, reference]))
        digit_prediction = self.row("version 2", "version 2")
        self.assertNotIn("reference_digits_but_neither_asr_returns_digits",
                         suspicion_reasons(reference, 2, [reference, digit_prediction]))

    def test_gigaam_duration_boundary(self):
        row = self.row()
        self.assertNotIn("duration_over_gigaam_25_seconds", suspicion_reasons(row, 25, [row, row]))
        self.assertIn("duration_over_gigaam_25_seconds", suspicion_reasons(row, 25.001, [row, row]))


if __name__ == "__main__":
    unittest.main()
