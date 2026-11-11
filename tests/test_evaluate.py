import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).parents[1] / "scripts/evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class EvaluationTests(unittest.TestCase):
    def test_known_edits(self):
        for hypothesis, expected in (("кот ест дома", (1, 0, 0)),
                                     ("кот спит", (0, 1, 0)),
                                     ("кот спит дома сейчас", (0, 0, 1))):
            with self.subTest(hypothesis=hypothesis):
                row = evaluate.score("кот спит дома", hypothesis)
                self.assertEqual(tuple(row[key] for key in
                                       ("word_substitutions", "word_deletions", "word_insertions")), expected)
                self.assertAlmostEqual(row["wer"], 1 / 3)

    def test_character_edits(self):
        for hypothesis in ("кот", "к", "китт"):
            self.assertEqual(evaluate.score("кит", hypothesis)["character_errors"], 1 if hypothesis != "к" else 2)

    def test_normalization_retains_content(self):
        self.assertEqual(evaluate.normalize("  Е\u0308Ж, НЕ 12 — нет, 13! "), "еж не 12 нет 13")

    def test_empty_reference(self):
        row = evaluate.score("!!!", "два слова")
        self.assertTrue(row["empty_reference"])
        self.assertEqual(row["word_insertions"], 2)
        self.assertEqual(row["character_insertions"], 9)
        self.assertIsNone(row["wer"])
        self.assertIsNone(row["cer"])
        self.assertEqual(evaluate.score("", "")["word_errors"], 0)

    def test_corpus_weighting(self):
        records = [evaluate.score("а", "б"), evaluate.score("а б в г", "а б в г")]
        for row, seconds, duration in zip(records, (1, 3), (1, 9)):
            row.update(elapsed_sec=seconds, duration_sec=duration)
        totals = evaluate.aggregate(records)
        self.assertEqual(totals["wer"], 1 / 5)
        self.assertEqual(totals["cer"], 1 / 8)
        self.assertEqual(totals["rtf"], 4 / 10)
        empty = evaluate.score("", "слово")
        empty.update(elapsed_sec=1, duration_sec=1)
        self.assertIsNone(evaluate.aggregate([empty])["wer"])
        self.assertEqual(evaluate.aggregate(records + [empty])["wer"], 2 / 5)

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text('{"audio_id":"a"}\n{"audio_id":"a"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluate.read_index(path, "audio_id")

    def test_invalid_links(self):
        samples = {"s": {"text": "текст"}}
        manifest = {"a": {"sample_id": "s", "duration_sec": 1}}
        prediction = {"a": {"sample_id": "s", "text": "текст"}}
        timing = {"a": {"sample_id": "s", "duration_sec": 1, "elapsed_sec": 0.5}}
        evaluate.check_links(samples, manifest, prediction, timing)
        for predictions in ({}, {**prediction, "extra": prediction["a"]},
                            {"a": {"sample_id": "wrong", "text": "текст"}}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate.check_links(samples, manifest, predictions, timing)
        with self.assertRaises(ValueError):
            evaluate.check_links({}, manifest, prediction, timing)
        with self.assertRaises(ValueError):
            evaluate.check_links(samples, manifest, prediction, {})


if __name__ == "__main__":
    unittest.main()
