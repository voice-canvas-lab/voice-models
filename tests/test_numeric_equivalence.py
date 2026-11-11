import inspect
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import score, aggregate, digest
from evaluate_numeric import evaluate as evaluate_numeric
from numeric_equivalence import canonicalize


class NumericEquivalenceTests(unittest.TestCase):
    def compare(self, ref, hyp):
        return score(canonicalize(ref)["transformed_text"], canonicalize(hyp)["transformed_text"])

    def test_equivalent_verified_date_time(self):
        ref = "Дата — шестое октября две тысячи двадцать третьего года; время — семнадцать часов сорок минут двадцать девять секунд"
        for hyp in ("Дата 6 октября 2023 года; время 17 часов 40 минут 29 секунд",
                    "Дата 6 октября 2023 года; время 17:40:29"):
            with self.subTest(hyp=hyp):
                self.assertEqual(self.compare(ref, hyp)["wer"], 0)
                self.assertEqual(self.compare(ref, hyp)["cer"], 0)
                self.assertFalse(canonicalize(hyp)["flags"])

    def test_equivalent_versions(self):
        for ref, hyp in (("версия два точка ноль", "версия 2.0"),
                         ("версия два точка ноль один", "версия 2.01"),
                         ("версия два точка ноль точка три", "версия 2.0.3")):
            with self.subTest(ref=ref):
                self.assertEqual(self.compare(ref, hyp)["wer"], 0)
                self.assertEqual(self.compare(ref, hyp)["cer"], 0)

    def test_equivalent_ids_and_explicit_clock_zeros(self):
        for ref, hyp in (("note_000001", "эн оу ти и подчёркивание ноль ноль ноль ноль ноль один"),
                         ("документ_000015", "документ подчёркивание ноль ноль ноль ноль один пять"),
                         ("document_001", "ди оу си ю эм и эн ти подчёркивание ноль ноль один"),
                         ("время 09:02:17", "время ноль девять часов ноль два минуты семнадцать секунд")):
            with self.subTest(ref=ref):
                self.assertEqual(self.compare(ref, hyp)["wer"], 0)

    def test_real_errors_stay_errors(self):
        cases = [
            ("версия 2.0", "версия 2.1"),  # Wrong digit.
            ("версия 2.0.3", "версия 2.3"),  # Missing component.
            ("версия 2.0", "версия 0.2"),  # Reordering.
            ("версия 2.01", "версия 2.1"),  # Lost zero.
            ("note_000001", "note_1"),
            ("документ_000015", "документ_000051"),
            ("дата 6 октября 2023 года", "дата 7 октября 2023 года"),
            ("дата 6 октября 2023 года", "дата 6 октября 2024 года"),
            ("дата 6 октября 2023 года", "дата 6 октября"),
            ("дата 06 октября 2023 года", "дата 6 октября 2023 года"),
            ("время 17:40:29", "время 17:29:40"),
            ("время 17:40:29", "время 17:40"),
            ("время 09:02:17", "время 9:02:17"),
            ("время семнадцать часов сорок минут двадцать девять секунд",
             "время семнадцать часов двадцать девять секунд сорок минут"),
        ]
        for ref, hyp in cases:
            with self.subTest(ref=ref, hyp=hyp):
                row = self.compare(ref, hyp)
                self.assertGreater(row["wer"], 0)
                self.assertGreater(row["cer"], 0)

    def test_ambiguous_time_not_repaired(self):
        text = "Время 17:4029."
        result = canonicalize(text)
        self.assertEqual(result["transformed_text"], text)
        self.assertTrue(result["needs_review"])
        self.assertIn("ambiguous_or_invalid_compact_time_no_repair", [f["reason"] for f in result["flags"]])
        self.assertGreater(self.compare("Время 17:40:29.", text)["wer"], 0)

    def test_unknowns_and_plain_text_preserved(self):
        for text in ("число 123", "2023-10-06", "17:40:29", "версия два точка", "версия 2..0",
                     "версия два точка один пять", "дата 31 февраля 2023 года"):
            with self.subTest(text=text):
                result = canonicalize(text)
                self.assertEqual(result["transformed_text"], text)
                self.assertTrue(result["needs_review"])
        text = "  Открой документ — Инструкция по ремонту!"
        self.assertEqual(canonicalize(text)["transformed_text"], text)
        self.assertFalse(canonicalize(text)["flags"])

    def test_component_boundaries_and_types(self):
        ref, hyp = "версия 2.03", "версия 20.3"
        self.assertNotEqual(canonicalize(ref)["transformed_text"], canonicalize(hyp)["transformed_text"])
        self.assertGreater(self.compare(ref, hyp)["wer"], 0)
        self.assertGreater(self.compare("note_20", "версия 2.0")["wer"], 0)

    def test_single_input_no_reference_dependency(self):
        self.assertEqual(list(inspect.signature(canonicalize).parameters), ["text"])
        text = "Время 17:4029; версия два точка ноль"
        first = canonicalize(text)
        self.compare("время 17:40:29", text)
        self.compare("время 09:02:17", text)
        self.assertEqual(canonicalize(text), first)

    def test_separate_evaluation_keeps_baseline_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, results, baseline = root / "dataset", root / "asr", root / "baseline"
            for path in (dataset, results, baseline):
                path.mkdir()
            sample = {"sample_id": "s", "text": "версия два точка ноль"}
            audio = {"audio_id": "a", "sample_id": "s", "duration_sec": 2}
            prediction = {"audio_id": "a", "sample_id": "s", "text": "версия 2.0"}
            timing = {"audio_id": "a", "sample_id": "s", "elapsed_sec": 1, "duration_sec": 2}
            paths = {"samples": dataset / "samples.jsonl", "manifest": dataset / "audio_manifest.jsonl",
                     "predictions": results / "predictions.jsonl", "timings": results / "timings.jsonl"}
            for key, row in (("samples", sample), ("manifest", audio), ("predictions", prediction), ("timings", timing)):
                paths[key].write_text(json.dumps(row) + "\n", encoding="utf-8")
            record = {**timing, **score(sample["text"], prediction["text"])}
            (baseline / "record_metrics.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            (baseline / "summary.json").write_text(json.dumps({**aggregate([record]), "input_sha256": {k: digest(p) for k, p in paths.items()}}), encoding="utf-8")
            before = {p: p.read_bytes() for p in [*paths.values(), *baseline.iterdir()]}
            output = root / "additional"
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_numeric(dataset, results, baseline, output)
            row = json.loads((output / "record_metrics.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["baseline"], record)
            self.assertGreater(row["baseline"]["wer"], 0)
            self.assertEqual(row["numeric_equivalent"]["wer"], 0)
            self.assertEqual(row["reference_transform"]["original_text"], sample["text"])
            self.assertEqual(row["hypothesis_transform"]["original_text"], prediction["text"])
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(FileExistsError):
                evaluate_numeric(dataset, results, baseline, output)


if __name__ == "__main__":
    unittest.main()
