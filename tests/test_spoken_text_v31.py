import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from spoken_text import analyze_sample, speak_filename

REPO = Path(__file__).resolve().parents[1]
ACTIONS = {a["action_id"]: a for a in map(json.loads, (REPO / "examples/zhanna-v1/filename-rule-review/actions.jsonl").read_text(encoding="utf-8").splitlines())}


class SimpleFilenameTests(unittest.TestCase):
    def sample(self, name):
        return {"sample_id": "s", "text": f"Открой файл «{name}», пожалуйста.",
                "args": {"object_type": "file", "name": name}, "action_id": "open"}

    def test_requested_example(self):
        spoken, segments = speak_filename("Отчёт_06.pdf")
        self.assertEqual(spoken, "Отчёт подчёркивание ноль шесть точка пи ди эф")
        self.assertEqual("".join(s["original"] for s in segments), "Отчёт_06.pdf")

    def test_ordinary_words_and_spaces_not_spelled(self):
        self.assertEqual(speak_filename("Итоговый отчёт.pdf")[0], "Итоговый отчёт точка пи ди эф")
        self.assertEqual(speak_filename("ОТЧЁТ.pdf")[0], "ОТЧЁТ точка пи ди эф")
        self.assertEqual(speak_filename("План-график_0001.txt")[0], "План дефис график подчёркивание ноль ноль ноль один точка ти экс ти")

    def test_technical_sequence_and_extension_spelling(self):
        self.assertEqual(speak_filename("Отчёт_AB007.PDF")[0], "Отчёт подчёркивание эй би ноль ноль семь точка пи ди эф")
        self.assertEqual(speak_filename("API_001.json")[0], "эй пи ай подчёркивание ноль ноль один точка джей эс оу эн")

    def test_date_time_in_filename_never_becomes_calendar_reading(self):
        spoken, _ = speak_filename("Список_2023-10-06_17.40.29")
        self.assertEqual(spoken, "Список подчёркивание два ноль два три дефис один ноль дефис ноль шесть подчёркивание один семь точка четыре ноль точка два девять")
        self.assertNotIn("октября", spoken)

    def test_case_metadata_outside_punctuation_and_args_preserved(self):
        sample = self.sample("Отчёт_06.PDF")
        before = copy.deepcopy(sample)
        result = analyze_sample(sample, ACTIONS)
        self.assertTrue(result["auto_tts_eligible"])
        self.assertEqual(result["spoken_text"], "Открой файл «Отчёт подчёркивание ноль шесть точка пи ди эф», пожалуйста.")
        binding = result["name_bindings"][0]
        self.assertEqual(binding["original_name"], "Отчёт_06.PDF")
        self.assertEqual("".join(c["character"] for c in binding["characters"]), "Отчёт_06.PDF")
        self.assertEqual(binding["characters"][0]["case"], "upper")
        self.assertEqual(sample, before)
        for label in ("латинская", "кириллическая", "русская", "заглавная"):
            self.assertNotIn(label, result["spoken_text"])

    def test_ambiguity_flagged_and_original_retained(self):
        for name in ("Отчeт.pdf", "Unknownword.pdf", "Отчёт  итог.pdf", "Отчёт\nитог.pdf"):
            with self.subTest(name=name):
                sample = self.sample(name)
                result = analyze_sample(sample, ACTIONS)
                self.assertFalse(result["auto_tts_eligible"])
                self.assertTrue(result["review_reasons"])
                self.assertEqual(result["spoken_text"], sample["text"])

    def test_existing_five_inputs_not_changed(self):
        path = REPO / "examples/zhanna-v1/spoken-v3.0.0/pilot/inputs.jsonl"
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            before = copy.deepcopy(row)
            result = analyze_sample(row, ACTIONS)
            self.assertTrue(result["auto_tts_eligible"])
            self.assertEqual(row, before)
            for binding in result["name_bindings"]:
                if binding["object_type"] == "file":
                    self.assertEqual("".join(s["original"] for s in binding["segments"]), binding["original_name"])


if __name__ == "__main__":
    unittest.main()
