import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from spoken_text_v3_0_0 import analyze_sample, derive, speak_filename

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "examples/zhanna-v1/filename-rule-review/actions.jsonl"
ACTIONS = {r["action_id"]: r for r in map(json.loads, CATALOG.read_text(encoding="utf-8").splitlines())}


class ExactFilenameTests(unittest.TestCase):
    def sample(self, name="отчёт-01.pdf", text=None):
        return {"sample_id": "s", "text": text or f"Открой файл «{name}», пожалуйста.",
                "args": {"object_type": "file", "name": name}, "action_id": "open",
                "context": {"objects": [], "selected_object_ids": []}, "group_id": "g", "split": "test"}

    def test_contract_binding_and_outside_punctuation(self):
        sample = self.sample()
        result = analyze_sample(sample, ACTIONS)
        self.assertTrue(result["auto_tts_eligible"])
        binding = result["name_bindings"][0]
        self.assertEqual(sample["text"][binding["start"]:binding["end"]], sample["args"]["name"])
        self.assertEqual(binding["argument"], "name")
        self.assertEqual(result["spoken_text"], 'Открой файл «' + binding["spoken_name"] + '», пожалуйста.')
        self.assertIn("дефис ноль один точка", binding["spoken_name"])
        self.assertNotIn("запятая", result["spoken_text"])

    def test_all_characters_extension_alphabet_case_and_zeros(self):
        name = "Аa_0001-Ё.pdf"
        spoken, chars = speak_filename(name)
        self.assertEqual("".join(c["character"] for c in chars), name)
        self.assertEqual([c["index"] for c in chars], list(range(len(name))))
        self.assertIn("заглавная русская а латинская эй", spoken)
        self.assertIn("подчёркивание ноль ноль ноль один дефис", spoken)
        self.assertIn("точка латинская пи латинская ди латинская эф", spoken)

    def test_filename_date_and_version_take_priority(self):
        name = "версия 2.0_2023-10-06_17.40.29"
        result = analyze_sample(self.sample(name), ACTIONS)
        self.assertTrue(result["auto_tts_eligible"])
        self.assertNotIn("октября", result["spoken_text"])
        self.assertNotIn("семнадцать", result["spoken_text"])
        self.assertIn("два ноль два три дефис один ноль дефис ноль шесть", result["spoken_text"])
        self.assertIn("один семь точка четыре ноль точка два девять", result["spoken_text"])

    def test_date_and_version_outside_file_keep_semantic_rules(self):
        sample = self.sample("отчёт.pdf", 'Открой «отчёт.pdf»; дата 2023-10-06; время 17:40:29; версия 2.0.')
        result = analyze_sample(sample, ACTIONS)
        self.assertTrue(result["auto_tts_eligible"])
        self.assertIn("шестое октября", result["spoken_text"])
        self.assertIn("семнадцать часов сорок минут двадцать девять секунд", result["spoken_text"])
        self.assertTrue(result["spoken_text"].endswith("версия два точка ноль."))

    def test_mismatch_repetition_and_boundaries_flagged_without_repair(self):
        for name, text in (("отчёт.pdf", "Открой отчёт.txt"),
                           ("a.txt", "Открой a.txt, затем a.txt"),
                           ("a.txt", "Открой ba.txt"), ("a", "Открой a.txt"),
                           ("a.txt", "Открой x-a.txt"), ("a.txt", "Открой a.txt-copy"),
                           ("a.txt", "Открой A.txt")):
            with self.subTest(name=name, text=text):
                result = analyze_sample(self.sample(name, text), ACTIONS)
                self.assertFalse(result["auto_tts_eligible"])
                self.assertEqual(result["spoken_text"], text)
                self.assertFalse(result["edits"])

    def test_rename_binds_distinct_roles_and_rejects_shared_occurrence(self):
        sample = self.sample()
        sample.update(action_id="rename", text="Переименуй a.txt в b.txt.",
                      args={"object_type": "file", "old_name": "a.txt", "new_name": "b.txt"})
        result = analyze_sample(sample, ACTIONS)
        self.assertTrue(result["auto_tts_eligible"])
        self.assertEqual([b["argument"] for b in result["name_bindings"]], ["old_name", "new_name"])
        sample["args"]["new_name"] = "a.txt"
        sample["text"] = "Переименуй a.txt."
        self.assertFalse(analyze_sample(sample, ACTIONS)["auto_tts_eligible"])

    def test_query_is_not_a_name_and_schema_controls_binding(self):
        sample = self.sample("a.txt", "Ищи a.txt")
        sample.update(action_id="search", args={"object_type": "file", "query": "a.txt"})
        self.assertFalse(analyze_sample(sample, ACTIONS)["name_bindings"])
        catalog = copy.deepcopy(ACTIONS)
        catalog["open"]["parameters"]["properties"]["name"]["type"] = "integer"
        self.assertFalse(analyze_sample(self.sample(), catalog)["auto_tts_eligible"])

    def test_unsupported_character_blocks_all_changes(self):
        result = analyze_sample(self.sample("report😀.pdf"), ACTIONS)
        self.assertFalse(result["auto_tts_eligible"])
        self.assertIn("unsupported_filename_character", result["review_reasons"])

    def test_derive_changes_only_text_and_separates_name_mapping(self):
        sample = self.sample()
        before = copy.deepcopy(sample)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.jsonl", Path(tmp) / "derived"
            source.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            raw = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                derive(source, output, CATALOG)
            row = json.loads((output / "samples.jsonl").read_text(encoding="utf-8"))
            self.assertEqual({k: v for k, v in row.items() if k != "text"}, {k: v for k, v in sample.items() if k != "text"})
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(sample, before)
            mapping = json.loads((output / "filename_mapping.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(mapping["original_name"], sample["args"]["name"])
            self.assertTrue(mapping["applied"])
            self.assertEqual("".join(c["character"] for c in mapping["characters"]), sample["args"]["name"])


if __name__ == "__main__":
    unittest.main()
