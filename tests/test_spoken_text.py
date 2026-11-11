import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
# Regression coverage of preserved v1.0.1; current safe policy is tested separately.
from spoken_text_v1_0_1 import RULES_VERSION, derive, transform


class SpokenTextTests(unittest.TestCase):
    def test_ordinary_text_byte_for_byte(self):
        for text in ("Открой документ Инструкция по ремонту", "Сохрани заметку. Спасибо!",
                     "  Открой «Мои  мысли» — пожалуйста.\n", "версия два точка ноль"):
            with self.subTest(text=text):
                self.assertEqual(transform(text), (text, [], []))

    def test_iso_date_and_timestamp(self):
        self.assertEqual(transform("Дата 2023-10-06")[0], "Дата шестое октября две тысячи двадцать третьего года")
        out, edits, flags = transform("Список_2023-10-06_17.40.29")
        self.assertEqual(out, "Список подчёркивание шестое октября две тысячи двадцать третьего года семнадцать часов сорок минут двадцать девять секунд")
        self.assertEqual(edits[-1]["kind"], "iso_datetime")
        self.assertFalse(flags)
        out, edits, _ = transform("Документ_2024-08-12_17.46.35")
        self.assertIn("двенадцатое августа две тысячи двадцать четвёртого года", out)
        self.assertEqual(edits[-1]["kind"], "iso_datetime")
        self.assertFalse(any(e["kind"] == "technical_id" for e in edits))
        self.assertIn("двадцать девятое февраля", transform("2024-02-29")[0])
        self.assertIn("invalid_or_unsupported_date_time_preserved_literally", transform("2023-02-29")[2])

    def test_explicit_versions_and_clock(self):
        self.assertEqual(transform("версия 2.0")[0], "версия два точка ноль")
        self.assertEqual(transform("Документация_версия_3.01")[0], "Документация подчёркивание версия подчёркивание три точка ноль один")
        self.assertEqual(transform("время 09:02:17")[0], "время девять часов две минуты семнадцать секунд")
        self.assertEqual(transform("время 00:01:01")[0], "время ноль часов одна минута одна секунда")
        self.assertIn("invalid_or_unsupported_date_time_preserved_literally", transform("время 25:61:00")[2])

    def test_numbers_in_names(self):
        self.assertEqual(transform("Подборка 4: Экология")[0], "Подборка четыре: Экология")
        self.assertEqual(transform("Отчёт 2023")[0], "Отчёт две тысячи двадцать три")
        out, _, flags = transform("Дневник мыслей.4")
        self.assertEqual(out, "Дневник мыслей точка четыре")
        self.assertIn("ambiguous_name_numeric_suffix", flags)
        self.assertEqual(transform("3D моделирование")[0], "три ди моделирование")

    def test_technical_id_keeps_each_leading_zero(self):
        self.assertEqual(transform("Открой note_000001")[0], "Открой эн оу ти и подчёркивание ноль ноль ноль ноль ноль один")
        self.assertEqual(transform("документ_000015")[0], "документ подчёркивание ноль ноль ноль ноль один пять")
        self.assertEqual(transform("файл_000000")[0], "файл подчёркивание ноль ноль ноль ноль ноль ноль")

    def test_ambiguity_is_flagged_not_repaired(self):
        out, _, flags = transform("Список_2023_10_06_17_40_29")
        self.assertIn("ambiguous_numeric_chain_no_semantic_guess", flags)
        self.assertIn("ноль шесть", out)
        self.assertNotIn("октября", out)
        out, _, flags = transform("Список_202023-10-06_17-40-29")
        self.assertIn("двести две тысячи двадцать три", out)
        self.assertTrue(flags)
        self.assertIn("ambiguous_numeric_chain_no_semantic_guess", transform("число 2.0")[2])

    def test_every_edit_is_reversible_and_non_numeric_names_retained(self):
        text = "Создай Фото_для_отчета_2023-07-02_09.02.17 из файла в папке Эволюция, с именем Таблица_2024-02-05_11.08.26"
        spoken, edits, _ = transform(text)
        # Rebuild original from logged original spans, never from args/context.
        restored = []
        pos = 0
        for edit in edits:
            self.assertGreaterEqual(edit["start"], pos)
            self.assertEqual(text[edit["start"]:edit["end"]], edit["original"])
            restored.extend((text[pos:edit["start"]], edit["original"]))
            pos = edit["end"]
        restored.append(text[pos:])
        self.assertEqual("".join(restored), text)
        self.assertIn("из файла в папке Эволюция, с именем", spoken)
        self.assertIn("Фото подчёркивание для подчёркивание отчета", spoken)

    def test_derived_fields_and_source_are_unchanged(self):
        row = {"sample_id": "s", "text": "версия 2.0", "action_id": "rename", "args": {"version": "9.9"},
               "context": {"objects": [], "selected_object_ids": []}, "group_id": "g", "split": "test"}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            original = source.read_bytes()
            output = Path(tmp) / "derived"
            derive(source, output)
            derived = json.loads((output / "samples.jsonl").read_text(encoding="utf-8"))
            mapping = json.loads((output / "mapping.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(derived, {**row, "text": "версия два точка ноль"})
            self.assertEqual(mapping["rules_version"], RULES_VERSION)
            self.assertEqual(mapping["original_text"], row["text"])
            with self.assertRaises(FileExistsError):
                derive(source, output)


if __name__ == "__main__":
    unittest.main()
