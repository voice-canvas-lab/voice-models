import contextlib
import copy
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from spoken_text_text_only import derive, transform_text


class TextOnlySpeakingTests(unittest.TestCase):
    def test_ordinary_text_and_punctuation_stay(self):
        text = "Открой «Документ»: проверь, сохрани — пожалуйста."
        result = transform_text(text)
        self.assertEqual(result['spoken_text'], text)
        self.assertTrue(result['tts_ready'])
        self.assertEqual(result['edits'], [])

    def test_date_values_all_retained_without_calendar_guess(self):
        result = transform_text("Список_2023-10-06_17.40.29")
        self.assertEqual(result['spoken_text'], "Список две тысячи двадцать три десять ноль шесть семнадцать. сорок. двадцать девять")
        components = [part for edit in result['edits'] for part in edit.get('numeric_components', [])]
        self.assertEqual(components, ['2023', '10', '06', '17', '40', '29'])
        self.assertNotIn('октября', result['spoken_text'])
        for word in ('дефис', 'подчёркивание', 'заглавная', 'латинская'):
            self.assertNotIn(word, result['spoken_text'])

    def test_no_typo_or_year_correction(self):
        result = transform_text("Список_202023-10-06")
        self.assertEqual(result['spoken_text'], "Список двести две тысячи двадцать три десять ноль шесть")
        self.assertEqual([p for e in result['edits'] for p in e.get('numeric_components', [])], ['202023', '10', '06'])

    def test_version_and_leading_zeros(self):
        self.assertEqual(transform_text("версия 2.0")['spoken_text'], "версия два точка ноль")
        self.assertEqual(transform_text("Документация_версия_3.01")['spoken_text'], "Документация версия три точка ноль один")
        self.assertEqual(transform_text("документ_000015")['spoken_text'], "документ ноль ноль ноль ноль один пять")
        self.assertTrue(transform_text("документ_000015")['tts_ready'])

    def test_latin_abbreviation_and_id_not_expanded(self):
        for text, literal in (("note_000001", "note"), ("3D моделирование", "D"), ("ООО", "ООО"), ("мастерpieces", "мастерpieces")):
            with self.subTest(text=text):
                result = transform_text(text)
                self.assertIn(literal, result['spoken_text'])
                self.assertTrue(result['pronunciation_flags'])
                self.assertFalse(result['tts_ready'])
                self.assertNotIn('эн оу', result['spoken_text'])
                self.assertNotIn('трёхмер', result['spoken_text'])

    def test_not_a_character_reversible_policy(self):
        self.assertEqual(transform_text("Имя_06")['spoken_text'], transform_text("Имя-06")['spoken_text'])

    def test_unknown_dotted_or_colon_numbers_flagged_not_fixed(self):
        result = transform_text("17:4029")
        self.assertTrue(result['needs_pronunciation_review'])
        self.assertIn('четыре тысячи двадцать девять', result['spoken_text'])
        self.assertNotIn('сорок минут', result['spoken_text'])
        self.assertTrue(transform_text("число 3.14")['needs_pronunciation_review'])
        self.assertTrue(transform_text("-5")['needs_pronunciation_review'])
        self.assertIn('-', transform_text("-5")['spoken_text'])

    def test_metadata_does_not_influence_transformation(self):
        self.assertEqual(list(inspect.signature(transform_text).parameters), ['text'])
        rows = [{'sample_id': 'a', 'text': 'Дневник мыслей 4', 'args': {'name': 'Иное.9'}, 'context': {'objects': [], 'selected_object_ids': []}, 'action_id': 'create', 'group_id': 'g1', 'split': 'test'},
                {'sample_id': 'b', 'text': 'Дневник мыслей 4', 'args': {'name': 'Ещё другое'}, 'context': {'objects': [], 'selected_object_ids': []}, 'action_id': None, 'group_id': 'g2', 'split': 'train'}]
        original = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.jsonl'
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            data = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                derive(source, Path(tmp) / 'out')
            derived = [json.loads(l) for l in (Path(tmp) / 'out/samples.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(derived[0]['text'], derived[1]['text'])
            self.assertEqual(derived[0]['text'], 'Дневник мыслей четыре')
            for before, after in zip(rows, derived):
                self.assertEqual({k: v for k, v in before.items() if k != 'text'}, {k: v for k, v in after.items() if k != 'text'})
            self.assertEqual(source.read_bytes(), data)
            self.assertEqual(rows, original)


if __name__ == '__main__':
    unittest.main()
