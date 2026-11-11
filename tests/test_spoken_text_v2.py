import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from spoken_text_v2_0_0 import RULES_VERSION, analyze, derive, transform


class SafeSpokenTextTests(unittest.TestCase):
    def test_ordinary_punctuation_unchanged(self):
        for text in ('Открой «План-график»: проверь, сохрани — пожалуйста.',
                     'Открой документ (черновик); затем — сохрани.',
                     'Подборка четыре: Экология. Спасибо!'):
            self.assertEqual(transform(text), (text, [], []))
        out, _, flags = transform('Подборка 4: Экология — итоги.')
        self.assertEqual(out, 'Подборка четыре: Экология — итоги.')
        self.assertFalse(flags)

    def test_structured_date_time_no_separator_names(self):
        cases = {
            'Дата: 2023-10-06; время — 17:40:29.': 'Дата: шестое октября две тысячи двадцать третьего года; время — семнадцать часов сорок минут двадцать девять секунд.',
            '2023-10-06_17.40.29': 'шестое октября две тысячи двадцать третьего года, семнадцать часов сорок минут двадцать девять секунд',
        }
        for text, expected in cases.items():
            result = analyze(text)
            self.assertEqual(result['spoken_text'], expected)
            self.assertTrue(result['auto_tts_eligible'])
            for name in ('дефис', 'подчёркивание', 'двоеточие', 'точка'):
                self.assertNotIn(name, result['spoken_text'])

    def test_explicit_version_preserves_meaningful_dots_and_zeros(self):
        self.assertEqual(transform('версия 2.0')[0], 'версия два точка ноль')
        self.assertEqual(transform('версия 2.01')[0], 'версия два точка ноль один')

    def test_ids_and_ambiguous_names_unchanged_and_blocked(self):
        for text in ('Открой note_000001', 'документ_000015', 'Список_2023-10-06_17.40.29',
                     'Список_2023_10_06_17_40_29', 'Дневник мыслей.4', '3D моделирование',
                     'Документация_версия_3.1', 'План-4'):
            with self.subTest(text=text):
                result = analyze(text)
                self.assertEqual(result['spoken_text'], text)
                self.assertFalse(result['auto_tts_eligible'])
                self.assertTrue(result['review_reasons'])
                self.assertEqual(result['edits'], [])
                self.assertTrue(result['decisions'])

    def test_invalid_values_not_guessed_or_dictated(self):
        for text in ('Дата 2023-02-29', 'время 17:4029', 'время 25:61:00', 'значение 0015', 'число 2.0'):
            result = analyze(text)
            self.assertEqual(result['spoken_text'], text)
            self.assertFalse(result['auto_tts_eligible'])

    def test_ambiguity_blocks_partial_conversion(self):
        text = 'Версия 2.0; открой note_000001'
        result = analyze(text)
        self.assertEqual(result['spoken_text'], text)
        self.assertEqual(result['edits'], [])
        self.assertTrue(result['proposed_safe_edits_not_applied'])

    def test_derive_keeps_all_rows_and_separates_queue(self):
        rows = [{'sample_id': 's' + str(i), 'text': text, 'context': {'objects': [], 'selected_object_ids': []},
                 'args': {'keep': '0001'}, 'action_id': None, 'group_id': 'g' + str(i), 'split': 'test'}
                for i, text in enumerate(('версия 2.0', 'note_000001'))]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.jsonl'
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            before = source.read_bytes()
            output = Path(tmp) / 'new'
            with contextlib.redirect_stdout(io.StringIO()):
                derive(source, output)
            actual = [json.loads(l) for l in (output / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(actual), 2)
            self.assertEqual(source.read_bytes(), before)
            for old, new in zip(rows, actual):
                self.assertEqual({k: v for k, v in old.items() if k != 'text'}, {k: v for k, v in new.items() if k != 'text'})
            safe = [json.loads(l) for l in (output / 'safe_samples.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['sample_id'] for r in safe], ['s0'])
            queue = [json.loads(l) for l in (output / 'technical_test_queue.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(queue[0]['sample_id'], 's1')
            self.assertFalse(queue[0]['automatic_dictation'])
            self.assertEqual(json.loads((output / 'rules.json').read_text(encoding='utf-8'))['rules_version'], RULES_VERSION)


if __name__ == '__main__':
    unittest.main()
