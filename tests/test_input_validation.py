import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from input_validation import load_samples


class SampleValidationTests(unittest.TestCase):
    def sample(self):
        return {
            "sample_id": "s1", "group_id": "g1", "split": "test",
            "text": "open selected document", "action_id": "open", "args": {},
            "context": {"objects": [], "selected_object_ids": []},
        }

    def context(self):
        return {"objects": [{"object_id": "doc1", "object_type": "document",
                             "name": "Report", "attributes": {"is_open": True,
                                                                  "nested": [1, {"a": "b"}]}}],
                "selected_object_ids": ["doc1"]}

    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return load_samples(path)

    def test_empty_context_accepted(self):
        row = self.sample()
        self.assertEqual(self.load([row]), [row])

    def test_nonempty_context_preserved(self):
        row = self.sample()
        row["context"] = self.context()
        self.assertEqual(self.load([row]), [row])

    def test_duplicate_object_ids_rejected(self):
        row = self.sample()
        row["context"] = self.context()
        row["context"]["objects"].append(copy.deepcopy(row["context"]["objects"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate object_id"):
            self.load([row])

    def test_missing_selected_object_rejected(self):
        row = self.sample()
        row["context"]["selected_object_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "selected_object_id not found"):
            self.load([row])

    def test_invalid_context_structure_rejected(self):
        invalid = [None, [], {}, {"objects": {}, "selected_object_ids": []},
                   {"objects": [], "selected_object_ids": "doc1"}]
        for change in (None, [], {"object_id": "x"},
                       {"object_id": "x", "object_type": "document", "name": " "},
                       {"object_id": "x", "object_type": "document", "name": "Report", "attributes": []}):
            invalid.append({"objects": [change], "selected_object_ids": []})
        invalid.append({"objects": [], "selected_object_ids": [{}]})
        for context in invalid:
            with self.subTest(context=context):
                row = self.sample()
                row["context"] = context
                with self.assertRaises(ValueError):
                    self.load([row])

    def test_existing_sample_checks_preserved(self):
        for field, value in (("sample_id", ""), ("text", " "), ("group_id", ""),
                             ("split", "other"), ("args", []), ("action_id", 1)):
            with self.subTest(field=field):
                row = self.sample()
                row[field] = value
                with self.assertRaises(ValueError):
                    self.load([row])
        row = self.sample()
        del row["context"]
        with self.assertRaises(ValueError):
            self.load([row])
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            self.load([self.sample(), self.sample()])
        with self.assertRaisesRegex(ValueError, "No samples"):
            self.load([])

    def test_group_labels_context_and_split_consistent(self):
        first = self.sample()
        first["context"] = self.context()
        second = copy.deepcopy(first)
        second.update(sample_id="s2", text="open this document")
        self.assertEqual(self.load([first, second]), [first, second])
        for field, value in (("action_id", "delete"), ("args", {"name": "Other"}),
                             ("context", {"objects": [], "selected_object_ids": []}),
                             ("split", "train")):
            with self.subTest(field=field):
                changed = copy.deepcopy(second)
                changed[field] = value
                with self.assertRaisesRegex(ValueError, "Group has inconsistent"):
                    self.load([first, changed])


if __name__ == "__main__":
    unittest.main()
