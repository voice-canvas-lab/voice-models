import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import read_index
from evaluate_fragments import main as evaluate_main
from fragment_metrics import load_annotations, report_lines, score_fragments, summarize
from input_validation import load_samples
from run_pipeline import build_stages, main as pipeline_main, preflight

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples/fragment-pilot"


class FragmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = read_index(EXAMPLES / "samples.jsonl", "sample_id")
        cls.annotations = load_annotations(EXAMPLES / "annotations.jsonl", cls.samples)

    def scored(self, sid, prediction):
        return {row["fragment_id"]: row for row in score_fragments(self.samples[sid]["text"], prediction, self.annotations[sid])}

    def test_manual_dataset_and_exact_predictions(self):
        self.assertEqual(len(load_samples(EXAMPLES / "samples.jsonl")), 6)
        for sid, sample in self.samples.items():
            with self.subTest(sample=sid):
                self.assertTrue(all(row["preserved"] for row in self.scored(sid, sample["text"]).values()))

    def test_lost_negation_despite_other_ne(self):
        rows = self.scored("fragment_003", "Переименовывай папку Архив в Не трогать.")
        self.assertFalse(rows["command_negation"]["preserved"])
        self.assertTrue(rows["new_name"]["preserved"])
        rows = self.scored("fragment_006", "Удаляй папку Архив и не открывай папку Черновики.")
        self.assertFalse(rows["delete_negation"]["preserved"])
        self.assertTrue(rows["open_negation"]["preserved"])

    def test_incorrect_number_and_explicit_variant(self):
        bad = self.scored("fragment_002", "Создай 22 заметки в папке Отчеты.")
        self.assertFalse(bad["count"]["preserved"])
        good = self.scored("fragment_002", "Создай 21 заметку в папке Отчеты.")
        self.assertTrue(good["count"]["preserved"])
        fragment = {"fragment_id": "n", "category": "number", "start_token": 1, "end_token": 2,
                    "reference": "21", "variants": ["двадцать одну"]}
        result = score_fragments("Создай 21 заметку.", "Создай двадцать одну заметку.", [fragment])
        self.assertTrue(result[0]["preserved"])

    def test_wrong_name_and_word_substrings(self):
        rows = self.scored("fragment_001", "Переименуй папку Черновики в Документы.")
        self.assertFalse(rows["new_name"]["preserved"])
        rows = self.scored("fragment_005", "Сохрани заметку Планета, не меняя название папки Архивный.")
        self.assertFalse(rows["note"]["preserved"])
        self.assertFalse(rows["folder"]["preserved"])
        rows = self.scored("fragment_003", "Непереименовывай папку Архив в Не трогать.")
        self.assertFalse(rows["command_negation"]["preserved"])

    def test_self_correction_requires_initial_spoken_value(self):
        rows = self.scored("fragment_004", "Создай тринадцать заметок в папке Идеи.")
        self.assertFalse(rows["spoken_initial_count"]["preserved"])
        self.assertFalse(rows["correction_marker"]["preserved"])
        self.assertTrue(rows["spoken_corrected_count"]["preserved"])

    def test_no_annotations_not_evaluated_and_counts(self):
        empty = summarize([{"fragments": []}])
        self.assertTrue(all(v["retention"] is None and v["status"] == "not_evaluated" for v in empty["categories"].values()))
        self.assertIn("не оценено", "\n".join(report_lines(empty)))
        details = [{"category": "name", "preserved": True}, {"category": "name", "preserved": False}]
        value = summarize([{"fragments": details}])["categories"]["name"]
        self.assertEqual((value["checked"], value["preserved"], value["retention"]), (2, 1, 0.5))

    def test_invalid_annotation_reference_and_duplicate_ids(self):
        row = {"sample_id": "fragment_001", "fragments": [{"fragment_id": "x", "category": "name", "start_token": 2, "end_token": 3, "reference": "Неверно"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_annotations(path, self.samples)
            row["fragments"][0]["reference"] = "Черновики"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_annotations(path, self.samples)

    def test_evaluation_cli_with_synthetic_predictions_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, results, output = root / "data", root / "asr", root / "evaluation"
            dataset.mkdir()
            results.mkdir()
            sample = self.samples["fragment_003"]
            (dataset / "samples.jsonl").write_text(json.dumps(sample), encoding="utf-8")
            (dataset / "audio_manifest.jsonl").write_text(json.dumps({"audio_id": "a", "sample_id": sample["sample_id"], "path": "unused.wav"}), encoding="utf-8")
            (results / "predictions.jsonl").write_text(json.dumps({"audio_id": "a", "sample_id": sample["sample_id"], "text": "Переименовывай папку Архив в Не трогать."}), encoding="utf-8")
            arguments = ["evaluate_fragments", "--dataset", str(dataset), "--results", str(results), "--annotations", str(EXAMPLES / "annotations.jsonl"), "--output", str(output)]
            # A separate sidecar for this fixture, never inferred from args.
            sidecar = root / "annotations.jsonl"
            sidecar.write_text(json.dumps({"sample_id": sample["sample_id"], "fragments": self.annotations[sample["sample_id"]]}), encoding="utf-8")
            arguments[arguments.index("--annotations") + 1] = str(sidecar)
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                evaluate_main()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["categories"]["negation"]["preserved"], 0)
            self.assertIsNone(summary["categories"]["number"]["retention"])
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("command_negation", report)
            self.assertIn("не оценено", report)
            without_annotations = ["evaluate_fragments", "--dataset", str(dataset), "--results", str(results), "--output", str(root / "unannotated")]
            with patch.object(sys, "argv", without_annotations), contextlib.redirect_stdout(io.StringIO()):
                evaluate_main()
            unannotated = json.loads((root / "unannotated/summary.json").read_text(encoding="utf-8"))
            self.assertTrue(all(v["retention"] is None for v in unannotated["categories"].values()))

    def test_invalid_annotations_rejected_by_preflight_before_probes(self):
        config = json.loads((REPO / "configs/pipeline-pilot.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "annotations.jsonl"
            annotation.write_text(json.dumps({"sample_id": "unknown", "fragments": []}), encoding="utf-8")
            config.update(samples=str(EXAMPLES / "samples.jsonl"), output="run", annotations=str(annotation))
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown or duplicate sample_id"):
                preflight(path, root, lambda *args: self.fail("Environment probe must not run"))

    def test_optional_pipeline_snapshot_and_no_annotations_to_asr(self):
        config = json.loads((REPO / "configs/pipeline-pilot.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            resolved = {**config, "output": str(root), "samples": str(EXAMPLES / "samples.jsonl"), "annotations": str(EXAMPLES / "annotations.jsonl")}
            config = {**config, "annotations": "examples/fragment-pilot/annotations.jsonl"}
            stages = dict(build_stages(resolved, root))
            self.assertIn("fragments_whisper", stages)
            self.assertIn("fragments_gigaam", stages)
            for name in ("tts_checks_and_noise", "whisper", "gigaam"):
                self.assertNotIn("--annotations", stages[name])
                self.assertNotIn(str(EXAMPLES / "annotations.jsonl"), stages[name])
                self.assertTrue(any(str(arg).endswith("config.models.json") for arg in stages[name]))
            with patch.object(sys, "argv", ["pipeline", "--config", "unused"]), \
                    patch("run_pipeline.preflight", return_value=(config, resolved, list(self.samples.values()), {})), \
                    patch("run_pipeline.execute_stages", return_value=0), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    pipeline_main()
            self.assertEqual(stopped.exception.code, 0)
            model_config = json.loads((root / "config.models.json").read_text(encoding="utf-8"))
            self.assertNotIn("annotations", model_config)
            self.assertNotIn("samples", model_config)
            self.assertEqual((root / "annotations.jsonl").read_bytes(), (EXAMPLES / "annotations.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
