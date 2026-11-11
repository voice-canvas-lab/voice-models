import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_conditions import group_metrics
from evaluate import score
from run_pipeline import build_stages, execute_stages, preflight

REPO = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def fixture(self, root):
        config = json.loads((REPO / "configs/pipeline-pilot.json").read_text(encoding="utf-8"))
        (root / "samples.jsonl").write_bytes((REPO / "examples/pilot/samples.jsonl").read_bytes())
        config.update(samples="samples.jsonl", output="results",
                      interpreters={"main": "python-main", "gigaam": "python-gigaam"})
        for name in config["interpreters"].values():
            (root / name).touch()
        return config

    def probe(self, path, packages):
        return {"prefix": str(path), "packages": {p: "test" for p in packages}}

    def test_preflight_and_separate_interpreters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            path = root / "run.json"
            path.write_text(json.dumps(config))
            _, resolved, samples, _ = preflight(path, root, self.probe)
            self.assertEqual(len(samples), 4)
            self.assertFalse((root / "results").exists())
            stages = dict(build_stages(resolved, root / "results", root))
            self.assertEqual(stages["gigaam"][0], resolved["interpreters"]["gigaam"])
            self.assertEqual(stages["whisper"][0], resolved["interpreters"]["main"])

    def test_invalid_inputs_fail_before_environment_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            path = root / "run.json"
            def forbidden(*args):
                self.fail("Environment probe/model execution must not occur")
            for mutate in (
                lambda c: c.update(samples="missing.jsonl"),
                lambda c: c.update(output="."),
                lambda c: c["interpreters"].update(gigaam="missing-python"),
                lambda c: c["ASR"]["whisper"]["decoding"].update(initial_prompt="reference"),
                lambda c: c["ASR"]["gigaam"].update(model="other-model"),
                lambda c: c["ASR"]["whisper"].update(revision="main"),
            ):
                case = copy.deepcopy(config)
                mutate(case)
                path.write_text(json.dumps(case))
                with self.assertRaises(ValueError):
                    preflight(path, root, forbidden)
            path.write_text(json.dumps(config))
            (root / "samples.jsonl").write_text('{"sample_id":"bad"}\n')
            with self.assertRaises(ValueError):
                preflight(path, root, forbidden)

    def test_failure_stops_and_preserves_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def runner(command, log, cwd):
                calls.append(command)
                return 7
            code = execute_stages(root, [("first", ["one"]), ("second", ["two"])], {}, runner, root)
            saved = json.loads((root / "status.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(calls, [["one"]])
            self.assertEqual(saved["status"], "failed")
            self.assertEqual([s["status"] for s in saved["stages"]], ["failed", "skipped"])
            self.assertEqual(saved["stages"][0]["exit_code"], 7)

    def test_missing_artifact_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = execute_stages(root, [("whisper", ["fake"])], {}, lambda *args: 0, root)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads((root / "status.json").read_text())["status"], "failed")

    def test_group_counts_and_corpus_weighting(self):
        samples = {"s1": {"action_id": "rename", "group_id": "g1", "split": "train"},
                   "s2": {"action_id": "open", "group_id": "g2", "split": "test"}}
        manifest, records = {}, {}
        for aid, sid, ref, hyp in (("a", "s1", "a", "b"), ("b", "s1", "a", "b"),
                                   ("c", "s2", "a b c d", "a b c d")):
            manifest[aid] = {"sample_id": sid, "voice": "xenia", "augmentations": []}
            records[aid] = {**score(ref, hyp), "sample_id": sid, "elapsed_sec": 1, "duration_sec": 2}
        groups = group_metrics(records, manifest, samples)
        self.assertEqual(groups["all"]["wer"], 2 / 6)
        self.assertEqual(groups["all"]["unique_samples"], 2)
        self.assertEqual(groups["all"]["unique_group_ids"], 2)
        self.assertEqual(groups["action_id=rename"]["records"], 2)
        self.assertEqual(groups["action_id=rename"]["unique_samples"], 1)
        self.assertEqual(groups["action_id=open"]["wer"], 0)
        self.assertEqual(groups["split=train"]["records"], 2)
        self.assertEqual(groups["split=train"]["wer"], 1)
        self.assertEqual(groups["split=test"]["records"], 1)
        self.assertEqual(groups["split=test"]["wer"], 0)


if __name__ == "__main__":
    unittest.main()
