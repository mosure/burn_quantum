"""Mutation checks for the narrowly scoped finite-shot fit comparison."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from compare_backends import compare


class FiniteShotFitComparison(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cpu_dir = Path(self.directory.name) / "cpu"
        self.gpu_dir = Path(self.directory.name) / "wgpu"
        self.cpu_dir.mkdir()
        self.gpu_dir.mkdir()
        self.cpu = {
            "example": "photonic_mz_shots", "seed": 7, "steps": 80,
            "dtype": "f32", "backend": "cpu", "fusion": False, "adapter": "Flex CPU",
            "metrics": {
                "worst_offset_error": 0.008868493077869744,
                "worst_test_mse": 6.215071604180634e-6,
                "uncalibrated_baseline_mse": 0.03139843404777426,
                "total_physical_attempts": 2949120.0,
            },
        }
        self.gpu = copy.deepcopy(self.cpu)
        self.gpu.update(backend="wgpu", adapter="test GPU")
        self.gpu["metrics"]["worst_offset_error"] = 0.009037666752629625
        self.gpu["metrics"]["worst_test_mse"] = 6.454434295631575e-6

    def compare_reports(self):
        name = self.cpu["example"]
        (self.cpu_dir / f"{name}.json").write_text(json.dumps(self.cpu))
        (self.gpu_dir / f"{name}.json").write_text(json.dumps(self.gpu))
        return compare(self.cpu_dir, self.gpu_dir, [name])[name]

    def test_observed_crossing_retains_quality_and_separate_provenance(self):
        result = self.compare_reports()
        fit = result["finite_shot_fits"]["worst_offset_error"]
        self.assertEqual(fit["cpu"], self.cpu["metrics"]["worst_offset_error"])
        self.assertEqual(fit["candidate"], self.gpu["metrics"]["worst_offset_error"])
        self.assertAlmostEqual(fit["absolute_difference"], 0.0001691736747598811)
        self.assertEqual(fit["upper_bound_exclusive"], 0.05)
        self.assertIn("threshold crossing", fit["rationale"])
        self.assertTrue(fit["evidence"].endswith("mz-shot-divergence/summary.json"))
        self.assertNotIn("worst_offset_error", result["diagnostics"])
        self.assertAlmostEqual(result["max_absolute_metric_difference"],
                               abs(self.cpu["metrics"]["worst_test_mse"]
                                   - self.gpu["metrics"]["worst_test_mse"]))

    def test_both_backends_enforce_finite_nonnegative_strict_fit_bound(self):
        for report in (self.cpu, self.gpu):
            original = report["metrics"]["worst_offset_error"]
            for invalid in (-1e-6, 0.05, 0.051, float("inf"), float("nan")):
                with self.subTest(backend=report["backend"], invalid=invalid):
                    report["metrics"]["worst_offset_error"] = invalid
                    with self.assertRaises(ValueError):
                        self.compare_reports()
            report["metrics"]["worst_offset_error"] = original

    def test_heldout_reference_and_physical_cost_checks_are_unchanged(self):
        for metric, deviation in (("worst_test_mse", 1e-3),
                                  ("uncalibrated_baseline_mse", 1e-3),
                                  ("total_physical_attempts", 1000)):
            with self.subTest(metric=metric):
                original = self.gpu["metrics"][metric]
                self.gpu["metrics"][metric] += deviation
                with self.assertRaises(ValueError):
                    self.compare_reports()
                self.gpu["metrics"][metric] = original

    def test_structured_preparation_cost_still_rejects_one_attempt(self):
        self.cpu["cost"] = {"attempts": 2949120, "modeled_joules": 2.94912e-6}
        self.gpu["cost"] = copy.deepcopy(self.cpu["cost"])
        self.gpu["cost"]["attempts"] += 1
        with self.assertRaises(ValueError):
            self.compare_reports()
        self.gpu["cost"]["attempts"] -= 1
        self.gpu["cost"]["modeled_joules"] += 1e-12
        with self.assertRaises(ValueError):
            self.compare_reports()

    def test_rule_does_not_apply_to_another_example_or_metric(self):
        self.cpu["example"] = self.gpu["example"] = "another_example"
        with self.assertRaises(ValueError):
            self.compare_reports()
        self.cpu["example"] = self.gpu["example"] = "photonic_mz_shots"
        for report in (self.cpu, self.gpu):
            report["metrics"]["another_offset_error"] = report["metrics"].pop("worst_offset_error")
        with self.assertRaises(ValueError):
            self.compare_reports()


if __name__ == "__main__":
    unittest.main()
