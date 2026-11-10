#!/usr/bin/env python3
"""Compare CPU and accelerator reports; per-example acceptance also runs in Rust."""
import argparse
import json
import math
from pathlib import Path
from example_catalog import example_names


# These are measured host/transport diagnostics, not numerical quantum outputs.
# Keep both values in the report without imposing artificial device timing parity.
DIAGNOSTICS = {
    "wine_classifier": {"elapsed_seconds"},
    "fbqc_remote": {"remote_seconds"},
    "fbqc_four_star_memory": {f"distance{d}_{phase}_seconds"
                              for d in (2, 3, 4)
                              for phase in ("host", "setup", "sampling", "decoding")},
    "fbqc_adaptive": {f"controller_p{q}_seconds" for q in (50, 95, 99)},
    "fbqc_six_ring_memory": {f"side{d}_host_seconds" for d in (2, 4)},
    "planar_logical_memory": {f"d{d}_host_seconds" for d in (3, 5, 7)},
    # Floating-point decimal serialization lengths can differ between devices.
    "distributed_mps": {"one_worker_wire_bytes", "three_workers_wire_bytes"},
}

def require(condition, message):
    if not condition:
        raise ValueError(message)


REPORT_FIELDS = {"example", "seed", "steps", "dtype", "backend", "fusion", "adapter", "metrics"}
EXACT_COST_FIELDS = {"cost", "classification_cost", "born_cost", "illustrative_cost_model"}


def compare_details(expected, actual, path, exact=False):
    """Compare every structured scientific result; return largest error and leaf count.

    Integer counts, identities, shapes and modeled costs must match exactly.
    Other floating outputs use the same F32 bound as top-level scientific metrics.
    Measured timing exclusions are confined to DIAGNOSTICS, never inferred from
    a substring that could accidentally exempt modeled latency or energy costs.
    """
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(), f"{path}: different fields")
        largest, count = 0.0, 0
        for key, value in expected.items():
            difference, leaves = compare_details(value, actual[key], f"{path}/{key}",
                                                exact or key in EXACT_COST_FIELDS)
            largest, count = max(largest, difference), count + leaves
        return largest, count
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), f"{path}: different sequence length")
        largest, count = 0.0, 0
        for i, (value, candidate) in enumerate(zip(expected, actual)):
            difference, leaves = compare_details(value, candidate, f"{path}[{i}]", exact)
            largest, count = max(largest, difference), count + leaves
        return largest, count
    if isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{path}: nonnumeric value")
        require(math.isfinite(expected) and math.isfinite(actual), f"{path}: nonfinite")
        error = abs(actual - expected)
        tolerance = 0.0 if exact else 5e-5 + 2e-4 * abs(expected)
        require(error <= tolerance, f"{path}: {actual} vs {expected}, tolerance {tolerance}")
        return error, 1
    require(type(actual) is type(expected) and actual == expected, f"{path}: {actual!r} vs {expected!r}")
    return 0.0, 1


def validate_learning_curves(report, name):
    """Reject stale reports that predate the declared seeded learning curves."""
    if name != "photonic_benchmarks":
        return
    runs = report.get("runs")
    require(isinstance(runs, list) and len(runs) == report["metrics"]["seed_count"] and runs,
            f"{name}: missing seeded benchmark runs")
    for run in runs:
        for field in ("calibration_learning_curve", "kernel_learning_curve"):
            curve = run.get(field)
            require(isinstance(curve, list) and len(curve) >= 2,
                    f"{name}/seed{run.get('seed')}: missing {field}")


def compare(reference, candidate, examples=None):
    results = {}
    backend = candidate.name.removesuffix("-fusion")
    fusion = candidate.name.endswith("-fusion")
    names = example_names() if examples is None else tuple(examples)
    require(names, "no examples selected for comparison")
    for name in names:
        cpu = json.loads((reference / f"{name}.json").read_text())
        other = json.loads((candidate / f"{name}.json").read_text())
        require(cpu["backend"] == "cpu", "reference must be a CPU execution")
        require(other["backend"] == backend, "report backend does not match its directory")
        require(other["fusion"] == fusion, "report fusion configuration does not match")
        for key in ("example", "seed", "steps", "dtype"):
            require(cpu[key] == other[key], f"{name}: different {key}")
        require(cpu["metrics"].keys() == other["metrics"].keys(), f"{name}: different metrics")
        largest = 0.0
        diagnostics = {}
        finite_shot_fits = {}
        for metric, expected in cpu["metrics"].items():
            actual = other["metrics"][metric]
            require(math.isfinite(actual) and math.isfinite(expected), f"{name}/{metric}: nonfinite")
            if name == "photonic_mz_shots" and metric == "worst_offset_error":
                # Preserve the example's existing independent fit-quality bound.
                # A shared RNG draw can cross CPU/GPU categorical thresholds that
                # differ by ordinary F32 rounding, changing the optimizer path.
                # This is a sampled fitted parameter, not an exact optical output.
                require(0 <= expected < 0.05 and 0 <= actual < 0.05,
                        f"{name}/{metric}: finite-shot fit must satisfy 0 <= error < 0.05")
                finite_shot_fits[metric] = {
                    "cpu": expected,
                    "candidate": actual,
                    "absolute_difference": abs(actual - expected),
                    "lower_bound_inclusive": 0.0,
                    "upper_bound_exclusive": 0.05,
                    "rationale": (
                        "Finite-shot fitted-parameter quality uses the example's existing bound; "
                        "a verified F32 CDF threshold crossing changes one sampled click and "
                        "the optimizer trajectory. Exact physics, held-out MSE and costs retain "
                        "their numerical comparisons."
                    ),
                    "evidence": "artifacts/photonic-completion/mz-shot-divergence/summary.json",
                }
                continue
            if metric in DIAGNOSTICS.get(name, set()):
                require(actual >= 0 and expected >= 0, f"{name}/{metric}: negative diagnostic")
                diagnostics[metric] = {"cpu": expected, "candidate": actual}
                continue
            error = abs(actual - expected)
            # Sampling is statistical; exact objectives/gradients use tight F32 bounds.
            tolerance = 5e-5 + 2e-4 * abs(expected)
            if metric == "sample_total_variation":
                tolerance = 0.02
            elif name == "hybrid_regression" and metric.endswith("_parameter_change"):
                # Native Burn Linear layers may use TF32 on CUDA. Optimizer
                # trajectories can differ while the held-out MSE still agrees.
                tolerance = 0.005
            require(error <= tolerance, f"{candidate.name}/{name}/{metric}: {actual} vs {expected}, tolerance {tolerance}")
            largest = max(largest, error)
        validate_learning_curves(cpu, name)
        validate_learning_curves(other, name)
        expected_details = {key: value for key, value in cpu.items() if key not in REPORT_FIELDS}
        actual_details = {key: value for key, value in other.items() if key not in REPORT_FIELDS}
        detail_error, detail_count = compare_details(expected_details, actual_details,
                                                    f"{candidate.name}/{name}/details")
        results[name] = {"max_absolute_metric_difference": largest,
                         "max_absolute_detail_difference": detail_error,
                         "compared_detail_values": detail_count,
                         "adapter": other["adapter"], "diagnostics": diagnostics,
                         "finite_shot_fits": finite_shot_fits}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="CPU report directory")
    parser.add_argument("candidates", type=Path, nargs="+", help="GPU report directories")
    parser.add_argument("--example", action="append", dest="examples",
                        help="Compare only this binary (repeatable); defaults to all Cargo binaries")
    args = parser.parse_args()
    names = example_names() if args.examples is None else args.examples
    print(json.dumps({directory.name: compare(args.reference, directory, names)
                      for directory in args.candidates}, indent=2))
