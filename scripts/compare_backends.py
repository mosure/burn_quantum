#!/usr/bin/env python3
"""Compare CPU and accelerator reports; per-example acceptance also runs in Rust."""
import json
import math
from pathlib import Path
import sys

EXAMPLES = (
    "vqe", "qaoa_maxcut", "variational_classifier", "quantum_kernel",
    "born_machine", "hybrid_regression", "qudit_classifier",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(reference, candidate):
    results = {}
    backend = candidate.name.removesuffix("-fusion")
    fusion = candidate.name.endswith("-fusion")
    for name in EXAMPLES:
        cpu = json.loads((reference / f"{name}.json").read_text())
        other = json.loads((candidate / f"{name}.json").read_text())
        require(cpu["backend"] == "cpu", "reference must be a CPU execution")
        require(other["backend"] == backend, "report backend does not match its directory")
        require(other["fusion"] == fusion, "report fusion configuration does not match")
        for key in ("example", "seed", "steps", "dtype"):
            require(cpu[key] == other[key], f"{name}: different {key}")
        require(cpu["metrics"].keys() == other["metrics"].keys(), f"{name}: different metrics")
        largest = 0.0
        for metric, expected in cpu["metrics"].items():
            actual = other["metrics"][metric]
            require(math.isfinite(actual) and math.isfinite(expected), f"{name}/{metric}: nonfinite")
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
        results[name] = {"max_absolute_metric_difference": largest, "adapter": other["adapter"]}
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python3 scripts/compare_backends.py CPU_REPORT_DIR GPU_REPORT_DIR [...]")
    print(json.dumps({path.name: compare(Path(sys.argv[1]), path)
                      for path in map(Path, sys.argv[2:])}, indent=2))
