#!/usr/bin/env python3
"""Compare CPU and accelerator reports; per-example acceptance also runs in Rust."""
import argparse
import json
import math
from pathlib import Path
from example_catalog import example_names


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="CPU report directory")
    parser.add_argument("candidates", type=Path, nargs="+", help="GPU report directories")
    parser.add_argument("--example", action="append", dest="examples",
                        help="Compare only this binary (repeatable); defaults to all Cargo binaries")
    args = parser.parse_args()
    names = example_names() if args.examples is None else args.examples
    print(json.dumps({directory.name: compare(args.reference, directory, names)
                      for directory in args.candidates}, indent=2))
