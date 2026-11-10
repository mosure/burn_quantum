#!/usr/bin/env python3
"""Independently check fixed-shot qualification with SciPy (no Rust import).

Run with a Python environment containing SciPy, for example:
  /tmp/burn-quantum-psiqdk312/bin/python scripts/check_threshold_intervals.py

All grids and sample counts come from the recorded predeclared protocols. This
checker does not generate samples, pool studies, change a stopping rule or fit
an asymptotic threshold. Its JSON includes input hashes and all recomputed
finite-size decisions, including unresolved results.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import scipy
from scipy.stats import beta, binom


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{name}: expected nonnegative integer")
    return value


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configurations(folder):
    common = read_json(folder / "protocol.json")
    names = {
        "six_ring_iid_outcome_erasure": ("six-ring-qualification", "sides", "erasure_probabilities"),
        "six_ring_shor_boosted_optical_loss": ("optical-qualification", "sides", "photon_loss_probabilities"),
        "planar_phenomenological": ("planar-qualification", "distances", "probabilities"),
    }
    result = []
    for sweep in common["sweeps"]:
        name, size, rate = names[sweep["name"]]
        result.append((name, {
            "protocol": "protocol.json", "sizes": sweep[size], "rates": sweep[rate],
            "groups": ["all"], "shots": sweep["shots_per_cell"], "seed": sweep["seed"],
            **common["statistics"],
        }))
    for name, protocol, groups in [
        ("planar-confirmation", "planar-confirmation-protocol.json", None),
        ("surgery-threshold", "surgery-threshold-protocol.json", "bases"),
        ("surface-threshold", "surface-threshold-protocol.json", "instruments"),
        ("surface-confirmation", "surface-confirmation-protocol.json", "instruments"),
    ]:
        p = read_json(folder / protocol)
        result.append((name, {
            "protocol": protocol, "sizes": p["distances"],
            "rates": p.get("rates", p.get("probabilities")),
            "groups": p[groups] if groups else ["all"], "shots": p["shots_per_cell"],
            "seed": p.get("base_seed", p.get("seed")),
            **(p["statistics"] if "statistics" in p else {
                "family_alpha": p["family_alpha"], "planned_intervals": p["planned_intervals"],
            }),
        }))
    return result


def exact_interval(events, trials, family_alpha, family):
    tail = family_alpha / (2 * family)
    lower = 0.0 if events == 0 else float(beta.ppf(tail, events, trials - events + 1))
    upper = 1.0 if events == trials else float(beta.isf(tail, events + 1, trials - events))
    # Independently check the defining binomial tail equations, not only beta endpoints.
    errors = []
    if events:
        errors.append(abs(float(binom.sf(events - 1, trials, lower)) - tail))
    if events < trials:
        errors.append(abs(float(binom.cdf(events, trials, upper)) - tail))
    return {"lower": lower, "upper": upper}, max(errors, default=0.0)


def trend(points):
    """Classify every adjacent pair of independently recomputed intervals."""
    directions = []
    diagnostics = []
    for left, right in itertools.pairwise(points):
        direction = (-1 if right[1]["upper"] < left[1]["lower"] else
                     1 if right[1]["lower"] > left[1]["upper"] else 0)
        directions.append(direction)
        diagnostics.append({
            "smaller_size": left[0], "larger_size": right[0],
            "direction": {-1: "decrease", 0: "overlap", 1: "increase"}[direction],
            "decrease_separation": left[1]["lower"] - right[1]["upper"],
            "increase_separation": right[1]["lower"] - left[1]["upper"],
        })
    if all(x == -1 for x in directions):
        decision = "Suppressed"
    elif all(x == 1 for x in directions):
        decision = "Increased"
    elif -1 in directions and 1 in directions:
        decision = "NonMonotone"
    else:
        decision = "Unresolved"
    return decision, diagnostics


def bracket(decisions):
    if any(t == "NonMonotone" for _, t in decisions):
        return None
    decreased = [p for p, t in decisions if t == "Suppressed"]
    increased = [p for p, t in decisions if t == "Increased"]
    if decreased and increased and max(decreased) < min(increased):
        return {"lower": max(decreased), "upper": min(increased)}
    return None


def check_run(folder, name, plan):
    path = folder / f"{name}.json"
    study = read_json(path)
    cells = study["cells"]
    sizes, rates, groups = plan["sizes"], plan["rates"], plan["groups"]
    require(sizes == sorted(set(sizes)) and rates == sorted(set(rates)), f"{name}: protocol order")
    expected = set(itertools.product(groups, sizes, rates))
    observed = set()
    intervals = {}
    maximum_error = maximum_padding_error = maximum_tail_error = 0.0
    all_valid = True
    totals = {"attempts": 0, "logical_failures": 0, "decoder_limits": 0,
              "inconsistent": 0, "ambiguous_erasures": 0}
    for i, cell in enumerate(cells):
        label = f"{name} cell{i}"
        group = cell.get("instrument", cell.get("basis", "all"))
        size = cell.get("distance", cell.get("size"))
        noise = cell.get("probability", cell.get("noise"))
        if isinstance(noise, dict):
            require(noise["data_x"] == noise["data_z"] == noise["measurement"], f"{label}: noise model")
            require(noise["data_erasure"] == noise["measurement_erasure"] == 0, f"{label}: unexpected erasure")
            noise = noise["data_x"]
        key = (group, size, noise)
        require(key in expected and key not in observed, f"{label}: duplicate or unplanned cell {key}")
        observed.add(key)
        require(cell["start"] == 0, f"{label}: unexpected shot range")
        counts = cell["study"]["counts"] if "study" in cell else cell["counts"]
        for field, value in counts.items():
            if field != "logical_rank_histogram":
                integer(value, f"{label} {field}")
        trials = counts["attempts"]
        require(trials == plan["shots"], f"{label}: changed sample count")
        # Erasure ambiguity is not counted a second time: the recorded sampled
        # logical-error event already includes the failed erased recoveries.
        failures = sum(counts[field] for field in ["logical_failures", "decoder_limits", "inconsistent"])
        require(failures <= trials, f"{label}: overlapping/invalid unsuccessful counts")
        rank_histogram = counts["logical_rank_histogram"]
        require(all(integer(n, f"{label} rank count") >= 0 for n in rank_histogram), f"{label}: rank counts")
        require(sum(rank_histogram) + counts["decoder_limits"] + counts["inconsistent"] == trials,
                f"{label}: histogram/denominator accounting")
        require(sum(rank_histogram[1:]) == counts["ambiguous_erasures"], f"{label}: ambiguity accounting")
        if "conditional_expected_failures" in cell:
            conditional = sum(n * (1 - 2 ** -rank) for rank, n in enumerate(rank_histogram))
            require(math.isclose(conditional, cell["conditional_expected_failures"], abs_tol=1e-9),
                    f"{label}: conditional erasure statistic")
        if "study" in cell:
            marginal = cell["study"]["logical_failures"]
            require(all(integer(n, f"{label} marginal") <= counts["logical_failures"] for n in marginal),
                    f"{label}: marginal exceeds union")
            require(counts["logical_failures"] <= sum(marginal), f"{label}: union exceeds marginal sum")
        for field in totals:
            totals[field] += counts[field]
        all_valid &= counts["decoder_limits"] == counts["inconsistent"] == 0
        oracle, tail_error = exact_interval(failures, trials, plan["family_alpha"], plan["planned_intervals"])
        maximum_tail_error = max(maximum_tail_error, tail_error)
        require(tail_error < 1e-11, f"{label}: SciPy binomial tail equation failed")
        recorded = cell["failure_interval"]
        padded = {"lower": max(0.0, oracle["lower"] - 1e-11),
                  "upper": min(1.0, oracle["upper"] + 1e-11)}
        for endpoint in ["lower", "upper"]:
            error = abs(recorded[endpoint] - oracle[endpoint])
            padding_error = abs(recorded[endpoint] - padded[endpoint])
            maximum_error = max(maximum_error, error)
            maximum_padding_error = max(maximum_padding_error, padding_error)
            require(error < 2e-11 and padding_error < 1e-12, f"{label}: wrong {endpoint} interval")
        intervals[key] = oracle
    require(observed == expected, f"{name}: missing declared cells")
    metrics = study["metrics"]
    for field, value in {"cells": len(expected), "shots_per_cell": plan["shots"],
                         "attempts": totals["attempts"], "family_alpha": plan["family_alpha"],
                         "planned_intervals": plan["planned_intervals"]}.items():
        require(metrics[field] == value, f"{name}: incorrect metric {field}")
    require(study["qualification"] is True and study["seed"] == plan["seed"], f"{name}: wrong protocol identity")
    if "qualification_valid" in study:
        require(study["qualification_valid"] == all_valid, f"{name}: incorrect qualification validity")
    recomputed_trends, recomputed_brackets, adjacent = {}, {}, {}
    for group in groups:
        decisions, pair_results = [], []
        for p in rates:
            decision, pairs = trend([(size, intervals[group, size, p]) for size in sizes])
            decisions.append([p, decision if all_valid else "Unresolved"])
            pair_results.append({"noise": p, "adjacent_comparisons": pairs})
        calculated = bracket(decisions) if all_valid else None
        recorded_trends = study["finite_size_trends"]
        reported = recorded_trends[group] if isinstance(recorded_trends, dict) else recorded_trends
        reported_bracket = (study["finite_size_brackets"][group] if "finite_size_brackets" in study
                            else study["finite_size_bracket"])
        require(decisions == reported, f"{name}/{group}: incorrect finite-size trends")
        require(calculated == reported_bracket, f"{name}/{group}: incorrect finite-size bracket")
        recomputed_trends[group] = decisions
        recomputed_brackets[group] = calculated
        adjacent[group] = pair_results
    return {
        "cells": len(cells), "sha256": sha256(path), "protocol": plan["protocol"],
        "protocol_sha256": sha256(folder / plan["protocol"]),
        "family_alpha": plan["family_alpha"], "planned_intervals": plan["planned_intervals"],
        "shots_per_cell": plan["shots"], "totals": totals, "qualification_valid": all_valid,
        "maximum_endpoint_error": maximum_error, "maximum_error_after_declared_padding": maximum_padding_error,
        "maximum_binomial_tail_error": maximum_tail_error,
        "finite_size_trends": recomputed_trends, "finite_size_brackets": recomputed_brackets,
        "adjacent_comparisons": adjacent,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "artifacts/photonic-surgery")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    runs = {name: check_run(args.results, name, plan) for name, plan in configurations(args.results)}
    require(sum(r["cells"] for r in runs.values()) == 296, "expected all 296 predeclared cells")
    report = {
        "schema": 2, "checker": "scripts/check_threshold_intervals.py",
        "checker_sha256": sha256(Path(__file__)), "scipy_version": scipy.__version__,
        "rows": sum(r["cells"] for r in runs.values()),
        "attempts": sum(r["totals"]["attempts"] for r in runs.values()),
        "maximum_endpoint_error": max(r["maximum_endpoint_error"] for r in runs.values()),
        "maximum_error_after_declared_padding": max(r["maximum_error_after_declared_padding"] for r in runs.values()),
        "maximum_binomial_tail_error": max(r["maximum_binomial_tail_error"] for r in runs.values()),
        "method": "Independent SciPy beta quantiles/isf, verified against binomial tail equations; Bonferroni families read from predeclared protocols; unconditional unsuccessful counts and full fixed grids verified; no resampling or pooling",
        "confidence_scope": "99% simultaneous within each separately declared family; not a joint 99% claim over all five families",
        "runs": runs,
    }
    output = args.output or args.results / "scipy-interval-oracle.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verified {report['rows']} cells; maximum endpoint discrepancy {report['maximum_endpoint_error']:.3g}")
    for name, run in runs.items():
        print(name, json.dumps(run["finite_size_brackets"], sort_keys=True))


if __name__ == "__main__":
    main()
