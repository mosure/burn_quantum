#!/usr/bin/env python3
"""Collect existing local qualification artifacts; never generate new samples.

Run after the documented tests, runtime matrix and independent interval checker.
Raw logs stay under artifacts/photonic-surgery; the resulting checked-in JSON
retains their hashes, numerical results, protocols and working-tree provenance.
"""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def tests(path):
    rows = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;", path.read_text())
    if not rows or any(r[0] != "ok" or int(r[2]) for r in rows):
        raise ValueError(f"Missing or failed test result: {path}")
    return {"binaries": len(rows), "passed": sum(int(r[1]) for r in rows),
            "failed": sum(int(r[2]) for r in rows), "ignored": sum(int(r[3]) for r in rows),
            "log": str(path.relative_to(ROOT)), "log_sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/photonic-surgery")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/photonic-surgery-results.json")
    args = parser.parse_args()
    folder = args.artifacts.resolve()
    names = ["six-ring-qualification", "optical-qualification", "planar-qualification",
             "planar-confirmation", "surgery-threshold", "surface-threshold", "surface-confirmation"]
    studies = {name: read(folder / f"{name}.json") for name in names}
    oracle = read(folder / "scipy-interval-oracle.json")
    for name in names:
        if oracle["runs"][name]["sha256"] != digest(folder / f"{name}.json"):
            raise ValueError(f"Stale independent oracle: {name}")
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "rust-toolchain.toml"]
    paths += [p for p in (ROOT / "crates").rglob("*") if p.is_file()
              and p.suffix in {".rs", ".toml", ".json", ".py", ".cc"} and "__pycache__" not in p.parts]
    paths += [ROOT / "scripts" / name for name in ["compare_backends.py", "check_threshold_intervals.py",
                                                  "plot_surgery_thresholds.py", "collect_surgery_evidence.py"]]
    source = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}
    runtime = {}
    for backend in ["cpu", "wgpu", "wgpu-fusion", "cuda", "cuda-fusion"]:
        status = read(folder / backend / "status.json")
        if not status["success"] or any(c["exit_code"] for c in status["commands"]):
            raise ValueError(f"Failed runtime matrix: {backend}")
        delta = [k for k, v in status["source_after"].items() if source.get(k) != v]
        allowed = {"crates/burn_quantum/Cargo.toml", "crates/burn_quantum/src/photonic/qec/optical_erasure.rs",
                   "crates/burn_quantum_example/src/bin/surface_threshold.rs"}
        if set(delta) - allowed:
            raise ValueError(f"Unqualified source changes after runtime: {backend}: {delta}")
        runtime[backend] = {
            "success": True, "commands": status["commands"],
            "native_tests": tests(folder / backend / "tests.log"),
            "reports": {c["name"]: read(folder / backend / (c["name"] + ".json"))
                        for c in status["commands"] if c["name"] != "tests"},
            "source_changes_since_matrix": delta,
            "qualification_of_changes": "Python-bytecode package exclusions; rustdoc code-span fix; confirmation-only CLI profile separately replayed on all five configurations. No numerical-core change.",
            "original_status_sha256": digest(folder / backend / "status.json"),
        }
    replay = read(folder / "surface-final-replay.json")
    if replay["source_sha256"] != source["crates/burn_quantum_example/src/bin/surface_threshold.rs"] or any(r["exit_code"] for r in replay["runs"]):
        raise ValueError("Stale or failed final surface replay")
    checks = read(folder / "checks-final.json")
    if len(checks) != 6 or any(c["exit_code"] for c in checks):
        raise ValueError("Incomplete final build checks")
    fixtures = {str(p.relative_to(ROOT)): {"sha256": digest(p), "bytes": p.stat().st_size}
                for p in sorted((ROOT / "crates/burn_quantum/tests/stabilizer").glob("*stim*.json"))}
    package = read(folder / "package-status.json")
    if package["library"]["status"] != "passed":
        raise ValueError("Core package not verified")
    logs = {str(p.relative_to(ROOT)): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*.log")) if "initial-runtime" not in p.parts}
    protocols = {p.name: read(p) for p in sorted(folder.glob("*protocol.json"))}
    result = {
        "schema": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local simulation and named finite-size noise models; no hardware qualification, release or exact-commit CI claim.",
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_working_tree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "source_sha256": source, "protocols": protocols, "studies": studies,
        "independent_statistics": oracle, "stim_provenance": read(folder / "stim-provenance.json"),
        "pinned_stim_fixtures": fixtures,
        "physical_record_distances": read(folder / "cubical-record-validation.json"),
        "validation": {"cpu_workspace": tests(folder / "cpu-workspace-final.log"),
                       "additional_signed_relation_regression": tests(folder / "output-relations-after-workspace.log"),
                       "checks": checks, "runtime": runtime, "final_surface_replay": replay,
                       "backend_comparison": read(folder / "backend-comparison.json"), "package": package},
        "logs": logs,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "source_files": len(source), "cells": oracle["rows"], "attempts": oracle["attempts"]}))


if __name__ == "__main__":
    main()
