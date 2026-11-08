#!/usr/bin/env python3
"""Discover canonical example binaries from Cargo's workspace target metadata."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def example_names():
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--locked", "--no-deps", "--format-version", "1",
        "--manifest-path", str(ROOT / "Cargo.toml"),
    ], cwd=ROOT, text=True))
    package = next((p for p in metadata["packages"]
                    if p["name"] == "burn_quantum_example"), None)
    if package is None:
        raise ValueError("burn_quantum_example is missing; run from a complete repository checkout")
    names = tuple(sorted(t["name"] for t in package["targets"] if "bin" in t["kind"]))
    if not names:
        raise ValueError("burn_quantum_example has no executable targets")
    return names


if __name__ == "__main__":
    print("\n".join(example_names()))
