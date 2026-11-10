#!/usr/bin/env python3
"""Run the actual pinned local PsiQDK parser and state-vector simulator.

Use Python 3.12 with these exact packages:
psiqdk==2.5.0 psiqdk-workbench==4.55.3 workbench-sim-native==0.3.0
antlr4-python3-runtime==4.13.2 numpy==2.5.4
This does not contact a provider or run quantum hardware.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

PINNED = {
    "psiqdk": "2.5.0",
    "psiqdk-workbench": "4.55.3",
    "workbench-sim-native": "0.3.0",
    "antlr4-python3-runtime": "4.13.2",
    "numpy": "2.5.4",
}
for name, expected in PINNED.items():
    actual = importlib.metadata.version(name)
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected}, found {actual}")
if platform.python_version_tuple()[:2] != ("3", "12"):
    raise RuntimeError("qualification requires Python 3.12")

import numpy as np
from psiqdk.workbench import QPU
from psiqdk.workbench.integrations.qasm import QasmString

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", type=Path, help="Previously exported psiqdk_interop JSON; otherwise runs Cargo")
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
if args.inputs:
    inputs = json.loads(args.inputs.read_text())
else:
    environment = dict(os.environ, BURN_QUANTUM_DEVICE="cpu", CARGO_NET_OFFLINE="true")
    inputs = json.loads(subprocess.check_output(
        ["cargo", "run", "--quiet", "--locked", "-p", "burn_quantum_example", "--bin", "psiqdk_interop"],
        cwd=root, env=environment, text=True))

cases = []
for case in inputs["cases"]:
    qasm = case["export"]["program"]
    n = case["export"]["qubits"]
    # Explicit local simulator pipeline, not default/hardware/provider selection.
    qpu = QPU(num_qubits=n, filters=[">>state-vector-sim>>", ">>buffer>>"])
    measured = [int(v) for v in QasmString(qasm).add_operations_to_qpu(qpu)]
    state = np.asarray(qpu.pull_state(), dtype=np.complex128).reshape(-1)
    assert len(state) == 1 << n
    assert np.isclose(np.vdot(state, state), 1., atol=1e-12, rtol=0)
    if case["expected_readouts"] is not None:
        decoded = [value != 0 for value in measured]
        assert decoded == case["expected_readouts"]
        assert all(v in (0, 1 << wire) for v, wire in zip(measured, case["export"]["readout_wires"]))
    case["psiqdk_amplitudes_little_endian"] = [[float(v.real), float(v.imag)] for v in state]
    case["psiqdk_weighted_readouts"] = measured
    case["qasm_sha256"] = hashlib.sha256(qasm.encode()).hexdigest()
    cases.append(case)

output = {
    "schema": 1,
    "python": platform.python_version(),
    "packages": PINNED,
    "execution": "local PsiQDK state-vector-sim and buffer filters; no provider/hardware",
    "wire_mapping": "Burn wire i is q[i]; Burn amplitude index is big-endian, PsiQDK index uses bit i",
    "global_phase_policy": inputs["global_phase_policy"],
    "cases": cases,
}
path = Path(__file__).with_name("psiqdk-2.5.0.json")
path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
print(f"Wrote {len(cases)} independently simulated circuits to {path}")
