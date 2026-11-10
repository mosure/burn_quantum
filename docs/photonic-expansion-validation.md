# Photonic local-simulation expansion: validation

Recorded 2026-11-09 on the local unreleased working tree based on
`3b4473801b1b972795140f0bac4ab100b1b8ef0e`, Rust 1.95.0 and Burn 0.22.1.
[Current status](photonic-status.md) preserves all incomplete roadmap acceptance
gates. This is not a publication, hardware-QPU or exact-commit CI claim.

## Scope

The CPU workspace run passed **130 tests and one doctest**. All **fifteen**
application binaries ran successfully. The new targets include 16 encoded
photonic tests, 7 optics tests, 7 compiler tests, 5 runtime/factory tests,
6 stabilizer tests and 3 binary-QEC tests. Existing simulator, SAT, QN-SPSA,
training and contract tests remain in the full run.

The accelerator matrix selects the NVIDIA RTX PRO 6000 Blackwell Workstation
Edition, driver 610.43.02, using WGPU/Vulkan and CUDA with kernel fusion off/on.
It runs `backend_parity`, `photonic`, `optics` and `photonic_compiler`: **39 core
checks per configuration**, followed by **six photonic example acceptance tests**
and each corresponding executable. Selected tensor work is F32; independent
scalar references use host F64. Routing and host algorithm checks do not become
GPU operations simply because their test target selects a GPU.

The six example binaries are `fbqc_bell_swap`, `fbqc_factory_pipeline`,
`fbqc_repetition_memory`, `photonic_hom`, `photonic_mz_learning` and
`photonic_teleportation`. Stabilizer trajectories, factory scheduling and QEC are
host algorithms. Their examples explicitly execute a small selected-device
dense oracle and report host-work metrics separately. The prior nine nonphotonic
application GPU runs were not repeated in this focused expansion qualification.

[Machine-readable evidence](photonic-expansion-results.json) records final matrix
completion, reports, CPU/GPU comparison, source hashes and package provenance.
All four configurations completed successfully. Maximum report differences from
CPU were 1.20e-7 on WGPU and 5.97e-8 on CUDA, with kernel fusion both off and on.
The comparison script and existing numerical tolerances are unchanged.

## Independent physical and compiler evidence

- Pinned **Stim 1.15.0**, built and executed as native C++, supplies 1,024 exact
  Pauli expectations across complex Clifford and postselected fixtures. The
  checked source hash, generator and reproducer are packaged under
  [`tests/stabilizer`](../crates/burn_quantum/tests/stabilizer).
- Pinned **Perceval 1.3.0 / Exqalibur 1.4.2 / NumPy 2.5.4** supplies all 100
  complex amplitudes for a three-mode, three-photon Fourier interferometer plus
  unfiltered asymmetric-loss HOM probabilities. See the
  [generator](../crates/burn_quantum/tests/optics_perceval_v1_3.py) and
  [fixture](../crates/burn_quantum/tests/optics_perceval_v1_3.json).
- Independent scalar-complex tests cover all four-qubit Paulis after random
  Clifford evolution/measurements; dense comparisons cover 40,000 lossy
  fusion trajectories and their output correlations.
- Measurement-pattern tests compare complete corrected channels on entangled
  inputs across nonzero branches, parameter/input derivatives and resource
  rejection. Physical one-qubit gate teleportation separately checks Choi
  channels, routed losses and surviving entangled references.
- Factory/runtime tests cover independent-attempt yield/cost, exhausted budgets,
  births, contention, reset, causal feed-forward, atomic merging/failure,
  cancellation, retries, changed calibration identity and checkpoint restoration.
- Binary-QEC tests exhaust erasure-supported faults at repetition distances
  3/5/7, logical ambiguity versus physical degeneracy, inconsistent syndrome and
  validated persistence. This is not full photon-loss error correction.

These external comparisons cover their stated fixtures. They do not establish
unrestricted equivalence with Stim or Perceval, physical fusion-network
calibration, or successful interoperability with PsiQuantum hardware.

## Example results on CPU

| Example | Independent check | Observed result |
| --- | --- | --- |
| HOM | Balanced splitter with indistinguishable two-photon input | Bunching 0.9999999404; coincidence below 5e-32 |
| Optical phase learning | Three initializations, held-out inputs, analytic detector probabilities | Worst test MSE 7.95e-8 against uncalibrated baseline 0.03140; max analytic error 1.64e-7 |
| Logical teleportation | Five measurements and 32 corrected branches | Total probability 0.9999998808; max conditional error 1.02e-7 |
| Factory pipeline | Independent 0.6-success factories with three attempts, then lossy Bell fusion | Availability 0.86743; end-to-end success 0.31030; 6.22949 source photons per attempted experiment |
| Repetition memory | Reference-entangled X-error memory, exact flagged-error enumeration | 2,289 corrected fault cases; logical XX/ZZ correlations 0.9999999404 in the dense oracle |

The optical learner is a small exact-probability calibration task, not a
finite-shot learning or predictive-advantage result. Factory results count
unsuccessful factories and fusion outcomes; their cost includes every source
attempt. Repetition coordinate erasure means an unknown X error at a flagged
site, not a physically absent photon.

## Scaling and engineering

A compiled 200-photon Bell-resource recipe executes with the host stabilizer
engine. A 2,048-qubit GHZ tableau uses 2,326,528 measured allocated bytes; one
local debug preparation/query observation was 70 ms. These demonstrate the
supported representation's polynomial memory, not broad performance or arbitrary
non-Clifford scalability. [Detailed evidence](photonic-stabilizer.md).

Formatting, strict all-feature/all-target Clippy, warning-free all-feature
rustdoc, isolated no-backend/Flex builds, shell syntax and local documentation
targets passed. The standalone package built and verified, contains 108 files
including all 96 core Rust files and both external-reference fixtures, and its
extracted targets compile. Inspection matches source bytes and confirms
`path_in_vcs = crates/burn_quantum`, `dirty = true`. Version remains 0.2.0 and
Cargo dependencies/features/lockfile are unchanged. This local dirty archive
is not release provenance.

## Reproduction

```sh
cargo test --locked --workspace --all-targets
cargo test --locked --workspace --doc
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --all-features --no-deps
bash scripts/run_examples.sh cpu
# Includes the expanded tests and the Cargo-discovered complete application catalog:
bash scripts/validate_backend.sh wgpu
bash scripts/validate_backend.sh wgpu --fusion
bash scripts/validate_backend.sh cuda
bash scripts/validate_backend.sh cuda --fusion
cargo package --locked --offline --allow-dirty -p burn_quantum
```

The recorded run used the focused six-example GPU matrix described above, not
the broader whole-catalog command. Raw logs and reports are retained under
ignored `artifacts/photonic-expansion/`; the saved `run-gpu.sh` records its exact
commands. Ordinary Cargo tests use committed fixtures and do not install Python
packages or run external simulators. Regeneration is a separate explicit step.
