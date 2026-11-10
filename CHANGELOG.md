# Changelog

## Unreleased

- Add local growing-distance XX/ZZ merge/split, measurement-only logical CNOT,
  complete noisy transition detectors and observed-record physical recovery.
- Add bounded sparse erasure/union-find decoding, exact logical-image ranks,
  correlated boosted/Shor optical loss laws and fixed-shot threshold qualification
  with simultaneous binomial intervals and retained decoder failures.
- Add actual six-ring surgery instruments and boundary deformations, independent
  signed Stim fixtures, explicit negative unboosted-yield scaling and CPU/GPU oracles.
- Make fusion-program consumption checks constant-time while preserving validated
  persistence and atomic rejection of reused or absent photons.
- Add a modular photonic stack: dual-rail/vacuum encoding, exact bounded Fock
  optics, calibrated multimode sources/detectors, standard and Grice fusion
  effects, signed resource networks and native Burn numerical execution.
- Add bounded universal gate synthesis, noisy heralded resource factories,
  physical VQE/QAOA, finite buffers, independent wavelength/rail reservations,
  explicit route objectives and verified retries of parity-encoded unknown inputs.
- Add packed Clifford tableaux and reference-frame sampling, four-star/six-ring
  networks, physical Shor encoding/fusions, open planar logical memories and
  Clifford protocols, and exact bounded magic-state injection/distillation.
- Add finite-shot mixed-state tomography/overlap, measured QN-SPSA probes,
  calibration-identifiability checks, minibatch shot allocation and seeded
  matched-record photonic learning benchmarks with physical cost accounting.
- Add mixed-qudit MPS execution through `QuantumBackend`, controlled Schmidt
  truncation, local Kraus trajectories and streaming observables. Native Burn
  contractions use CPU/GPU devices; complex SVD remains bounded host work without AD.
- Add distributed ownership of entangled MPS tensors, recoverable boundary
  transactions, bounded event/shot workers and immutable asynchronous local jobs.
  Keep raw unsuccessful attempts, calibration/provenance and resource limits.
- Bound GPU fusion planning for wide precise contractions using hierarchical
  concatenation; validate outputs and all input/operator gradients independently.
- Add finite six-ring boundary measurements/logical correlations and bounded
  OpenQASM interchange qualified against the actual pinned PsiQDK local simulator.
- Add independent pinned Stim/Perceval/PsiQDK fixtures, expanded single-file applications,
  automatic GPU-test discovery, and source-specific CPU/GPU/package evidence.
- Promote `serde_json` to a library dependency for bounded worker/event persistence.
  Compiled photonic inspection plans use schema 3; existing circuit/source schemas
  retain their versions. The `fusion` feature continues to mean Burn kernel fusion.
- Keep hardware deployment and provider-specific fault-tolerant architecture
  qualification separate from the bounded local models. PsiQuantum remains the
  long-term hardware target; no local simulator output is labeled a QPU result.
- Keep the root virtual workspace and library in `crates/burn_quantum`, beside
  `crates/burn_quantum_example`. Each crate owns its manifest/README; shared
  dependencies, lockfile, toolchain, guides and validation tooling stay at root.

## 0.2.0 — 2026-11-08

### Added

- Reversible Boolean CNF/DIMACS oracles, Grover amplification, bounded seeded
  SAT search, and classical verification of returned witnesses.
- Structured `ControlledX` and `PhaseFlip` circuit operations, plus
  `Circuit::run_from` for execution on existing states.
- Fidelity-based quantum natural SPSA with a Fubini–Study metric (QFI/4), spectral
  regularization, decay schedules, resampling, loss blocking and checkpoints.
- Single-file `grover_sat` and `qnspsa_vqe` examples, bringing the suite to nine.

### Organization

- Circuit, SAT and QN-SPSA implementations are separated into focused private
  modules behind their existing public imports.
- Shared workspace metadata, dependency versions and lints are centralized.
- SAT/QN-SPSA tests are grouped by responsibility, and validation scripts
  discover example binaries through Cargo metadata.
- Repository and contribution guides document extension boundaries and validation.

### Compatibility and limits

- **Update exhaustive matches on `Operation`** to handle the new `ControlledX`
  and `PhaseFlip` variants. This public enum extension is why the release moves
  from 0.1 to 0.2 rather than using a patch version.
- Circuits containing those operations serialize as schema v2. Legacy-only
  circuits retain v1, and the reader accepts both versions. Version 0.1 readers
  cannot execute v2 instructions.
- Rust 1.95 and Burn 0.22.1 remain the tested baseline; dependency versions and
  existing tensor/module interfaces are unchanged by this release preparation.
- Dense SAT simulation remains exponential. Exhausted search returns
  `Inconclusive`, not an UNSAT proof. QN-SPSA's dense host metric requires O(d²)
  storage and an O(d³) solve even though evaluation count is dimension-independent.
- `burn_quantum_example` remains an unpublished workspace package.

### Validation

The implementation passed 80 CPU tests and one doctest. WGPU and CUDA, with
fusion off and on, each passed 45 checks including all nine example acceptance
tests on an NVIDIA RTX PRO 6000 Blackwell. GPU execution used F32; strict F64
references remain on CPU. SAT and QN-SPSA reports were unchanged by the module
reorganization. See [the validation records](docs/README.md#validation-evidence)
for commands, source snapshots, tolerances and hardware scope.

## 0.1.0

Initial Burn-native qubit/qudit simulator, differentiable quantum modules,
density matrices/channels, seven canonical QML examples, and CPU/GPU qualification.
