# Changelog

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
