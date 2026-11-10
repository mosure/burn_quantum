# Repository organization

The repository root is a virtual Cargo workspace with two sibling packages:

```text
Cargo.toml                    Workspace members, shared metadata/dependencies/lints
Cargo.lock                    One locked dependency graph for the workspace
rust-toolchain.toml           Shared Rust toolchain
crates/
  burn_quantum/
    Cargo.toml                Published library package and execution features
    README.md                 Package usage and links to repository guides
    LICENSE-*                 License texts included in the standalone archive
    src/                      Public library and private implementation modules
    tests/                    Integration tests and independent references
    examples/                 Basic simulator demos and device probes
    support/device.rs         Executable/test device selection
  burn_quantum_example/
    Cargo.toml                Unpublished application suite
    README.md                 Algorithm catalog and execution instructions
    src/lib.rs                Example-only reporting and evaluation utilities
    src/bin/                  One complete algorithm and acceptance test per file
docs/                         Repository guides and historical qualification records
scripts/                      Workspace validation and example runners
.github/workflows/            CPU, packaging and GPU qualification workflows
```

Both packages inherit common metadata, dependency versions and lints from the
root workspace. The example package inherits its core dependency path from
`[workspace.dependencies]`. Each package selects its own Burn features; the
library can compile without an execution backend or autodiff. Both are default
members, so ordinary root `cargo test`/`cargo check` commands select the workspace.
Use `-p burn_quantum` for isolated library checks.

Repository documentation, the changelog, contribution guide and license texts
remain at the root. The library includes its own README and copies of the root
licenses in the publishable archive, together with its sources, tests, demos
and device helper. Its extracted package builds independently of the workspace
and sibling example crate. CI verifies that boundary with `cargo package`.

## Ownership and boundaries

| Location | Responsibility | Keep elsewhere |
| --- | --- | --- |
| `crates/burn_quantum/src/layout.rs`, `complex.rs`, `gate.rs` | Register conventions, tensor representation, validated operators | Application models and datasets |
| `crates/burn_quantum/src/backend.rs`, `control.rs`, `precision.rs` | Quantum execution contract, dense contractions, structured controls, precision handling | Environment-selected devices and CLI behavior |
| `crates/burn_quantum/src/state.rs`, `noise.rs`, `observable.rs`, `sampling.rs` | Pure/density states, channels, readouts, measurement | Task-specific objectives and acceptance criteria |
| `crates/burn_quantum/src/circuit/` | Validated circuit IR, serialization, tensor binding and execution | Optimizer policy and hardware job submission |
| `crates/burn_quantum/src/gradient.rs`, `quantum_module.rs`, `qml.rs` | Differentiation, native Burn modules, reusable quantum ML operations | Example training loops and synthetic datasets |
| `crates/burn_quantum/src/sat/` | CNF representation, reversible oracle compilation, bounded search | DIMACS file I/O and CLI reporting |
| `crates/burn_quantum/src/optim/qnspsa/` | Fidelity estimator, metric solve, optimizer state and updates | Circuit ansatz, data, loss/fidelity evaluator implementations |
| `crates/burn_quantum/src/tensor_network/` | Canonical MPS evolution, measurement, streaming readout, bounded host SVD, distributed state ownership | Application circuits and deployment transport policy |
| `crates/burn_quantum/src/transport.rs` | Shared bounded frames and absolute socket deadlines | Quantum semantics and credentials |
| `crates/burn_quantum/src/photonic/` | Encoding, Fock optics, Clifford/dense execution, resource/fusion compilation, finite fabric schedules, QEC, learning and event/worker runtimes | Example datasets/objectives; provider-specific hardware adapters |
| `crates/burn_quantum_example/src/bin/` | One complete algorithm and acceptance test per file | Repeated quantum operations shared by applications |
| `crates/burn_quantum_example/src/lib.rs` | Small example-only reporting and evaluation utilities | Public quantum library APIs |
| `crates/burn_quantum/examples/` | Cargo examples for basic simulator use, integration and device probes | The canonical application suite |
| `crates/burn_quantum/support/device.rs` | Explicit device selection shared by executables and tests | The public library, which accepts ordinary Burn devices |
| `crates/burn_quantum/tests/` | External API conformance and independent mathematical references | Production implementations of their own test oracles |
| `scripts/`, `.github/workflows/` | Reproducible execution and validation | Quantum algorithms |
| `docs/`, `artifacts/` | Committed guides/evidence and ignored local logs respectively | Build output in source control |

Small cohesive modules stay as single files. Directory modules are used where
responsibilities have separate reasons to change; there is no artificial file
size limit or extra crate for each algorithm. GPU numerical operations continue
through the shared precision path, including SAT and optimizer evaluations.

## Circuit module

```text
crates/burn_quantum/src/circuit/
  mod.rs            Circuit specification, builders, instruction validation, public exports
  operation.rs      Angle, Rotation, Operation: device-independent instruction definitions
  serialization.rs  Versioned wire format and checked conversions
  execution.rs      Tensor bindings, resource preflight, zero-state/continuation execution
```

Instruction definitions do not import Burn tensors. Execution owns the binding
and device-specific work. Deserialization routes instructions through the same
builder validation, so loading cannot bypass invariants. `CircuitData` remains
private; existing schemas and root/module import paths remain unchanged.

## SAT module

```text
crates/burn_quantum/src/sat/
  mod.rs     Public exports
  cnf.rs     Normalization, DIMACS parsing, evaluation, checked serialization
  oracle.rs  GroverSat, reversible compilation, preparation and fixed iterations
  search.rs  SearchConfig, SatOutcome, SearchReport, bounded randomized search
```

Oracle compilation reads the normalized formula through its accessors. Search
uses the oracle's execution methods rather than reaching into its private
circuits or engine. A new search policy can reuse the compiled oracle without
changing CNF parsing. File access remains in the `grover_sat` executable.

## QN-SPSA module

```text
crates/burn_quantum/src/optim/
  mod.rs                 Public optimizer exports
  qnspsa/
    mod.rs               Private implementation-module exports
    config.rs            Burn Config and configuration validation
    estimator.rs         Pure scalar gradient/metric estimator
    metric.rs            Host F64 spectral regularization and linear solves
    optimizer.rs         Initialization, callbacks, tensor steps, restoration
    state.rs             Checkpoint, update result and diagnostic types
    validation.rs        Shared numeric and allocation guards
```

Only `metric.rs` knows about nalgebra. The scalar estimator has no tensor or
device dependency. The optimizer orchestrates fresh evaluations and owns state
transitions; callers supply the quantum model and evaluators. Checkpoints retain
their schema, and model weights remain normal Burn records.

Implementation submodules are private, with narrowly scoped internal access.
Users continue importing `burn_quantum::{Circuit, Angle, Rotation, Operation}`,
`burn_quantum::sat::{Cnf, GroverSat, SearchConfig, SatOutcome, SearchReport}`, and
`burn_quantum::optim::{QnSpsa, QnSpsaConfig, ...}`. No wrapper API or new crate is
needed to use the reorganization.

## Photonic module

The current [photonic stack](photonic-status.md) retains the encoded foundation
modules `encoding`, `loss`, `resource`, `fusion`, `fabric`, `program` and `simulator`.
Separate `optics`, `stabilizer`, `compiler`, `factories`, `qec`, `learning`, `runtime`
and `interop` APIs own their physical models and execution contracts. Routing,
compilation, optical instruments, QEC geometry/decoding and worker transports
have focused implementation files. Typed recipes and compiled plans remain
independent of dense tensor allocation; numerical executors reuse the core's
precision and resource guards. The [optical](photonic-optics.md),
[compiler](photonic-compiler.md), [QEC](photonic-stabilizer.md),
[interchange](photonic-interop.md) and [runtime](photonic-runtime.md) guides describe
the current public surfaces; the original [module roadmap](photonic-roadmap.md#module-ownership)
records their planned boundaries.

Within `qec`, `surgery/` separates geometry, ideal execution, spatial distances,
transition detectors and observed-record recovery. `threshold/` owns sparse graph
decoders, noise studies and fixed-shot statistics. `fusion_surgery/` owns actual
six-ring optical instruments and boundary geometry. Examples contain study grids
and reporting; shared physical laws, decoding and confidence intervals stay in
the library. See [surgery](photonic-surgery.md) and
[threshold decoding](photonic-threshold-decoder.md).

## Shared executable support

The library's basic demos and integration tests include `support/device.rs`
within their own package. The unpublished example crate includes the same file
from its sibling `burn_quantum` directory, so device selection is maintained in
one place. It is development support rather than part of the public quantum API;
ordinary library consumers always pass Burn devices explicitly.

## Tests and example discovery

The `sat`, `qnspsa` and `photonic` integration test targets use root
files that hold shared fixtures and include focused test modules:

- `crates/burn_quantum/tests/sat/`: CNF, oracle, search, and serialization/resource/continuation contracts.
- `crates/burn_quantum/tests/qnspsa/`: scalar estimators/solves, optimizer state transitions, and quantum geometry/device parity.
- `crates/burn_quantum/tests/photonic/`: encoding/loss, independent fusion-instrument oracles, fabric scheduling, program and sampling contracts.
- `crates/burn_quantum/tests/common/`: independent scalar-complex reference operations shared by the suite.

These are external tests using the public library API. They cannot access the
library's private implementation modules. Existing assertions and tolerances
were retained during the split.

```sh
cargo test --locked -p burn_quantum --test sat oracle::
cargo test --locked -p burn_quantum --test qnspsa geometry::
```

`scripts/example_catalog.py` reads Cargo target metadata for
`burn_quantum_example`. Both `run_examples.sh` and `compare_backends.py` use it,
so a new binary is automatically included without editing parallel name lists.
An empty/missing example package fails discovery. All twenty-two current examples have
one source file each, including physical VQE, optical learning, QEC memory and
process/control experiments.

```sh
python3 -B scripts/example_catalog.py
bash scripts/run_examples.sh cpu
python3 -B scripts/compare_backends.py artifacts/examples/cpu artifacts/examples/wgpu
# Explicit subset for a focused qualification or archived reports:
python3 -B scripts/compare_backends.py --example qnspsa_vqe CPU_DIR GPU_DIR
```

The default comparison requires every current example report. Missing reports
are errors, not silent omissions. `--example` is repeatable and intentionally
selects a narrower set; it does not imply whole-suite validation.

## Workspace relocation verification

The **2026-11-08** relocation was checked against release commit `90443a8`:

- All 54 relocated Rust files are byte-for-byte unchanged. All nine algorithm
  sources are unchanged; the example library only updates its device-helper
  include path.
- Cargo metadata retains the package names, target names, feature mappings,
  dependency requirements and default-member selection. `Cargo.lock` is
  unchanged, and both packages now resolve beneath `crates/`.
- The CPU run passed 80 tests, one doctest, four basic demos and all nine
  application executables. The nine JSON reports exactly match the earlier
  module-split qualification.
- WGPU and CUDA, each with fusion off/on, passed 45 checks per configuration
  on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition, driver 610.43.02.
  These comprise the 36 backend/SAT/QN-SPSA checks and all nine application
  acceptance tests. GPU execution uses F32; host analytic/control cases and
  F64 reference fixtures remain CPU checks. The SAT and QN-SPSA executables
  also passed CPU/GPU report comparisons; their reports exactly match the
  earlier qualification on all four configurations.
- Formatting, strict all-feature Clippy, warning-free all-feature rustdoc,
  isolated no-backend/Flex builds, shell syntax and local Markdown links passed.
- The standalone package contains 61 files, including all 54 Rust files, its
  README and both license texts. Its normalized manifest resolves workspace
  inheritance, and its VCS metadata identifies `crates/burn_quantum`. Cargo
  independently built the archive and checked all extracted package targets.

These checks describe the local working tree. Logs, source comparisons, package
inspection and hardware reports are retained under ignored
`artifacts/workspace-layout/`. The package version remains 0.2.0; the repository
layout change is recorded under Unreleased in the changelog.

## Earlier module-split verification

[organization-results.json](organization-results.json) records the earlier internal
module split, before the library moved under `crates/burn_quantum`. Paths and
source hashes in that snapshot describe the old root-package layout. The earlier
SAT, QN-SPSA, and initial release reports remain historical records of the file
layouts and numerical results they qualified; their hashes were not rewritten
to imply they tested later sources.

The **2026-11-08** regression run passed 80 CPU tests, one doctest, all nine CPU
example executables, and 45 checks per hardware configuration: WGPU and CUDA,
each with fusion off/on, on an NVIDIA RTX PRO 6000 Blackwell. Those 45 checks
include all nine example acceptance tests, plus backend, SAT, and QN-SPSA tests
(some mathematical/control cases intentionally remain CPU references). GPU
execution used F32. SAT and QN-SPSA reports exactly matched their pre-refactor
records on CPU and all four GPU configurations. Dependency versions, feature
mappings, the lockfile, and all example algorithm sources were unchanged.

See [the documentation index](README.md) and [contributing guide](../CONTRIBUTING.md)
for navigation and extension workflows.

## Tensor-network implementation

`tensor_network` separates state/checkpoint invariants (`state.rs`), gauge/gate
updates (`evolution.rs`), Kraus trajectories (`measurement.rs`), observables and
bounded export (`readout.rs`), native contractions/host SVD (`tensor.rs`), and the
`QuantumBackend` adapter (`engine.rs`). Distributed protocol, ownership and
coordinator transactions have separate files under `distributed/`. Local and
remote observable contractions share the same core implementation. The reusable
frame transport is shared with photonic shot workers; no provider dependency or
CLI environment selection enters the public core.
