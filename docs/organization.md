# Repository organization

The workspace has two packages: the published `burn_quantum` library at the
repository root, and the unpublished `burn_quantum_example` application suite
under `crates/`. Both inherit package metadata, shared dependency versions, and
lints from the root workspace. Each package still selects its own Burn features;
the library can compile without an execution backend or autodiff.

## Ownership and boundaries

| Location | Responsibility | Keep elsewhere |
| --- | --- | --- |
| `src/layout.rs`, `complex.rs`, `gate.rs` | Register conventions, tensor representation, validated operators | Application models and datasets |
| `src/backend.rs`, `control.rs`, `precision.rs` | Quantum execution contract, dense contractions, structured controls, precision handling | Environment-selected devices and CLI behavior |
| `src/state.rs`, `noise.rs`, `observable.rs`, `sampling.rs` | Pure/density states, channels, readouts, measurement | Task-specific objectives and acceptance criteria |
| `src/circuit/` | Validated circuit IR, serialization, tensor binding and execution | Optimizer policy and hardware job submission |
| `src/gradient.rs`, `quantum_module.rs`, `qml.rs` | Differentiation, native Burn modules, reusable quantum ML operations | Example training loops and synthetic datasets |
| `src/sat/` | CNF representation, reversible oracle compilation, bounded search | DIMACS file I/O and CLI reporting |
| `src/optim/qnspsa/` | Fidelity estimator, metric solve, optimizer state and updates | Circuit ansatz, data, loss/fidelity evaluator implementations |
| `crates/burn_quantum_example/src/bin/` | One complete algorithm and acceptance test per file | Repeated quantum operations shared by applications |
| `crates/burn_quantum_example/src/lib.rs` | Small example-only reporting and evaluation utilities | Public quantum library APIs |
| `examples/` | Cargo examples for basic simulator use, integration and device probes | The canonical application suite |
| `support/device.rs` | Explicit device selection shared by executables and tests | The public library, which accepts ordinary Burn devices |
| `tests/` | External API conformance and independent mathematical references | Production implementations of their own test oracles |
| `scripts/`, `.github/workflows/` | Reproducible execution and validation | Quantum algorithms |
| `docs/`, `artifacts/` | Committed guides/evidence and ignored local logs respectively | Build output in source control |

Small cohesive modules stay as single files. Directory modules are used where
responsibilities have separate reasons to change; there is no artificial file
size limit or extra crate for each algorithm. GPU numerical operations continue
through the shared precision path, including SAT and optimizer evaluations.

## Circuit module

```text
src/circuit/
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
src/sat/
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
src/optim/
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

## Tests and example discovery

The `sat` and `qnspsa` integration test targets retain their names. Their root
files hold shared fixtures and include focused test modules:

- `tests/sat/`: CNF, oracle, search, and serialization/resource/continuation contracts.
- `tests/qnspsa/`: scalar estimators/solves, optimizer state transitions, and quantum geometry/device parity.
- `tests/common/`: independent scalar-complex reference operations shared by the suite.

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
An empty/missing example package fails discovery. All nine algorithms retain
one source file each; the reorganization did not alter those implementations.

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

## Verification record

[organization-results.json](organization-results.json) records the reorganization
checks and source hashes. The earlier SAT, QN-SPSA, and initial release reports
remain historical records of the file layouts and numerical results they
qualified; their hashes were not rewritten to imply they tested later sources.

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
