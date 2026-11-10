# burn_quantum

Differentiable qubit and qudit simulation using Burn 0.22's native tensors,
devices, modules, parameters, optimizers, and weight records. Core: Rust 1.95+.
The visualization workspace uses Rust 1.97.1 / Bevy 0.20.

This is a working finite-dimensional quantum execution layer with a dense
reference engine. Burn 0.22's dispatch catalog does not support registering
external tensor backends, so `QuantumBackend` describes quantum execution while
`DenseSimulator` uses supported Burn tensor devices. The public API uses
`Tensor<D>` and `Device`, with no legacy backend generic or private Burn fork.
See [Burn's migration guide](https://burn.dev/books/burn/migrating-to-0.22.html).

## Quick start

```toml
[dependencies]
burn_quantum = "0.4.0"
burn = { version = "~0.22.1", default-features = false, features = ["std", "flex"] }
```

The core crate is available on [crates.io](https://crates.io/crates/burn_quantum).
The `burn_quantum_example` workspace package is run from a repository checkout.
See the [changelog](CHANGELOG.md) for release and migration notes. Version 0.4
updates the visualization API; the core quantum API is unchanged from 0.3.

```rust
use burn::tensor::Device;
use burn_quantum::{Circuit, DenseSimulator, QuantumBackend, Register, Unitary};

fn main() -> burn_quantum::Result<()> {
    let mut circuit = Circuit::new(Register::qubits(2)?, 0, 0);
    circuit.unitary(vec![0], Unitary::h())?;
    circuit.unitary(vec![0, 1], Unitary::cnot())?;

    let state = DenseSimulator::default()
        .execute(&circuit, None, None, 1, &Device::flex())?;
    state.validate(1e-5)?;
    // [0.5, 0.0, 0.0, 0.5] in |00>, |01>, |10>, |11> order.
    println!("{}", state.probabilities());
    Ok(())
}
```

Wire zero is the most significant digit. `Register::new(vec![2, 3, 4])` has
24 amplitudes; `|1,2,3>` is flat index 23. Ordered targets define a gate's local
basis: applying a two-site matrix to `[2, 0]` gives wire 2 local significance
before wire 0. Matrices use row-major `[output, input]` storage.

## Implemented

- Qubits, qudits and heterogeneous registers, with checked mixed-radix indexing.
- Batched F32/F64 split-complex pure states; exact density matrices and partial traces.
- Arbitrary validated local unitary matrices; Pauli/H/S/T, Fourier, shift/clock,
  SUM/CNOT, SWAP, level-controlled gates, and differentiable two-level rotations.
- Local Hermitian expectations, variances, Hamiltonian sums and marginals.
- Arbitrary validated Kraus channels, amplitude damping, phase flip and depolarization.
- Versioned serializable circuits with shared parameters and batched input encoding.
- Canonical algorithm builders and explicit measurement/feed-forward branches,
  including differentiable normalization of a fixed conditional state.
- Native reverse-mode gradients and correct occurrence-wise parameter-shift
  references for both qubit and embedded qudit rotations.
- `QuantumLayerConfig` / `QuantumLayer` using Burn `Config`, `Module`, `Param`,
  optimizer traversal, device movement, and burnpack weight records.
- Reproducible terminal measurement with a local seeded RNG and allocation budgets.
- Reusable QML encoding/circuit templates, probability losses, pure-state fidelity
  and differentiable fidelity kernels in `burn_quantum::qml`.
- Shared application benchmarks: real UCI Wine classification with train-only
  preprocessing and held-out/classical controls, plus multi-qubit Ising VQE.
  Enable the core `benchmarks` feature for these bounded async training pipelines.
- Fidelity-based quantum natural SPSA with a regularized Fubini–Study metric,
  seeded perturbations, loss blocking, schedules, and portable checkpoints.
- Boolean CNF/DIMACS SAT search with reversible clause oracles, Grover amplification,
  bounded unknown-solution-count search, and verified candidate assignments.
- Matrix-product states for large low-entanglement qubit/qudit circuits, with
  explicit truncation budgets, Kraus trajectories and distributed tensor ownership.
- Photonic computing: dual-rail/Fock optics,
  calibrated standard/boosted fusion, universal bounded gate synthesis, finite
  fabric routing, encoded QEC references, measured learning protocols and local jobs.

## Interactive quantum explorer

Open the [live quantum explorer](https://mosure.github.io/burn_quantum/).
The workspace catalog contains eighteen experiments, from Bell/GHZ entanglement
and canonical algorithms to real-data Wine classification, Ising energy
minimization, exact-3SAT constraints and configurable photonic Bell-resource chains.
The [`burn_quantum_visualization`](crates/burn_quantum_visualization/README.md)
crate shares a Bevy/Feathers interface between native and WebGPU browsers:
compact parallel Bloch panels, canonical circuit diagrams, physical routing,
optimizer history and synchronized timelines. **2D probability disks** encode
`P(1)` as filled area, phase as hand angle, and coherence as hand length; purity
remains explicit. **3D spheres** and Cartesian projections are also available.
Settings groups secondary controls, and the experiment list uses aligned,
wrapping rows. Configure algorithm size before running or import validated JSON.

The URL preserves the selected example, applied and pending configuration,
quantum/fabric time, selections, camera, playback and display settings. Imported
experiments require their exported JSON alongside the link.

```sh
cargo install burn_quantum_visualization --version 0.4.0 --locked
quantum_explorer
# Or run the viewer and embedding example from this workspace:
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer
cargo run --locked -p burn_quantum_visualization --example embedded
```

Snapshots are computed once and bounded explicitly. An optional shared-WGPU
bridge lets a renderer read eligible Burn tensor buffers on the same GPU device;
the tutorial gallery itself uses portable CPU simulation. See the
[architecture and acceptance plan](docs/visualization-plan.md) for the data,
rendering and future scale-out contracts.

## Canonical quantum AI/ML examples

The workspace includes [`burn_quantum_example`](crates/burn_quantum_example/README.md),
with one complete source file per implementation: **VQE**, **QAOA MaxCut**,
**variational classification**, **quantum-kernel classification**, a **Born
machine**, **hybrid regression**, **qutrit multiclass classification**,
**Grover SAT search**, **quantum natural SPSA VQE**, and **lossy photonic Bell swapping**.
Additional configurable examples cover **teleportation**, **Toffoli**, **QFT**,
**Deutsch–Jozsa**, **phase estimation**, **Grover search** and the **bit-flip code**.
The larger application examples add **4–8-qubit Wine classification**, **4–8-spin
Ising VQE**, and **exact-three-literal SAT** with an independent small-instance
reference. Wine uses 178 real chemical analyses, all thirteen features, a fixed
stratified train/validation/test split, and both majority and linear-ridge controls.
Each has an explicit problem, execution or training procedure, numerical
acceptance criteria, a test, and a JSON result report.

```sh
cargo run --locked -p burn_quantum_example --bin vqe
cargo run --locked -p burn_quantum_example --bin born_machine
cargo run --locked -p burn_quantum_example --bin qnspsa_vqe
cargo run --locked -p burn_quantum_example --bin wine_classifier -- 6 2 64 731
cargo run --locked -p burn_quantum_example --bin ising_vqe -- 6 2
cargo run --locked -p burn_quantum_example --bin three_sat -- 5 sat
cargo run --locked -p burn_quantum_example --bin fbqc_bell_swap
bash scripts/run_examples.sh
bash scripts/validate_backend.sh wgpu
bash scripts/validate_backend.sh cuda --fusion
cargo test --locked --workspace --all-targets
```

Shared quantum operations live in the core crate; model-specific datasets and
training logic remain in each example. See the [suite guide](crates/burn_quantum_example/README.md)
and [example validation record](docs/examples-validation.md). The core crate's
[`examples/`](crates/burn_quantum/examples/) directory contains basic simulator
demonstrations and device probes.

## Photonic and fusion-based computing

`burn_quantum::photonic` separates encoded states and loss, ideal resource
recipes, destructive Bell measurements, resource-level programs, timed fabric
routes and simulation. `FusionProgram` compiles small resource/fusion algorithms
without allocating a quantum state. `DenseFusionSimulator` executes all exact
branches on native Burn devices, retaining failure/loss probabilities, surviving
states and unconditional shot records. Local qubit circuits can be encoded with
their parameter/input bindings and autodiff intact.

The local stack includes calibrated multimode optics, standard and Grice boosted
Bell instruments, universal bounded gate synthesis, noisy resource factories,
finite buffers, independent wavelength/rail routing and input-preserving parity
retries. Four-star/six-ring networks, physical Shor fusions, open-boundary planar
logical protocols and magic-state injection/distillation have explicit model,
noise, decoder and cost contracts. Photonic learning includes measured mixed-state
overlap protocols, calibration identifiability and matched-record controls.

[`MpsState` and `MpsSimulator`](docs/tensor-network.md) add scalable low-entanglement
non-Clifford execution. Contractions use native Burn devices; complex SVD is
bounded host work with no reverse-mode claim. Local process workers can own
separate tensors of the same entangled state, with checked transaction recovery.
Asynchronous local jobs preserve raw outcomes, provenance and physical costs.
[OpenQASM interchange](docs/photonic-interop.md) is checked against the actual
pinned PsiQDK local SDK simulator.

Read the [capability and acceptance map](docs/photonic-status.md),
[optical API](docs/photonic-optics.md), [compiler guide](docs/photonic-compiler.md),
[runtime guide](docs/photonic-runtime.md), [QEC/boundary guide](docs/photonic-stabilizer.md),
[growing-distance surgery and thresholds](docs/photonic-surgery.md),
[magic-state guide](docs/photonic-magic.md)
and [original staged roadmap](docs/photonic-roadmap.md). Model-specific logical
references do not establish a hardware threshold or a production fault-tolerant
fabric compiler. PsiQuantum is the long-term hardware target; execution here is
local. The application suite and qualification scripts run from a repository checkout.

## Quantum natural SPSA

`burn_quantum::optim::QnSpsaConfig` initializes a quantum natural optimizer for
native Burn parameter tensors. Loss and pure-state fidelity callbacks provide
fresh evaluations on the selected device; no backward pass is needed. Each
resampling uses two loss and four fidelity calls to estimate the gradient and
Fubini–Study metric (QFI/4). Optional blocking, positive spectral regularization,
decay schedules and versioned checkpoints support reproducible training.

The dense metric and solve run in host F64 with O(d²) storage and O(d³) work for
d parameters. See the [algorithm, API, numerical limits and validation](docs/qnspsa.md)
and the [single-file example](crates/burn_quantum_example/src/bin/qnspsa_vqe.rs).
Run the optimizer example from a repository checkout.

## Boolean satisfiability

`burn_quantum::sat::{Cnf, GroverSat, SearchConfig, SatOutcome}` provides the solver
API. Run its single-file `grover_sat` example from a repository checkout:

```sh
cargo run --locked -p burn_quantum_example --bin grover_sat
cargo run --locked -p burn_quantum_example --bin grover_sat -- problem.cnf
BURN_QUANTUM_DEVICE=wgpu cargo run --locked -p burn_quantum_example --features wgpu --bin grover_sat
BURN_QUANTUM_DEVICE=cuda cargo run --locked -p burn_quantum_example --features cuda,fusion --bin grover_sat
```

The solver compiles clause logic without precomputing satisfying assignments,
uncomputes its work qubits, and checks every sampled candidate. A failed bounded
search returns `Inconclusive`; only an explicit empty clause produces a proven
`Unsatisfiable` result. Dense simulation uses exponentially many amplitudes and
does not establish quantum advantage. See the [SAT API and algorithm guide](docs/sat.md).

## Differentiable quantum layer

```rust
use burn::tensor::Device;
use burn_quantum::{Angle, Circuit, Observable, QuantumLayerConfig, Register,
                   Rotation, gate::Axis};

fn main() -> burn_quantum::Result<()> {
    let device = Device::flex().autodiff();
    let mut circuit = Circuit::new(Register::qubits(1)?, 1, 0);
    circuit.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))?;
    let config = QuantumLayerConfig::new(circuit, vec![Observable::z(0)], vec![0.3]);
    let layer = config.init(&device)?;
    let features = layer.forward(None, 8)?; // [batch=8, readouts=1]
    let loss = (features + 1.0).square().mean();
    let gradients = loss.backward();
    assert!(layer.parameters.val().grad(&gradients).is_some());
    Ok(())
}
```

Use `Angle::input(i)` with `forward(Some(features), batch)` for classical input
encoding. Input tensors and quantum parameters remain in the same autodiff
graph, including through surrounding classical layers. Declare feature counts
on `Circuit::new`; no implicit reshaping or dtype conversion is performed.
Only first-order reverse-mode derivatives are promised by Burn.

Save the serialized `QuantumLayerConfig` beside `layer.into_record()` or
`layer.save_file(...)`. The weight record contains parameters; circuit structure,
readout order, and simulator limits live in the config. Reload the config, call
`init`, then use Burn's `try_load_record` / `try_load_file`.

## Run and validate

```sh
cargo run --locked -p burn_quantum --example bell
cargo run --locked -p burn_quantum --example qudit
cargo run --locked -p burn_quantum --example train
cargo run --locked -p burn_quantum --example hybrid
cargo run --locked -p burn_quantum --release --example profile
cargo test --locked --workspace --all-targets
cargo test --locked --workspace --doc
cargo clippy --locked --workspace --all-targets -- -D warnings
```

`train` optimizes a variational expectation with Adam. `hybrid` trains a
classical encoder, entangled two-qubit circuit, and classical output head end to
end on a small deterministic regression problem. These are integration
demonstrations, not evidence of quantum advantage or a research benchmark.

Default features select the portable `flex` CPU backend and `autodiff`.
Consumers can select devices with `default-features = false` and enable
`flex`, `wgpu`, `cuda`, `fusion`, or `autodiff` as appropriate. The library itself
can compile without an execution backend; actually creating tensors requires
one. Enable the same features in your direct Burn dependency where needed.

```sh
cargo check --locked -p burn_quantum --no-default-features
cargo check --locked -p burn_quantum --features wgpu,fusion --all-targets
cargo run --locked -p burn_quantum --features wgpu --example wgpu_smoke
```

`BURN_QUANTUM_DEVICE=wgpu` selects a discrete GPU in all example executables;
enable `--features wgpu` on their package. `cuda` and `wgpu-integrated` are also
supported selectors, and GPU requests fail if the requested hardware is absent.
The validation scripts select features automatically and record the adapter.
See [CPU/GPU runtime validation](docs/backend-validation.md) for tested hardware,
precision, parity tolerances and reproducible commands.

## Numerical and resource boundaries

For `N = product(local_dimensions)` and batch `B`, a pure state stores `2BN`
real scalars; a density matrix stores `2BN²`. Applying a local dense gate of
dimension `K` costs `O(BNK)` for the contraction plus `O(Nm)` for current index
planning over `m` wires. Local application does not build an
`N x N` operator. Autodiff adds depth-dependent retained activations, and the
reference implementation constructs gather plans at each application.

Quantum contractions preserve operand precision: Flex uses native GEMM; other
devices use full-precision multiply/reduce contractions. This avoids the locked
CubeCL stack's accelerated F32-to-TF32 conversion, which can break state
normalization. The portable GPU path may require more kernel launches; optimized
full-precision quantum kernels remain future work.

`ResourceLimits` checks complex state entries, batched operator entries, and
per-operation gather indices before allocation. Defaults are `2^22`, `2^20`
and `2^24` respectively. A stored pure state at the default limit alone takes
32 MiB in F32 or 64 MiB in F64; intermediates, host copies, plans and the
autodiff graph require more. Limits are not a total process/device memory cap.
Built-in dense gate factories separately cap allocation at 65,536 matrix
entries; larger explicit matrices can use `Matrix` / `Unitary` subject to
engine limits. Dense gather indices are restricted to signed 32-bit range.

Constant unitaries, Hermitian observables and Kraus completeness are validated.
`from_amplitudes`, `from_entries`, and `apply_tensor` validate metadata without
synchronizing tensor values. Call state `validate(tolerance)` explicitly to
check physical values. Arbitrary tensor operators may be nonunitary; users
must ensure physical semantics and finite parameter/input values. Sampling is
an explicit synchronous, nondifferentiable readback and leaves the state intact.

There is no blanket guarantee of efficient simulation for arbitrary quantum
applications. Dense states, MPS, Clifford trajectories and truncated Fock models
have different explicit limits. The checkout includes adaptive photonic control,
asynchronous local jobs and distributed MPS ownership; remote hardware execution
and multi-host performance remain unqualified. Continuous-variable results
require convergence within the declared cutoff and optical model. See the
[capability map](docs/photonic-status.md) for those distinctions.

Read the [architecture and initial plan](docs/architecture.md),
[engineering and research roadmap](docs/roadmap.md), and
[validation record](docs/validation.md).

## Repository organization

The repository root is a virtual Cargo workspace. The packages live under
`crates/`: [`burn_quantum`](crates/burn_quantum/README.md) owns the public library,
integration tests and simulator demos; [`burn_quantum_example`](crates/burn_quantum_example/README.md)
contains one executable source file per application;
[`burn_quantum_visualization`](crates/burn_quantum_visualization/README.md) owns
experiment traces, visualization adapters and the native/browser workbench.
Circuit, SAT, QN-SPSA and photonic
internals are split by responsibility behind stable public exports. Shared
metadata, dependency versions and lints live in the root manifest. Repository
guides, validation scripts and CI stay at the root; runner and comparison
scripts discover example binaries through Cargo metadata.

See the [repository map](docs/organization.md), [contributing guide](CONTRIBUTING.md),
and [documentation index](docs/README.md) for module ownership and validation workflows.

## License

MIT OR Apache-2.0.
