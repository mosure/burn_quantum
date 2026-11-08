# burn_quantum_example

Nine small, executable quantum AI/ML and search implementations using Burn's native
tensors, optimizers, and the `burn_quantum` simulator. Every algorithm is a single file
under `src/bin/`, containing its circuit, data/problem definition, training or
fit procedure, evaluation, `main`, and a test. The package is a local workspace
member and is not published.

## Examples

| Binary / source | Implementation | Independent evaluation |
| --- | --- | --- |
| [`vqe`](src/bin/vqe.rs) | Two-qubit VQE for `H = XX + YY + ZZ`, RY/CNOT ansatz, Adam | Energy against the analytic -3 ground energy and fidelity against the singlet |
| [`qaoa_maxcut`](src/bin/qaoa_maxcut.rs) | Depth-one QAOA with shared gamma/beta, ZZ cost evolution and X mixer | Exact enumeration of triangle cuts, optimal probability and sampled best cut |
| [`variational_classifier`](src/bin/variational_classifier.rs) | Angle encoding, variational entangling layer, Pauli-Z binary readout, cross-entropy | Disjoint continuous-parity test grid and majority baseline |
| [`quantum_kernel`](src/bin/quantum_kernel.rs) | Product RY feature map, fidelity Gram matrix, kernel ridge classifier | Gram invariants, classical solve residual and disjoint parity test grid |
| [`born_machine`](src/bin/born_machine.rs) | Three-qubit Born machine, soft-target cross-entropy, Adam | KL divergence, total variation and seeded sampled frequencies |
| [`hybrid_regression`](src/bin/hybrid_regression.rs) | Burn linear encoder, entangled quantum layer, linear head | Disjoint regression test grid and parameter updates in all three stages |
| [`qudit_classifier`](src/bin/qudit_classifier.rs) | Single-qutrit three-class model using explicit two-level rotations and Born readout | Disjoint angle-cluster test data and multiclass majority baseline |
| [`grover_sat`](src/bin/grover_sat.rs) | Reversible CNF oracle, Grover amplification, bounded search with unknown solution count; optional DIMACS file | Analytic success probability, clean work qubits, verified witness |
| [`qnspsa_vqe`](src/bin/qnspsa_vqe.rs) | Fidelity-based quantum natural SPSA, two-qubit RY/CNOT/RZ ansatz, Heisenberg energy | Analytic -3 ground energy, singlet fidelity, normalization and evaluator-call budget |

The implementations use established algorithm families on intentionally small
problems, with explicit models and objectives. VQE and QAOA cover variational
optimization; the other examples cover supervised, kernel, generative, and
hybrid learning. The qutrit example adapts a variational classifier to a
three-level register; it is not a claim to reproduce a particular published
qudit architecture.

## Run

From the repository root:

```sh
cargo run --locked -p burn_quantum_example --bin vqe
cargo run --locked -p burn_quantum_example --bin qaoa_maxcut
cargo run --locked -p burn_quantum_example --bin variational_classifier
cargo run --locked -p burn_quantum_example --bin quantum_kernel
cargo run --locked -p burn_quantum_example --bin born_machine
cargo run --locked -p burn_quantum_example --bin hybrid_regression
cargo run --locked -p burn_quantum_example --bin qudit_classifier
cargo run --locked -p burn_quantum_example --bin grover_sat
cargo run --locked -p burn_quantum_example --bin grover_sat -- problem.cnf
cargo run --locked -p burn_quantum_example --bin qnspsa_vqe
```

Runner and comparison scripts discover the binary targets through Cargo metadata
using [`example_catalog.py`](../../scripts/example_catalog.py); adding an example
does not require maintaining separate executable lists.

Each command prints a JSON report and returns an error if its stated numerical
acceptance criteria fail. Counts, learning rates, and seeds are visible in the
single source file. `seed = 0` denotes an explicitly initialized deterministic
run with no random data. VQE/Born use local ChaCha8 initialization; QAOA/Born
sampling uses an explicit seed. QN-SPSA uses a local seed for its perturbation
directions. No datasets are downloaded.

```sh
bash scripts/run_examples.sh cpu                   # artifacts/examples/cpu/*.json
cargo test --locked -p burn_quantum_example --bins  # runs all nine checks
cargo test --locked --workspace --all-targets       # core and example regressions
bash scripts/validate.sh                           # full workspace qualification
```

The examples use F32 on the explicitly selected device, defaulting to Flex CPU.
The same source, initialization, training budget and acceptance criteria run on
GPU. Select hardware and compile its feature together:

```sh
BURN_QUANTUM_DEVICE=wgpu cargo run --locked -p burn_quantum_example --features wgpu --bin vqe
BURN_QUANTUM_DEVICE=cuda cargo run --locked -p burn_quantum_example --features cuda --bin qudit_classifier
bash scripts/validate_backend.sh wgpu              # parity tests, example tests and executables
bash scripts/validate_backend.sh wgpu --fusion     # same checks with kernel fusion
bash scripts/validate_backend.sh cuda
bash scripts/validate_backend.sh cuda --fusion
python3 scripts/compare_backends.py artifacts/examples/cpu artifacts/examples/wgpu
```

`wgpu` requires a discrete GPU; `wgpu-integrated` selects an integrated GPU.
`BURN_QUANTUM_DEVICE_INDEX` selects the adapter within that class, defaulting to
zero. CUDA also uses that index. Requested hardware that is unavailable fails
the run. `wgpu-cpu` explicitly selects a software WGPU adapter for diagnostics;
it is never an automatic fallback and is not evidence of hardware GPU execution.
The JSON includes backend, actual adapter, precision and the example's `fusion`
feature selection. Reports are separated into `artifacts/examples/<backend>/`
or `<backend>-fusion/` directories.

Shared selection lives in [`support/device.rs`](../../support/device.rs), used
by the example package, basic simulator demos and backend parity tests. The core
library continues to accept ordinary Burn devices without reading environment
variables. The original F64 CPU oracle tests retain their strict tolerances;
[`backend_parity.rs`](../../tests/backend_parity.rs) compares portable F32
execution and derivatives against CPU/scalar/analytic references.

Training uses exact simulator gradients/expectations. QN-SPSA instead estimates
its updates from perturbed energies and state fidelities, without backpropagation;
see the [algorithm and validation guide](../../docs/qnspsa.md). Terminal shot sampling is
shown in QAOA, the Born machine and SAT search. These tests do not establish shot-based QPU
training, large-scale performance, or quantum advantage. See the
[hardware validation record](../../docs/backend-validation.md) for actual runtime
coverage and the [manual GPU CI workflow](../../.github/workflows/gpu.yml).

## What belongs in the core

Reusable quantum operations live in [`burn_quantum::qml`](../../src/qml.rs):

- `angle_embedding`: ordered qubit feature encoding with atomic validation.
- `ry_cnot_layer`: an explicitly defined reusable real-amplitude circuit layer.
- `pauli_z_probabilities`: consistent P(0)/P(1) conversion from an expectation.
- `distribution_cross_entropy`: differentiable probability loss for one-hot or
  soft targets, shared by binary, generative and qutrit models.
- `fidelity_kernel`: differentiable batched Gram/cross-kernel construction.

[`StateVector::fidelity`](../../src/state.rs) supports state-overlap objectives
and reference checks. `QuantumLayerConfig` accepts empty observables for
state-based objectives; use `state` for energies/probabilities and `forward` for
configured Hermitian readouts. Calling `forward` without readouts returns an
error. Existing weight-record parameter paths remain unchanged.

[`burn_quantum::sat`](../../src/sat/mod.rs) contains CNF parsing, reversible oracle
compilation and bounded search. Controlled gates and circuit continuation also
live in the core. The SAT example contains only its problem, analytic evaluation
and command-line reporting.

[`burn_quantum::optim`](../../src/optim/mod.rs) contains the reusable QN-SPSA
estimator, schedules, metric regularization/solve, blocking and checkpoints.
The example supplies its ansatz, loss/fidelity callbacks and independent
evaluation. Quantum evaluations stay on the selected device; the dense metric
is processed in host F64.

Quantum tensor helpers validate shapes, layouts, dtypes, contexts and relevant
budgets. They do not synchronously validate probability values: callers provide
normalized states/distributions, and the examples explicitly validate states at
evaluation boundaries. Cross-entropy clamps the logarithm at the specified
epsilon; gradients below that floor are zero.

The example library contains device selection, reporting, seeded initialization,
readback and classification metrics. Synthetic datasets, classical fitting algorithms,
and model-specific evaluation stay beside their examples. In particular, the
kernel example contains its small classical Cholesky solve rather than adding
a generic linear solver to the quantum library.

## Algorithm references

These references establish the algorithm families; the small problems and Rust
implementations here are original educational implementations, not reproductions
of the referenced experiments.

- VQE: [Peruzzo et al., A variational eigenvalue solver on a quantum processor](https://arxiv.org/abs/1304.3061).
- QN-SPSA: [Gacon et al., Simultaneous Perturbation Stochastic Approximation of the Quantum Fisher Information](https://arxiv.org/abs/2103.09232).
- QAOA: [Farhi, Goldstone and Gutmann, A Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028).
- Variational classification: [PennyLane's variational classifier tutorial](https://pennylane.ai/demos/tutorial_variational_classifier).
- Fidelity kernels: [Qiskit's fidelity-kernel definition](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.TrainableFidelityQuantumKernel.html).
- Born machines: [PennyLane's quantum circuit Born-machine tutorial](https://pennylane.ai/demos/tutorial_qcbm).

See [the original seven-example validation](../../docs/examples-validation.md)
and [the SAT guide](../../docs/sat.md) for recorded outcomes and limitations.
