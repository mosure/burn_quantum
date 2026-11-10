# burn_quantum_example

Small, executable quantum AI/ML, search and photonic implementations using Burn's native
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
| [`fbqc_bell_swap`](src/bin/fbqc_bell_swap.rs) | Two Bell resources, contended fabric routing, rail loss and destructive fusion; unconditional shots | Analytic 0.36 success / 0.28 loss, corrected Bell fidelity and schedule checks |
| [`photonic_hom`](src/bin/photonic_hom.rs) | Bosonic two-photon interference, loss and detection | Analytic bunching, number conservation, unfiltered detection probabilities |
| [`photonic_mz_learning`](src/bin/photonic_mz_learning.rs) | Burn Module/Param/Adam optical phase learning | Three initializations, disjoint held-out inputs and analytic baseline |
| [`photonic_teleportation`](src/bin/photonic_teleportation.rs) | Logical measurement compilation with outcome-dependent corrections | Direct circuit equivalence and absolute branch normalization |
| [`fbqc_factory_pipeline`](src/bin/fbqc_factory_pipeline.rs) | Heralded factories, timed fabric, host tableaux and resumable local shards | Unconditional success, source cost and selected-device dense oracle |
| [`fbqc_repetition_memory`](src/bin/fbqc_repetition_memory.rs) | Host Clifford repetition-memory experiment for a specified binary error channel | Logical correlations, erasure decoding and dense device reference |
| [`physical_vqe`](src/bin/physical_vqe.rs) | VQE through routed heralded local/CNOT channels | Singlet energy, native gradient and abort-aware photon costs |
| [`fbqc_adaptive`](src/bin/fbqc_adaptive.rs) | Correlated source outages, finite buffers and bounded fresh-resource retries | Exact unconditional yield/cost, causal feedback and checkpoint replay |
| [`fbqc_remote`](src/bin/fbqc_remote.rs) | Two real process/TCP photonic trajectory workers | Byte-identical local replay, dense device oracle and communication cost |
| [`fbqc_four_star_memory`](src/bin/fbqc_four_star_memory.rs) | Four-star toric memory with loss, Pauli noise and bounded decoding | Multiple distances, unconditional Wilson intervals and device resource-state oracle |
| [`photonic_mz_shots`](src/bin/photonic_mz_shots.rs) | Finite-shot lossy phase training with parameter-shift estimates | Three starts, held-out MSE, frozen control and attempted-shot budget |
| [`photonic_kernel`](src/bin/photonic_kernel.rs) | Lossy mixed-state Hilbert-Schmidt kernel classifier | Held-out accuracy, majority and matched classical controls |
| [`photonic_born_qnspsa`](src/bin/photonic_born_qnspsa.rs) | Full-rank lossy Born model with squared-Uhlmann QN-SPSA | Three starts, disjoint sampled holdout, analytic Bures metric and evaluator cost |

| [`physical_qaoa`](src/bin/physical_qaoa.rs) | QAOA through calibrated resource factories, optical effects and finite buffers | Exact conditional objective, false-herald/failure semantics and physical cost |
| [`photonic_benchmarks`](src/bin/photonic_benchmarks.rs) | Five-seed classification, mixed kernels and Born fitting from finite records | Disjoint calibration/holdout, matched-record classical controls, photons and illustrative time/energy proxies |
| [`photonic_magic_distillation`](src/bin/photonic_magic_distillation.rs) | Reed-Muller 15-to-1 distillation and T injection | Native encoder/transversal/decoder oracle, exhaustive phase faults, noisy acceptance and resource cost |
| [`fbqc_six_ring_memory`](src/bin/fbqc_six_ring_memory.rs) | Bare and physical Shor-encoded six-ring toric studies | Correlated loss, independent stabilizer fixtures, multiple sizes and unconditional confidence intervals |
| [`planar_logical_memory`](src/bin/planar_logical_memory.rs) | Open planar code memories and H/CNOT/S circuit references | Entangled logical channels, single-fault/loss recovery and explicit gate-level cost |
| [`fbqc_lattice_surgery`](src/bin/fbqc_lattice_surgery.rs) | Growing local merge/split with actual six-ring lowering | Native resource oracle, encoded Choi channels, distance certificates and retained physical aborts |
| [`fbqc_threshold`](src/bin/fbqc_threshold.rs) | Periodic six-ring outcome erasure and encoded optical loss | Exact joint erasure law, logical homology and fixed-shot simultaneous intervals |
| [`planar_threshold`](src/bin/planar_threshold.rs) | Repeated noisy-check planar memories | Bounded union-find, unconditional failures and independent confirmation study |
| [`surgery_threshold`](src/bin/surgery_threshold.rs) | Complete noisy XX/ZZ merge/split transitions | Four entangled-channel labels, spacetime distances and explicit finite-size brackets |
| [`surface_threshold`](src/bin/surface_threshold.rs) | Actual six-ring surface memories and fusion-seam merges | Resource-derived detectors, growing record distance and a fixed outcome-noise study |
| [`mps_qaoa`](src/bin/mps_qaoa.rs) | Non-Clifford QAOA on an 80-qubit MPS ring | Analytic cut law, random control, Schmidt storage and approximation bounds |
| [`distributed_mps`](src/bin/distributed_mps.rs) | One entangled state owned by one/three real worker processes | Complete dense amplitudes, streamed correlations and injected commit recovery |
| [`photonic_jobs`](src/bin/photonic_jobs.rs) | Immutable asynchronous lossy optical jobs | Native phase oracle, exact raw-shot replay and counted detector misses |

| [`fbqc_open_boundaries`](src/bin/fbqc_open_boundaries.rs) | Finite six-ring strips/boxes with actual boundary measurements | Independent branch amplitudes, signed logical identity/H correlations and correlated-loss records |
| [`psiqdk_interop`](src/bin/psiqdk_interop.rs) | Bounded logical export to pinned PsiQDK local simulation | Actual SDK complex-state fixtures, bit ordering, global phase and readout mapping |

The implementations use established algorithm families on intentionally small
problems, with explicit models and objectives. VQE and QAOA cover variational
optimization; the other examples cover supervised, kernel, generative, and
hybrid learning. The qutrit example adapts a variational classifier to a
three-level register; it is not a claim to reproduce a particular published
qudit architecture.
Photonic examples cover resource-level primitives, physical algorithm training,
optical learning, scheduling/control, logical-memory studies and process workers.
Each names its physical assumptions; none demonstrates quantum advantage.

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
cargo run --locked -p burn_quantum_example --bin fbqc_bell_swap
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
cargo test --locked -p burn_quantum_example --bins  # runs every binary's acceptance checks
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

Shared selection lives in [`burn_quantum/support/device.rs`](../burn_quantum/support/device.rs), used
by the example package, basic simulator demos and backend parity tests. The core
library continues to accept ordinary Burn devices without reading environment
variables. The original F64 CPU oracle tests retain their strict tolerances;
[`backend_parity.rs`](../burn_quantum/tests/backend_parity.rs) compares portable F32
execution and derivatives against CPU/scalar/analytic references.

Training uses exact simulator gradients/expectations. QN-SPSA instead estimates
its updates from perturbed energies and state fidelities, without backpropagation;
see the [algorithm and validation guide](../../docs/qnspsa.md). Terminal shot sampling is
shown in QAOA, the Born machine and SAT search. These tests do not establish shot-based QPU
training, large-scale performance, or quantum advantage. See the
[hardware validation record](../../docs/backend-validation.md) for actual runtime
coverage and the [manual GPU CI workflow](../../.github/workflows/gpu.yml).

## What belongs in the core

Reusable quantum operations live in [`burn_quantum::qml`](../burn_quantum/src/qml.rs):

- `angle_embedding`: ordered qubit feature encoding with atomic validation.
- `ry_cnot_layer`: an explicitly defined reusable real-amplitude circuit layer.
- `pauli_z_probabilities`: consistent P(0)/P(1) conversion from an expectation.
- `distribution_cross_entropy`: differentiable probability loss for one-hot or
  soft targets, shared by binary, generative and qutrit models.
- `fidelity_kernel`: differentiable batched Gram/cross-kernel construction.

[`StateVector::fidelity`](../burn_quantum/src/state.rs) supports state-overlap objectives
and reference checks. `QuantumLayerConfig` accepts empty observables for
state-based objectives; use `state` for energies/probabilities and `forward` for
configured Hermitian readouts. Calling `forward` without readouts returns an
error. Existing weight-record parameter paths remain unchanged.

[`burn_quantum::sat`](../burn_quantum/src/sat/mod.rs) contains CNF parsing, reversible oracle
compilation and bounded search. Controlled gates and circuit continuation also
live in the core. The SAT example contains only its problem, analytic evaluation
and command-line reporting.

[`burn_quantum::optim`](../burn_quantum/src/optim/mod.rs) contains the reusable QN-SPSA
estimator, schedules, metric regularization/solve, blocking and checkpoints.
The example supplies its ansatz, loss/fidelity callbacks and independent
evaluation. Quantum evaluations stay on the selected device; the dense metric
is processed in host F64.

[`burn_quantum::photonic`](../burn_quantum/src/photonic/mod.rs) owns the dual-rail,
loss, resource/fusion, fabric and execution APIs. The Bell-swap example supplies
its fabric/problem and checks success, loss, scheduling and corrected fidelity.
See [photonic model assumptions](../../docs/photonic-fbqc.md) and its
[CPU/GPU qualification](../../docs/photonic-validation.md).

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
- FBQC: [Bartolucci et al., Fusion-based quantum computation](https://arxiv.org/abs/2101.09310).
- QAOA: [Farhi, Goldstone and Gutmann, A Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028).
- Variational classification: [PennyLane's variational classifier tutorial](https://pennylane.ai/demos/tutorial_variational_classifier).
- Fidelity kernels: [Qiskit's fidelity-kernel definition](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.TrainableFidelityQuantumKernel.html).
- Born machines: [PennyLane's quantum circuit Born-machine tutorial](https://pennylane.ai/demos/tutorial_qcbm).

See [the original seven-example validation](../../docs/examples-validation.md)
and [the SAT guide](../../docs/sat.md) for recorded outcomes and limitations.

The reusable MPS engine and its local/distributed streaming contractions live in
`burn_quantum::tensor_network`. Optical tomography, calibrated likelihoods,
shot allocation, resource costs, logical code protocols and immutable job contracts
also live in the core. Their examples contain only problem setup, training/study
policy and independent acceptance checks. Complex MPS SVD, tableau/decoder work
and classical scheduling remain host operations even when native numerical
oracles/contractions run on the requested GPU.

## Configurable canonical circuit examples

Each algorithm has one executable file and reuses `burn_quantum::algorithms`.
The optional positional size argument is qubits, controls, input/precision bits,
code length, or independent transfers as documented at the top of each file.

```sh
cargo run -p burn_quantum_example --bin qft -- 8
cargo run -p burn_quantum_example --bin deutsch_jozsa -- 8
cargo run -p burn_quantum_example --bin phase_estimation -- 8
cargo run -p burn_quantum_example --bin grover_search -- 8
cargo run -p burn_quantum_example --bin toffoli -- 6
cargo run -p burn_quantum_example --bin bit_flip_code -- 7
cargo run -p burn_quantum_example --bin teleportation -- 2
```

The existing `qaoa_maxcut` training executable now uses the shared symbolic graph
builder. All these programs honor `BURN_QUANTUM_DEVICE`, including explicit WGPU
selection with `--features wgpu`. They report independent numerical acceptance
criteria, and teleportation checks all measurement branches. Interactive versions
are available in the visualization gallery, including configurable QAOA layers.
