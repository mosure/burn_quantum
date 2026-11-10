# Application benchmarks and scaling examples

The workspace combines small teaching circuits with inspectable application
pipelines. The eighteen-entry explorer includes real chemical classification,
multi-spin energy minimization, exact-3SAT constraints and a lossy photonic
resource chain. The examples crate contains 47 single-file executables, including
tensor-network workloads with much larger registers. These are local simulator
experiments with explicit assumptions and reference checks, not quantum advantage
or hardware qualification claims.

## Real data: wine cultivar classification

`burn_quantum::qml::benchmarks::wine` uses the public
[UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/winedataset): 178 actual
chemical analyses, thirteen measured features and three cultivar labels. The
original `wine.data` and `wine.names` are bundled; runtime execution has no download
or network dependency. Attribution is Aeberhard & Forina (1992),
[DOI 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J), CC BY 4.0. Original-file
SHA-256 hashes and the separate data license are recorded in
[the fixture README](../crates/burn_quantum/src/qml/benchmarks/data/README.md).

The model uses four to eight qubits, one to three entangling/data-reuploading
layers, and all thirteen chemical measurements in every layer. Each qubit starts
in `|+>`. Feature-dependent Y/Z rotations, mixing CNOTs, trainable RZ→RY rotations
and a CNOT ring form each layer. The measured Z expectations of every qubit feed a
jointly trained three-class linear head. Class probabilities are the classical
head's softmax outputs; they are not three raw computational-basis probabilities.
Adam trains both circuit angles and readout weights with exact Burn autodiff.

Evaluation is deliberately separated from training:

1. A local seeded ChaCha8 RNG creates disjoint class-stratified partitions:
   105 training, 34 validation and 39 test observations, retaining original row IDs.
2. `qml::preprocessing::Standardizer` fits feature means and population standard
   deviations on training rows only. The angle map is the fixed bounded function
   `0.8 * atan(z)`. All thirteen features remain present; no test-driven selection
   or dimensionality reduction occurs.
3. Training records every actual parameter iterate and train/validation loss.
   Minimum validation cross entropy selects the checkpoint. All configured
   updates still execute, so the selected checkpoint can precede the last iterate.
4. After checkpoint selection, that model evaluates the test partition once.
   The report preserves every predicted probability and true label, the confusion
   matrix (true rows, predicted columns), accuracy and cross entropy.
5. Controls use exactly the same partitions: the training majority class and a
   classical linear-ridge classifier using all thirteen standardized features.
   Ridge selects regularization from `[0.001, 0.01, 0.1, 1, 10]` using validation
   loss. Its intercept is unpenalized; the fixed softmax logit multiplier is four.
   No control hyperparameter is selected using test performance.

The default is four qubits, two layers, 64 updates and seed 731. The bounds are
4–8 qubits, 1–3 layers and 1–128 updates. There are `2 * qubits * layers` quantum
angles and `3 * qubits + 3` classical parameters. A full run executes
`2 * (updates + 1) + 1` batched quantum forward passes, including validation and
final test evaluation; backward work is additional. Dense state storage scales
with batch size times `2^qubits`, and autodiff retains intermediate tensors.

```sh
cargo run --release --locked -p burn_quantum_example --bin wine_classifier -- 4 2 64 731
cargo run --release --locked -p burn_quantum_example --bin wine_classifier -- 6 2 64 732
BURN_QUANTUM_DEVICE=wgpu cargo run --release --locked -p burn_quantum_example --features wgpu --bin wine_classifier -- 4 2 64 731
cargo test --locked -p burn_quantum_example --bin wine_classifier
```

The `wine` explorer entry calls this same pipeline. Its Bloch panels show one
fixed held-out input chosen by the seeded partition order, through each actual
training iterate. That representative quantum state is distinct from dataset-wide
accuracy. The study reports the validation-selected checkpoint, test predictions
and baselines; the last timeline frame remains the actual last optimizer iterate.

`WineRun::model()` exports a serializable `WineModel` containing the architecture,
training scaler, selected quantum angles and classical head. The CLI includes
this checkpoint as `model` in its JSON report. Restore it to classify new raw
chemical measurements without retraining:

```rust
use burn::tensor::Device;
use burn_quantum::qml::benchmarks::wine::WineModel;

// `report` is the CLI JSON and `measurements` is Vec<Vec<f64>> in FEATURE_NAMES order.
let model: WineModel = serde_json::from_value(report["model"].clone())?;
let cultivar_probabilities = model.predict(&measurements, &Device::flex()).await?;
```

Inference validates the checkpoint version, shapes, finite F32 parameters and
training statistics. Each call accepts 1–4096 thirteen-feature observations and
at most 1,048,576 complex state elements. It never refits preprocessing. Tests
roundtrip the checkpoint through JSON, reproduce all 39 held-out predictions and
reject invalid shapes, nonfinite values and oversized batches.

## Many-body task: Ising ground-state preparation

`qml::benchmarks::ising` minimizes the open-chain Hamiltonian
`H = -sum_i Z_i Z_(i+1) - sum_i X_i`, in units `J = h = 1`. Its real-amplitude
ansatz prepares `|+>` on each site, alternates trainable RY rotations with CNOT
chains, and measures local X/ZZ terms. The bounds are four to eight spins and one
to three entangling layers; the default takes 80 Adam updates. The API permits
1–128 updates.

An independently assembled dense Hamiltonian supplies the exact smallest
eigenvalue. A separate analytic reference minimizes the spatially uniform product
state. The report gives actual energy, the variational bound and error per spin.
Neither exact diagonalization nor dense statevector training is a scalable
many-body solver; this example makes approximation quality visible as system size
and ansatz depth change.

```sh
cargo run --release --locked -p burn_quantum_example --bin ising_vqe -- 6 2
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo vqe --size 6 --layers 2 --view learning
```

## Constraint task: exact-three-literal satisfiability

`sat::benchmarks::ThreeSatBenchmark` produces explicitly displayed three-literal
clauses with distinct variables. A reversible compute/mark/uncompute oracle uses
one work qubit per clause. One actual Grover round amplifies satisfying assignment
probability; reported values come from the resulting distribution, with work
qubits marginalized and their cleanup checked separately.

The CLI supports three to six Boolean variables; the explorer caps the retained
trace at three to five. Satisfiable and contradictory variants have independent
bounded exhaustive references. A returned witness is checked against every
clause. An `UNSAT` claim comes from that exhaustive reference, not from a failed
quantum sample or an arbitrary Grover budget. For general bounded search, the
existing `sat` API retains its `Inconclusive` outcome when no proof is available.

```sh
cargo run --locked -p burn_quantum_example --bin three_sat -- 5 sat
cargo run --locked -p burn_quantum_example --bin three_sat -- 5 unsat
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo three-sat --size 5 --view circuit
```

Dense storage is exponential in variables **plus clause work qubits**. This
educational reference is not intended to compete with classical industrial SAT
solvers; it exposes the circuit, readout, witness and resource consequences.

## Photonic qubits, loss and resource delivery

Photonic dual-rail computation encodes one logical qubit in two optical modes.
The extra local level in a loss-aware model is vacuum, not a requirement to
compute with logical qutrits. The explorer defaults to qubit resource states and
offers an optional `|0>, |1>, |vac>` representation. Both use the same physical
resource graph and compiled fabric plan. See
[PsiQuantum's description of its photonic qubits](https://www.psiquantum.com/quantum-computing)
for the physical dual-rail encoding motivation.

The configurable chain uses two to six Bell resources, a shared switch and
destructive fusion analyzer, finite route/storage waits, rail attenuation and
detector transmission. It samples 2,048 unconditional attempts with a fixed seed,
retaining erased fusions, ordinary failures, heralded-but-lost outputs and
successful delivery. An independent analytic law checks this specific balanced,
independent-loss Bell chain; its standard unboosted fusions have 50% ideal success
per connection. Conditional corrected Bell fidelity is reported alongside
unconditional yield, never substituted for it.

The state timeline is **ideal resource preparation**. It does not invent a pure
state trajectory for mixed photon loss, and animated packets represent scheduled
routing rather than successful detection. Larger optional vacuum representations
can exceed the explicitly displayed trace budget even when logical qubits fit.

```sh
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo photonic --size 4 --variant 0 --view fabric
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo photonic --size 3 --variant 1 --view fabric
```

These are transparent local FBQC ingredients relevant to future photonic systems.
They do not specify PsiQuantum's proprietary architecture, calibrations or device
access. See the [photonic status](photonic-status.md), [local interoperability](photonic-interop.md)
and [growing-distance qualification](photonic-surgery-validation.md) for the
implemented models and their separate evidence.

## Larger registers through tensor networks

`mps_qaoa` accepts 4–256 qubits (80 by default). It executes non-Clifford p=1 QAOA
on a ring, checks the analytic cut fraction against a random-cut control, and
reports retained complex elements and approximation error. A nine-point small
reference search precedes the configured large-ring evaluation. Maximum bond 16
and accumulated trace-distance budget `1e-4` are explicit.

```sh
cargo run --release --locked -p burn_quantum_example --bin mps_qaoa -- 80
cargo run --release --locked -p burn_quantum_example --bin mps_qaoa -- 128
```

MPS contractions use the selected Burn device; complex SVD remains explicit host
work without autodiff. The ability to run this low-entanglement family with many
qubits does not imply efficient simulation of arbitrary deep circuits. Other
examples cover distributed MPS ownership, bounded Clifford/frame sampling and
photonic QEC. Each representation has different assumptions and costs.

## Reuse and runtime qualification

Enable the core `benchmarks` feature to use `qml::benchmarks::{wine, ising}`.
Their pure `ansatz` builders validate complexity and construct ordinary `Circuit`
definitions before simulation. Async `run(&config, &device)` functions accept
native Burn devices; all tensor readbacks are asynchronous for WASM compatibility.
The example CLI and explorer adapt their shared outputs rather than duplicating
training logic. Preprocessing is independently reusable outside these datasets.

Measurements on 2026-11-10 used F32 Flex CPU on an Intel Core i9-12900K. These
fixed-configuration examples expose their reference results directly:

| Experiment and setup | Measured result | Independent control |
| --- | --- | --- |
| Wine: 4 qubits, 2 layers, 64 updates, seed 731, selected iterate 53 | 35/39 held-out correct (89.74%) | Ridge 39/39; training-majority class 15/39 |
| Ising: 4 spins, 2 layers, 80 updates, fixed deterministic angles | Energy -4.75747585 | Exact -4.75877048; uniform product -4.33333333 |
| MPS QAOA: 256 qubits, p=1, beta=pi/8, gamma=pi/4, fixed angles | Cut fraction 0.749998977; 32,168 retained complex values | Analytic 0.75; random cut 0.5 |

Additional Wine runs kept two layers and 64 updates, changing only the declared
register size or split/initialization seed:

| Qubits | Seed | Validation accuracy | Held-out accuracy | Ridge held-out |
| --- | --- | --- | --- | --- |
| 4 | 731 | 91.2% | 89.7% | 100.0% |
| 4 | 732 | 91.2% | 84.6% | 100.0% |
| 4 | 733 | 88.2% | 82.1% | 97.4% |
| 6 | 731 | 91.2% | 89.7% | 100.0% |
| 8 | 731 | 97.1% | 100.0% | 100.0% |

The default selected iteration is 53; training cross entropy falls from 1.0893
to 0.3505 there. The training-majority test accuracy is 38.5%. The strong classical
control matches or exceeds this QML model, as the reports make explicit. The
default Ising run reaches -4.75747585 against exact -4.75877048, or 0.00032366 error
per spin; its initial and uniform-product energies are -3.92773342 and -4.33333333.

An actual F32 WGPU/Vulkan default Wine run on an NVIDIA RTX PRO 6000 Blackwell
Workstation Edition selected the same iteration and produced the same accuracies
and confusion matrix. Selected train cross entropy differed from Flex by
`8.0e-7`. Full-report comparison of 1,871 structured values, including parameter
history, predictions and the portable checkpoint, had maximum absolute error
`4.36e-6` (unrounded: `4.351139e-6`). The 80-qubit MPS QAOA CPU/WGPU comparison
had maximum scientific metric difference `1.59e-7`.
Small circuit batches can be dominated by GPU launch/readback overhead, so this
is runtime/numerical qualification, not a speedup claim. This observation does
not qualify every size, accelerator, fusion configuration or browser runtime.

Tests cover dataset integrity/partitions, scaler leakage boundaries, all-feature
encoding across supported sizes/depths, actual quantum-angle updates,
reproducibility, held-out quality and the classical control. Device qualification
requires actual execution, not just a successful `--features wgpu` compilation.
The Wine tests also ran on that WGPU adapter, including restored-checkpoint
inference and reproducible updates. `run_with_progress` provides an optional
async callback after each recorded iterate so a browser can yield to its event
loop without introducing a web dependency into the core. The existing backend
tools discover all executable targets automatically and
compare their scientific reports; elapsed execution time is retained but excluded
from numerical parity. Historical full-suite records apply only to their stated
source snapshots, not automatically to these new applications.
