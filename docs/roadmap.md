# Research and engineering roadmap

The current implementation includes the native dense reference, Burn QML,
MPS/Clifford/Fock engines and the local photonic stack. The sections below retain
longer-term engineering directions; implemented contracts are identified here
and in the [photonic acceptance map](photonic-status.md).

The [photonic acceptance map](photonic-status.md) links the current implemented
Clifford, optical, compiler, factory, QEC, QML and scale-out models. The
[original staged roadmap](photonic-roadmap.md) records their planned acceptance
criteria; its M0 baseline is historical.

## Current contracts

| Concern | Implemented contract | Boundary |
| --- | --- | --- |
| State space | Nonempty tensor product of dimensions >= 2 | Finite dimensions, checked `usize` products |
| Pure states | Split-complex `[B,N]` Burn tensors | Exponential N; configurable element budget |
| Mixed states | Split-complex `[B,N,N]` | Quadratic in N; small noisy problems |
| Gate set | Arbitrary checked local unitaries; Pauli, Fourier, Weyl, SUM, SWAP, controls | Dense local matrices; built-in allocation cap |
| Unitary circuit bindings | Affine shared parameter / input references; analytic X/Y/Z level rotations | No expression DAG or dynamic control |
| Training | Burn first-order autodiff, `Module`, `Config`, `Param`, optimizers | Retained graph grows with circuit depth |
| Gradient oracle | Occurrence-wise 2/4-term shifts | Implemented rotation generators only |
| Quantum natural optimization | Fidelity-based QN-SPSA, spectral regularization, blocking, seeded schedules/checkpoints | Fidelity-defined geometry; dense host O(d²) storage / O(d³) solve; synchronous callbacks |
| Dense readouts | Local Hermitian operators, variance, local Hamiltonian sums, marginals | Exact analytic values |
| Density-state noise | Arbitrary checked Kraus operators, three qubit channel factories | Constant channel parameters; density path |
| Dense terminal sampling | Seeded terminal computational-basis shots | Host synchronization; no gradient / collapse |
| Persistence | Versioned circuit JSON, validated deserialization, Burn weights | Config and weight record stored together by caller |
| Tensor device dispatch | Native Burn device dispatch; optional WGPU/CUDA features | Each device requires separate qualification |
| Boolean SAT | Reversible CNF oracles, Grover amplification, bounded unknown-count search | Exponential dense simulation; failed search is inconclusive |
| Photonics / FBQC | Fock/dual-rail models, calibrated fusion, bounded universal compilation, finite routing, QEC/learning/local runtime | Explicit model-specific bounds; hardware qualification deferred |
| Tensor networks | Mixed-qudit MPS, non-Clifford gates, Kraus trajectories, selective readouts and distributed ownership | Entanglement-dependent rank; bounded host SVD without reverse-mode AD |

## 1. Prepared execution and accelerator kernels

First measure the supplied profile harness and representative variational
circuits. Record warmup, device synchronization, repeated median/p95 wall time,
peak host/device memory, batch, depth, gate arity, precision and output error.
The initial harness runs one warmup and seven synchronized timings per size,
reporting median/min/max and state-storage lower bounds. It is a small
end-to-end timing probe, not a broad benchmark or a quantum-advantage claim.

The reference path regenerates gather indices and uploads small operators for
each call. Add a bounded cache keyed by layout, ordered targets, device identity
and precision; separate static matrices from trainable tensors so the cache
never retains a stale autodiff graph. A prepared plan should expose a resource
estimate and reject unsupported kernels before execution. Cache eviction must
account for all simultaneously retained plans, not just one tensor.

Structured controlled X and phase flips now have portable gather/diagonal paths
for [SAT search](sat.md). Next implement strided one/two-site and specialized
diagonal/permutation kernels using Burn backend extensions. Require independent CPU-reference agreement,
forward/backward GPU parity, fusion-on/off parity, mixed-radix correctness,
batch isolation, and measured improvements including readout costs. A custom
forward kernel needs a real VJP and checkpoint recomputation implementation;
the extension routing macro does not synthesize gradients.

## 2. Memory-aware differentiation

Compare reverse-mode graphs, Burn gradient checkpointing, and an adjoint-state
method for reversible pure-state circuits. Preserve parameter tying, affine
scales and differentiable input encoding. Adjoint methods need defined behavior
at nonunitary operations; they cannot be applied unchanged across sampling or
Kraus mixtures. Add generator metadata before supporting new parameter-shift
families. Stochastic estimators must report variance, shot budgets and seeds.

Acceptance: gradients agree with finite differences and independent shifts;
every shared occurrence contributes; all unused parameters have explicit zero
Jacobian columns; memory vs depth and runtime vs parameter count are measured.
Do not advertise higher-order derivatives through Burn's first-order engine.

[Quantum natural SPSA](qnspsa.md) now estimates pure-state geometry with constant
loss/fidelity callback count per resampling and needs no backward pass. The
classical metric remains dense. Future block or low-rank metrics, batched probe
execution, and additional shot-allocation policies must preserve tested metric conventions
and report accuracy, estimator variance, synchronization and full solve costs.

## 3. Simulation representations and further extensions

MPS, Clifford and finite-cutoff optical implementations now have their own
validated APIs. Sparse states and distributed dense state vectors remain further
extensions; distributed MPS ownership is already implemented.

| Engine | Useful structure | Costs / acceptance evidence |
| --- | --- | --- |
| MPS / tensor networks | Limited entanglement and favorable circuit geometry | Bond dimensions, truncation error, fidelity, contraction plan; dense parity at small size |
| Stabilizer | Clifford circuits | Tableau correctness; reject unsupported non-Clifford gates or use explicit decomposition |
| Sparse states | Low support in the chosen basis | Density crossover and sparsity growth; no automatic speed claim |
| Quantum trajectories | Noisy circuits with modest shot budgets | Statistical agreement with density matrices and confidence intervals |
| Distributed state vectors | Exact high-entanglement states | Communication volume, network topology, sharding and collective gradient correctness |
| Truncated bosonic systems | Continuous-variable applications with finite cutoff | Cutoff-convergence study and occupation leakage diagnostics |

`MpsSimulator` uses the associated state type on `QuantumBackend`; direct MPS
readouts avoid materializing a full distribution. Dense export and probability
vectors have separate budgets and may be rejected while local expectations
remain feasible. New engines should preserve that separation and expose their
supported operations before execution.

## 4. Dynamic programs and hardware

The unitary `Circuit` remains a straight-line v1/v2 IR. Measurement patterns and
causal event programs now provide separate measured/adaptive execution, including
bounded retry protocols. A future general dynamic circuit IR must still define
classical registers, reset/loop semantics, execution limits and migration rules;
unsupported engines must reject its instructions explicitly.

[LocalSimulationJobs](photonic-runtime.md#immutable-asynchronous-local-jobs)
implements bounded asynchronous submit/status/result/cancel with immutable job
identity, calibration leases, raw outcomes and attempted-shot accounting.
Actual QPU providers must add hardware topology, basis translation, provenance,
mitigation and uncertainty contracts without silently falling back to simulation.
Hardware gradients need occurrence-resolved shift jobs or justified estimators,
rather than treating a remote measurement as an ordinary differentiable tensor.

## 5. Quantum ML application suite

The `burn_quantum_example` workspace crate now provides small, reproducible VQE,
QAOA MaxCut, variational classification, fidelity-kernel classification,
Born-machine, hybrid-regression, qutrit-classification, Grover SAT search, and
quantum-natural-SPSA VQE implementations, plus a lossy photonic Bell-swap primitive, each
in one file with numerical acceptance checks. Shared encoding, ansatz, readout,
loss and fidelity operations live in the core crate's `qml` module. These are
integration prototypes on small problems; broader application benchmarks and
differentiable-channel learning still require task-specific protocols.

For each workload save data generation code and seeds, split manifests,
preprocessing, circuit/config JSON, Burn weight and optimizer records,
dependency lock, device/dtype, training budget, raw curves and shot logs. Compare
with a matched classical baseline, a frozen circuit, and a random-feature
control. Report confidence intervals and full resource costs. Evaluate
generalization, robustness to noise, calibration and gradient concentration;
training loss reduction alone is not useful task performance or quantum
advantage.

## 6. Release and maintenance

Before a public release, qualify hardware backends on actual devices, record
exact revision CI, verify packaged content with an unpatched consumer and
review API compatibility. Keep the tested Burn minor pinned until migration
tests pass. Add benchmarks and fuzzing of serialized circuits separately from
the deterministic unit suite. Property tests should use independent oracles,
not duplicate the gather-plan implementation.

The project is deliberately extensible, but no design can guarantee support for
all future Quantum AI/ML applications. Finite-dimensional universal operators
provide expressiveness; computational cost, physical models and execution
semantics remain explicit constraints.
