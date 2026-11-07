# Research and engineering roadmap

The current deliverable is an executable, tested dense simulator and Burn QML
integration. The roadmap below describes additional work, not capabilities
already implemented. Priority is driven by measured application requirements.

## Current contracts

| Concern | Implemented contract | Boundary |
| --- | --- | --- |
| State space | Nonempty tensor product of dimensions >= 2 | Finite dimensions, checked `usize` products |
| Pure states | Split-complex `[B,N]` Burn tensors | Exponential N; configurable element budget |
| Mixed states | Split-complex `[B,N,N]` | Quadratic in N; small noisy problems |
| Gate set | Arbitrary checked local unitaries; Pauli, Fourier, Weyl, SUM, SWAP, controls | Dense local matrices; built-in allocation cap |
| Parameters | Affine shared parameter / input references; analytic X/Y/Z level rotations | No expression DAG or dynamic control |
| Training | Burn first-order autodiff, `Module`, `Config`, `Param`, optimizers | Retained graph grows with circuit depth |
| Gradient oracle | Occurrence-wise 2/4-term shifts | Implemented rotation generators only |
| Readouts | Local Hermitian operators, variance, local Hamiltonian sums, marginals | Exact analytic values |
| Noise | Arbitrary checked Kraus operators, three qubit channel factories | Constant channel parameters; density path |
| Sampling | Seeded terminal computational-basis shots | Host synchronization; no gradient / collapse |
| Persistence | Versioned circuit JSON, validated deserialization, Burn weights | Config and weight record stored together by caller |
| Runtime | Native Burn device dispatch; optional WGPU/CUDA features | Each device requires separate qualification |

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

Then implement strided one/two-site kernels and diagonal/permutation fast paths
using Burn backend extensions. Require independent CPU-reference agreement,
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

## 3. Additional simulation representations

| Engine | Useful structure | Costs / acceptance evidence |
| --- | --- | --- |
| MPS / tensor networks | Limited entanglement and favorable circuit geometry | Bond dimensions, truncation error, fidelity, contraction plan; dense parity at small size |
| Stabilizer | Clifford circuits | Tableau correctness; reject unsupported non-Clifford gates or use explicit decomposition |
| Sparse states | Low support in the chosen basis | Density crossover and sparsity growth; no automatic speed claim |
| Quantum trajectories | Noisy circuits with modest shot budgets | Statistical agreement with density matrices and confidence intervals |
| Distributed state vectors | Exact high-entanglement states | Communication volume, network topology, sharding and collective gradient correctness |
| Truncated bosonic systems | Continuous-variable applications with finite cutoff | Cutoff-convergence study and occupation leakage diagnostics |

Use the associated state type on `QuantumBackend` for alternate exact analytic
engines. Do not force MPS or hardware results into a dense amplitude buffer.
The present full-distribution method may be unsupported on such engines; add
capability negotiation and selective-readout contracts before integration.

## 4. Dynamic programs and hardware

Version the IR to add classical registers, measurement destinations, resets,
conditions, loops and execution limits. Define trajectory vs ensemble semantics
and the nondifferentiable boundaries. New instructions must fail explicitly on
engines that do not support them; serialized version upgrades need fixtures and
migration policy. Existing v1 structure remains a straight-line unitary circuit.

For QPUs, introduce a separate asynchronous submit/poll/cancel contract with
immutable compiled jobs, topology mapping, basis translation, batch/shot
budgets, calibration identifiers, provider provenance, mitigation configuration
and result uncertainty. Never silently fall back from hardware to simulation.
Hardware gradients use occurrence-resolved shift jobs or justified estimators,
not a claim that a remote measurement is an ordinary differentiable tensor.

## 5. Quantum ML application suite

The `burn_quantum_example` workspace crate now provides small, reproducible VQE,
QAOA MaxCut, variational classification, fidelity-kernel classification,
Born-machine, hybrid-regression and qutrit-classification implementations, each
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
