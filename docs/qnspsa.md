# Quantum natural SPSA

`burn_quantum::optim` implements fidelity-based quantum natural simultaneous
perturbation stochastic approximation (QN-SPSA). It estimates a loss gradient and
the **Fubini–Study metric, equal to one quarter of the pure-state quantum Fisher
information matrix**, using randomly perturbed evaluations. The metric comes
from quantum-state overlaps, independently of the objective's Hessian.

The algorithm follows [Gacon et al., *Simultaneous Perturbation Stochastic
Approximation of the Quantum Fisher Information*](https://arxiv.org/abs/2103.09232),
especially Eq. (13), with the metric convention and four fidelity points also
used by [Qiskit's QNSPSA implementation](https://qiskit-community.github.io/qiskit-algorithms/_modules/qiskit_algorithms/optimizers/qnspsa.html).
This implementation is included in `burn_quantum` 0.2.0.

## Burn integration and example

The API uses native Burn `Config`, `Tensor<1>`, and `Device`. It accepts two
fallible `FnMut` callbacks returning `Result<f64>`:

- `loss(parameters)` evaluates the scalar objective to minimize.
- `fidelity(left, right)` evaluates `|<psi(left)|psi(right)>|²`.

The callbacks can execute qubit, qudit, or mixed-register circuits on the
parameter tensor's device. They control data, shot sampling, and objective
construction. `StateVector::fidelity` supplies exact pure-state overlaps.
The optimizer requires neither a backward pass nor an autodiff device.

The complete [single-file VQE example](../crates/burn_quantum_example/src/bin/qnspsa_vqe.rs)
prepares a two-qubit RY/CNOT/RZ ansatz and minimizes `H = XX + YY + ZZ`, whose
ground state is the singlet with energy -3. Its training loop is:

```rust,ignore
use burn_quantum::optim::QnSpsaConfig;

let mut optimizer = QnSpsaConfig::new()
    .with_seed(17)
    .with_learning_rate(0.08)
    .with_learning_rate_decay(0.2)
    .with_perturbation(0.12)
    .with_regularization(0.02)
    .with_resamplings(2)
    .with_blocking_tolerance(Some(0.0))
    .init(2)?;
for _ in 0..80 {
    let update = optimizer.step(model.parameters.val(), &mut loss, &mut fidelity)?;
    model.parameters = model.parameters.map(|_| update.parameters);
}
```

`Param::map` preserves Burn's parameter identity and model-record layout.
Accepted updates are new tensor leaves with the original dtype, device, and
gradient-requirement flag; the optimizer update itself is nondifferentiable.
Probe tensors do not require gradients. Put jointly optimized parameters into
one flat vector to retain their metric cross terms. Separate optimizer instances
would instead approximate separate blocks.

Burn's ordinary gradient-consuming optimizer trait cannot request new
loss/fidelity evaluations at perturbed parameters. This optimizer therefore has
an explicit evaluation-based step API alongside normal Burn optimizers. The
scalar estimator `qn_spsa_sample` is also public for inspection or callers that
schedule their own evaluations; it does not implement asynchronous QPU jobs.

```sh
cargo run --locked -p burn_quantum_example --bin qnspsa_vqe
BURN_QUANTUM_DEVICE=wgpu cargo run --locked -p burn_quantum_example --features wgpu --bin qnspsa_vqe
BURN_QUANTUM_DEVICE=cuda cargo run --locked -p burn_quantum_example --features cuda,fusion --bin qnspsa_vqe
cargo test --locked -p burn_quantum --test qnspsa
```

The photonic stack also supplies [measured fidelity and mixed-state geometry
protocols](photonic-optics.md), including their shot accounting and independent
geometry checks. [Asynchronous local jobs](photonic-runtime.md#immutable-asynchronous-local-jobs)
can schedule caller evaluations; the optimizer's step interface remains synchronous,
and this is not a hardware-QPU provider integration.

## Estimator and update

For independent Rademacher vectors `u,v` (each component independently ±1),
parameters `theta`, perturbation `c`, and fixed first fidelity argument, define:

```text
L+ = L(theta + c*u)              L- = L(theta - c*u)
F0 = F(theta, theta + c*u)       F1 = F(theta, theta - c*u)
F2 = F(theta, theta + c*(u+v))   F3 = F(theta, theta + c*(-u+v))

gradient_hat = (L+ - L-) * u / (2*c)
deltaF = (F2 - F0) - (F3 - F1)
g_hat = -deltaF * (u*v^T + v*u^T) / (8*c²)
```

The metric is
`g_ij = Re(<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>)`.
For `RY(theta)|0>`, it is 1/4, and the estimator at finite `c` gives
`sin²(c)/(4*c²)`. For a global-phase direction it is zero. Individual estimates
can be indefinite even though the exact metric is positive semidefinite.

Each attempt averages `resamplings` independent gradient/metric estimates.
Starting with `g_bar_0 = initial_metric_scale * I`, completed attempt `t >= 1`
updates `g_bar_t = (t*g_bar_(t-1) + mean(g_hat_t))/(t+1)`. The initial metric
has weight one; resamplings reduce variance within each attempt, not its weight
in the running average.

The preconditioner is the **spectral** absolute value plus a positive ridge:
`M = |g_bar| + beta*I`. With `g_bar = Q diag(lambda) Q^T`, solve using
`Q diag(1/(abs(lambda)+beta)) Q^T gradient_hat`, then propose
`theta_new = theta - learning_rate * direction`. This is a symmetric
eigendecomposition, not elementwise absolute values or an explicit matrix
inverse. The implementation uses a bounded nalgebra eigensolve in host F64.

The first `metric_delay` attempts learn the metric but use the ordinary SPSA
gradient direction. Subsequently every attempt uses the natural direction.

## Configuration and state semantics

| Setting | Default | Meaning |
| --- | --- | --- |
| `learning_rate` | 0.05 | Numerator `a` |
| `perturbation` | 0.1 | Numerator `c` |
| `learning_rate_decay` | 0.602 | Exponent `alpha` |
| `perturbation_decay` | 0.101 | Exponent `gamma` |
| `stability_constant` | 0 | Nonnegative learning-rate offset `A` |
| `regularization` | 0.01 | Positive metric ridge `beta` |
| `initial_metric_scale` | 1 | Nonnegative identity prior |
| `resamplings` | 1 | Independent direction pairs per attempt |
| `metric_delay` | 0 | Attempts before natural preconditioning starts |
| `blocking_tolerance` | `None` | Optional allowed loss increase |
| `seed` | 0 | Local ChaCha8 perturbation seed |
| `max_metric_elements` | 65,536 | Per-matrix element budget, allowing 256 parameters |

For zero-based attempt `k`, the schedules are `a/(k+1+A)^alpha` and
`c/(k+1)^gamma`. Zero exponents select constant schedules. These defaults are
starting configurations, not a convergence guarantee or an automatic tuning
procedure. Positive finite rates, perturbation, and ridge are required; the
other numeric configuration values are checked for their stated domains.

With blocking enabled, each attempt evaluates the current and proposed losses
afresh, accepting when `loss_proposed <= loss_before + tolerance`. A blocked
proposal retains the original tensor but advances the schedule/random stream
and learns its metric estimate. No cached objective is reused across changed
data or noisy evaluations. The tolerance is in objective units, not a relative
error or an automatically estimated confidence interval.

`QnSpsaStepReport` records the actual rates, one-based attempt number, acceptance,
proposed displacement norm, and callback counts. `loss_estimate` is the average
of the **perturbed** losses, not the objective at the center or accepted point.
`loss_before` and `loss_proposed` exist only when blocking is enabled. Callers
choose stopping criteria and evaluation budgets explicitly.

`optimizer.state()` produces a versioned Serde checkpoint. Restore it using
`QnSpsa::from_state`, which checks the configuration, dimensions, finite values,
and metric symmetry. Save model weights/configuration separately using Burn's
normal records. Also save any data cursor and the evaluator's RNG state when
resuming noisy training. The optimizer's seed and completed-attempt count
determine the next local perturbation stream without touching global RNG state.
Equivalent callbacks and restored parameters reproduce the continuation within
floating-point tolerance; cross-device bitwise reproducibility is not promised.

Shape errors, unresolvable perturbations, nonfinite results, invalid fidelities,
failed callbacks, and numerical-solve errors return `Err` without changing the
optimizer state. Callback side effects or already consumed hardware shots
cannot be rolled back. Only successful attempts, including blocked proposals,
advance the optimizer.

## Cost, precision, and scope

For `R` resamplings, an attempt uses **2R loss and 4R fidelity callback calls**,
plus two loss calls if blocking is enabled, independent of parameter count.
These are evaluator calls, not circuit preparations or shots. The example's
fidelity callback simulates two states per call; its 80 attempts at `R=2` and
with blocking consume 480 loss and 640 fidelity calls. Initial/final evaluation
and state validation are separate from those training counters.

The dense metric still needs **O(d²) host storage and O(d³) classical work** per
natural step for `d` parameters. Scratch matrices and cloned checkpoints add
storage beyond `max_metric_elements`; it is not a total process memory cap.
Quantum evaluations run on the selected CPU/GPU device, while parameter
readback, scalar callbacks, averaging, regularization, and the solve synchronize
with the host. The reference simulator retains its exponential Hilbert-space
cost. No claim of arbitrary-scale optimization or quantum advantage follows
from constant evaluator count. Block/low-rank metrics, prepared batched probe
execution, and asynchronous QPU scheduling remain future work.

F32 and F64 parameters are supported where the selected Burn device supports
them. Fidelity errors/noise are amplified by `1/c²`, and finite perturbations
bias the geometric estimate. The optimizer rejects a perturbation that rounds
to the unchanged parameter, but it cannot guarantee a well-resolved fidelity
difference. The F32 CPU/GPU geometry comparison uses `c=0.08`; the strict CPU
geometry oracle uses F64 and `c=0.002`. Smaller values are not automatically
more accurate.
Squared overlaps must lie in `[0,1]`, with an absolute `1e-5` roundoff allowance;
accepted roundoff is not clipped. Mitigated estimates outside that interval
need an explicit caller policy.

The contract here is a smooth normalized **pure-state** family and its squared
overlap. Reusing the API with density-matrix similarities does not establish a
Bures or mixed-state quantum Fisher implementation. Exact-state and seeded
finite-shot callback tests do not establish convergence for arbitrary noisy
objectives, real QPUs, or learning tasks.

## Validation

[`tests/qnspsa.rs`](../crates/burn_quantum/tests/qnspsa.rs) checks the single-rotation metric factor,
exhaustive perturbation averages with off-diagonal rank-one geometry, independent
complex tangent-vector geometry for qubits and qutrits, a global-phase null
direction, rotated indefinite-matrix regularization, and a closed-form coupled
2x2 natural solve. It also checks scalar update/averaging recurrences, schedules,
blocking, constant evaluator counts, invalid inputs, failure atomicity,
checkpoint continuation, and Burn parameter identity/gradient flags.

The selected-device tests compare F32 qubit/qutrit estimates against CPU at the
same finite perturbation: absolute metric tolerance `2e-4` and gradient tolerance
`5e-5`. Agreement with infinitesimal geometry allows `0.004` finite-perturbation
bias. A one-parameter training trajectory agrees with CPU within `2e-4` and
reaches expectation below `-0.9999`. Strict F64 tangent comparisons use `8e-6`.
The finite-shot fixture uses seeded Bernoulli sampling of analytic probabilities
(512 shots per callback), and checks the underlying exact final loss below
`-0.98`; it is a CPU noise fixture, not a hardware experiment.

The VQE acceptance test requires energy in `[-3.00001,-2.99)`, singlet fidelity
above `0.997`, normalized final state within `1e-5`, and exact callback counts.
The same budgets, seeds, and acceptance thresholds apply to CPU and GPU.
Detailed measured results and commands are in [qnspsa-results.json](qnspsa-results.json).

Recorded on **2026-11-08**, with Rust 1.95.0, Burn 0.22.1, and an **NVIDIA RTX
PRO 6000 Blackwell Workstation Edition**, driver **610.43.02**:

| Configuration | Checks | VQE energy | Singlet fidelity |
| --- | --- | --- | --- |
| Flex CPU | Full workspace: 80 tests + 1 doctest | -3.00000048 | 1.00000005 |
| WGPU / Vulkan, fusion off | 26 targeted checks + executable | -2.99999976 | 0.99999988 |
| WGPU / Vulkan, fusion on | 26 targeted checks + executable | -2.99999976 | 0.99999988 |
| CUDA, fusion off | 26 targeted checks + executable | -3.00000048 | 1.00000013 |
| CUDA, fusion on | 26 targeted checks + executable | -3.00000048 | 1.00000013 |

Each GPU configuration runs the 16 optimizer tests (including 12 host-only
reference/control cases), nine existing selected-device backend-parity tests,
and the VQE acceptance test. The four optimizer tests that use the selected
device exercise qubit/qutrit geometry, phase invariance, training trajectories,
and native Burn parameter updates. The eight older example executables were
not rerun on GPU for this addition; their CPU acceptance tests passed in the
full workspace run. GPU F64, other vendors, and other operating systems are
not qualified by these results.

All reported example metrics agree with CPU under the existing comparison
tolerances; the largest absolute difference is `7.16e-7`. Slightly super-unit
fidelities and energies below -3 are F32 roundoff within the stated acceptance
bounds. All-feature strict Clippy, warning-free rustdoc, formatting, no-backend
compilation, and Flex compilation without autodiff also passed.

The standard validation scripts include the optimizer tests and `qnspsa_vqe`:

```sh
bash scripts/validate.sh
bash scripts/validate_backend.sh wgpu
bash scripts/validate_backend.sh wgpu --fusion
bash scripts/validate_backend.sh cuda
bash scripts/validate_backend.sh cuda --fusion
```

## Mixed photonic states

The evaluator determines the geometry. Supplying squared Uhlmann fidelity
`F(rho,sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))²` makes the same finite-difference
estimator approach the Bures metric. For smooth full-rank families this is QFI/4;
it reduces to Fubini–Study on smooth pure-state families. Rank-changing points
need separate treatment and must not be silently assigned the same regularity
assumptions. See [Zhou and Jiang](https://arxiv.org/abs/1910.08473) and
[Safranek](https://arxiv.org/abs/1612.04581).

The photonic optics API supplies a bounded host density-matrix fidelity reference.
This accesses simulated density entries and performs a host eigensolve; it is not
an executable hardware overlap protocol or a native autodiff operation. Finite-shot
optical parameter-shift probes and measurement Fisher estimates are separately
named APIs with shot counts, uncertainty and regularization. Their measurement
geometry is not substituted for the mixed-state quantum metric.
