# Quantum AI/ML example validation

Recorded **2026-11-07**, using Rust 1.95.0, Burn 0.22.1, and Flex CPU. Executable
examples use F32; core reference tests also exercise F64. This extends the
[initial simulator qualification](validation.md) with
the `burn_quantum_example` workspace crate and shared core QML operations.
Subsequent CPU/GPU runtime checks and the expanded test count are recorded in
[backend validation](backend-validation.md).

The tables below retain the original seven-example results. The later
[Grover SAT](sat.md) and [quantum natural SPSA VQE](qnspsa.md) additions have
their own validation records; the current suite contains nine examples.

## Scope and organization

Each algorithm has one complete source file under
[`crates/burn_quantum_example/src/bin`](../crates/burn_quantum_example/src/bin).
Its circuit, synthetic data or problem definition, optimization, evaluation,
executable entry point, and acceptance test remain together. The small example
library provides reporting, initialization and evaluation utilities.

Shared quantum encoding, RY/CNOT layers, Pauli-Z probability conversion,
distribution cross-entropy, and fidelity kernels live in
[`burn_quantum::qml`](../crates/burn_quantum/src/qml.rs). Pairwise state fidelity is available on
`StateVector`. `QuantumLayer` permits empty observable lists for state-based
objectives while retaining its existing Burn parameter/record structure;
`forward` reports an error when no readouts were configured.

## Executable results

All seven commands completed successfully and emitted finite JSON metrics via
`bash scripts/run_examples.sh`. The script retains the reports under the ignored
`artifacts/examples/cpu/` directory. The following values are rounded from those
reports; they are observations, not hard-coded outputs.

| Example | Training / fitting budget | Observed evaluation |
| --- | --- | --- |
| VQE | 160 Adam steps, seed 7 | Energy `-2.99999952` against exact `-3`; singlet fidelity `0.99999976` |
| QAOA MaxCut | 140 Adam steps, seed 11 for 512 shots | Expected cut `1.99999952` against enumerated optimum `2`; optimal probability `0.99999976`; best sampled cut `2` |
| Variational classifier | 120 Adam steps, explicit initialization | Cross-entropy `0.472220 -> 0.270040`; train/test accuracy `1.0 / 1.0`, majority baseline `0.5` |
| Fidelity kernel classifier | Kernel ridge solve, regularization `0.001` | Train/test accuracy `1.0 / 1.0`, majority baseline `0.5`; Gram symmetry error `0`, diagonal error `2.38e-7` |
| Born machine | 250 Adam steps, seed 19 | Target-to-model KL `2.026800 -> 1.16e-7`; total variation `1.14e-6`; 4,096-shot total variation `0.011768` |
| Hybrid regression | 120 Adam steps, explicit initialization | Train MSE `0.136066 -> 0.000120172`; disjoint test MSE `0.0000946092` |
| Qutrit classifier | 120 Adam steps, explicit initialization | Cross-entropy `0.204146 -> 0.011130`; train/test accuracy `1.0 / 1.0`, majority baseline `1/3` |

The classical encoder, quantum parameters, and classical head in hybrid
regression all changed: their parameter L1 differences were `1.19391`,
`0.328244`, and `0.533005`. Each must exceed `1e-3`, in addition to the MSE
requirements, so the example checks training across all three stages.

Classification uses disjoint deterministic train/test inputs: 16/36 parity
examples for each binary classifier, and 27/48 angle-cluster examples for the
qutrit classifier. Hybrid regression uses 24 training grid points and 31
different test points. The Born machine fits a specified eight-outcome
distribution and evaluates both exact and sampled probabilities. VQE and QAOA
use analytic or exhaustively enumerated references. The kernel example checks
its classical solve residual below `1e-8`.

## Checks

These checks passed locally:

- `cargo fmt --all --check`.
- `cargo clippy --locked --workspace --all-targets -- -D warnings`.
- `cargo test --locked --workspace --all-targets`: **42 passing tests**,
  consisting of 35 core integration tests and seven algorithm acceptance tests.
- `cargo test --locked --workspace --doc`: **one passing doctest**.
- `cargo check --locked -p burn_quantum --no-default-features` and the same
  command with `--features flex`.
- `cargo check --locked -p burn_quantum --features wgpu,fusion --all-targets`:
  compile coverage only.
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps`.
- All seven example binaries, via `bash scripts/run_examples.sh`.
- The existing `bell`, `qudit`, `train`, and `hybrid` simulator demos.
- `cargo package --locked --allow-dirty -p burn_quantum`: the core source
  archive built and compiled independently of the example workspace member.

Nine new core tests cover atomic template validation, probability conversion,
soft-target loss values and gradients, invalid metadata/epsilon, complex-valued
fidelity oracles, phase invariance, kernel gradients and resource limits, and
training without observable readouts. Existing simulator, qudit-gradient,
serialization, channel, and context-validation tests remain in the workspace
suite. CI and `scripts/validate.sh` include the example package; isolated core
feature checks explicitly select `-p burn_quantum` to avoid workspace feature
unification hiding missing features.

## Interpretation and limits

These are small executable implementations of established algorithm families,
with explicit numerical acceptance criteria. They establish that the circuits,
losses, Burn optimizers, and evaluation paths work together on the stated
problems. Synthetic test accuracy does not establish real-world predictive
performance or quantum advantage. One seed per stochastic example provides a
reproducible smoke test, not a multi-seed statistical study.

Training uses exact dense simulator probabilities/expectations and native
reverse-mode autodiff. Shot sampling is terminal evaluation, not shot-based
training. GPU runtime is qualified separately in [backend validation](backend-validation.md).
The suite does not qualify quantum hardware, noisy learning, distributed
execution, or large-register scalability. The dense
engine retains exponential state size. Probability losses assume normalized,
nonnegative inputs and clamp logarithms at `epsilon`; they check metadata
without introducing host synchronization. Fidelity kernels require normalized
states, materialize the pairwise output, and enforce per-allocation budgets
rather than a total memory bound.
