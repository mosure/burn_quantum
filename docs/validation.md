# Validation record

This is the original 2026-10-29 qualification record. The subsequent workspace
example suite and its shared core operations are covered by
[the 2026-11-07 example validation record](examples-validation.md).
Actual GPU runtime qualification subsequently became possible outside the
sandbox; see [CPU/GPU backend validation](backend-validation.md) for current results.

Validated locally on 2026-10-29 against the working tree, Burn **0.22.1** and
the repository's `Cargo.lock`. This document records implementation
qualification, not a published release, remote CI result, or quantum advantage.

## Environment

- Linux x86_64; Intel Core i9-12900K.
- Rust 1.95.0 (`59807616e`, 2026-04-14), Cargo 1.95.0.
- Flex CPU, explicit F64 reference tests and F32 examples.
- Local dependencies were available offline. In the execution environment the
  snap Rust shim was unavailable, so the standalone toolchain's `bin` directory
  was prepended to PATH. No nightly or local Burn patches were required.

Reproduce the main checks with `bash scripts/validate.sh`. Raw logs are written
to `artifacts/validation/` and intentionally excluded from the source package.
Use `CARGO_NET_OFFLINE=true` when dependencies are already cached.

## Passed checks

| Check | Result |
| --- | --- |
| `cargo fmt --all --check` | Pass |
| `cargo clippy --locked --all-targets -- -D warnings` | Pass |
| `cargo test --locked --all-targets` | 26 integration tests pass; example targets compile |
| `cargo test --locked --doc` | 1 executable documentation test passes |
| `cargo check --locked --no-default-features` | Pass, no execution backend required to compile |
| `cargo check --locked --no-default-features --features flex` | Pass, inference without the autodiff feature |
| `cargo check --locked --features wgpu,fusion --all-targets` | Pass |
| `cargo check --locked --all-features --all-targets` | Pass, including CUDA and WGPU |
| `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps` | Pass |
| `bell`, `qudit`, `train`, `hybrid` examples | Executed successfully, including convergence assertions |
| Release `profile` example | Executed at 4, 8, 12 qubits, one warmup and seven timings each |
| `cargo package --locked --allow-dirty` | Source archive built and independently compiled; generated logs excluded |

The test suite covers multiple cases per test, including:

- Bell/GHZ states at 2/3/5 qubits; maximally entangled qudits at dimensions 3/4/5.
- Independent scalar complex arithmetic against mixed-radix tensor application,
  reversed and nonadjacent targets, two distinct batch entries, and gate inverses.
- Generalized Weyl relations, level-controlled Y gates, Pauli-Y coherences,
  Hamiltonian sums, variances, marginals and seeded measurement statistics.
- X/Y/Z rotation gradients at dimensions 2/3/4 against finite differences and
  appropriate parameter shifts, including spectator-level coherences.
- Shared affine parameters, per-example input gradients, and balanced Burn
  checkpointing; incompatible complex-component execution contexts are rejected.
- Pure-to-density gradients through amplitude damping, Kraus completeness,
  mixed-state positivity checks, complex reduced states, noise endpoints and
  arbitrary qutrit dephasing.
- Burn `Config` serialization, burnpack weight-record round trips, module
  device movement and actual Adam updates.
- Invalid matrix/IR invariants, versions, dimensions, duplicate wires, tensor
  contexts, nonphysical values, overflow and resource-limit rejection.

## Executed prototype outcomes

| Prototype | Observed result |
| --- | --- |
| Bell state | Probabilities approximately `[0.5, 0, 0, 0.5]`; only 00/11 sampled |
| Qutrit Bell state | Equal 00/11/22 probabilities; reduced purity approximately 1/3 |
| One-angle variational optimization | Initial loss 3.8233407; final Z expectation -0.997102 after 100 Adam steps |
| Classical encoder + entangled circuit + classical head | Regression MSE 0.1360664 -> 0.0001202 after 120 Adam steps |

Both training problems are deterministic integration demonstrations. They do
not establish useful generalization, quantum advantage, or trainability at
large depth.

## Small release-mode timing probe

Flex CPU, F32, batch 1, H on each wire followed by a nearest-neighbor CNOT chain.
Each timed execution includes zero-state creation, gather planning, gate
execution, probability readback and normalization validation. The state-byte
column is a storage lower bound, **not measured peak memory**.

| Qubits | Hilbert dimension | Gates | State bytes | Median ms | Min ms | Max ms |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 7 | 128 | 0.034 | 0.032 | 0.048 |
| 8 | 256 | 15 | 2,048 | 0.180 | 0.176 | 0.283 |
| 12 | 4,096 | 23 | 32,768 | 4.065 | 4.019 | 4.124 |

These small bounded measurements establish that the prototype runs and provide
a baseline for future prepared-plan/kernel work. They do not characterize
large-state throughput, training memory, tensor-network scaling, or GPU speed.

## Remaining qualification

- GPU runtime and numerical parity are **unverified**. `nvidia-smi` could not
  communicate with the NVIDIA driver and `/dev/dri` was unavailable in this
  execution environment. Feature compilation is separate from hardware proof.
  The optional `wgpu_smoke` example checks a forward value and its analytic
  derivative when run on an available adapter.
- CUDA runtime, browser/Wasm execution, distributed simulation and remote QPUs
  have not been tested. No hardware fallback is implied by this record.
- CI is configured in `.github/workflows/ci.yml`; no remote run was triggered.
- This working tree has not been committed, pushed or published.
