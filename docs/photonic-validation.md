# Photonic foundation qualification

This is the historical M0 snapshot. See [expansion validation](photonic-expansion-validation.md)
for the later local-simulation modules and tests.

Recorded **2026-11-09** for the local, unreleased implementation on top of
`3b4473801b1b972795140f0bac4ab100b1b8ef0e`. Package versions remain 0.2.0;
these additions are not in the published 0.2.0 release. This is local validation,
not an exact-commit CI or publication claim. [Machine-readable evidence](photonic-results.json)
retains the source hashes, hardware, counts, example reports and package inspection.

The [model guide](photonic-fbqc.md) specifies the implemented physics and API.
The [roadmap](photonic-roadmap.md) separates future capabilities from this result.

## Executed matrix

Rust 1.95.0 and Burn 0.22.1; default Flex CPU and an **NVIDIA RTX PRO 6000
Blackwell Workstation Edition**, driver **610.43.02**. WGPU selected the discrete
Vulkan adapter. CUDA selected the same physical GPU. Photonic tensor execution
uses **F32**; independent scalar references and planner arithmetic use host F64.
The existing workspace also retains its stricter CPU F64 reference tests.

| Configuration | Passing checks in this run | Executable/report coverage |
| --- | --- | --- |
| CPU | 96 workspace tests and one doctest | All ten current application binaries |
| WGPU | 9 backend-parity + 15 photonic + 1 Bell-swap acceptance | `fbqc_bell_swap` |
| WGPU + kernel fusion | Same 25 checks | `fbqc_bell_swap` |
| CUDA | Same 25 checks | `fbqc_bell_swap` |
| CUDA + kernel fusion | Same 25 checks | `fbqc_bell_swap` |

The photonic target includes ten tensor-execution tests and five host planner,
serialization or preflight tests. Counts do not imply that host graph planning
runs on the GPU. Hardware selection is explicit and unavailable hardware is an
error. No software adapter was used as an accelerator substitute.

The prior nine application GPU acceptance tests were not rerun in this focused
photonic qualification. Their earlier evidence remains in the historical
[workspace](organization.md#workspace-relocation-verification) and algorithm
records. The shared validation script now includes the photonic target, and
Cargo-based example discovery includes the new binary automatically.

## Mathematical and runtime coverage

- Asymmetric rail attenuation: populations, complex coherences, trace/positivity,
  channel composition, absorbing vacuum, invalid probabilities and storage powers.
- Local logical encoding: shared parameter/input binding, analytic output and
  gradient checks, unsupported entangler rejection and allocation overflow.
- Resource preparation: all six-ring amplitudes against an independent graph-sign
  formula, invalid edges and checked serialization.
- Full fusion instrument: an independent scalar-complex partial-inner-product
  oracle on batched four-site states, including vacuum and reversed nonadjacent
  targets. Every branch weight and surviving density entry is compared.
- State-dependent Bell success, preserved failure parity, explicit loss erasures,
  impossible conditioning, input gradients and the success denominator in a
  conditional objective. The latter compares its derivative to an analytic quotient.
- Routing: earliest arrival, directed reachability, horizon/overflow rejection,
  link contention, synchronization and source/output storage attenuation.
- Program contracts: atomic consumed-ID rejection, version checks, rejection of
  unknown noise/control fields, deterministic compilation, typed recipe access,
  retained options and separate compilation/execution budgets.
- End-to-end Bell swapping: analytic unconditioned probabilities, conditional
  entangled coherences, seeded joint shots, complete loss with counted vacuum
  outputs, repeated destructive fusions, all-consumed classical results and
  branch/shot limits.

The compiler accepts a 200-photon recipe without tensor allocation; the dense
executor rejects its simulation. This is a separation-of-responsibilities test,
not evidence of large-system quantum execution. Dense simulation remains
exponential and the current defaults fit at most six sites at batch one.

## Bell-swap results

Two ideal Phi+ resources supply four photons. The consumed photons have net
transmissions 0.9 and 0.8, with ideal storage/detection in this fixture. Shared-link
contention gives arrivals at ticks 3 and 5, then completion at tick 6.

| Quantity | Analytic reference | CPU result |
| --- | --- | --- |
| Resolved fusion success | 0.36 | 0.3599998951 |
| Unresolved fusion failure | 0.36 | 0.3599998951 |
| Loss record | 0.28 | 0.2799999118 |
| Total probability | 1 | 0.9999997020 |
| Corrected conditional Bell fidelity | 1 | 1 |
| Sampled success, 4096 attempts, seed 23 | Approximately 0.36 | 0.3688964844 |

**All reported metrics match CPU exactly on all four GPU configurations.** This
is an observation for this fixture/hardware, not a general bitwise-reproducibility
guarantee. The scalar oracle checks use F32 absolute tolerances down to `2e-6`
or `3e-6`; gradient checks use up to `5e-5`. The example accepts probability and
fidelity errors below `2e-5` and sampled-success error below 0.025. The existing
comparison script passed unchanged; no tolerance was relaxed for photonics.

## Engineering checks and package

Formatting, strict workspace/all-target/all-feature Clippy, warning-free
all-feature rustdoc, isolated no-backend and Flex-only builds all passed.
The full CPU suite and all ten application binaries passed. Shell syntax,
local Markdown targets and whitespace were also checked.

`cargo package --locked --offline --allow-dirty -p burn_quantum` built and
verified the standalone archive. Inspection found **77 files**, including all
**70 core-package Rust files** byte-for-byte equal to the checkout, README,
both licenses and resolved workspace manifest inheritance. All extracted
package targets compiled. Its VCS metadata correctly records
`path_in_vcs = crates/burn_quantum` and `dirty = true`; this local archive is
not release provenance. No dependency, lockfile or Cargo-feature change was needed.

## Reproduce

```sh
cargo test --locked --workspace --all-targets
cargo test --locked --workspace --doc
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --all-features --no-deps
cargo check --locked -p burn_quantum --no-default-features
cargo check --locked -p burn_quantum --no-default-features --features flex
bash scripts/run_examples.sh cpu

# Repeat for wgpu, wgpu,fusion, cuda and cuda,fusion; choose the matching device.
BURN_QUANTUM_DEVICE=wgpu cargo test --locked -p burn_quantum --features wgpu \
    --test backend_parity --test photonic -- --test-threads=1 --nocapture
BURN_QUANTUM_DEVICE=wgpu cargo test --locked -p burn_quantum_example --features wgpu \
    --bin fbqc_bell_swap -- --test-threads=1 --nocapture
BURN_QUANTUM_DEVICE=wgpu cargo run --locked -p burn_quantum_example --features wgpu \
    --bin fbqc_bell_swap

# With reports saved under their backend/fusion directory names:
python3 -B scripts/compare_backends.py --example fbqc_bell_swap \
    artifacts/photonic/cpu artifacts/photonic/wgpu artifacts/photonic/wgpu-fusion \
    artifacts/photonic/cuda artifacts/photonic/cuda-fusion
cargo package --locked --offline --allow-dirty -p burn_quantum
```

Local logs and JSON reports are in ignored `artifacts/photonic/`. These checks
establish the specified encoded model, compilation contracts and native-device
execution. They do not establish full optical/Fock physics, Perceval or Stim
runtime parity, realistic resource generation, boosted fusion, universal
algorithm lowering, QEC thresholds, hardware execution, distributed scalability
or a photonic learning advantage. Those each have acceptance gates in the roadmap.
