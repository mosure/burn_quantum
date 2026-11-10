# Photonic milestone qualification

Recorded 2026-11-09 for the local unreleased checkout, based on Git commit
`3b4473801b1b972795140f0bac4ab100b1b8ef0e`. Exact source/log/archive hashes are in
[the machine-readable record](photonic-milestones-results.json). This record
supersedes the earlier expansion results for the new implementation; historical
qualification files remain unchanged. It is not an exact-commit CI result,
a publication record, a threshold reproduction or hardware qualification.

## Acceptance coverage

The CPU workspace suite passes 180 tests and one doctest. Four subsequently
added encoded-fusion tests and the wide-contraction regression also pass, for
**185 tests plus one doctest**.
The 22 example binaries each have acceptance tests; the process-worker example
also has a helper entry-point test. New coverage includes:

- Complete reference-entangled CNOT/CZ/SWAP channels, reversed operands,
  symbolic gradients, all joint failure/loss outcomes, latent output erasure,
  abort probability and expected physical cost.
- Exact optical standard-fusion effects/coherences, dark-count false heralds,
  partial distinguishability, finite sources, detector dynamics and a published
  Grice analyzer with complete signal/ancilla/detector loss accounting.
- Independent pinned Perceval output for 1,320 Grice probabilities and partial
  Hong–Ou–Mandel distributions, in addition to earlier Fock and Stim fixtures.
- Reference-frame sampling/continuation, signed stabilizer checks, four-star
  toric geometry, exhaustive small matching/erasure oracles and fault injection.
- Actual Shor-encoded graph resources and four physical fusions per encoded
  Bell attempt; all 8,192 tested record/erasure combinations and lossy encoded
  swapping retain correct recovered correlations and consumed-photon costs.
- Finite source/switch/analyzer buffers, configuration timing, independent and
  correlated factories, queue rollback, causal checkpoint validation and fresh
  resource retries. Actual TCP/process tests exercise identity rejection,
  timeouts, cooperative cancellation, stable seeds and byte-identical replay.
- Finite-shot optical gradients/Fisher estimates, mixed-state kernels and
  bounded squared-Uhlmann/Bures geometry, including numerical guard cases.

## Devices and commands

CPU is Burn Flex. GPU execution uses the NVIDIA RTX PRO 6000 Blackwell
Workstation Edition, driver 610.43.02; WGPU selects Vulkan explicitly through
our shared device helper. Requested device failures are errors, not skips.
The matrix uses WGPU and CUDA with Burn kernel fusion disabled and enabled.
Each of those four configurations passes **106 selected core checks and 23
example checks across all 22 binaries** (one extra check is the worker-process
entry point). Every current core target that selects its runtime device is
included. Host-only targets additionally run in the CPU workspace suite.
Host tableau/frame sampling, graph decoding, control and network operations
remain host algorithms. The exact tensor reference in each host-oriented
example runs on the selected native Burn device.

Primary commands are `cargo test --locked --workspace --all-targets`, workspace
doctests, and the core/backend-specific targets recorded in the JSON. All
example binaries run their acceptance tests for each GPU configuration. The
public `scripts/validate_backend.sh` includes the new targets and automatically
discovers example binaries. Loopback network tests need local socket access.

Strict all-target/all-feature Clippy, rustfmt, warnings-as-errors rustdoc,
backend-free and Flex-only library checks pass. `cargo package --locked
--allow-dirty -p burn_quantum` verifies the standalone archive; its VCS record
correctly identifies `crates/burn_quantum` and the dirty working tree. The
archive contains 139 files, including 125 Rust files and all checked fixture
inputs. The extracted archive is additionally compiled with all targets.
`serde_json` is now a regular library dependency for bounded event/network
persistence; existing package version remains 0.2.0 and this work is unreleased.

## Fusion planner regression

The 330-mode Grice fixture exposed a long planning stall in Burn's fused execution
of a single 330-input concatenation. The final precision contraction dispatches
bounded groups of 16 columns and uses a concatenation hierarchy with at most 16
inputs per node. It retains the same F32 arithmetic, allocation shapes, device and
native autodiff; Flex still uses its original GEMM path. No tolerance was widened.

A new dimension-330 batched complex regression compares outputs, input gradients
and all 435,600 real/imaginary operator-gradient entries with independent scalar
oracles at `3e-6`. In the targeted WGPU-fusion run, forward finished in 2.52 seconds,
complete forward/backward in 35.97 seconds, and the original Grice fixture in
4.15 seconds. These are single local diagnostics under concurrent machine load,
not throughput benchmarks. The complete affected backend suites run after the fix.

## Representative CPU results

These are controlled small examples with their stated noise models, not
quantum-advantage or production-throughput demonstrations.

| Example | Result and independent reference |
| --- | --- |
| Physical VQE | Conditional energy -3.00000024 versus exact -3; herald 0.11073025 versus 0.11073030; expected resource photons 3.920799 |
| Shot-trained interferometer | Worst held-out MSE 6.22e-6 versus frozen-control 0.031398; 2,949,120 attempted physical evaluations |
| Mixed-state optical kernel | Accuracy 0.96923 versus majority 0.50769; matches declared classical kernel control |
| Mixed Born QN-SPSA | Worst true TVD 0.005145 versus uniform-photon 0.108863; held-out sampled TVD 0.008576; analytic Bures metric error 0.000322 |
| Adaptive Bell distribution | 2,048 experiments; success 0.530762 versus exact bounded 0.549276; 9.55469 photons/experiment; identical checkpoint/window traces |

The Born optimizer's metric uses 840 exact **host spectral fidelity evaluations**;
these are not counted as an implemented physical overlap-measurement protocol.
Sampling, fidelity geometry, normalization and photon accounting retain separate
contracts. Raw example reports are embedded in the JSON evidence.

The four-star study uses 256 attempts per distance/model, per-photon loss 0.005
and independent X/Y/Z probabilities 0.0005 each. Its ideal complete-Bell reference
has unconditional failure Wilson intervals `[0.1514,0.2482]`, `[0.0419,0.1038]`,
`[0.0242,0.0753]` for distances 2, 3, 4. Two distance-4 attempts exhaust the decoder
budget and count as failures. The actual fixed-ZZ unboosted analyzer has erased
logical information in every tested attempt at these distances; its failure
interval is `[0.985216,1]`. These are different declared measurement models,
not evidence of a threshold for a physically boosted/encoded implementation.

See [current status](photonic-status.md), [optical assumptions](photonic-optics.md),
[compiler assumptions](photonic-compiler.md), [QEC assumptions](photonic-stabilizer.md)
and [runtime contracts](photonic-runtime.md) for unfulfilled roadmap gates.
