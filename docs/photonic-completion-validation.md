# Local photonic roadmap completion qualification

Recorded 2026-11-09 against the unreleased working tree based on
`3b4473801b1b972795140f0bac4ab100b1b8ef0e`. The [acceptance map](photonic-status.md)
identifies the bounded local implementations of M0–M7 and their model boundaries.
The [machine-readable record](photonic-completion-results.json) identifies the
exact source, independent fixtures, commands, device results and package archive.
Historical foundation/expansion/milestone records retain their original hashes.

## Completed implementation paths

- Mixed-qudit MPS evolution with native Burn contractions, explicit host complex
  SVD, bounded Schmidt truncation, conditional/Kraus trajectories, selective
  observables and an 80-qubit non-Clifford QAOA example. Approximation distance,
  allocation limits, checkpoint gauge and full-chain normalization are checked.
- Actual multi-process ownership of one entangled MPS. Two-phase boundary
  updates survive worker/coordinator restart and ambiguous commit persistence;
  reads remain blocked until a durable decision is reconciled. This is distinct
  from the existing independent-shot sharding and event-window protocols.
- Bounded universal qubit synthesis, parameter-preserving physical lowering,
  calibrated complete optical POVMs, noisy Choi resources, finite factory/fabric
  schedules, independently reserved rails and parity-encoded fresh-resource
  retries. Physical VQE and QAOA retain unsuccessful outcomes and physical cost.
- Correlated multimode source/phase/loss ensembles, calibrated Grice analyzers,
  passive finite-shot tomography and mixed-state overlap, identifiable held-out
  calibration, bounded shot allocation and seeded classification/kernel/Born
  learning curves with matched-record controls.
- Rotated six-ring networks, finite six-ring boundaries with signed logical
  correlations, planar open-code memory and Clifford protocols, physical encoded
  fusions, and a specified 15-to-1 magic-state protocol and entangled T injection.
- Immutable asynchronous local jobs with bounded queues, provenance, calibration
  expiry, cancellation, raw-attempt accounting and explicit simulator identity.
  OpenQASM interoperability is checked against the actual pinned PsiQDK local
  simulator, including every complex amplitude, endian order and weighted readout.

Reusable mathematics, channels, scheduling and execution contracts live in
`crates/burn_quantum`; each application remains one source file in
`crates/burn_quantum_example/src/bin`.

## Numerical and protocol regression coverage

The suites compare complete entangled channels and detector branches, native
gradients, exact small-code faults, independently enumerated boundary signs,
finite-shot confidence controls, host/network restart behavior and invalid-input
budgets. The finite six-ring oracle independently enumerates all 1,024 branches
of its two-resource case. It is not a threshold estimate.

The calibrated-channel optimization uses the exactly maximally mixed measured
half of a noiseless unitary Choi resource to contract rejected optical effects.
Accepted branches still execute the physical Kraus channel; noisy-resource and
nonideal output-storage paths retain the general implementation. Tests compare
both complete states/records and input/gate gradients against the general path.
The QAOA loop synchronizes at each optimizer boundary to bound queued work.
Neither the objective, iteration count, physical noise model nor tolerance was
weakened. These are Cargo's unoptimized host dev/test builds with other active
GPU workloads; timings are diagnostics, not throughput benchmarks.

Checkpoint regressions reject accumulated full-chain normalization drift that
individual local gauge checks can miss. Job regressions retain a completed final
record while correctly reporting cancellation or expired calibration. Durable
commit tests include a persistence callback that writes and then returns an error.

Actual CUDA-fusion execution exposed a separate layout defect: scaling a
noncontiguous `[2,2,1]` MPS center after a Bell-to-T circuit removed its complex
branch, giving norm and endpoint ZZ of 0.5. This was amplitude corruption, not
ordinary rounding. The MPS scalar helper now uses a native logical flatten,
multiply and reshape, forcing the necessary layout materialization. The new
two-qubit regression checks every stored/dense complex amplitude, norm,
correlations and checkpoint immutability; the 80-qubit GHZ/T check remains.
No CPU fallback, host detachment, changed physical model or relaxed tolerance is
used. The original failed log is preserved in the evidence. Every GPU core
target and both affected MPS example binaries are requalified after this fix;
the full example reports are produced afterward. This validates the local
workaround, not a patch or complete diagnosis of the upstream generated kernel.

The finite-shot interferometer fit has a separately traced stochastic difference:
at seed 7, step 37, a shared draw lies between CPU/WGPU cumulative probabilities
separated by 5.79e-8, changing one click. The standalone paired probe reproduces
both final phase errors, 0.00886849 and 0.00903767; complete native probabilities
remain within 3.35e-7 of the analytic oracle. This fitted-parameter diagnostic
therefore retains both values and is checked against each run's existing
nonnegative `<0.05` learning-quality bound. It is not treated as a timing
diagnostic or as bit-identical arithmetic. Physics, held-out MSE and physical-cost
comparisons keep their previous bounds. The raw trace and reproduction sources
are embedded in the JSON evidence. Five comparator mutation tests reject invalid
fits and unrelated numerical/cost deviations; local validation and CPU CI run them.

## Device and package qualification

The CPU workspace passes **272 tests plus one doctest**, all **32 example
binaries**, strict Clippy, formatting, documentation, backend-free/Flex-only
checks and the standalone package-consumer build. Each of the four GPU
configurations passes **135 core tests and 34 example tests across 32 binaries**.
Both MPS example binaries have an additional three-test revalidation after the
layout fix. All **160 fresh reports** (32 per configuration, including CPU) pass
the final comparison, covering 865 structured scientific values per GPU report
set in addition to the top-level metrics. The finite-shot fit exception above
retains its independent quality bound and both measured values.
Requested devices must run: missing GPU access is never a successful skip.

The native matrix is Flex CPU plus NVIDIA RTX PRO 6000 Blackwell, driver
610.43.02, through WGPU/Vulkan and CUDA with Burn kernel fusion disabled and
enabled. Host routing, tableau/frame evolution, decoding, event control and
complex SVD remain host algorithms. Their native numerical reference paths use
the selected device where applicable.

`scripts/validate_backend.sh` discovers every integration target using the shared
runtime-device selector and every example binary. The complete CPU workspace
also covers host-only targets. `scripts/compare_backends.py` checks structured
learning curves and modeled costs as well as top-level metrics; only explicitly
listed measured timing/transport diagnostics are excluded from numerical parity.

Strict Clippy, formatting, doctests, warnings-as-errors documentation,
backend-free/Flex-only builds and a verified standalone crate archive are part
of the recorded checks. Archive source/fixture bytes are compared with the
qualified checkout: all **183 source/fixture files** in the **187-file archive**
match byte for byte. The remaining four files are Cargo's normalized/original
manifest pair, lockfile and VCS metadata. The extracted package is compiled with all targets;
its VCS record correctly identifies the dirty tree and `crates/burn_quantum`.

## Representative controlled results

The full JSON reports retain seeds, physical costs and assumptions. These small
examples establish numerical behavior, not quantum advantage or throughput.

| Example | CPU result and reference |
| --- | --- |
| MPS QAOA | 80 qubits, 9,640 complex stored values, cut fraction 0.75000016 versus analytic 0.75 and random 0.5 |
| Calibrated physical QAOA | Conditional cut 0.99740726 versus analytic 0.99740716; herald 0.0016633788 versus physical reference 0.0016633815 |
| Seeded photonic benchmarks | Five seeds, classification and kernel accuracy at least 0.96, worst Born true TVD 0.014135, 203,520 shared physical preparations |
| Magic distillation | Input independent Z-error 0.01, output error 0.00004014, acceptance 0.797477; source availability 0.995, syndrome-readout error 0.0001, ideal Clifford gates |
| Asynchronous optical jobs | 8,192 attempts per worker layout, 1,642 missed-photon attempts retained, identical one/three-worker records |
| PsiQDK interchange | Four actual SDK cases, 311 emitted gates, 9,581 QASM bytes; full complex amplitudes checked at 3e-6 |

Learning-curve points reuse nested measurement pools, so their costs are not
additive across points. Matched classical controls use the same observed records;
no unrecorded extra physical evaluations are granted to either side. Physical
time and energy columns are explicitly modeled proxies. Four-star studies
separate setup, sampling and decoding times; adaptive-fabric reports separate
source yield, analyzer success, buffer occupancy, latency distribution and useful
output rate. Measured host/controller timings remain diagnostic under shared load.

## Scope of completion

This completes bounded **local reference implementations**, not a claim that
arbitrary large photonic quantum computation is now practical. MPS rank and
exact optical sectors can still grow exponentially. Finite six-ring patches,
planar circuits and magic protocols have separate stated noise assumptions;
growing-distance rough/smooth six-ring surgery and noisy-check hardware threshold
qualification require additional architecture and evidence. No multi-host
throughput, production fault-tolerant fabric, quantum advantage or hardware
execution is claimed. PsiQuantum hardware access remains the future target
requested by the user; the SDK integration here executes its local simulator.

The working tree remains uncommitted and unreleased. This is source-qualified
local evidence, not exact-commit CI or a crates.io publication record.
