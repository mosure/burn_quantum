# Photonic local-simulation acceptance map

Updated 2026-11-09. The M0–M7 **bounded local reference implementations** are now
implemented, including the integration work described below. The [original
roadmap](photonic-roadmap.md) remains the acceptance specification; the
[surgery/threshold qualification](photonic-surgery-validation.md) and the earlier
[completion qualification record](photonic-completion-validation.md) identify
which source snapshot, tests, examples and actual devices were checked.

PsiQuantum hardware execution is deferred as requested. Local reference models
have explicit boundaries: no growing-distance hardware threshold, production
fault-tolerant fabric compiler, arbitrary continuous-variable simulator or
multi-host performance claim follows from these implementations.

## Milestones and acceptance evidence

| Milestone | Implemented local path and acceptance evidence | Explicit model boundary |
| --- | --- | --- |
| M0 semantic reference | Dual-rail/vacuum states, complete destructive fusion, partial failure parities, asymmetric CPTP loss, unconditional branches, native gradients and exact small resource/fabric examples | Dense reference storage and branch enumeration are bounded and exponential |
| M1 scalable execution | Packed tableau, 64-shot reference frames, signed expectations, complete continuation checkpoints, independent Stim fixtures, increasing lattice sizes and separated execution diagnostics. MPS adds mixed qudits, non-Clifford gates, controlled Schmidt truncation, Kraus trajectories and an 80-qubit QAOA oracle | Tableau/frame work is host-side. MPS contractions are native Burn, complex SVD is host-side without reverse AD; entanglement can require exponential rank |
| M2 optical physics | Finite Fock sectors, passive interference, number/loss/detection checks, internal modes, correlated source/phase/loss ensembles, standard and calibrated Grice analyzers, dark-count false heralds and pinned Perceval fixtures | Finite calibrated passive models, with explicit source support/cutoffs and detector assumptions; no universal optical-device model |
| M3 algorithm lowering | Existing Circuit/Angle bindings, bounded algebraic universal synthesis, causal measurement patterns, complete entangled-input channel tests, noisy heralded Choi factories, calibrated Bell effects, finite physical schedules and actual physical VQE/QAOA objectives/costs | Dense opaque synthesis is bounded; numerical reconstruction error is distinct from backend/noise error. Non-Clifford resources and correction apparatus are explicitly supplied models |
| M4 factories/fabric | Correlated availability/quality, finite switch/link/buffer/detector constraints, wavelength/rail reservations and skew, explicit arrival/survival policies, causal events, fresh-resource retries and parity-encoded unknown-input retries. Adversarial reservations and exact tiny stochastic controls | Greedy scheduling can reject a feasible global alternative. Independently routed rails require the declared phase stabilization/matching; retries have a bounded encoded domain and abort on loss |
| M5 logical references | Four-star and rotated six-ring networks, physical Shor orientations/fusions, correlated-loss memory studies, exact small matching/GF(2) oracles, pinned independent stabilizers, finite six-ring patches with real boundary measurements and signed logical correlations, planar open-code memory/H/CNOT/S, local growing-distance merge/split/CNOT with full noisy transition recovery, actual six-ring boundary deformation, and 15-to-1 magic distillation/T injection | Fixed-shot periodic erasure, encoded optical loss, planar-memory and surgery studies have separate explicit models. Ideal check backaction and boundary closure do not imply a calibrated circuit-level or hardware threshold |
| M6 learning | Native optical/physical gradients, explicit conditioning denominators, passive finite-shot mixed-state tomography/overlap with cost, Bures QN-SPSA probes, identifiable held-out calibration, bounded minibatch allocation, seeded learning curves and matched-record controls for classification/kernel/Born tasks | The overlap protocol targets its stated vacuum-plus-single-photon family. Cost/time/energy proxies state their assumptions. Small examples do not establish advantage or large-model trainability |
| M7 execution/interop | Stable component seeds, disjoint threaded/TCP shot accounting, bounded causal checkpoints, actual multi-process ownership of one entangled MPS, durable boundary reconciliation, asynchronous immutable local jobs, cancellation/expiry/backpressure tests and pinned actual PsiQDK local simulation interchange | Trusted local TCP or authenticated tunnels, cooperative callbacks, caller-owned durable storage and a single coordinator. Multi-host/QPU qualification remains outside the requested local pass |

Every engine rejects unsupported physics or exhausted budgets explicitly. No
failed detector/factory/decoder outcome is converted into a successful logical
sample. Finite benchmark limits, confidence intervals, seeds and physical cost
assumptions are part of each study.

## API ownership

- `photonic::optics`: Fock states/densities, passive components, internal modes,
  calibrated ensembles, full optical Bell effects, detection, tomography and
  finite-shot estimation. [Optical guide](photonic-optics.md).
- `photonic::compiler`: logical measurement calculus, `GateSynthesis`,
  `PhysicalCircuit`, calibrated emitted resources, finite placement and
  `ParityRetryPlan`. [Compiler guide](photonic-compiler.md).
- `photonic::fabric`, `program`, `factories`: typed physical topology, packet and
  rail reservations, timing/capacity provenance, correlated factories and finite
  production/control. [Runtime guide](photonic-runtime.md).
- `photonic::stabilizer`, `qec`: independent state/check/logical representations,
  resource and encoded-fusion trajectories, four-star/six-ring/finite patches,
  bounded decoding, open planar logical channels and magic protocols.
  [Clifford/QEC guide](photonic-stabilizer.md), [magic guide](photonic-magic.md).
- `tensor_network`: canonical MPS state, native contractions, host SVD, selective
  observables, pure trajectories and distributed entangled-state ownership.
  [Tensor-network guide](tensor-network.md).
- `photonic::runtime`: immutable identities, result ledgers, causal events, remote
  shot workers and asynchronous local jobs. Readouts retain raw unsuccessful
  attempts; fingerprints must identify the actual program/model/calibration.
- `photonic::interop`: bounded OpenQASM 3 export with explicit global-phase,
  endian, parameter and readout contracts. [PsiQDK interoperability](photonic-interop.md).

Root is a virtual workspace. The library remains in `crates/burn_quantum`,
applications in `crates/burn_quantum_example`. Reusable physical/math/protocol
operations belong in the library; every application remains one executable file.

## Qualifications and extension boundaries

Native Fock/density/pattern/physical-circuit work, MPS contractions and numerical
oracles run on the selected Burn CPU/WGPU/CUDA device. Routing, decoding,
tableaux/frames, event control and complex SVD are host operations. A requested
GPU that is unavailable causes an error rather than an implicit CPU fallback.
The validation runner automatically discovers tests that select a runtime device
and all example binaries; host-only contracts are covered by the workspace run.

The [growing-distance surgery API](photonic-surgery.md) now supplies local XX/ZZ
merge/split, a measurement-only CNOT, complete transition detectors and physical
recovery from observed records. Distance certificates and full entangled-input
channels accompany the implementation. Fixed-shot studies qualify finite-size
suppression/crossings under explicitly named periodic erasure, encoded optical
loss and noisy-check phenomenological models; they do not extrapolate a device
threshold. See the [current evidence](photonic-surgery-validation.md).

The [six-ring optical API](photonic-fusion-surgery.md) distinguishes actual
resource gate teleportation from direct boundary deformation. It retains all
failed attempts and the negative scaling of unboosted whole-circuit heralding.
Specific boundary networks have independent signed logical-channel fixtures;
end-code distance, fusion-record distance and the chosen syndrome closure are
separate certificates. Generic finite boxes never acquire a fault-tolerance
claim solely because their size increases.

The public PsiQDK SDK is qualified through its **actual local simulator**, using
pinned versions, generated QASM hashes and complete complex-state fixtures.
PsiQuantum's current [QPU documentation](https://docs.construct.psiquantum.com/psiqdk-workbench/new-tutorials/QPU-Object.html)
explicitly places hardware access in future support. This implementation targets
the documented logical interchange boundary, not invented hardware endpoints.

## Evidence history

[Surgery qualification](photonic-surgery-validation.md) and its
[raw results](photonic-surgery-results.json) cover the latest pass.
[Completion qualification](photonic-completion-validation.md) and its
[raw results](photonic-completion-results.json) retain the preceding snapshot.
[Previous milestone qualification](photonic-milestones-validation.md),
[expansion qualification](photonic-expansion-validation.md) and
[foundation qualification](photonic-validation.md) retain their original source
hashes and counts. They are historical evidence, not substitutes for validation
of the current source.
