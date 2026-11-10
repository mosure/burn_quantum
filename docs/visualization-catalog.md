# Configurable quantum experiment catalog

The 0.3.0 catalog combines configurable scientific experiments with parallel
subsystem views. Release qualification covers the delivery gates below.

## Presentation contracts

- Default to a register overview with simultaneous reduced-state panels. Every
  qubit uses its actual partial-trace Bloch vector; qudits use populations. A
  selection opens focused inspection and the joint computational distribution.
- Keep panels legible with an adaptive grid and bounded groups on small screens.
  All panels share the timeline and camera orientation. Selection is direct,
  with keyboard/touch-compatible controls retained.
- Render controlled gates, open/closed controls, SWAP, controlled phases and
  multi-wire unitaries according to their operation semantics. Classical
  measurements and feed-forward must be distinguished from coherent controls.
- Offer complexity settings before simulation, showing register size, gate
  count and retained-state budget. Reject excessive settings before allocation;
  never silently change the algorithm or truncate its quantum state.
- Retain the small Bell, interference and qutrit demonstrations alongside
  larger algorithm and protocol examples.

## Algorithm contracts and independent checks

| Experiment | Complexity | Scientific check |
| --- | --- | --- |
| Mixed register | Qubit/qudit groups | Reduced purities and marginals of independent entangled pairs |
| Teleportation | Independent transfers within the budget | Every measurement branch preserves the input state at the receiver after classical correction |
| Toffoli / controlled X | Control count | Computational truth table and superposition amplitudes |
| QFT | Register size and direction | Independent discrete Fourier matrix, ordering and inverse round trip |
| Deutsch–Jozsa | Input width and oracle class | Constant versus balanced input-register readout |
| Phase estimation | Counting precision | Exact binary phases and off-grid probability distribution |
| Grover search | Search width and iteration count | Analytic marked-state success probability |
| QAOA MaxCut | Graph size and ansatz layers | Independently evaluated cut expectation and parameter bindings |
| Bit-flip code | Code size and injected error | Logical recovery, syndrome and failure assumptions |

## Delivery gates

1. Shared scientific APIs and owned, validated experiment/configuration records.
2. Canonical diagram primitives and an executable, explicitly modeled measurement
   protocol wherever a demonstration uses classical feed-forward.
3. CPU mathematical oracles, actual WGPU numerical parity and resource rejection.
4. Native/browser visual review of small, large and mixed-dimensional registers;
   selection, resizing, timeline, complexity changes and JSON interchange checks.
5. Clean package dry runs, exact-commit CI/Pages, dependency-ordered publication,
   archive provenance and an independent registry-only consumer.

## Implemented structure and limits

- `burn_quantum::algorithms`: one module per algorithm; standard `Circuit` IR,
  structured controls, symbolic QAOA angles and composable QFT.
- `burn_quantum::protocol`: checked unitary references, fresh classical bits,
  explicit outcome branches and actual conditional-state execution.
- `demo::DemoConfig` / `preview` / `build_configured`: shared native/browser
  configuration, bounded preflight, actual traces and output interpretation.
- `model::diagram`: semantic gate classification independent of prose labels.
  `view/circuit` conservatively packs disjoint vertical spans into columns.
- `view/state/{bloch,overview}`: one projection for focus and register views;
  adaptive groups retain direct selection and bounded geometry.
- `view/fabric/{components,timeline}`: physical roles, resource ownership, paired
  rails, held outputs, travel/wait intervals and destructive measurement occupancy.

Defaults retain at most 257 frames, 65,536 basis amplitudes per frame and
1,048,576 aggregate amplitude/local-density entries. These constraints combine:
16-wire GHZ exceeds the default retained-trace budget, although a shorter
16-wire trace could fit. Callers may set explicit `TraceLimits`; the viewer does
not infer free GPU memory. Cached host traces are deliberately separate from the
shared-WGPU buffer bridge for dense live readouts. Browser tracing yields to the
event loop between groups of operations.

Teleportation uses three wires per transfer and two classical bits. A selected
branch's probability is retained at every step. The bit-flip example is an odd
repetition code with coherent majority decoding and retained syndrome information;
phase errors and repeated noisy error-correction rounds are outside that example.
The QAOA gallery evaluates fixed parameters; the executable training example uses
the same symbolic circuit. Oracle examples specify their actual promise family.

## Validation evidence

The algorithm tests compare Fourier amplitudes to an independent DFT, phase
estimation to the finite geometric-series distribution, Grover to its rotation
formula, and QAOA to independent diagonal-cost and tensor-product mixer evolution.
They check every one/two-transfer teleportation branch and all correctable error
patterns for repetition lengths 3, 5 and 7, including a beyond-distance failure.
Full-document CPU/WGPU comparison includes all fifteen default demos, branch
likelihoods, local density matrices, amplitudes and QN-SPSA history.

Browser qualification covers every catalog entry, 12 simultaneous Bloch panels,
direct selection, focus, explicit 13-qubit execution, oversize rejection, narrow
viewport grouping, timeline/JSON controls, touch input and device-loss handling.
The fabric's resource IDs, route waits, output locations and destructive occupancy
are checked separately from the quantum resource-preparation state.

## Scientific references

Conventions are implemented and tested explicitly above; these primary teaching
and research sources explain the corresponding protocols:

- [IBM: quantum teleportation](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/entanglement-in-action/quantum-teleportation)
- [IBM: Fourier transform and phase estimation](https://quantum.cloud.ibm.com/learning/en/courses/utility-scale-quantum-computing/quantum-phase-estimation)
- [IBM: Grover's algorithm](https://quantum.cloud.ibm.com/docs/en/tutorials/grovers-algorithm)
- [IBM: repetition codes](https://quantum.cloud.ibm.com/docs/en/tutorials/repetition-codes) (measurement-based reference; the gallery labels its coherent decoder explicitly)
- [Bartolucci et al.: fusion-based quantum computation](https://www.nature.com/articles/s41467-023-36493-1)


## Parallel Bloch presentation refinement

The register grid uses centered compact cells with restrained selection brackets,
shared orientation and optional trails. Spatial mode is an orthographic 3D
representation with analytically shaded sphere surfaces and front/back grid cues.
Planar mode preserves all coordinates through an XY/XZ/YZ disk plus an explicit
hidden-component label. Shared surface images are built/uploaded once and reused;
there is no render target or camera per subsystem. Plot clipping applies to both
the surface images and line geometry. Secondary controls live in a single Settings panel by default to
protect plotting space, and browser mobile canvases retain useful vertical room.

The default control surface has contextual view tabs, Focus, Settings and Details,
plus three playback buttons. A scrollable settings panel holds projection, guides,
trails, speed and complexity. The native catalog uses quiet list rows; phones use
one experiment selector. UI diagnostics are sampled after layout so hit rectangles
match the state being reported.
