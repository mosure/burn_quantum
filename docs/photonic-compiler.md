# Logical measurement-pattern compiler

`burn_quantum::photonic::compiler` now implements a causal measurement layer
between the existing gate circuit and physical resource/fusion scheduling.
It provides an **ideal MBQC compiler and exact reference executor**, plus a
routed single-qubit primitive, bounded universal synthesis and a composable
physical compiler with calibrated source/optical instruments, finite rail-aware
routing and verified parity-code retries. Ideal and calibrated resource models
are explicit alternatives. Fault-tolerant code selection remains a separate
layer. Logical graph CZ commands describe ideal entanglement; physical execution
uses the separately checked resource/fusion lowering.

## Supported lowering and public API

`MeasurementPattern::compile(&Circuit, PatternCompileLimits)` accepts qubits,
bounded dense qubit unitaries, symbolic Rx/Ry/Rz rotations and structured
multicontrolled X/phase predicates. Dense gates and predicates are first
synthesized into local/CNOT operations; CZ and SWAP have direct native paths.
Dimension and expansion limits are explicit, including for Toffoli/SAT circuits.
Constants, parameter/input indices, affine scales and reversed rotation levels
retain their semantics. Constant-unitary decomposition preserves the quantum
channel; an irrelevant global phase may differ.

The immutable result exposes its original circuit, commands, stable site IDs,
output map, prepared-site count and measurement count. Source circuits are the
persistence input. Serialized compiled patterns are for inspection and have no
unchecked deserialization path.

Commands are `PreparePlus`, `Entangle`, `Measure`, and `CorrectX`. Every classical
correction references a preceding measurement. Each prepared site is fresh;
measured sites are consumed and final outputs are restored to logical wire order.
Compilation bounds both prepared sites and command expansion before execution.

The compiler uses the one-bit teleportation identity

```text
|psi> --CZ-- measure after Rz(a), H; outcome s
|+>   --CZ-- X^s correction                   -> H Rz(a) |psi>
```

Each branch has probability 1/2 for normalized inputs, including an input
entangled with reference qubits. The correction is performed before dependent
commands. Consequently the implemented runtime uses actual outcome-dependent
feed-forward, rather than silently dropping byproducts. Rz and Rx require two
such J(a)=H Rz(a) steps; Ry uses conjugation by fixed Rz rotations. CNOT is H on
the target, a graph CZ edge, then H. These are measurement-calculus primitives;
a graph edge is not an assertion of deterministic optical entanglement.
[Measurement calculus](https://arxiv.org/abs/quant-ph/0412135).

## Native execution and differentiation

`PatternSimulator::execute_all` enumerates every transcript. `execute_branch`
executes one caller-supplied transcript and retains its absolute probability.
Neither method samples or postselects silently. `PatternBranch::state` is a
subnormalized corrected state; `conditional_state` explicitly normalizes after
host validation and retains denominator gradients.

Execution uses native Burn tensors and the existing precision-preserving dense
contractions. Symbolic bindings are validated through the existing circuit
binding path. Gradient flow through data and parameters is preserved.
Preparations append one |+> ancilla and measurement removes one live site, so
peak live width is logical width plus one, even for a long pattern. The exact
classical transcript count remains exponential in the number of measurements.
This is a streaming reference implementation, not a scalable non-Clifford engine.
Very long selected transcripts are rejected before absolute branch weights lose
the dtype exponent range; a future log-weight trajectory API needs its own contract.

## Validation

The `photonic_compiler` target verifies the complete Choi channel for every
outcome of H teleportation, reversed-operand CNOT, CZ and SWAP. Input qubits are
entangled with untouched references; this checks arbitrary inputs and coherences,
not only basis-state populations. Selected transcripts additionally check
arbitrary complex local unitaries, reversed generators and zero-control polarity.
An analytic gradient test covers affine parameters, encoded inputs and explicit
conditioning. Invalid semantic/expansion/transcript/resource inputs are rejected.

The single-file `photonic_teleportation` example compiles a symbolic phase circuit
on a Bell input. Every one of its 32 corrected transcripts is checked against
independent analytic output probabilities. It records ideal pattern size,
measurement/branch counts, total probability and output error.

```sh
cargo test --locked -p burn_quantum --test photonic_compiler
cargo run --locked -p burn_quantum_example --bin photonic_teleportation
```

## Routed heralded gate teleportation

`HeraldedGatePlan::compile(gate, fabric, ports, options)` realizes a bounded,
physical resource/fusion protocol for any constant single-qubit unitary U:

1. Accept an injected dual-rail input at an explicit source port.
2. Supply the ideal two-photon Choi resource `(I tensor U)|Phi+>` at another port.
3. Route the input and one resource half to a standard fusion analyzer, with
   actual shared-link reservations, waiting loss and detector attenuation.
4. Preserve all five destructive outcomes. Successful Psi+/- observations
   condition local corrections `U P† U†` on the output resource photon.

Execution accepts a native `DensityMatrix` and an input-site index, so inputs
may be mixed, parameterized or entangled with untouched reference sites.
Output ordering is restored to the original register, with the input site
replaced by the output resource photon. Gradients through injected tensors are
retained. All preparation occurs before physical loss; ideal resource supply,
local correction and zero controller latency are explicit model assumptions.

Lossless success implements U on arbitrary inputs. Under loss, a success herald
can still accompany an erased output; unequal rail attenuation can distort the
conditional logical state. The returned states retain those effects rather than
promising ideal logical execution from a detector success alone.

Complete entangled-reference tests check both successful outcomes for H, T and
Y, with either input wire replaced. Failure/loss records, shared-link timing,
unconditional lossy probabilities and native gradients are separately tested.
The model uses an ideal Choi resource; factory yield and calibrated physical
preparation remain separate costs, not assumed probabilities of one.

## Composable physical circuits

`PhysicalCircuit::compile` now lowers supported `Circuit` definitions into an
actual resource/fusion schedule. `PhysicalPlacement` specifies injected input
ports, a just-in-time ideal-resource source, an analyzer and classical feedback
latency. `PhysicalGate` retains each gate's resource birth, routed photons,
logical wire map and completion. The compiler tracks output photon locations
between gates and charges storage once from birth/release through readout.
Input and resource-half routes contend for the same exclusive fabric links.

The native universal basis is arbitrary local constant unitaries, symbolic
Rx/Ry/Rz, and CNOT, with direct CZ and SWAP resources. `GateSynthesis` now lowers
dense unitaries on up to four qubits by QR/two-level elimination, and structured
multicontrolled X/phase predicates by exact phase-polynomial circuits. Control
polarity, target order and native affine bindings survive lowering. Both the
measurement-pattern and physical compilers use this checked synthesis. Existing
`GroverSat::oracle()` and diffusion circuits therefore use the same frontend.
A native local gate consumes a two-photon Choi resource and one fusion; a native
two-qubit gate consumes a four-photon resource and two fusions. The compiler never
assumes that a non-Clifford entangling byproduct can be corrected deterministically.

Expansion is exponential in a general dense operator or large predicate, and
is bounded before execution. Defaults allow at most four dense-unitary qubits,
12 predicate qubits and 32,768 native gates; physical compilation additionally
applies its photon/fusion budgets. There is no intentional approximate synthesis.
For every opaque dense gate the implementation reconstructs the whole native
operator on the host and measures the Frobenius residual modulo global phase.
The reported `numerical_error`/`synthesis_error` accumulates those residuals plus
floating-point allowances and must fit the requested budget. This is a numerical
reconstruction diagnostic, not interval-arithmetic certification or a bound on
subsequent tensor precision or physical noise. The algebraic constructions follow
[Barenco et al.](https://arxiv.org/abs/quant-ph/9503016).

For a one-qubit U, each successful fusion uses the local correction U P† U†.
For CNOT/CZ/SWAP, conjugation maps the measured Pauli byproduct to a product of
local Paulis, including when an output has become vacuum. The runtime uses
these physically local corrections; it does not execute an ideal entangler on
the data as a substitute for fusion.

### Exact channel execution without a global resource density

`PhysicalCircuit::execute` accepts arbitrary pure qubit inputs, including
reference entanglement, and native parameter/input tensors. The resource Choi
amplitudes are built on the selected Burn device. Analytically contracting the
specified Bell bras produces local Kraus operators acting on the data density.
For one gate with local dimension d, a resolved record has operator

```text
K = R C U B P A / d
```

where A/B are input and measured-resource attenuation, C is an output-loss
Kraus operator, P is the Bell byproduct and R its physical correction.
The implementation sums the complete output-loss channel before correction.
All unsuccessful joint detector records use their derived POVM effects, retain
absolute probabilities, and terminate the attempt after that gate. The two
fusions of a two-qubit gate both execute before this decision.

Corrected success histories are marginalized before proceeding. This preserves
exact mixed output states and parameter gradients, including asymmetric loss,
without exponentially many successful transcripts. The result reports marginal
joint detector records at each reached gate, not reconstructed full histories.
Resources are requested only after preceding gates succeeded; expected resource
photons and attempted fusions include that survival probability.

Output erasure is latent. It does not trigger a controller action until a later
physical detector observes loss. The all-heralds-success state can therefore
contain vacuum. `herald_probability`, `logical_survival_probability`, abort
probability and expected physical costs are separate quantities. Conditional
objectives must retain and validate their success denominator. Very deep
unencoded teleportation circuits have exponentially small herald probability;
finite-precision results do not imply a practical large-scale architecture.

The retained data density still scales as 9 to the logical-site count; no extra
global density of all supplied resources is formed. This enables complete
entangled-reference two-qubit channel checks without constructing an eight-site
encoded resource density. Larger supported workloads use the separate
[Clifford/frame](photonic-stabilizer.md) and [MPS/distributed](tensor-network.md)
representations. A compiled physical density channel is not automatically
executable by those engines; its operations and noise require a compatible adapter.

### Physical validation and VQE

`physical_compiler` checks complete CNOT/CZ/SWAP Choi channels with both operand
orders, all 25 joint detector outcomes against scalar formulas on a nonuniform
input, symbolic gate/input gradients and expected abort cost. An independent
explicit resource-state/fusion implementation agrees with the contracted
single-gate channel under asymmetric loss. A latent-output-erasure test verifies
that the next gate still consumes resources and learns of loss from detection.

`physical_vqe` is a complete single-file VQE that trains through this physical
compiler, with routed resource preparation, destructive fusion, correction and
loss in every evaluation. It starts from |01>, teleports Ry(theta) and CNOT,
and minimizes conditional Heisenberg energy using native Burn SGD. The analytic
singlet energy is -3. At link transmission 0.98 the herald probability is
`(0.5 * 0.98^2)^3`; expected resource photons are
`2 + 4 * (0.5 * 0.98^2)`, accounting for early abort. The example checks energy,
heralding and resource cost independently; conditional optimization alone does
not establish a photonic advantage.

```sh
cargo test --locked -p burn_quantum --test physical_compiler
cargo run --locked -p burn_quantum_example --bin physical_vqe
```

## Calibrated sources, instruments and physical costs

`compile_calibrated` takes a complete `PhysicalCalibration` snapshot. Its local
and entangling `ResourceFactoryModel` values specify bounded heralded attempts,
photons consumed by every attempt, cycle duration and conditional emitted-state
quality. An optional CPTP channel acts on the complete two- or four-photon Choi
resource in measured-halves-then-output-halves order, supporting coherent,
correlated and vacuum defects. The ideal logical Choi state is the declared
preparation target; the library does not infer a laboratory preparation recipe.
Factory failures emit no resource and abort before data fusion. Factory windows
are reserved at their worst-case duration and early emissions are released at
the window end. Conditional resource noise is specified at that release point,
including internal source storage managed by the source model.

The selected `OpticalBellInstrument` supplies its complete POVM and calibration
identity, including the standard and first-level Grice optical implementations.
The compiler contracts each effect's spectral bras W with each native noisy
resource pure-state component R to obtain K = R W. It retains every positive
spectral component, including false-herald output channels; a declared resolved
Bell result does not certify an ideal Bell projection. Source quality, measured
rail propagation, detector effects, output propagation and feed-forward correction
occur in that order. The two-qubit native Clifford corrections use both declared
XX and ZZ parities, including the Phi outcomes resolved by the Grice analyzer.
No global resource/data density is allocated. Work is bounded by the configured
maximum number of contracted Kraus terms before tensor execution.

`PhysicalRun` separately reports source-abort probability, all observed aborts,
herald probability, logical output survival, expected generation photons,
analyzer ancillas, attempted fusions and duration. Ancillary photons declared by
an analyzer are supplied resources; their preparation yield is not invented.
Initial encoded data photons remain separate inputs. Generation photons include
exhausted attempts and all costs are weighted by the probability of actually
reaching that operation. Conditional objectives differentiate both state and
success denominator. Corrections remain ideal local optical unitaries.

The calibrated regression suite checks standard-instrument agreement with the
independent primitive, full CNOT/Grice Choi channels, emitted-resource CPTP noise,
source exhaustion, cost and analytic native gradients. A deliberately destroyed
unknown Bell input plus dark-count false heralds retains only 1/4 conditional
entanglement fidelity, rather than being silently upgraded to a successful gate.

## Finite buffers and independent optical rails

`compile_finite` and calibrated finite compilation reserve every photon from
birth through consumption or final output. This includes newly emitted resource
outputs held during fusion, and original logical inputs waiting for other gates.
Configuration and feedback are causal, capacities are checked transactionally,
and storage is charged once per segment. Fixed consumption ticks are explicit
inputs; failure to find a schedule never expands a physical capacity implicitly.

`PhysicalCircuit::compile_rails` integrates `RailPacketScheduler` with wavelength
channel capacities and bounded rail skew. Both optical modes are reserved even
when the photon is in a coherent superposition: these reservations describe
configured modes, not a sampled which-rail trajectory. Each wavelength has its
own link/buffer calendar; same-wavelength conflicts remain exclusive, and
wavelengths declared independent may run concurrently. Corresponding analyzer
input rails use matched channel IDs. Both actual rail routes and their aggregate
attenuation remain in compiled provenance. Destination delay aligns rail arrivals,
with that delay's asymmetric attenuation retained.

`RoutePolicy::EarliestArrival` prioritizes arrival time. `MaximumSurvival`
minimizes negative log geometric-mean rail transmission, including waiting until
the fixed consumption deadline. It uses bounded space-time search and reports
infeasibility/work exhaustion. These are ordered greedy packet policies, not a
guarantee of globally optimal throughput or joint scheduling completeness.
Wavelength channels are explicitly independent capacity domains; shared hardware
constraints spanning wavelengths need to be declared in a richer topology.
Paths/storage are assumed phase stabilized. Frequency conversion, uncalibrated
phase drift and coherent interferometers are not modeled by classical routing.

## Unknown-input-safe bounded retries

`ParityRetryPlan` implements a real two/three-port retry channel for a supplied
parity encoding: |0_L> is the uniform even-parity superposition and |1_L> the
odd-parity superposition. A known Z measurement removes one encoding photon and
preserves the unknown logical state up to a known X. A standard-fusion
`Failure { bit }` therefore permits retrying another port with a fresh Choi
resource. `Loss` immediately aborts. After a resolved fusion, destructive Z
measurements of surviving encoding photons and outcome-dependent local
corrections decode the teleported state. Storage during feedback precedes both
Bell and decoding corrections. This realizes the parity-encoding principle in
[Gilchrist, Hayes and Ralph](https://arxiv.org/abs/quant-ph/0505125).

Every terminal history, decode bit, reference state, absolute weight, completion
time and fresh-resource cost is retained. Complete entangled-reference H/T tests
verify every accepted retry branch, including branches preceded by known failures;
additional tests cover exhaustion, total loss and asymmetric feedback storage.
Ideal two/three-port herald yields are 3/4 and 7/8, with expected supplied resource
photons 3 and 3.5. Encoding photons are separately counted inputs. The protocol
assumes a supplied parity encoding and ideal resource/local measurement machinery;
it does not implement a deterministic photonic encoder or recovery after unknown
photon loss. It never reuses a destroyed unencoded input.

## Physical QAOA example

`physical_qaoa` trains a two-qubit single-edge MaxCut QAOA circuit using calibrated
standard optical effects, bounded 0.9-success factories, 0.995-transmission routes,
finite buffers and explicit feedback on every objective evaluation. Both native
CNOT entanglers pass through physical resource/fusion execution. Native Burn SGD
is checked against the exact conditional objective
`(1 + sin(4 beta) sin(gamma)) / 2` and complete herald probability
`0.99^7 * (0.5 * 0.995^2)^9`. Resource counts and source/detector aborts are reported
separately. This is a physical-cost-accounted conditional optimization example,
not evidence of photonic speedup or a multi-seed application benchmark.

```sh
cargo test --locked -p burn_quantum --test gate_synthesis --test calibrated_physical --test parity_retry --test rail_routing
cargo run --locked -p burn_quantum_example --bin physical_qaoa
```

## Capability boundaries

`fbqc_adaptive` retains the complete terminal latency histogram and compares it
with exact enumeration of its three-round source/fusion process. It reports
source invocation/pair yield, conditional analyzer success and unconditional
experiment success separately. Buffer occupancy counts reserved packet slots
from the compiled routes through delayed feedback, including slots whose photon
may already have been lost; it excludes factory-internal preparation and links.
The example checks each node's capacity and reports peak slots, slot-ticks and
mean occupancy. Useful output rate is accepted Bell pairs divided by total
elapsed ticks for sequential experiments, not a claim of globally optimal
throughput. Tick quantiles describe modeled completion latency; separate host
p50/p95/p99 seconds include checkpoint creation/restoration and exclude duplicate
validation replay. Host timings do not affect seeded scientific records.

The integrated local paths now cover bounded universal synthesis, finite/rail-aware
fabric schedules, calibrated standard/Grice instruments, arbitrary conditional
resource CPTP noise, bounded heralded factories and verified parity-code retries.
Global optimal scheduling, shared cross-wavelength hardware constraints, optical
encoding-factory synthesis, automatic fault-tolerant code selection/lattice surgery
and arbitrary dynamic control remain separate capabilities. Source models and
calibration snapshots are explicit effective inputs; successful tests are not
hardware calibration or a fault-tolerance threshold. The dense physical reference
still scales exponentially in live logical sites, while an unencoded long chain
has exponentially small herald probability. Finite precision and resource budgets
remain part of every execution contract.
