# Photonic and fusion-based computing foundations

This guide describes the original encoded M0 model. Additional Fock, stabilizer,
compiler, factory, QEC, learning and runtime modules are documented in the
[current implementation status](photonic-status.md).

`burn_quantum::photonic` is included in published releases since 0.3. This guide
focuses on the encoded-state model and its default scheduler. The separate
[Fock optics](photonic-optics.md), [physical compiler](photonic-compiler.md),
[QEC and finite boundaries](photonic-stabilizer.md), and [runtime](photonic-runtime.md)
guides describe the expanded local stack. These bounded models do not establish
a universal fault-tolerant photonic computer or remote hardware access.

The central design choice is to separate optical state physics, quantum
measurement records, resource-level programs, fabric schedules, and simulation
representations. FBQC uses small entangled resource states and destructive
entangling measurements; unsuccessful optical fusions can reveal a partial
parity, whereas photon-loss erasures can remove both parities. Those distinctions
matter to later decoding. [FBQC paper](https://arxiv.org/html/2101.09310v1).

## Encoded-model surface

| Module | Public entry points | Contract |
| --- | --- | --- |
| `encoding` | `DualRail` | Three-level code, local qubit-circuit lowering, unconditional readouts |
| `loss` | `RailTransmission` | Rail-specific intensity attenuation, composition, storage loss, CPTP channel |
| `resource` | `ResourceState` | Ideal graph states, Phi+ pairs, six-rings; checked descriptions |
| `fusion` | `StandardFusion`, `FusionOutcome`, `FusionBranch` | Complete five-outcome destructive instrument; partial parities and subnormalized states |
| `fabric` | `Fabric`, `FabricNode`, `FabricLink`, `Route` | Directed packet routing with exclusive link intervals and waiting attenuation |
| `program` | `FusionProgram`, `PhotonId`, `CompileOptions`, `CompiledFusionNetwork` | Friendly resource/fusion builder, validated serialization, static compilation |
| `simulator` | `DenseFusionSimulator`, `SimulationLimits`, `FusionRun` | Exact small-system density branches, explicit conditioning, seeded joint sampling |

These modules reside under
[`crates/burn_quantum/src/photonic`](../crates/burn_quantum/src/photonic).
No new dependency or Cargo feature is required. The existing `fusion` feature
continues to mean **Burn GPU kernel fusion**, not a photonic fusion protocol.
The new executor does not misrepresent destructive measurement as an operation
of the existing synchronous unitary `QuantumBackend` contract.

## State, loss and measurement semantics

One site has ordered basis

```text
0 = |10>  logical zero: photon in the first rail
1 = |01>  logical one:  photon in the second rail
2 = |00>  vacuum:       photon absent
```

This represents the zero/one-photon subspace of two optical modes. It cannot
represent two photons in a rail pair, spectral distinguishability, arbitrary
mode coupling, or bosonic bunching. A dimension-three register alone is not a
physical optical specification. `DualRail::unitary` embeds a one-qubit unitary
as `U ⊕ 1`; arbitrary three-level gates need not preserve the photonic encoding.
Compare [Perceval's encoding tutorial](https://perceval.quandela.net/docs/v1.3/notebooks/Encoding_Tutorial.html).

For rail intensity transmissions eta0 and eta1, the loss channel uses

```text
K0 = diag(sqrt(eta0), sqrt(eta1), 1)
K1 = sqrt(1-eta0) |vac><0|
K2 = sqrt(1-eta1) |vac><1|
```

It preserves trace, makes vacuum absorbing, and traces separate environment
records for loss from each rail. Logical-zero population is never used to hide
lost photons. Consecutive independent attenuators compose by multiplying their
intensity transmissions. `over_ticks` uses integer exponentiation, including
exact identity for zero waiting time. Rail asymmetry changes the conditional
logical state, so a generic scalar survival coin is insufficient.

`DualRail::z` is diag(1,-1,0) and reports an unconditional expectation.
`DualRail::survival` is diag(1,1,0). Normalizing a conditional readout requires
retaining the corresponding survival probability; it changes the objective.
Constant loss channels retain gradients through input tensors. Transmission
parameters are currently host scalars, not trainable Burn tensors.

The standard analyzer resolves Psi+ and Psi- in the one-photon-per-site code:

| Outcome | Quantum action before tracing both inputs | Revealed information |
| --- | --- | --- |
| Success, XX even | Project onto Psi+ | XX=+1, ZZ=-1 |
| Success, XX odd | Project onto Psi- | XX=-1, ZZ=-1 |
| Failure, bit false | Project onto 00 | Individual Z results +1,+1; ZZ=+1, XX erased |
| Failure, bit true | Project onto 11 | Individual Z results -1,-1; ZZ=+1, XX erased |
| Loss | Project onto the sector where either site is vacuum | Coarse record; both parities discarded |

The projectors sum to the nine-dimensional identity. Both input sites are
removed in every branch. Output sites retain their original order; compiler
photon IDs remain stable even as dense register indices change. The loss record
intentionally discards any surviving single-detector information.

**The 50% success rate is not a state-independent random switch.** Independent
maximally mixed code inputs give 50%; a Psi+ input gives 100% and a Phi+ input
gives 0% in this fixed analyzer basis. The separate
[Grice optical instruments](photonic-optics.md#physically-derived-bell-instruments)
include their own ancillary states, detector model and loss accounting;
`StandardFusion` retains its original instrument.

`FusionBranch::probability()` is an absolute per-batch weight.
`unnormalized_state()` has that weight as its trace. Impossible branches remain
present, so differentiable exact execution has no value-dependent pruning.
`conditional_state(min_probability)` explicitly reads back weights and rejects
conditioning if any batch entry is impossible or too small. Conditioning
retains quotient gradients. When all sites have been consumed, the state is
`None` and the classical branch probability remains available.

## Encoded-model construction and local lowering

`DualRail::encode_circuit(&Circuit)` lowers existing **local** qubit unitaries
and rotations to the code space. It preserves parameter/input indices, affine
bindings and native Burn differentiation. It rejects entangling operations
with the offending instruction index. It is neither an optical mesh synthesizer
nor a general gate-model-to-FBQC compiler.

`FusionProgram` expresses algorithms using supplied resources and fusions:

```rust
use burn_quantum::photonic::{FusionProgram, ResourceState, NodeId};

let mut program = FusionProgram::new();
let a = program.add_resource(ResourceState::bell_pair(), NodeId(0))?;
let b = program.add_resource(ResourceState::bell_pair(), NodeId(1))?;
program.fuse(a[1], b[0], NodeId(3))?;
// Compile against a validated Fabric with source nodes 0/1 and fusion node 3.
// Remaining outputs have stable IDs a[0] and b[1].
# Ok::<(), burn_quantum::Error>(())
```

Compilation produces routes, link reservations, fusion ticks, accumulated
transmissions, and output IDs. It retains the source program, fabric snapshot,
compiler options and schema marker. It allocates **no quantum state**. The test
suite compiles a 200-photon program and separately verifies that the dense
reference refuses that simulation before allocation.

`FusionProgram::resources` and `fusions` expose typed read-only recipes, and
`ResourceState::edges` / `hadamards` describe preparation without building dense
matrices. Alternate executors can inspect these recipes directly.

Program and fabric JSON use independent schema version 1 with checked
reconstruction. Unknown versions and unknown recipe/fabric fields fail. Photon
reuse, repeated operands, invalid graph edges, wrong node roles, malformed
transmissions, overflow and budget violations are errors. Deserialization does
not bypass constructors. Bound input byte sizes before parsing untrusted JSON;
owned vectors and serde parsing are not a streaming memory limit.

The compiled plan is read-only and serializable for inspection; it deliberately
has no unchecked deserialization path. Persist recipes and recompile. The core
`Circuit` and QN-SPSA checkpoint schemas are independently versioned.

## Default packet fabric model and scheduling policy

The following describes `FusionProgram::compile` with default timing options.
`compile_timed` adds explicit resource births, dependencies and concurrent
scheduling. `compile_finite` additionally checks packet buffers and switches;
independent wavelength/rail scheduling belongs to the
[physical compiler](photonic-compiler.md) and [runtime](photonic-runtime.md).

- Each directed link transports one synchronized dual-rail packet at a time.
  Reservation intervals are half-open `[departure, arrival)`. Parallel links
  have independent occupancy. Opposite directed links are independent;
  half-duplex coupling is not represented.
- Route selection minimizes arrival time, with deterministic node/link traversal
  tie breaks. It does not optimize success probability, congestion globally,
  source throughput, or optical footprint.
- Sources supply complete ideal resources at tick zero. Builder calls do not
  model probabilistic resource factories or resource-generation timing.
- Fusions follow program order. The two inputs are greedily routed against
  existing reservations; the earlier arrival waits for the other. One analyzer
  tick is reserved before the next fusion starts. Different fusions do not run
  concurrently in this baseline scheduler.
- Storage attenuation includes time before release, waits for busy links,
  synchronization at the analyzer, and output storage through the finish tick.
  Buffer capacity, coherence noise and switch reconfiguration are not modeled.
- Link transmission describes net traversal loss. Detector efficiency is a
  separate attenuator before otherwise ideal photon-number-resolving detection.
  Avoid double-counting losses between these calibrations.

This default packet model does not infer coherent optical propagation from
transport links. The expanded rail scheduler and optical executor separately
represent rail/wavelength identity, bounded buffers, skew and controller timing.
Hardware calibration and multi-clock synchronization remain model-specific.
Switch/multiplexing networks are a material architectural cost in photonic FBQC;
an abstract shortest path does not remove that cost. [Switch-network study](https://arxiv.org/abs/2109.13760).

## Execution, scale and QML boundaries

The reference prepares the supplied resources in a joint Burn state, forms its
density matrix, applies each photon's compiled loss, and performs the scheduled
instruments. Moving independent local attenuation before measurements on other
sites is valid for this static model: there are no intervening local gates or
adaptive retransmissions, and each photon is transported/consumed at most once.
That optimization must be revisited when those assumptions change.

For n supplied sites and batch B, the initial density uses `2 B 9^n` real
scalars. F fusions produce `5^F` retained branch records, including zero-weight
branches. Defaults allow at most 125 branches, and existing `ResourceLimits`
preflight states, operators and plans. Under the default density-entry limit,
only up to six sites fit at batch one. These are element preflights, not a total
host/device/autodiff-memory guarantee. Larger Clifford resource plans can use
the implemented [tableau/frame engines](photonic-stabilizer.md); non-Clifford
low-entanglement circuits have a separate [MPS engine](tensor-network.md).
Their capability and noise contracts differ from this exhaustive density path.

Exact tensor operations use native Burn devices, dtype checks and the existing
precision-preserving contraction implementation. Sampling and conditional
validation are explicit host boundaries. `sample` draws the joint branch/output
distribution with local ChaCha8 randomness and a total-shot budget. Every attempt
counts, including complete loss; final output digits can also contain vacuum.
Tiny floating roundoff is tolerated during CDF construction, without discarding
failed attempts or changing quantum branch weights.

Native gradients through encoded rotations, loss and fusion branch weights are
tested. A discrete routing choice or sampled click is not a differentiable
operation. The existing QN-SPSA optimizer requires a specified pure-state
fidelity evaluator; it must not be presented as a mixed-state or noisy-hardware
natural metric without a separate justified geometry and measurement protocol.

## Run the complete first example

[`fbqc_bell_swap.rs`](../crates/burn_quantum_example/src/bin/fbqc_bell_swap.rs)
contains the complete problem, fabric, program, evaluation and acceptance test.
Two Bell pairs feed a shared transport link, forcing arrivals at ticks 3 and 5
and a finish tick of 6. Input transmissions 0.9 and 0.8 give success 0.36,
failure 0.36 and loss 0.28. Corrected successful outputs have unit Bell fidelity;
the example also checks unconditional seeded sampling.

```sh
cargo run --locked -p burn_quantum_example --bin fbqc_bell_swap
cargo test --locked -p burn_quantum --test photonic
BURN_QUANTUM_DEVICE=wgpu cargo test --locked -p burn_quantum --features wgpu --test photonic
BURN_QUANTUM_DEVICE=cuda cargo run --locked -p burn_quantum_example --features cuda,fusion --bin fbqc_bell_swap
```

The [validation record](photonic-validation.md) distinguishes host-only planner
checks, CPU references, accelerator execution and unverified optical capabilities.
