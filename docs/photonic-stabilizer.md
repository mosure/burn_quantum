# Clifford execution and erasure-decoding references

The `photonic::stabilizer` module executes Clifford resource/fusion trajectories
with a host bit-packed tableau. It removes the dense state-size restriction for
this supported family. Fixed measurement programs additionally support 64-shot
bit-sliced Pauli-frame batches and complete continuation checkpoints. These are
host engines; the dense resource oracles in the examples execute on the selected
Burn CPU/GPU device. No photonic hardware threshold is claimed.

## State and measurement API

`Tableau::zero(n, limits)` allocates a pure stabilizer state with n stabilizer
and n destabilizer rows. Each row stores packed X/Z bits and a phase modulo four,
using the convention `i^phase X^x Z^z`. H, S, X, Z, CNOT and CZ preserve this
representation. `PauliProduct` checks distinct site indices; `expectation`
returns exactly -1, 0 or +1. `measure` accepts a random bit and reports whether
that bit was actually needed. It supports deterministic replay and exhaustive
small-case branch enumeration without silently forcing a physically impossible
outcome. Indices stay fixed after measurement.

This follows the stabilizer/destabilizer approach described by
[Aaronson and Gottesman](https://arxiv.org/abs/quant-ph/0406196). The Rust code
uses its own phase representation and is tested against independent scalar
complex amplitudes and a pinned Stim reference.

`PauliFrame` stores classical X/Z corrections and propagates them through
Clifford operations. `flips` tells whether a frame reverses a requested Pauli
readout. Global frame phase is intentionally omitted because it does not affect
measurement outcomes. Frame operations are separate from state preparation.

```rust
use burn_quantum::photonic::stabilizer::{Pauli, PauliProduct, Tableau, TableauLimits};
let mut state = Tableau::zero(2, TableauLimits::default())?;
state.h(0)?;
state.cnot(0, 1)?;
let zz = PauliProduct::new(vec![(0, Pauli::Z), (1, Pauli::Z)])?;
assert_eq!(state.expectation(&zz)?, 1);
assert!(!state.measure(&zz, true)?.negative); // The supplied random bit is irrelevant.
# Ok::<(), burn_quantum::Error>(())
```

## Resource/fusion trajectories

`StabilizerSimulator::prepare` consumes the same checked compiled resource/fabric
plan as the dense executor. It prepares graph resources and their local
Hadamards without building amplitudes. `sample(plan, start, shots, seed, readouts)`
returns complete attempts with global shot IDs, fusion records, computational
output digits and optional conditional-trajectory Pauli expectations. Readouts
use stable global photon IDs and can reference only unconsumed outputs.

The supported noise model is **balanced independent rail loss**. Unequal rail
transmissions fail before tableau allocation: their no-loss filter generally
leaves the stabilizer family. A lost entangled photon is traced through a hidden
Z measurement, with its result averaged over trajectories. Simply deleting the
vertex would incorrectly preserve or invent survivor correlations. Vacuum output
digits are 2 and requested logical Pauli factors on lost sites contribute zero.

For two surviving fusion inputs, the executor first measures ZZ. Odd ZZ leads
to an XX measurement and a resolved Psi+/Psi- outcome. Even ZZ leads to a Z
measurement that resolves 00/11 failure. A loss record discards all measured
information and traces both consumed sites. This is the same complete effective
instrument as the dense reference, including state-dependent success and partial
failure parity; it is not a state-independent 50% coin.

Random streams use the runtime's versioned shot/component derivation. The
stabilizer engine has a fixed domain tag `0x53544142494c0001`, separating it from
factory randomness even when callers use the same global seed. Within that
domain, loss components use `2*photon`, fusion components `2*n+fusion_index`, and
terminal output components `2*n+F+output_index`. Splitting an unchanged experiment
into disjoint shot ranges preserves each result exactly. Changing the program
or physical model requires a new job identity; these IDs do not promise random
alignment between different experiments.

The returned `pauli_expectations` are exact values for each sampled environment
trajectory **before** terminal Z readout. Their averages estimate unconditional
encoded expectations. They are not an extra compatible set of physical detector
outcomes. `StabilizerShot` serializes for the runtime's checked result ledgers.

## Limits and scaling evidence

Tableau storage is O(n²), gate application O(n), and general Pauli measurement
O(n²) in bit operations. Each tableau is guarded by maximum qubits and packed
word count. The sampler holds one prepared and one working tableau, and bounds
returned records separately. Allocation limits exclude allocator overhead and
do not establish a total process-memory cap. Consumed indices remain allocated
in this trajectory executor. The separate reference sampler below streams fixed
measurement windows while retaining every live frame plane; it does not partition
entanglement or discard causal boundaries.

A local debug-build GHZ preparation and correlation query produced the following
single observations on 2026-11-09. The timings are diagnostic measurements, not a
statistically qualified performance benchmark:

| Qubits | Tableau allocated bytes | Preparation and query |
| --- | --- | --- |
| 256 | 61,440 | 1 ms |
| 512 | 188,416 | 3 ms |
| 1,024 | 638,976 | 15 ms |
| 2,048 | 2,326,528 | 70 ms |

The public resource executor also samples an actual 200-photon recipe containing
100 independent Bell resources and checks every pair's output correlations.
This qualifies a structured Clifford workload beyond the dense reference limit,
not arbitrary 200-photon optical or non-Clifford simulation.

## Independent validation

`cargo test --locked -p burn_quantum --test stabilizer -- --nocapture` runs six
checks covering:

- All 256 Pauli expectations on twelve seeded four-qubit circuits, then repeated
  measurements, against an independent scalar-complex state calculation.
- Frame propagation against explicitly applied state corrections.
- 40,000 lossy Bell-swap trajectories, with normal and reversed operands, against
  the dense joint branch/output distribution and conditional XX correlations.
- Exact shard replay, unsupported asymmetric loss, invalid readouts and budgets.
- Polynomial storage and the large structured resource recipe.
- **1,024 exact expectations from Stim 1.15.0** across Clifford and postselected
  states, including complex/Y phases.

The pinned source archive SHA-256 is
`95236006859d6754be99629d4fb44788e742e962ac8c59caad421ca088f7350e`.
The reference was compiled and executed as native C++ because the local Python
3.14 installation lacked the headers needed by Stim's bindings. This is actual
Stim runtime parity for the declared fixtures, not merely a hand-written
expected-value file. The fixture, C++ generator and isolated regeneration script
live under [`tests/stabilizer`](../crates/burn_quantum/tests/stabilizer).
The regeneration script verifies the source archive hash and needs a C++20
compiler; ordinary Cargo tests consume the saved fixture without Python or Stim.
To fetch a source archive even when a compatible wheel exists, use
`python3 -m pip download --no-deps --no-binary=stim stim==1.15.0`, then run
`python3 crates/burn_quantum/tests/stabilizer/regenerate_stim.py /path/to/stim-1.15.0.tar.gz`.
Both Rust fixture tests pin the archive hash and complete case/record counts,
including all 48 measurements in each six-ring branch. Truncated fixtures fail.

## Binary erasure decoder and memory example

`photonic::qec::ParityCode` describes binary relations `syndrome = H e` and
`logical_flips = L e`, with checked widths, bounded storage and versioned
serialization. `decode_erasures` assumes errors occur only on the supplied
unknown coordinates. GF(2) elimination distinguishes:

- `Decoded`: all compatible corrections have the same logical action.
- `Ambiguous`: at least one declared logical value is undetermined.
- `Inconsistent`: the syndrome cannot arise from the supplied erasures.

Dependent checks and physical degeneracy are permitted; only *logical*
ambiguity prevents decoding. Correlated erasures can be supplied as one mask.
This generic decoder does not transform joint fusion erasures into independent
parity errors. `MeasurementRelations` now derives checks from actual resource
stabilizers, while `SyndromeGraph` below supplies a separate bounded decoder for
unknown non-erased flips on the explicitly constructed four-star geometry.

`ParityCode::repetition(distance)` supplies an adjacent-ZZ-check reference for
one binary X-error channel. Tests exhaust all erasure-supported fault patterns
at distances 3, 5 and 7, plus dependent-check and persistence cases.

The single-file [`fbqc_repetition_memory`](../crates/burn_quantum_example/src/bin/fbqc_repetition_memory.rs)
example retains an untouched reference qubit entangled with the encoded memory.
It enumerates the erasure-supported X-fault ensemble at all three distances,
applies each decoded correction and verifies both logical Bell correlations.
It checks 2,289 correctable fault cases. With independent coordinate-erasure
probability 0.2, exact ambiguity probabilities are 0.008, 0.00032 and 0.0000128.
A small four-qubit dense oracle runs on the selected Burn CPU/GPU device; the
larger frame/decoder work is explicitly host work.

**Coordinate erasure in this repetition example means a flagged unknown X error,
not a physically missing photon.** The distinct four-star study below includes
physical photon loss and actual fusion parity erasures.


## Batched reference trajectories and continuation

`ReferenceSampler::compile(initial, observables, max_terms)` executes one exact
reference tableau trajectory. Each genuinely random measurement records a Pauli
kickback. `sample(start, shots, seed, noise, max_bits)` samples independent initial
X/Y/Z faults and propagates 64 shot frames per packed word. Measurement randomness
remains independent per shot and component. This follows the reference/frame
sampling approach described in the [Stim paper](https://arxiv.org/abs/2103.02202),
with an independent Rust implementation. The fixed program may include
noncommuting Pauli measurements. Adaptive measurement-program changes require
recompilation; this does not simulate arbitrary non-Clifford gates.

`start` creates a `ReferenceSession`; `advance(measurements, max_bits)` emits the
next window without retaining old output records. `checkpoint()` preserves the
entire X/Z frame planes, fixed compiled program, cursor, global shot range and RNG
namespace. Checkpoints serialize as schema 1. `resume` matches the whole compiled
program and checks dimensions, padding, cursor and allocation bounds. Its bit
budget covers both complete planes, new output rows and random scratch space.
A failed budget check leaves the session unchanged. Persistence input byte sizes
must also be bounded before deserialization. This is causal *measurement-window*
continuation; no quantum state is split across workers or silently truncated.

Storage of the frame planes is `2*n*ceil(shots/64)*8` bytes. The initial reference
tableau and compiled kickback program have separate polynomial storage. RNG
initialization still runs per shot/component, so bit slicing is not a claim of
SIMD RNG or a qualified speedup. Tests compare complete joint distributions to
exact branch enumeration for noncommuting measurements with identity, pure-X and
mixed-X/Y/Z faults, and verify exact replay across shot shards and JSON restarts.

## Checks derived from resource and fusion operators

`MeasurementRelations::derive` computes the intersection of a pure resource
stabilizer group with a commuting measurement group. GF(2) nullspace elimination
finds a complete check basis; explicit Pauli multiplication and the tableau fix
all signs, including Y phases. `correlation(target)` solves the corresponding
coset problem for an output Pauli observable. `infer(records, target)` eliminates
unknown parities and distinguishes known, erased, unconstrained and inconsistent
results. It shares the GF(2) implementation with `ParityCode`.

Records use `Some(false/true)` for +1/-1 eigenvalues and `None` for erasure.
`fusion_records` maps standard fusion outcomes to ordered XX/ZZ records while
preserving the known ZZ parity on ordinary failure. Caller-side physical output
loss must still be tracked; a relation cannot recreate a lost output photon.
Tests derive signed XX, YY and ZZ Bell-swap output correlations and verify known
checks, inconsistent records and erased-value inference.

## Actual periodic four-star network

`FourStarTorus::new(L, source, analyzer, max_photons)` implements the cubic
edge/face incidence construction from Appendix A of the
[FBQC paper](https://arxiv.org/html/2101.09310v1). Each unit cell has three edge and
three face resources, each with four photons; each face fuses with its four
boundary edges. A local Hadamard frame on edge resources converts the published
XZ/ZX fusion convention to this library's XX/ZZ convention. Thus cube checks use
ZZ records and vertex checks use XX records. Each check contains 24 physical
fusion parities. Periodic side length L uses `24*L^3` photons and `12*L^3` fusions.

`ResourceState::four_star` prepares the X-basis GHZ resource with stabilizers
ZZZZ and adjacent XX. `with_local_hadamards` supplies the explicit local-frame
conversion. Both syndrome graphs carry three geometric toric cut parities.
`erasure_logicals` detects nontrivial cycles supported on the supplied erased-edge
mask, including correlated XX/ZZ losses from the same physical fusion.
`decode` assigns zero cost to erased edges and unit cost to others, computes
shortest paths and solves minimum-cost defect pairing exactly by subset dynamic
programming. The **explicit maximum is 20 defects**, with `BudgetExceeded` rather
than a misleading correction if the configured bound is exceeded. This is a
bounded reference decoder, not a production blossom/MWPM implementation.

Validation checks every geometric check against the actual prepared resource
stabilizers, then checks complete sampled records against both the geometric
checks and the independently derived complete relation basis. Erasure homology
is compared against independent dense binary decoding over 400 random masks.
Every single fault at L=3 is corrected without changing a logical cut. For L=2,
all 128 possible syndromes are compared with exhaustive subgraph minimum costs,
with and without zero-cost erased edges.

## Finite physical-noise memory study

`run_four_star_memory` samples all physical attempts on the closed periodic
network. Balanced photon attenuation is sampled once per input photon; loss of
either input erases both fusion parities. Each photon also receives an independent
validated I/X/Y/Z mixture. In actual unboosted mode, XX is unavailable on ZZ-even
failure, retaining the state dependence of the standard analyzer. The optional
complete-Bell mode is an explicit ideal effective-instrument reference, **not a
claimed deterministic linear-optical device**.

Fully commuting Bell records are sampled and unavailable parities are erased.
On this fully consumed network this gives the joint marginal of available
records, including the loss/failure cases. The decoder sees no counterfactual
erased values. Coupled noiseless reference records are used only afterward to
score logical residuals. Independent graph matching uses unit costs and is not
a joint maximum-likelihood decoder for correlated X/Y/Z errors.

The single-file
[`fbqc_four_star_memory`](../crates/burn_quantum_example/src/bin/fbqc_four_star_memory.rs)
first checks the four-photon resource's ZZZZ and XX expectations on the selected
Burn device. It then runs the host lattice study. CPU debug observations on
2026-11-09 used seed 815, 256 attempts per model, physical photon loss 0.005 and
pX=pY=pZ=0.0005. The ideal complete-Bell reference produced:

| L | Photons | Corrected | Pauli logical errors | Erased logicals | Decoder budget | 95% failure Wilson interval |
|---|---:|---:|---:|---:|---:|---|
| 2 | 192 | 206 | 38 | 12 | 0 | [0.1514, 0.2482] |
| 3 | 648 | 239 | 15 | 2 | 0 | [0.0419, 0.1038] |
| 4 | 1,536 | 245 | 9 | 0 | 2 | [0.0242, 0.0753] |

The denominator is **all 256 attempts**, including erasure and budget outcomes.
These finite samples do not establish a threshold or asymptotic suppression.
The actual unboosted fixed-ZZ analyzer corrected **0/256 at each distance**:
its missing XX parities are a substantial limitation, not removed by
postselection. This is why boosted/encoded and failure-basis-randomized hardware
models must be qualified separately before importing published threshold claims.

The two complete frame planes for 256 shots occupied 12,288 / 41,472 / 98,304 bytes
at L=2/3/4. Whole per-distance example work (ideal check, noisy complete and
unboosted runs, additional frame allocation) took 3.12 / 15.48 / 44.20 seconds in
one debug CPU run. These are diagnostic observations, exclude unrelated process
allocations, and are not a statistically qualified throughput benchmark.

The periodic model supports closed homological memories. The sections below
add an independently checked six-ring network, physical inner encoding, and
open-boundary circuit references. These models retain distinct preparation,
measurement and noise assumptions; a successful finite study does not establish
a hardware threshold.

## Four-photon Shor encoded resources and fusion

`ShorEncodedResource::new` encodes each vertex of a logical graph into
the [[4,1,2]] code used in the FBQC paper: block stabilizers XXXX, Z0Z2, Z1Z3 and
logical representatives X0X2, Z0Z1. Its explicit graph construction starts each
block as two Bell pairs on (0,2),(1,3) and applies the logical CZ edges between
ports {0,1} of neighboring blocks. The resulting resource uses the existing
`ResourceState` recipe and every physical executor. The 64-photon resource cap
permits 16 logical sites; the six-ring becomes an actual 24-photon resource.
Declared local logical Hadamards switch the whole block to `ShorOrientation::Dual`.
The dual stabilizers are ZZZZ, X0X2 and X1X3, with logical X0X1 and Z0Z2.
Physical H on all four sites, together with these canonical representatives,
implements logical H. Mixed-orientation fusions permute the right block's
coordinates as [0,2,1,3], so their recovered observables are the same canonical
logical XX and ZZ. Tests independently encode all four orientation pairings and
all four logical Bell states.

`ShorBlock` preserves ordered stable photon identities and exposes checked code
stabilizers and logical Pauli products. `EncodedFusion::append` atomically appends
four standard physical fusions between orientation-matched code coordinates. Every attempt
consumes **eight signal photons and four analyzer operations**, including failed
and lost attempts; no ancillary source or ideal success coin is inserted.

`recover` consumes the actual program's physical `FusionOutcome` records.
Recovery solves the code-induced parity checks, retaining independent logical
XX/ZZ erasures whenever the data are insufficient. Physical identical-Z failure
bits also check the available individual code stabilizers. Inconsistent input
records are explicit. This is erasure recovery assuming valid input code
stabilizers, not correction of arbitrary unknown Pauli faults or a threshold
claim. `recover_parities` also permits explicitly declared complete-Bell reference
records; it does not change the physical analyzer used by the program.

Dedicated tests verify all code and encoded-graph stabilizers of the 24-photon
six-ring, logical anticommutation and detection of every weight-one Pauli fault.
An independently specified Clifford encoding circuit enumerates all physical
branches of all four logical Bell inputs. With ideal unboosted analyzers, logical
Psi inputs fully resolve with probability 1 and Phi inputs with probability 1/2,
so the uniform logical Bell ensemble has success probability 3/4 at the declared
eight-photon cost. This is encoded-fusion redundancy, distinct from the optical
Grice boost and its ancillary-photon model.

All 32 ideal physical records across the four logical Bell states are tested
against all 256 parity-erasure masks, with recovery independently classified by
exact quantum branch completion. Additional compiled physical trajectories use
16-photon encoded entanglement swapping, ideal and 0.8-transmission routes, and
verify every recovered parity against surviving logical-output correlations.
`PhysicalToricMemory::shor` now uses these actual resources and fusion records in
both four-star and six-ring toric memories, including the four-star logical H
frames. No effective independent logical-erasure coin replaces the physical
records. Detected inner-code inconsistencies conservatively erase both logical
parities; undetected faults remain in the outer memory's failure count.

## Rotated six-ring lattice and independent oracle

`SixRingTorus` implements the rotated cubic-cell construction in Supplementary
Note VC of [Bartolucci et al., *Fusion-based quantum computation* (2023)](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-36493-1/MediaObjects/41467_2023_36493_MOESM1_ESM.pdf).
Each cell contains a six-ring resource; ring ports in order are
(+z,+x,+y,−z,−x,−y). Each opposed neighboring face pair is fused. Each face
contributes one diagonal to each of the two checkerboard-colored vertex check
graphs, with the physical XX/ZZ record selected from the actual ring stabilizers.
Every bulk check therefore contains 12 fusion parities. Periodic side lengths
must be even to preserve the checkerboard boundary convention. This implements
the paper's rotated periodic coordinates rather than silently mixing them with
its earlier body-centered-cubic coordinates.

There are 6L³ bare photons, 3L³ physical fusions, L³/2 check vertices per color,
and 3L³ edges per color. Three cut parities per graph retain winding information.
Tests multiply every geometric check into actual resource stabilizers at L=2,4,6;
an independently eliminated resource/measurement relation basis constrains the
same sampled records. Every single parity fault at L=4 is decoded without
changing a logical cut. A separate **Stim 1.15.0 C++ oracle** independently builds
all eight L=2 ring resources and checks 192 sequential conditional Bell
measurements, including deterministic signs and random branches. The fixture,
generator and pinned archive hash live beside the earlier Stim tests; the
regeneration script rebuilds both fixtures from the pinned source archive.

`ToricFusionCode` separates this exact resource/record/check correspondence from
physical routing. `PhysicalToricMemory::bare` retains it directly;
`PhysicalToricMemory::shor` expands every resource photon into four physical code
coordinates and every fusion into four real standard analyzers. Inner recovery
uses the resource's declared orientation. The outer decoder receives only known
logical parities and erasure flags. Complete-Bell operation remains an explicit
ideal control, never substituted for the unboosted physical experiment.

The single-file [`fbqc_six_ring_memory`](../crates/burn_quantum_example/src/bin/fbqc_six_ring_memory.rs)
checks X0 Z1 Z5 of a six-ring on the selected Burn device, then performs a host
physical experiment. CPU debug observations on 2026-11-09, seed 281, 64 attempts,
pLoss=0.003 and pX=pY=pZ=0.0005, with fixed-ZZ-failure unboosted analyzers:

| Rotated side | Encoded photons/attempt | Physical fusions/attempt | Bare corrected | Shor corrected | Shor logical errors | Ambiguous | 95% all-attempt failure interval |
|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 192 | 96 | 1 | 15 | 0 | 49 | [0.6487, 0.8525] |
| 4 | 1,536 | 768 | 0 | 13 | 1 | 50 | [0.6829, 0.8773] |

No decoder budget was exceeded. The source/fusion model detected 15/104 inner
check inconsistencies across all attempts. Whole per-size debug host work took
0.36/9.82 seconds; noiseless complete-Bell controls corrected 4/4 at each size.
These observations demonstrate an actual encoded network and its limitations:
this small fixed-basis unboosted experiment does **not** show logical suppression.
Published boosted, randomized-basis or different source-noise thresholds are not
transferred to it.

## Open-boundary memories and logical Clifford reference circuits

`PlanarCode` supplies an explicit [[d²+(d−1)²,1,d]] open-boundary CSS patch for odd
distance 3 through 31, subject to a caller data-qubit budget. Vertical and
horizontal data edges give independent weight-three/four star and plaquette
checks and canonical weight-d logical X/Z strings. This uses the planar CSS
surface-code setting described by [Fowler et al. (2012)](https://arxiv.org/abs/1208.0928);
its bounded gate references and decoder are implemented and validated here,
without attributing their particular encoder to that paper.

The deterministic encoder row-reduces X stabilizers and one logical X row,
then synthesizes reversible CNOT column elimination. It accepts an arbitrary
input, including one entangled with an external reference, at `input_site()`;
other patch sites must be |0>. `encoding_plus_sites` and `encoding_cnots` expose
the executable primitives for dense-device validation and physical compilation.
There is no unreported state-projection probability.

The implemented operations are:

- **Memory:** actual one-ancilla CNOT check circuits, with explicit reset and reuse.
  Lost data sites are traced through unobserved environment records and replaced
  with |0>; loss flags, not hidden outcomes, enter decoding.
- **H:** transversal H followed by an exact 90-degree patch permutation. Reference
  SWAPs are implemented as three CNOTs and counted in the physical primitive cost.
- **CNOT:** one physical CNOT per corresponding data pair in two disjoint patches.
  The required cross-patch connectivity is explicit.
- **S:** an encoded |Y> ancillary patch, transversal CNOT, destructive physical Z
  readout, and conditional logical Z. Both branches are accepted. `readout_z`
  additionally performs erasure-aware classical correction of physical readout
  bits, preserving ambiguity and budget outcomes for noisy ancillary protocols.

`PlanarDecodeBudget` bounds exact minimum-weight enumeration separately for X
and Z error channels; erased sites have zero cost. Equal-minimum explanations
must agree on the logical action. It is not an unbounded or optimal correlated
Pauli decoder. `PlanarRecovery` and `PlanarReadout` distinguish correction/known
answers, ambiguity and exceeded search budgets. All widths, input-state
assumptions and patch overlaps are checked before gate mutation.

Tests verify unknown-input encoding at d=3,5,7; all 39 single Pauli faults at d=3;
all one-/two-site erasure masks with alternative hidden environment branches;
and exact logical Choi correlations for H, CNOT and both S branches at d=3,5.
The noisy S protocol additionally corrects every single ancillary X/Y/Z fault
using actual physical readout and subsequent data correction. Loss-aware encoded
readout checks every two-site erasure against the untouched reference. These
are nontrivial quantum channels, not only classical syndrome examples.

`run_planar_memory` repeats data-noise/check/correction rounds with stable global
shot seeds, complete attempted-input accounting, and preparation/CNOT/H/Pauli/
readout costs. `PlanarMemoryNoise::storage` converts explicitly supplied equal
rail transmission and dwell ticks into one physical loss probability; asymmetric
attenuation is rejected. Encoder, syndrome gates and detector readout are ideal
in this reference model. Hardware scheduling and noisy-check spacetime decoding
are separate capabilities, rather than implicit accuracy assumptions.

[`planar_logical_memory`](../crates/burn_quantum_example/src/bin/planar_logical_memory.rs)
checks a real 14-qubit encoded Bell state on the selected Burn device, then runs
host d=3/5/7 memories. It also lowers that same encoder through `PhysicalCircuit`
for auditable resource and timing costs. This last step is **schedule validation**;
the 14-input noisy physical density channel is not executed in that example.
CPU debug observations on 2026-11-09, seed 987, 64 attempted inputs per distance,
two rounds, per-round loss 1−0.998² and pX=pY=pZ=0.002:

| d | Data qubits | Corrected/attempted | Replaced data photons | Total CNOTs | Total preparations | Host seconds |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 13 | 64/64 | 10 | 6,848 | 2,378 | 0.01 |
| 5 | 41 | 64/64 | 23 | 28,032 | 7,767 | 0.09 |
| 7 | 85 | 64/64 | 44 | 68,160 | 16,236 | 3.90 |

Each zero-failure observation has the same 95% Wilson failure interval
[0,0.05662]; these finite results do not estimate a threshold. The costs exclude
the external Choi-test reference. Memory cost scales with actual completed
rounds, including the final failed round, rather than postselected successes.
The multi-distance test checks deterministic shard replay and full-loss aborts.

The periodic FBQC networks and these planar open-boundary circuit protocols have
different physical representations. The next section describes the separately
implemented finite FBQC boundary instruments. A planar H/CNOT/S reference is not a
compiled six-ring lattice-surgery/deformation layout, and ideal syndrome gates
do not establish circuit-level fault tolerance. Resource-aware non-Clifford
injection/distillation is documented in [the magic-state guide](photonic-magic.md)
with its own noise and cost assumptions.

## Finite physical FBQC boundaries and retained quantum outputs

`SixRingPatch` now builds a finite rectangular box of the same rotated six-ring
bulk connectivity, with **actual terminal boundary instruments**. Face order is
+x,+y,+z,−x,−y,−z. Each exterior port is either a retained quantum output or a
specified destructive X/Y/Z measurement; each internal face pair is a standard
physical fusion. These choices are explicit `PatchBoundary` values, not synthetic
syndrome edges added after sampling. `program()` supplies the physical resources,
fusions and fabric routes; `boundary_measurements()` supplies the ordered terminal
instruments; `outputs()` excludes consumed boundary photons.

The source is the rotated bulk construction described above. Its finite boundary
patterns here are independently declared and validated, rather than presented as
a reproduction of the paper's complete rough/smooth boundary layout. Arbitrary
face choices need not encode a growing-distance surface code.

`MeasurementRelations::output_relations` derives a complete independent commuting
basis of retained-output correlations, with exact signed record parities. It
solves the resource/measurement/output Pauli algebra; it does not infer logical
information from lost measurement values. A 2×2×2 example contains 48 resource
photons, 12 actual fusions, 16 terminal side measurements, eight retained photons,
one independent bulk/boundary record check, and eight output correlation
stabilizers. Both endpoint surfaces have a weight-four local correlation,
leaving three entangled logical degrees of freedom across the surfaces. This
small boundary instance has no asserted increasing quantum error distance.

`bell_strip(length)` selects X on +x/+y side ports, Z on −x/−y ports and retained
quantum outputs on both z ends. Each six-ring reduces to a length-three cluster
wire under these measurements. Odd strip lengths yield signed XZ/ZX endpoint
correlations (an H Choi resource); even lengths yield XX/ZZ (identity Choi).
Known measurement signs specify the necessary Pauli frame. Both correlations
must be known before accepting the complete heralded Bell/Clifford reference.
An unboosted fusion failure can remove one of them; it remains in the result.
This demonstrates actual finite-boundary photonic correlations, rather than only
scheduling a separate abstract logical gate circuit.

`sample` uses the **same physical trajectory function** as
`StabilizerSimulator`: balanced compiled rail attenuation, hidden environment
traces, state-dependent odd-ZZ success, resolved individual-Z failure, and actual
boundary measurements. Local Pauli faults are applied at the declared source
state location. `PhotonLossBurst` adds a single physical event that removes a
set of photons; overlapping events combine by union. A middle-resource burst
therefore correlates its two neighboring fusion-loss records and four boundary
erasures. Physically lost output factors give expectation zero and an erased
inference. `PatchShot` retains all records, presence flags, inferred parities and
actual conditional output expectations. Faulty known inferences are visible;
this algebraic erasure decoder does not claim to correct arbitrary Pauli noise.

Independent validation includes a real-amplitude 12-qubit simulator that
constructs the two six-ring resources directly and enumerates **all 1,024
physical standard-fusion/side-boundary branches**. Total probability is one;
fully recoverable Bell-correlation probability is one half; every inferred
output sign agrees with that independent conditional state. Other tests check
complete output rank/commutation on finite boxes, every known inferred sign
against actual noisy-instrument tableaus, all-output loss, 1,024 correlated-burst
attempts and exact replay of split global shot ranges. The existing six dense/
Stim/physical-loss stabilizer regressions still pass after sharing the trajectory
implementation.

The single-file
[`fbqc_open_boundaries`](../crates/burn_quantum_example/src/bin/fbqc_open_boundaries.rs)
first validates the signed one-ring boundary correlations with a native
six-qubit Burn state on the selected device, then samples actual open strips.
CPU observations on 2026-11-09, seed 884, 256 attempts per length,
pX=pY=pZ=0.002 and one whole-resource loss event of probability 0.1:

| Rings | Photons | Fusions | Boundary measurements | Correct correlations | Wrong known correlations | Erased | 95% all-attempt failure interval |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 6 | 0 | 4 | 218 | 9 | 29 | [0.1101, 0.1972] |
| 2 | 12 | 1 | 8 | 100 | 6 | 150 | [0.5484, 0.6671] |
| 3 | 18 | 2 | 12 | 52 | 7 | 197 | [0.7434, 0.8416] |

These are deliberately unencoded, fixed-analyzer boundary protocols. The loss of
useful correlations with strip length is measured rather than hidden through
postselection. Resource, fusion and boundary columns count scheduled operations
per attempted input; actual missing photons and failed measurements remain in
`PatchShot`. Combining finite boundary patterns with a demonstrated growing code
distance and a lattice-surgery gate schedule requires its own geometric proof;
the generic boundary API does not assert that proof for arbitrary faces.

The later [surgery implementation](photonic-surgery.md) adds a proved local
merge/split construction, full noisy transition graphs and physical recovery.
Its [six-ring implementation](photonic-fusion-surgery.md) separately validates
resource-level lowering and specific boundary deformations. The original generic
boxes above retain their original bounded claims and historical measurements.

In planar syndrome reference costs, a negative measured-site X reset represents
fresh |0> preparation at a reused tableau index after destructive readout. That
preparation is counted once. A hidden environment outcome after photon loss is
not a physical feed-forward control and is never charged as an observable gate.
