# Six-ring boundary networks and gate-teleportation surgery

`photonic::qec::fusion_surgery` provides three explicit physical constructions.
`SixRingSurfaceNetwork` preserves the bulk check colors at rough/smooth side
walls and exposes a declared local-H frame for its surface-code outputs.
`SixRingBoundaryNetwork` creates a local six-ring fusion graph with derived end
codes and a time-dependent parity seam. `SixRingSurgery` lowers the bounded-weight checks of `PlanarSurgery`
into real six-ring preparation, single-photon measurements and destructive
standard Bell analyzers. Accepted instruments preserve arbitrary entangled data.
Rejected instruments stop the attempt and retain its physical records and costs.
The gate-teleportation lowering has its own conditional success and cost model;
it is separate from the actual color-preserving lattice deformation.

## Color-preserving surface boundaries

`SixRingSurfaceNetwork` places six-ring sources on an axis-aligned rotated
`L x L x T` lattice (or `2L x L x T` for two patches). At a resource corner,
the three selected ports have a positive graph-stabilizer factor: XXX for
alternating ring ports, or ZXZ for three consecutive ports. The corner color is
`(x+y+z+corner_x+corner_y+corner_z) mod 2`. Both x walls preserve color1, and
both y walls color0. The terminal Pauli on each dangling side port is its factor
in the preserved-color corner check. Consequently the physical X/Z basis
alternates with cell parity and face orientation. Constant X/Z walls do not
implement this construction.

Time-end ports remain quantum outputs. Their code is XZZX in physical axes;
a declared local Hadamard on cells with odd `x+y` expresses it as CSS.
`hadamard_frame` is a coordinate convention, not an inserted physical gate.
Every returned check, logical representative and minimum-weight witness is
mapped back to the physical photon Pauli axes. `logical_phase_negative` includes
both Pauli multiplication phases and `H Y H = -Y`. The signed relation algebra
always refers to the actual physical measurements.

For the qualified memory cases, `L=3,T=3` has nine photons per end and exact
X/Z distances 3/3; `L=5,T=5` has 25 and distances 5/5; `L=7,T=7`
has 49 and distances 7/7. The two logical Choi
stabilizers certify the transmitted encoded state, including local frames.
Full merged-lattice certificates also pass at L5 and L7 with one pre layer,
L active layers and one post layer. They derive all four end codes at exact
distance L and the complete signed four-qubit logical Choi stabilizer group,
including exactly one joint parity on the two input blocks and one on the two
output blocks. The large proof uses explicit 128M relation-entry/candidate budgets.
A two-patch interface is closed by the same color-preserving single-photon
measurements. During its active source-layer window `[start,end)` these ports
instead undergo actual XX/ZZ fusions. `parity_merge` requires nonempty pre/post
slabs and at least L active layers; `with_window` also represents shorter
geometric windows for negative controls. A repeated-layer count alone does not
certify its noisy record distance.

`threshold::CubicalRecordModel` supplies the separate physical-record proof.
It constructs local detectors directly from positive resource-corner products,
splits them into actual fusion-connected components at disabled seam bonds,
and returns the extra end-code observables used to close the syndrome. Those
closing measurements are explicitly perfect in this model; their recorded
signs are retained and tested on actual tableau measurement branches. The
builder does not silently execute them. The complete local detector row space
agrees with full resource elimination for small memory/seam instances.

Under that declared record model, exact graph distances grow as 3,5,7,9 for the
corresponding memories and full-duration seams. All four signed seam logical
relation bits have distance L; a one-layer seam exposes distance 1 for two of
them. Complete signed check-space and physical Choi-label equivalence are
independently verified at L3 and L5; the scalable local construction supplies
the larger graph certificates. See [the decoder guide](photonic-threshold-decoder.md)
for the exact fault coordinates, closure assumptions and sampled studies. These are measurement-coordinate fault distances with ideal closing
readouts, not a threshold against every source, detector and optical circuit
fault. Physical standard-fusion sampling separately retains failed analyzers,
route losses and partial/erased output inferences. The optional unencoded
six-ring check instrument below can realize a closing Pauli check conditionally,
with its own source costs and failure probability; it is not included free in
the ideal closure model.

| Construction | Six-ring sources | Source photons | Fusions | Terminal side measurements |
| --- | ---: | ---: | ---: | ---: |
| L3 memory, T3 | 27 | 162 | 54 | 36 |
| L5 memory, T5 | 125 | 750 | 300 | 100 |
| L7 memory, T7 | 343 | 2,058 | 882 | 196 |
| L3 merge, T5, window(1,4) | 90 | 540 | 201 | 102 |
| L5 merge, T7, window(1,6) | 350 | 2,100 | 885 | 230 |
| L7 merge, T9, window(1,8) | 882 | 5,292 | 2,345 | 406 |

All sources, fusion ports and terminal/output identities are checked for complete
ownership. The new surface builder accepts an explicit photon budget capped at
16,384; existing rectangular and experimental BCC constructors keep their 4,096
caps. The certificate separately limits binary-relation work and distance
candidate enumeration. Larger allocations do not silently change those budgets.

`surface-stim-1.15.0.json` independently verifies five constructions: L3 and L5
memories, two separated L3 patches, the three-layer L3 seam and a one-layer seam.
It pins the Stim 1.15.0 source archive and circuit SHA256 hashes, supplies 32 actual
trajectories per case, and includes the full physical output stabilizer basis,
logical generators and every nontrivial logical stabilizer product. These last
products exercise Y-bearing operators and negative lift/physical signs. Rust
checks geometry, record order, complete CSS-frame check rank, physical logical
representatives, all signed output values and lift phases in all 160 trajectories.
The generator imports no Burn code.

```sh
python3 crates/burn_quantum/tests/stabilizer/generate_surface_fixture.py \
  --stim /path/to/stim-1.15.0 \
  --source-archive /path/to/stim-1.15.0.tar.gz \
  --output crates/burn_quantum/tests/stabilizer/surface-stim-1.15.0.json
cargo test --locked -p burn_quantum --test six_ring_surface
```

## Experimental mixed-basis boundary and seam network

`SixRingBoundaryNetwork::{memory,separated,parity_merge}` place actual six-ring
resource states on same-parity integer coordinates `(u,v,w)`, representing twice
the unrotated BCC coordinates. The three rotated-neighbor directions are
`(1,1,-1)`, `(1,-1,1)` and `(-1,1,1)`, using ring ports 1,2,0 respectively;
the opposite ports are 4,5,3. The finite crop is `0 <= u,v <= width` and
`0 <= w <= depth`, with odd width. Both u walls terminate in single-photon X,
both v walls in Z, and the remaining time-end ports are retained. Side-wall
measurements have priority at edges and corners. Every photon belongs to exactly
one fusion, terminal measurement or output, validated before construction.

This mixed X/Z staircase boundary is independently derived and validated here.
It uses the paper's bulk six-ring connectivity, but is not an assertion that the
paper's all-Z boundary drawing implements this exact coordinate crop.

Two adjacent blocks use `0 <= u <= 2*width+1`. A closed interface measures each
crossing port in X. During the active interval it instead performs actual XX/ZZ
fusion of the neighboring ports. Activity uses fusion-midpoint time; off-window
interface X takes priority even at a time-end corner. The tested merge windows
start at an internal odd coordinate and end at an internal even coordinate.
The API rejects other parity choices: changing `(1,4)` to `(2,4)` at width 3,
depth 10 destroys both transmitted logical qubits in the independent reference.

`certify` derives the full signed relation group from the prepared graph and
physical measurement observables. It reports per-end CSS checks, anticommuting
logical representatives, exact finite X/Z minimum distances with witnesses and
candidate counts, and complete logical Choi stabilizers. Encoded coordinates
are explicit in `logical_observables`; physical Pauli targets and outcome masks
are in `logical_relations`. `logical_phase_negative` retains the encoded-Y
lift sign; XOR it into the corresponding physical relation sign for an encoded
Pauli expectation. A depleted search budget returns an error rather
than an unsupported distance claim. Bounds are 4,096 source photons, 128 photons
per end block, and explicit GF(2) and distance-enumeration work limits.

| Memory width, depth 3 | Photons per end | Minimum X/Z distances at one end | Transmitted logical qubits |
| --- | ---: | --- | ---: |
| 3 | 8 | 2,2 | 1 |
| 5 | 21 | 3,4 | 1 |
| 7 | 40 | 4,6 | 1 |

The opposite time end exchanges X/Z axes for these odd-depth memory cases.
Even time extents cut a different output boundary: at width 3, depth 10 the input
blocks have eight photons and distances 2/2, while the output blocks have five
photons and distances 1/2. This asymmetry is exposed, not assigned the input
distance. At width 3, depth 10, separated patches
transmit two logical qubits. The `(1,4)` seam transmits one and has the independent
encoded Choi generators `XXII`, `IIZZ`, `XIZI`, `ZZXX`, ordered as inputA,inputB,
outputA,outputB. Their signs are computed from actual measurement records.
Thus it measures input XX and applies a Hadamard to each output; it is not an
identity-frame ZZ parity channel. Applying the reported frame convention is
necessary when composing it with another computation.

Odd time extents preserve the larger output boundary. The independently sampled
width 3/depth 11/window(1,4) and width 5/depth 13/window(1,6) cases have generators
`XXII`, `IIXX`, `XIXI`, `ZZZZ`: an XX parity channel with unchanged logical axes.
Both time conventions remain tested; the builder does not hide their difference.

Growing end-code distance does **not** establish growing spacetime fault
distance. In particular, the raw recorded logical frame has weight-one
undetected fusion-record errors even in a model with perfect additional closing end-code stabilizer readouts.
These readouts do not repair the observed measurement-record bottleneck.
This boundary certificate and a periodic bulk erasure threshold do not qualify
a fault-tolerant open surgery threshold. The physical graph can nevertheless be
compiled through `patch().program()` and sampled through `patch().sample()`:
standard analyzer failures, missing parities, loss bursts, faulty resources and
output-presence flags are retained. The certificate concerns ideal complete
measurement relations; it never replaces unresolved physical outcomes with
successful fusions.

The independent fixture `tests/stabilizer/boundary-stim-1.15.0.json` uses integer
symplectic algebra in `generate_boundary_fixture.py` plus actual Stim 1.15.0
sampling. It contains eight memory/seam/timing-control cases, 32 trajectories per
case, exact geometry, complete check masks, signed output relations and SHA256
hashes of each Stim circuit. Rust matches the source coordinates, every fusion
and terminal measurement, output partitions, and every signed output in all 256
trajectories. It independently verifies that the rejected timing has no
transmitted Bell information. Regeneration uses a locally built pinned Stim CLI:

```sh
python3 crates/burn_quantum/tests/stabilizer/generate_boundary_fixture.py \
  --stim /path/to/stim-1.15.0 \
  --output crates/burn_quantum/tests/stabilizer/boundary-stim-1.15.0.json
cargo test --locked -p burn_quantum --test six_ring_boundary
```

## Optical derivation

Start with graph edges `(0,1),...,(5,0)`. Z-measure ports 4 and 5, retaining both
bits, and correct `Z3^m4 Z0^m5`. The surviving graph is the four-vertex path
0–1–2–3. Apply H on 0 and 2. Its stabilizers become

```text
Z0 Z1, X0 X1 X2, Z1 Z2 Z3, X2 X3.
```

These are precisely the CNOT Choi stabilizers with input-control 0, input-target
3, output-control 1 and output-target 2. Standard Bell fusion of the two data
inputs with ports 0 and 3 consumes the old photons. An odd ZZ result in each
analyzer is required; each successful analyzer also records XX. On success,
correct output 1 by X and by `Z^(xx0 xor xx1)`, and output 2 by `Z^xx1`.
The output data occupy fresh resource photons, not the destructively measured
inputs. Storage-index permutations in the tableau preserve logical indices;
they are not uncounted physical SWAP gates.

The general gate-teleportation principle is from
[Gottesman and Chuang](https://arxiv.org/abs/quant-ph/9908010). The six-ring and
fusion conventions follow [Bartolucci et al.](https://www.nature.com/articles/s41467-023-36493-1).
The path extraction and local Clifford/frame identities above are derived and
checked here; they are not an assertion that this is the paper's bulk boundary.

A weight-w Pauli check prepares one syndrome photon in |0>, rotates its data
factors into Z, executes w of these teleported CNOTs into the syndrome, reads its
Z result and restores the data bases. Supports X, Y and Z, with `1 <= w <= 6`.
The resource's internal CZs specify the supplied entangled state. They are not
claimed deterministic online photonic gates; all six emitted photons and every
source invocation are counted.

## Physical and conditioned execution

`SixRingSampling::Physical` samples the complete process. Missing sources, lost
trimming photons, fusion loss, resolved even-ZZ failure and missing final
syndrome detection remain distinct. There is no input rollback or automatic
retry after a destructive failure. `OpticalSurgeryRun` also retains a controller
error when noisy records cannot be interpreted by the ideal-code controller.

The model admits heralded source availability, balanced measured-photon and
signal survival, and independent Pauli defects on emitted resource photons.
An accepted optical declaration can still have a wrong logical channel; source
faults are not silently converted into heralded loss. Storage of resource outputs
is ideal. Dark counts, asymmetric rail filtering and unheralded output loss are
outside this particular tableau instrument.

For source availability a, measured-photon survival m and signal survival s,
the complete weight-w check herald probability is

```text
[a m^4 s^2 / 4]^w m.
```

The four measured resource photons per gate are two trims and two Bell inputs.
Each gate also consumes two signal photons, including the current syndrome
carrier. The final factor m is the terminal syndrome readout. All failed
prefixes retain their actual consumed resources rather than the full scheduled
cost.

`ConditionedReference` is explicitly separate and requires ideal source/loss/noise
parameters. It forces odd-ZZ analyzer outcomes while retaining random XX,
trimming and check outcomes. A check carries log importance weight `-w ln(4)`.
This verifies a conditional channel; it is not a physical successful attempt or
an efficient repeat-until-success scheme.

## Growing-distance evidence and limitation

The single-file `fbqc_lattice_surgery` example executes full conditioned optical
checks at planar distances 3, 5 and 7, starting from supplied encoded Bell inputs.
The complete original code checks and four logical Choi witnesses certify the
resulting logical parity channel. All witnesses pass. The merged ZZ code has
protected X/Z distances `(6,3)`, `(10,5)` and `(14,7)` respectively, with d repeated
merge and split rounds. These spatial certificates concern the logical geometry,
not arbitrary faults in this optical realization.

| d | Six-ring resources | Emitted resource photons | Bell analyzers | log ideal herald probability |
| --- | ---: | ---: | ---: | ---: |
| 3 | 498 | 2,988 | 996 | -690.374592 |
| 5 | 2,930 | 17,580 | 5,860 | -4,061.842478 |
| 7 | 8,834 | 53,004 | 17,668 | -12,246.524386 |

This is negative physical scaling evidence: unencoded standard fusions make
completion exponentially unlikely. The example separately executes 128
unconditioned d=3 attempts and retains every abort and consumed resource. It
makes no six-ring threshold, fault-tolerant surgery or practical yield claim.

The native device oracle prepares the full six-ring state, projects all four
trimming branches, and compares every resulting CNOT Choi state with an
independent analytic vector at tolerance `3e-6`. Other tests verify X/Y/Z
projectors on complete entangled references through weight six, loss/yield
against independent probabilities, deterministic source-fault false heralds,
resource preflights and full d=3/5/7 conditioned surgery. Tableau work is host
execution; the resource oracle runs on the explicitly requested Burn device.

```sh
cargo test --locked -p burn_quantum --test fusion_surgery
cargo run --locked -p burn_quantum_example --bin fbqc_lattice_surgery
```
