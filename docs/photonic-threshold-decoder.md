# Detector graphs, decoding, and fixed-shot qualification

`photonic::qec::threshold` supplies reusable host graph decoders and statistical
inference. It does not require a Burn device. Physical quantum-channel oracles
remain separate tests; compiling with GPU features does not move this graph
work to a GPU.

## Sparse graph contract

`DecoderGraph::new` accepts detector vertices and ordered `DetectorEdge` fault
coordinates. Each edge carries two detector endpoints and an XOR mask on at
most 64 logical observables. Endpoint `graph.detectors()` is a virtual boundary
with no parity constraint. Parallel edges and self-loops are retained. Logical
labels are supplied by the independently validated model; they cannot establish
a physical model by themselves.

The decoder receives a detector syndrome and known erasure locations. It does
not receive the simulated true fault vector. `syndrome` and `logical_flips`
are also available for model construction and tests; the studies use true
logical flips only to assess recovery after the decoder has returned.

`analyze_erasures` uses union by size, path compression, and XOR potentials.
Every cycle outside the spanning forest contributes its logical label to a
binary basis. Identifying all spatial boundary endpoints with the virtual
vertex includes boundary-to-boundary relative cycles. The result is the full
logical image, including its rank, rather than just an erased-edge count.
`decode_erasures` additionally constructs a forest correction and detects
inconsistent syndromes. For independent fair erased bits, a rank-r logical
image gives conditional logical failure probability `1 - 2^-r`. The six-ring
runner samples that Bernoulli event from an independent seed component and
also reports the conditional mean and ambiguity count separately.

`decode_union_find` follows simultaneous half-edge growth of invalid clusters,
then spanning-forest peeling. It uses unweighted edges and implements the
synchronous algorithm, including a spatial boundary. Global scans make its
work `O((V + E) R alpha(V))`, where R is the number of growth rounds; storage
is `O(V + E)`. It does not claim minimum-weight matching or the optimized
boundary-list complexity of the published almost-linear implementation.

`GraphLimits` bounds vertices and edges. `DecoderBudget` bounds edge inspections
and growth rounds. `GraphStatus::WorkLimit` and `Inconsistent` never provide a
usable correction. `Corrected` means a syndrome-consistent recovery with no
intrinsic erasure ambiguity; unknown Pauli noise can still leave a logical
error. `Ambiguous` retains a consistent forest correction but explicitly marks
unrecoverable erasure information. Studies count every attempted shot, including
limits and inconsistencies, in the unconditional failure denominator.

## Models

- `SixRingErasureModel` reuses `SixRingTorus`'s actual resource/fusion incidence
  and XX/ZZ record ordering. `run` independently erases outcome coordinates;
  it does not identify outcome erasure probability with photon loss.
  `run_with_masks` accepts a caller-supplied correlated full-record mask for a
  separately specified optical law. Both color graphs and all six torus-cut
  logical coordinates enter the error event.
- `RepeatedMeasurements` constructs spacetime fault signatures from graphlike
  binary checks. Data noise occurs before each noisy round. A wrong measurement
  connects adjacent time layers. There is a known initial code syndrome and an
  ideal closing measurement with no extra data-noise interval.
- `PhenomenologicalMemory` uses the existing open `PlanarCode`, both Pauli
  channels, and these complete repeated-check histories. Independent X/Z
  probabilities p imply Y probability p² and total Pauli probability 2p-p².
  Heralded data erasure shares its location between both channels and replaces
  their error bits by independent fair bits. No circuit-level hook noise is
  inferred from this phenomenological model.
- `DecoderGraph::run_independent_faults` supports models with a separately
  validated catalog of equally likely independent binary locations. It returns
  both the any-logical-error event and overlapping marginal logical counts.
  `surgery_threshold` uses the complete `SurgeryDetectorModel` through both
  merge and split, including unknown initial seam values, paired boundary
  transition detectors, and all four output-Choi instrument correlations.

All studies use versioned `runtime::shot_seed` domains and individual shot
indices. Splitting a fixed range into disjoint shards reproduces the same
counts. The aggregate API deliberately leaves shard provenance and duplicate
range detection to the caller; use the existing job ledger when scheduling
persistent remote/local jobs.

## Statistical contract

`BinomialInterval::clopper_pearson` inverts exact binomial tails using bounded
f64 incomplete-beta inversion. The supported range is 1–10,000,000 fixed trials
and alpha in `[1e-12, 1)`. Endpoints include outward numerical padding.
`simultaneous` applies Bonferroni alpha/K across every predeclared interval,
including all examined sizes and noise values. These are fixed-sample intervals,
not optional-stopping confidence sequences. Fractional conditional error means
are not treated as binomial observations.

`compare_sizes` calls suppression or growth only when every adjacent tested
size has disjoint simultaneous intervals in the same direction. Overlap is
unresolved; conflicting directions are nonmonotone. `crossing_bracket` returns
only the corresponding finite-size suppression-to-growth bracket. It does not
fit or prove an asymptotic threshold.

The executable qualification grids, fixed stopping rules, seed domains, noise
assumptions, and interval families are recorded before sampling in
[the qualification record's `protocols` field](photonic-surgery-results.json),
including the bulk/optical/planar and independent surgical protocols.
The surgical runner always emits completed cells, marks invalid decoder
qualification explicitly, and suppresses bracket acceptance if any decoder
limit or inconsistent result occurred. Smoke tests use smaller fixed grids.

## Independent checks and sources

The tests exhaustively enumerate fault kernels on small graphs to compare the
entire logical image and rank, including boundary paths, parallel edges, and
self-loops. Forest corrections are checked against every supported small-graph
syndrome. All single faults at distance three and all double faults at distance
five satisfy the expected recovery bound. Repeated-check signatures are
compared with actual planar tableaus. Surgical tests include fresh independent
quantum measurement randomness, without forcing a predicted Pauli-frame
trajectory, and compare all four entangled-reference output correlations.

Statistical tests compare with NIST's published example, independent direct
binomial-tail inversion up to one million trials, and exhaustive small-sample
binomial coverage. The confidence code is independent of the quantum decoder.

Primary references:

- [Delfosse and Nickerson, *Almost-linear time decoding algorithm for topological codes*](https://arxiv.org/abs/1709.06218), especially Algorithms 1 and 2.
- [Delfosse and Zémor, *Linear-Time Maximum Likelihood Decoding of Surface Codes over the Quantum Erasure Channel*](https://arxiv.org/abs/1703.01517).
- [Dennis et al., *Topological quantum memory*](https://arxiv.org/abs/quant-ph/0110143), for repeated-measurement spacetime decoding.
- [Bartolucci et al., *Fusion-based quantum computation*](https://doi.org/10.1038/s41467-023-36493-1), with the published rotated six-ring convention and explicitly separate outcome-erasure and optical-loss models.
- [NIST/SEMATECH handbook, §7.2.4.1](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm), for exact binomial confidence equations.

The recorded [complete surgical qualification](photonic-surgery-results.json)
contains 1,600,000 attempts, 20,000 in each of 80 cells, with no decoder-limit or
inconsistent-syndrome outcomes. Both XX and ZZ families resolve decreasing
failure at p=0.015 and p=0.02 and increasing failure at p=0.03 and p=0.04 under
the stipulated 99% simultaneous intervals. Each finite-size bracket is
`[0.02, 0.03]`; the other grid points retain their unresolved classification.
This is the declared independent-Pauli/readout, ideal-check phenomenological
instrument model, not a circuit-level or optical threshold. The 213.53-second
recorded wall time includes ten concurrent size/basis workers and is not a
single-core decoder benchmark.

## Actual finite six-ring measurement records

`CubicalRecordModel` builds a decoder directly from
`SixRingSurfaceNetwork`'s physical six-ring sources, Bell XX/ZZ measurements,
and color-preserving side-terminal measurements. Each cube corner is a positive
resource stabilizer: an alternating triple of X operators, or the ring graph
stabilizer with X at the center and Z at its two neighbors. At a seam that is
closed, incident corners are split into components using the actual fusion
connections. Multiplying each compatible component cancels measured factors
and leaves a local detector relation. Every original fusion/terminal record is
retained as a possible fault coordinate. No extra incidence is discarded to
make the graph fit a decoder.

A corner product at a time end can leave a local physical Pauli operator on
retained photons. The model appends that operator to `observables()` as an
**explicit ideal nondestructive closing measurement**. `closure()` returns
these operators, and `syndrome()` requires their actual measured signs together
with every raw record. It does not substitute positive closing outcomes.
Closing measurements are a stated perfect boundary condition of this model;
this decoder does not silently charge them to the physical fusion program or
claim to include their hardware noise. Their physical implementation can be
modeled separately with the existing check instruments.

The constructor uses linear storage and `O(volume log volume)` ordered-map
work. Its corner-term budget counts 24 single-photon terms per six-ring before
allocation. `memory_graph()` supplies two geometric logical cuts;
`surgery_graph()` supplies four for a strictly internal active merge window.
The latter are three color-zero rough-boundary components (outer right wall,
pre-merge seam, post-merge seam) and the upper color-one wall. They are a basis
of the instrument's logical flow classes, not named individual canonical
output Paulis. Any-nonzero logical failure is invariant under a change of this
basis. An all-separated pair has a different topology and is explicitly
rejected by `surgery_graph()`. For named output observables,
`graph_from_relations()` uses bounded complete signed stabilizer elimination.

`DecoderGraph::minimum_logical_distance()` finds exact unweighted logical
support using a parity double cover. It includes boundary paths, parallel
coordinates and self-loops, returns the actual fault-coordinate witness, and
charges setup and adjacency inspections to an explicit work bound. Exhaustion
or absence of an odd witness returns an error, never a partial distance claim.

The independent tests establish the following:

- Local detector rows span the entire check space of global resource
  stabilizer elimination for distance-3 and distance-5 memory and merge cases,
  including actual closing observables. Each row's sign is checked against a
  separately prepared resource tableau.
- Geometric logical cut classes span exactly the signed physical Choi-flow
  classes modulo this complete check space at distances 3 and 5. Actual
  quantum measurements supply nonzero closing signs, and every physical raw
  single-record fault has the predicted detector signature. At distance 3,
  syndrome-only recovery restores each named Choi prediction to the actual
  quantum output for every single fault.
- For sides 3, 5, 7 and 9, single-patch memories have both raw-record distances
  equal to the side length. Merges lasting that many resource layers have all
  four distances equal to the side length. One-layer merges instead give
  `[d, 1, 1, d]`, retaining the short-time failure as a negative control. The
  growing-size checks use only local construction and sparse graph search,
  without global stabilizer elimination or a dense Hilbert-space state.

This is a model of errors or erasures in physical fusion and terminal
**measurement outcomes**, with perfect resources and perfect closing-check
measurements. An independent flip probability on those coordinates is not a
photon-loss probability or a full noisy optical instrument. The separate
fixed-shot physical-record qualification reports its assumptions explicitly.

`RecordCode` is a general bounded adapter for complete `MeasurementRelations`:
it retains signed checks, selected noisy coordinates and named logical
correlations. `try_graph()` applies only invertible check-row operations and
rejects a remaining hyperedge; `minimum_distance()` exhaustively verifies all
smaller supports within its declared weight and candidate budgets. This
adapter also retains the older constant-wall mixed-BCC crop as a negative
control: that geometry has undetectable single-fusion-record logical faults,
even after explicit perfect end-check measurements. Growing end-code distance
alone would therefore have been insufficient evidence of a useful spacetime
memory.
