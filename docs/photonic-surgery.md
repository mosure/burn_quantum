# Growing-distance surgery and threshold studies

The local QEC API now includes explicit planar merge/split instruments, a
measurement-only logical CNOT, noisy transition detector graphs, scalable graph
decoders, and reproducible fixed-shot threshold studies. These components have
different declared physical models. Their results must retain those model names.

## Planar merge/split instrument

`photonic::qec::PlanarSurgery` joins two existing `PlanarCode` patches. It uses a
doubled seam: `d` weight-two boundary gauges replace paired opposing checks by
weight-six products. Other bulk checks remain local. Measuring the seam product
gives the requested logical `ZZ` or `XX` parity. The compiler repeats merged
checks for `d` rounds, restores the original patch checks for `d` rounds, and
applies the recorded split frame. It exposes every check, coordinate, phase,
measurement, random branch, and physical correction.

This implements the merge/split instrument described by
[Horsman et al.](https://arxiv.org/abs/1111.4022), using the stated doubled seam
instead of their inserted data strip. It is an independently checked realization,
not a claim of a byte-for-byte reproduction of that circuit.

`MergeBasis::ZZ` joins smooth boundaries; `MergeBasis::XX` is the dual rough
operation. The merged protected spatial distances are `(2d,d)` and `(d,2d)`,
respectively. `distance_certificate()` computes shortest nontrivial paths in
the actual check graph and returns witness supports and bounded search work.
`PlanarSurgery::cnot` composes `ZZ(control,ancilla)`, `XX(ancilla,target)`,
destructive ancillary `Z`, and physical feedforward. It does not call a
transversal logical CNOT.

The ideal interpreter checks the input code space. Its callback interface accepts
actual check instruments: a failed check returns a partial transcript and a
consumed/mutated input, never a restored retry of an unknown state. Ideal-code
execution rejects unexplained noisy records; noise-aware decoding uses the
separate transition model below.

## Noisy transition model

`SurgeryDetectorModel` models independent physical data `X` and `Z` faults before
each round and flips of the recorded check bits. Quantum check backaction is
ideal. Both the initial state and the final closing readout are ideal. There are
`2d` noisy rounds, including the merge/split transition. There is no gate-hook,
source, leakage, dark-count, or optical loss channel hidden inside this model.

The first seam signs are random, so they are **not** treated as known-zero
syndromes. Detectors use differences between repeated seam measurements. At the
split, a pair of newly random opposing checks is compared jointly with its
previous merged check. Subsequent individual differences and the final closure
complete the detector history. Split feedback is retained in the classical
frame. Each elementary fault must have at most two detector endpoints; a
hyperedge is rejected instead of silently split into independent edges.

Four logical labels test the full entangled-input instrument, including the
reported parity and output/reference correlations. A successful decoder must
recover all four. Single-fault signatures are checked against the physical
tableau transcript; tests also use fresh quantum randomness on noisy entangled
trajectories. The decoder receives only detector records and declared erasures.
The sampled true fault is used afterward to classify the result.

`SurgeryDetectorModel` also produces shortest undetected logical fault witnesses
for its complete spacetime graph. These certify the declared phenomenological
model; they do not certify an unspecified noisy ancilla circuit.

## Decoder and statistical API

`photonic::qec::threshold` owns sparse detector graphs, decoding, repeated-check
memory models, fixed-shot counts, and statistical intervals. Allocation and work
limits are explicit. A boundary endpoint is unconstrained; logical labels remain
attached to paths through it.

- Erasure decoding uses union by size, path compression, XOR potentials, and
  forest peeling. The full erased logical image is retained. Rank `r` means
  `2^r` equally likely logical classes under the declared uniform erasure channel.
  Logical ambiguity and a sampled logical error are different report fields.
- Mixed decoding implements synchronous unweighted union-find growth and forest
  peeling. It has linear storage and bounded global edge sweeps. It is not MWPM,
  and does not claim the optimized boundary-list implementation's runtime.
- Exhausted decoder budgets and inconsistent syndromes stay in the attempted-shot
  denominator. Raw failed qualification results remain available. An invalid run
  cannot produce an accepted finite-size bracket.
- Clopper–Pearson intervals use fixed sample counts and a predeclared Bonferroni
  family. A suppression/growth statement requires disjoint intervals at **every
  adjacent tested size**. Overlap is reported as unresolved. Crossings are finite
  size evidence, not a rigorous asymptotic threshold theorem.

The union-find reference is
[Delfosse and Nickerson](https://arxiv.org/abs/1709.06218); exact erasure decoding
is independently compared with small GF(2) solution spaces. Interval tests use
direct binomial tails, including large trial counts and exact small-sample
coverage checks.

## Direct six-ring surface surgery

`SixRingSurfaceNetwork` constructs actual six-ring sources, destructive XX/ZZ
fusions and color-preserving side-wall measurements. Its quantum time ends
carry surface-code patches in an explicitly declared local Hadamard frame.
Enabling cross-interface fusions for `d` layers implements the joint-parity
merge; the preceding and following closed interfaces separate the patches.
The full signed encoded channel is derived from the physical resource graph.

`CubicalRecordModel` multiplies positive three-port resource corner stabilizers
to obtain local detector checks. At a disabled seam it keeps separate
fusion-connected components. This creates a sparse model with `O(V log V)`
construction and `O(V)` storage in resource volume `V`, without a dense state
or global binary elimination.
Terminal outcome bits remain fault coordinates. Nondestructive end-code
checks close the history: these are explicit extra ideal measurements whose
**actual signs** enter the syndrome, rather than assumed positive values.

The complete local detector space and geometric logical labels agree with the
full signed resource/Choi calculation for small memories and merges. Exact
shortest-path witnesses certify memory distances `(d,d)` and four merge
distances `(d,d,d,d)` for `d=3,5,7,9`. A one-layer merge instead has minimum
distance one. Independent Stim fixtures check the physical channel, while
fresh quantum trajectories and single-record faults check the decoder input
and corrected output predictions.

The corresponding threshold study flips every physical fusion/terminal record
independently with probability `p`. It assumes ideal resources, successful
quantum measurements and perfect end closure. This is an explicit outcome-noise
reference, separate from the periodic optical-loss model below. It does not
silently omit unknown fusion failures from a physical attempt. See the
[physical construction and negative controls](photonic-fusion-surgery.md).

## Six-ring and optical threshold models

`SixRingErasureModel` uses the existing actual rotated `SixRingTorus` connectivity
and its XX/ZZ record partition. Both color graphs and all three periodic logical
cuts contribute to failure. This reproduces the **erasure-only slice** of the
hardware-agnostic model in
[Bartolucci et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9938229/), Supplement
VII. It does not reproduce their mixed Pauli/erasure curve or Blossom V decoder.
For pure erasure, the exact logical cycle image gives the uniform residual-class
distribution without exponential matching. The report retains both sampled
logical errors and their conditional mean; only integer sampled errors receive
binomial intervals.

`OpticalFusionLoss` separately declares independent loss of signal and boosting
photons, randomized failure basis, and optional [[4,1,2]] Shor encoding. A lost
photon erases both parity records. A loss-free failed fusion erases exactly one.
The joint XX/ZZ law is kept, including after enumerating all 256 four-fusion
patterns and both common code orientations. Its marginal matches Supplement
II B/C, but the implementation does not replace the joint law with independent
outcome coins. Ideal sources must supply the requested orientations. Source
generation yield, Pauli noise, detector imperfections and calibration are absent
from this particular loss-only reference.

`photon_loss_at_marginal` is an algebraic model conversion. It returns `None`
when a requested outcome erasure is below the zero-loss failure floor. It must
not be presented as a measured device threshold.

## Executable studies

Each application is a separate file in `crates/burn_quantum_example/src/bin`:

- `fbqc_lattice_surgery`: native Burn six-ring resource oracle, full conditioned
  optical merge/split channels on entangled encoded inputs, distance and cost
  certificates, and retained rejected physical attempts.
- `fbqc_threshold`: periodic six-ring erasure and correlated encoded optical-loss
  sweeps. `--qualify six-ring` and `--qualify optical` each run 28 cells with
  20,000 trials per cell on sides 12, 16, 20 and 24.
- `planar_threshold`: independent X/Z faults and readout flips each at `p`,
  `d` noisy memory rounds, and ideal endpoints. `--qualify` runs the initial
  40-cell 20,000-trial study. `--qualify-confirmation` uses a separately declared
  fresh seed and 100,000 trials per cell; the two stages are not pooled.
- `surgery_threshold`: the complete noisy merge/split transition model, including
  both parities and all four entangled-channel labels.
- `surface_threshold`: actual color-preserving six-ring memories and fusion-seam
  merges at `d=3,5,7,9`. `--qualify` runs 64 fixed cells with 50,000 shots each,
  under the independent outcome-flip model with perfect observed closure.
  `--qualify-confirmation` is a separately declared memory-only study with fresh
  seeds, 16 cells and 100,000 shots per cell; the stages are never pooled.

Default runs are bounded smoke examples. Full qualification uses `--release`
and records the exact seeds, models, intervals, budgets and failure counts.
Tableaux, graph decoders and Monte Carlo control execute on the host. Native
quantum numerical oracles explicitly use the selected Burn CPU/WGPU/CUDA device;
GPU availability never causes silent CPU fallback.

The optical gate-teleportation lowerer consumes actual six-ring resources and
standard destructive fusions. Its complete instrument and its exponentially
small unboosted whole-circuit heralding probability are documented in the
[six-ring surgery guide](photonic-fusion-surgery.md). A successful conditioned
channel alone is not evidence of an optical fault-tolerance threshold.

The [qualification record](photonic-surgery-validation.md) supplies numerical
results, source provenance, independent oracles and tested hardware. Earlier
photonic qualification records remain historical snapshots.
