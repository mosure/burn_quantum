# Exact Fock optics and native photonic learning

The unreleased `burn_quantum::photonic::optics` module implements a bounded
bosonic reference engine. It is separate from the three-level encoded-photon
engine: optical modes have integer occupations, including multiphoton bunching.
Tensor execution and phase derivatives use ordinary Burn devices and the shared
precision-preserving contractions. Host F64 prepares constant optical matrices,
occupation maps and detector responses; it does not replace the selected tensor
device with a CPU simulator.

Wide GPU contractions preserve full-dtype products/reductions while bounding
fusion-planner work: groups of 16 columns are dispatched through Burn's public
`Device::flush`, and a hierarchical concatenation limits each node to 16 inputs.
Dispatching does not read tensor values, move devices or detach autodiff. Each
column product keeps the existing allocation bound, and each concatenation
intermediate fits the final output shape. The Flex CPU GEMM path is unchanged.
The `precision_fusion` regression checks dimension-330 batched complex outputs,
input gradients and every trainable operator-gradient entry against scalar
oracles. Qualification records include measured execution times; dense Fock
reference simulation still has substantial compilation and backward overhead.

## Representation and limits

`FockBasis::new(modes, PhotonSector::Fixed(n), max_states)` enumerates exactly n
photons over m modes, with dimension `binomial(m+n-1,n)`.
`PhotonSector::UpTo(n)` retains every total from vacuum through n, with dimension
`binomial(m+n,n)`. Occupations are ordered first by total, then lexicographically.
`ModeId` identifies one coherent optical mode within a basis. The vacuum-only
sector has dimension one and remains a valid state.

Combinatorics and occupation storage are checked before enumeration. This
reference permits 1–256 modes and 0–64 photons, subject to the much tighter
caller-provided basis and dense tensor budgets. Dense passive lifting also checks
matrix size and a conservative 2^24 scalar-work cap before expansion. These are
per-operation guards, not a total process/device/autodiff-memory limit.

Passive optical transformations conserve total photon number. These complete
sectors avoid the artificial boundary leakage of independently truncated
per-mode occupations. The engine does not add photons, model active squeezing,
or silently normalize a truncated source distribution.

## Passive components and state evolution

`Interferometer` validates a unitary mode matrix indexed `[output,input]`.
Creation operators transform as `a_input† -> sum_output U[output,input] a_output†`.
The Fock lift expands normalized creation operators and includes the square-root
occupation/factorial factors required for bosons. It is tested against an
independent permanent formula and pinned Perceval amplitudes.

Components include:

- `beam_splitter(modes, a, b, theta)`: the ordered Ry block
  `[[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]`;
  theta=pi/2 is balanced.
- `phase(modes, mode, phi)`: multiplication by `exp(i phi)` on that mode.
- `permutation(outputs)`: a checked bijection from input to output modes.
- `adjoint()`: the inverse passive transformation.

These conventions match [Perceval's component definitions](https://perceval.quandela.net/docs/v1.3/reference/components/unitary_components.html).

`FockState` holds batched split-complex amplitudes. `number` prepares a Fock
occupation; `from_amplitudes` imports native tensors with metadata checks.
`FockDensity` holds batched density matrices and supports explicit mixed-state
imports. Their `validate` methods perform synchronous physical-state checks.
The import methods themselves preserve graphs and do not synchronize values.

Both state types support `phase_shift(mode, angle_tensor, limits)` with one
shared angle or one angle per batch. It applies `exp(i n phi)` to amplitudes,
or `exp(i (n_row-n_column) phi)` to a density matrix. Inputs, phase parameters,
loss propagation and detector probabilities remain in the native autodiff graph.
Angles must be finite. Constant `Interferometer` matrices are host parameters;
only the explicit tensor phase API currently trains an optical component.

## Loss and detection

`FockDensity::lose(mode, eta, limits)` implements independent pure attenuation
with intensity transmission eta. Its Kraus operator for losing l photons maps
occupation n to n-l with amplitude
`sqrt(binomial(n,l) eta^(n-l) (1-eta)^l)`. The density result sums all loss records,
including vacuum, and retains surviving coherences. Actual loss on a fixed
positive-photon sector is rejected; use `UpTo` to include the environment's
possible outputs. Consecutive attenuators compose by multiplying transmissions.

`Detector::new(efficiency, resolution, dark_probability)` specifies independent
binomial detection followed by an independent Bernoulli extra dark count.
Counts saturate at the supplied resolution; resolution one is a threshold
(on/off) detector. This is explicitly an at-most-one-dark-count-per-window model,
not a Poisson dark-count process. Stateful dead time is provided separately below.

`detect` returns a `DetectionDistribution` with per-mode count records and
native `[batch,record]` probabilities. Zero-click records remain present. It
checks the joint outcome count and response-matrix storage before allocation.
No rejection sampling or implicit herald conditioning occurs. Loss in the
physical circuit and inefficiency in the detector are separate effects; do not
include the same attenuation in both calibrations.

## Internal modes, sources and detector dynamics

`WavepacketLayout` factors each fine mode into a spatial port and an orthogonal
internal label (for example, a chosen spectral/time-bin basis). `PhotonPacket`
contains a normalized complex ket over those labels. `FockState::wavepackets`
prepares their bosonically symmetrized product; it does not infer the probability
of producing that normalized resource. `layout.lift(interferometer)` applies
`U_spatial tensor I_internal`. `detect_spatial` traces unresolved internal labels
before applying the detector response. This is an exact finite internal-mode
model, not a phenomenological interpolation between classical and bosonic
probabilities. For two photons the HOM coincidence law is
`P(1,1) = (1 - |<wavepacket_a|wavepacket_b>|²)/2`. Tests cover overlaps 0, 0.3,
0.8 and 1, including a pinned independent SLOS comparison.

`CalibratedSource` stores a complete finite `P(n)` and a caller-supplied calibration
identity. `vacuum_single_pair(mean,g2,identity)` explicitly assumes support
`n=0,1,2`, with `p2=g2*mean²/2`, `p1=mean-2*p2` and `p0=1-p1-p2`; infeasible
moments are rejected. `FockDensity::source_product` prepares independent mode
sources as a diagonal mixture. Structural support checks reject even very small
nonzero tails outside the requested basis. A source identity is provenance
metadata, not a hardware-verified calibration or a resource-factory yield model.

`DetectorBank` adds causal, nonparalyzable dead time to calibrated detectors.
`observe(tick, incident_counts, seed, shot, window)` requires increasing ticks,
records blinded windows and zero counts, and uses stable per-window/detector
sampling identities. A click at tick t blinds that detector on `[t,t+dead_ticks)`;
blind events do not extend the interval. Invalid chronology, overflow and budget
errors leave the bank unchanged. Checked JSON checkpoints preserve response,
availability and event counters. This is a classical event-response layer driven
by incident photon counts; a multi-time quantum field is not inferred from those
counts. It does not automatically attach detector timing to every Fock circuit.

## Physically derived Bell instruments

`OpticalBellInstrument::standard` compiles the two balanced rail splitters into
Fock amplitudes, including explicit internal wavepackets and four independently
calibrated detector responses. For each coarse record it constructs the positive
effect `E_record = V† U† D_record U V` on two encoded dual-rail sites (dimension
nine, including vacuum). `branches` applies those effects on the selected native
Burn device and traces both measured inputs. The retained remote state has its
absolute branch weight; conditional normalization is explicit. This construction
is equivalent to tracing the full destructive optical measurement, including
unobserved internal labels. It matches all ideal `StandardFusion` probabilities
and remote coherences, and tests false heralds induced by vacuum dark counts.
A `Resolved` label reports the classical declaration; noise can make it wrong.

`OpticalBellInstrument::grice` implements the first Grice boost using the exact
four-spatial-mode matrix in [Wein et al., Eq. (3)](https://arxiv.org/html/1509.00088v2),
independently on both rails, with an explicitly supplied ideal `Phi+` ancillary
pair. The eight-mode/four-photon sector contains 330 states. Output-pattern
supports determine the Bell labels from amplitudes; success is not an inserted
Bernoulli event. The ideal resolved probabilities in the order
`[Psi+,Psi-,Phi+,Phi-]` are `[1,1,1/2,1/2]`, giving 3/4 for a uniform Bell prior.
Unresolved four-photon patterns retain the `AmbiguousEven` record.

`GriceConfig` accepts balanced rail transmissions for each of the two signal and
two ancillary photons, plus eight output detector efficiencies. The model assumes
perfect indistinguishability, number resolution through four photons, zero dark
counts and acceptance only after detecting all four photons. Under these explicit
assumptions, signal/ancilla survival is the product of all four transmissions and
an output occupation n has detection weight `product_j efficiency_j^n_j`.
Every other attempt contributes to the complementary `Rejected` effect. Thus
extra ancilla losses cannot be hidden by normalization. Tests check per-Bell
labels, complete probabilities and ancillary survival costs. Preparing the ideal
ancillary Bell pair itself is not a free deterministic factory claim: its upstream
preparation probability must be accounted for by the caller/factory layer.

## Finite shots and mixed-state geometry

`DetectionDistribution::sample_counts` draws complete categorical records with
stable per-batch seeds. `sample_readout` reports the sample mean, unbiased sample
variance, estimated standard error and attempted shot count for a finite
classical readout. No-click outcomes are ordinary outcomes; sampling never retries
until a photon is observed. Sampling synchronizes probabilities and runs on the
host; it is not a remote hardware execution provider.

`SinglePhotonPhaseExperiment` fixes preparation, one phase occurrence and final
optics/readout. It rejects any basis permitting two or more photons, because its
phase generator must have spectrum contained in `{0,1}`. Under this restriction,
`gradient` uses the exact `+/- pi/2`, coefficient-1/2 parameter-shift rule with
independent streams, propagates estimator standard errors and charges both
physical probe budgets. The rule is an instance of the two-eigenvalue result of
[Schuld et al.](https://arxiv.org/abs/1811.11184); it must not be applied unchanged
to multiphoton generators, repeated parameter occurrences or tied phases.

`classical_fisher` uses three independent histograms (central, plus and minus)
to estimate `sum_k derivative(p_k)^2 / max(p_k, probability_floor)`. It reports
attempts and the number of regularized outcomes. This nonlinear plug-in estimate
is biased at finite shots, especially for rare outcomes; it is not an unbiased
quantum Fisher estimator. The lossy single-photon Mach-Zehnder reference has
classical Fisher information eta away from singular endpoints, checked by an
independent analytic test.

`FockDensity::hilbert_schmidt_kernel` computes the native differentiable PSD
feature kernel `Tr(rho† sigma)` across two batches. It is a density feature inner
product, not Uhlmann fidelity. `uhlmann_fidelity` instead computes **squared**
Uhlmann fidelity `F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))²` for paired states,
using synchronous host-F64 spectral decomposition bounded to at most dimension
256. This exact reference validates physical density matrices and has no autodiff.
The separately costed `OpticalTomography::overlap` protocol below obtains its
input estimates from physical phase/splitter/PNR measurements. `bures_metric_from_fidelities`
uses symmetric probes to estimate
`g = [2 - sqrt(F_plus) - sqrt(F_minus)] / delta²` (stable rationalized arithmetic).
Its small-step limit is Bures metric = quantum Fisher information / 4; finite
step and numerical fidelity errors remain. For a rotating qubit of Bloch radius
r, independent tests check `F=1-r² sin²(delta/2)` and `g=r²/4`, as well as the
commuting-state fidelity formula. The pure-state limit is Fubini-Study geometry.

## Examples

```sh
cargo run --locked -p burn_quantum_example --bin photonic_hom
cargo run --locked -p burn_quantum_example --bin photonic_mz_learning
cargo run --locked -p burn_quantum_example --bin photonic_mz_shots
cargo run --locked -p burn_quantum_example --bin photonic_kernel
cargo run --locked -p burn_quantum_example --bin photonic_born_qnspsa
cargo test --locked -p burn_quantum --test optics --test optical_noise --test optical_calibration --test optical_learning
```

[`photonic_hom`](../crates/burn_quantum_example/src/bin/photonic_hom.rs) sends
`|1,1>` through a balanced splitter. It checks zero coincidences and equal
`|2,0>`/`|0,2>` bunching. Physical transmission 0.8 and detector efficiency 0.9
then yield no-click probability `(1-0.8*0.9)^2 = 0.0784` and two-detection
probability `(0.8*0.9)^2 = 0.5184`. Every attempt remains represented.

[`photonic_mz_learning`](../crates/burn_quantum_example/src/bin/photonic_mz_learning.rs)
learns a Mach–Zehnder phase offset through a native Burn `Module`, `Param` and Adam
optimizer. Three seeds initialize the parameter; each receives 120 updates on
12 training phases and evaluation on 17 disjoint phases. The synthetic targets
are independently generated from `0.8 cos²((x+0.65)/2)`. The model computes those
count probabilities through Fock preparation, two splitters, a tensor phase and
inefficient detectors, with no postselection. It compares held-out MSE against
an uncalibrated phase baseline and checks the learned model against the analytic
optical law. This is exact-gradient calibration on a small synthetic experiment,
not finite-shot training, a broad QML benchmark or evidence of quantum advantage.

[`photonic_mz_shots`](../crates/burn_quantum_example/src/bin/photonic_mz_shots.rs)
uses independent finite-shot residual and parameter-shift experiments, making
the product an unbiased gradient estimate for its MSE objective. Three seeds,
eight training phases, 17 disjoint validation phases and 80 updates at 512 shots
per probe charge **2,949,120** physical attempts in total, including all losses.
Held-out model quality is evaluated exactly against the analytic count law;
this evaluation is distinct from the noisy training protocol.

Cross-device replay can select a different click when an identical random draw
falls between probabilities separated by ordinary F32 rounding. A paired native
Flex/WGPU probe reproduced the reported final offset errors exactly:
`0.008868493077869744` and `0.009037666752629625`. Their first divergence was
seed 7, zero-based step 37, minus-shift probe, training row 4, shot 409. The uniform
draw `0.29361625907661315` lies between normalized categorical thresholds
`0.293616280783921` (CPU) and `0.29361622289125305` (WGPU), changing one
`[0,1]` click into `[1,0]`. Parameters were identical before this event. Across
the full probe, native probabilities matched the independent analytic law within
`1.90e-7` on CPU and `3.35e-7` on WGPU; the diagnostic sample lists exactly
reproduced the production sampler. The altered click changes subsequent updates;
it is not evidence of inaccurate optical evolution.

The comparison script therefore checks only this example's `worst_offset_error`
against its existing independent acceptance condition, `0 <= error < 0.05`, on
each backend. It retains both values, their difference, the bound and rationale
in `finite_shot_fits`, separate from timing diagnostics. Exact numerical outputs,
held-out MSE, reference values and physical costs retain their existing comparison
rules. Probe source, locked dependencies, all 240 paired optimizer steps and
SHA256 provenance are retained locally in
`artifacts/photonic-completion/mz-shot-divergence/`; the
[completion record](photonic-completion-results.json) embeds that evidence.
Run the comparison's mutation checks with
`python3 -m unittest discover -s scripts -p 'test_compare_backends.py'`.

[`photonic_kernel`](../crates/burn_quantum_example/src/bin/photonic_kernel.rs)
classifies disjoint circular inputs with a mixed-state Hilbert-Schmidt kernel
after physical transmission 0.6. A nearest-class-mean decision is compared with
a majority baseline and an exactly matched classical feature kernel
`(1-eta)² + eta² cos²((x-y)/2)`. The vacuum contribution and mixed-state purity
remain visible. Agreement with the classical control is expected; no quantum
advantage is asserted.

[`photonic_born_qnspsa`](../crates/burn_quantum_example/src/bin/photonic_born_qnspsa.rs)
fits the complete lossy detector distribution from 8,192 sampled training records
and checks 16,384 independently sampled held-out records, including no-click
records. Three starts use the existing QN-SPSA evaluator interface with exact
native probability losses and host squared-Uhlmann fidelity. For photon Bloch
radius r=0.6 and transmission eta=0.8, the learned metric is checked against
`eta*r²/4 = 0.072`. Evaluator calls and data shots are reported separately. This
is a simulator-based Bures natural-gradient demonstration with sampled training
data, **not** a finite-shot hardware fidelity/metric protocol.

All examples select the same explicit CPU/WGPU/CUDA device mechanism as the
existing suite. Native probabilities, loss channels, kernels and differentiable
phases run on that selected device. Fock lifting, random draws and the explicit
Uhlmann spectral reference are host work. GPU kernel fusion remains independent
of photonic fusion.

## Independent qualification

The `optics` integration target covers:

- combinatorial preflights and the one-dimensional vacuum sector;
- all matrix elements of a three-mode, three-photon Fourier lift against a
  separately implemented permanent formula;
- native-device execution of HOM, inverse transforms and mode permutations;
- binomial loss, asymmetric attenuation of complex coherences and composition;
- analytic native phase derivatives through a density matrix, loss and detection;
- detector saturation, dark counts, complete no-click output and allocation bounds;
- pinned Perceval SLOS amplitudes and unconditional asymmetric-loss probabilities.

The independent runtime fixture was generated with **Perceval 1.3.0**, **Exqalibur
1.4.2**, and **NumPy 2.5.4** in an isolated Python environment. It contains all 100
complex output/input amplitudes for the three-mode, three-photon Fourier gate,
and the full lower-sector distribution for HOM with transmissions 0.8 and 0.4.
The fixture generator uses `SimulatorFactory` without a photon-count filter;
its loss probabilities sum to one. See
[Perceval's nonunitary simulation model](https://perceval.quandela.net/docs/v1.3/notebooks/Simulation_non-unitary_components.html).

```sh
python3 -m venv /tmp/burn_quantum-perceval
/tmp/burn_quantum-perceval/bin/python -m pip install \
    perceval-quandela==1.3.0 exqalibur==1.4.2 numpy==2.5.4
MPLCONFIGDIR=/tmp/burn_quantum-matplotlib /tmp/burn_quantum-perceval/bin/python \
    crates/burn_quantum/tests/optics_perceval_v1_3.py
MPLCONFIGDIR=/tmp/burn_quantum-matplotlib /tmp/burn_quantum-perceval/bin/python \
    crates/burn_quantum/tests/optical_noise_perceval_v1_3.py
cargo test --locked -p burn_quantum --test optics --test optical_noise --test optical_calibration --test optical_learning
```

The [generator](../crates/burn_quantum/tests/optics_perceval_v1_3.py) and
[fixture](../crates/burn_quantum/tests/optics_perceval_v1_3.json) are included in
the core package. Ordinary Rust tests require no Python dependency or network.
Rust guards require all three package pins, SLOS identity, complete occupation
sectors, all four Bell/HOM cases, and the declared basis conventions. The
ancilla-only loss/dark-count test requires all 36 two-photon outcomes.
Native F32 amplitude/probability checks use absolute tolerance 2e-6, and the
phase gradient check uses 3e-5. Host permanent comparisons use 1e-12.

The second [pinned generator](../crates/burn_quantum/tests/optical_noise_perceval_v1_3.py)
and [fixture](../crates/burn_quantum/tests/optical_noise_perceval_v1_3.json) extend
SLOS qualification to all **1,320** Grice output probabilities (four Bell inputs
by 330 occupations) and the complete coarse partial-HOM distribution at four
wavepacket overlaps. The independent Python runtime produced per-Bell resolved
probabilities `[1,1,0.5,0.5]`; Rust compares the full distributions after native
Fock execution, not only those summary success rates.

The `optical_noise` target also checks source moment feasibility and sector
completeness, restartable detector dynamics, ideal instrument remote coherence,
noisy vacuum false heralds, Bell declarations and ancillary loss. The
`optical_learning` target checks shot budgets, deterministic replay, analytic
parameter-shift derivatives within estimated statistical error, finite-shot
classical Fisher, mixed-state kernels and squared-Uhlmann/Bures reference math.
These tests select the runtime device through the same environment as the rest
of the workspace. See the repository qualification record for actually executed
CPU/GPU combinations; selecting a feature alone does not establish device parity.

## Declared physical model boundaries

The exact engine remains bounded and dense. Internal-mode dimension must be
chosen explicitly; it does not solve a continuous spectral field or automatically
fit wavepackets from measured time traces. Joint finite emission and correlated phase/loss are explicit. Continuous source-pump field dynamics are outside the declared finite model. Detector dead time is nonparalyzable, with at most one
Bernoulli dark event per live window; afterpulsing, time jitter and correlated
noise are not inferred.

The first Grice boost has both a simple balanced-loss calibration and the complete
bounded `grice_calibrated` model described below. It assumes a supplied ancillary
Bell pair; its generation cost remains a separate factory responsibility.
`PhysicalCircuit` can explicitly select calibrated instrument effects and noisy
emitted-resource ensembles, preserving false heralds and complete outcomes.
Hardware providers and continuous quantum fields are outside this local finite
simulation scope. A higher recursive boost is a different resource recipe, not
an implicit success-probability adjustment.

## Joint calibration and complete noisy Grice effects

`JointPhotonSource` prepares correlated finite occupation mixtures, including
paired emission that cannot be reproduced by independent source marginals.
`CalibratedNoiseEnsemble` propagates a complete joint distribution over per-mode
phase offsets and attenuation. It preserves correlations between phase, loss and
spatial modes through native differentiable density operations.

`GriceNoiseConfig` supplies eight rail transmissions/phases, four photon internal
kets in a shared basis of up to four labels, and eight independent detector
responses. Sparse creation-operator expansion retains lost-input-rail environment
records and unresolved internal labels until interference is complete. Detector
corruption is applied to classical output records, following the separation of
loss and measurement noise in [Wein et al.](https://arxiv.org/abs/1509.00088).
Terms and scalar work have independent checked budgets.

The ideal published click classifier stays fixed under imperfections. A noisy
record can declare incorrect Bell parities; those false heralds remain in the
complete positive effects. Tests compare all effects with the earlier ideal and
balanced model, asymmetric signal filters, orthogonal photon labels, and vacuum
false heralds from a pinned Perceval ancilla-only distribution followed by an
independent eight-detector dark-count convolution.

`MachZehnderCalibration::fit` estimates effective survival, visibility and phase
from complete count probes using weighted sine/cosine regression. It rejects
rank-deficient designs and unresolved phase at zero visibility. Source brightness,
path transmission and detector efficiency produce the same multiplicative count
response: these probes identify their product, not each factor separately.
Independent factor calibration is therefore required before attributing loss to
a particular component. Fresh phases and count samples validate the fitted model.

## Measured mixed-state overlap protocol

`OpticalTomography` executes one number-basis experiment and two phase/splitter
experiments per pair of modes. A balanced splitter measures the real coherence
through half the output-count difference; preceding it with a pi/2 phase measures
the imaginary coherence. This is the phase-shifter/splitter path-projection
construction used in [single-photon path tomography](https://www.sciencedirect.com/science/article/abs/pii/S0375960111011157).

The identifiable sector is photon-number-superselected vacuum plus one photon
(or exactly one photon), over at most eight modes. Vacuum/photon coherence needs
an external phase reference and is rejected. Every setting retains no-click
records. For m modes, one state costs `1+m*(m-1)` settings times the declared shots
per setting; overlap uses two independent tomographies and charges both budgets.
Reports include all histograms, measurement settings and preparation attempts.

Linear inversion is projected to a positive photon block with the measured
survival trace. The projection displacement and finite-sample bias are explicit.
A simultaneous Hoeffding bound on measured probabilities, combined with that
displacement, gives a conservative trace-distance bound. Squared-Uhlmann fidelity
is computed **only from reconstructed matrices** as classical post-processing.
The overlap error bound uses the trace-distance/fidelity relations of
[Fuchs and van de Graaf](https://arxiv.org/abs/quant-ph/9712042), with the two
failure probabilities combined by a union bound. The bound can be loose; the API
does not advertise an unsupported narrow confidence interval.

`optical_tomography` validates complex reconstruction and an independent lossy
Bloch fidelity formula across four seeds. Another test passes measured overlaps
to QN-SPSA, samples the loss evaluations, verifies objective improvement, and
charges **198,656** physical simulated preparations across two updates. This is
a real local measurement protocol for its declared sector, separate from exact
spectral reference geometry.

## Minibatch allocation and matched-cost benchmarks

`ShotAllocationPolicy` reserves a next minibatch round from completed counts,
objective coefficients and pilot variances. Box-constrained Neyman allocation
minimizes the continuous plug-in objective variance; rounding preserves minimum,
per-item, per-round and cumulative bounds. The result distinguishes completed
observations from additional reservations and planned totals, and reports
estimated standard error. `ShotAccumulator` merges only completed experiments.
Adaptive stopping can introduce bias; plug-in standard errors are not labeled
rigorous confidence intervals.

`OpticalCostModel` converts complete attempts into expected injected photons,
serial/ideal-lane duration and energy under caller-declared assumptions. These
are model outputs, not measured hardware telemetry. Routing/reset/feedback costs
must be included in the supplied period or separately in the runtime schedule.

[`photonic_benchmarks`](../crates/burn_quantum_example/src/bin/photonic_benchmarks.rs)
records five seeds for measured-feature classification, mixed-state kernel
classification and calibrated Born-distribution prediction. It uses pilot-based
allocation over four calibration phases, 12 labeled training states, 25 disjoint
test states, and independent Born holdout counts. Native kernels are compared
with scalar classical density features using the **identical measured records
and optical query budget**. The control requires zero additional preparations.
JSON records retain each seed, allocation/progress snapshot, held-out metrics
and illustrative photon/time/energy assumptions. Agreement with this classical
control is expected; no quantum advantage is claimed.

```sh
cargo test --locked -p burn_quantum --test optical_calibration --test optical_tomography
cargo run --locked -p burn_quantum_example --bin photonic_benchmarks
```

Each seed now includes measured learning curves with explicit preparation budgets.
`calibration_learning_curve` compares the actual 1,024-shot pilot fit with the
8,192-shot completed allocation, using the identical disjoint test feature records
and Born holdout counts. It reports classification accuracy, true/empirical Born
TVD, and separate total preparation/photon/time/energy costs for each workload.
The final classification cost is 27,392 attempts; the final Born cost is 12,288.

`kernel_learning_curve` fits three genuinely different models on nested balanced
subsets of 4, 8 and 12 training examples, and evaluates each on the same 25 unseen
examples. Their respective total preparation budgets are 22,272, 25,344 and
28,416 attempts, including all tomography settings and no-click records. Every
point has an independently evaluated scalar density-feature control using the
same measured features and its own centroid normalization. All five seeds have
zero control disagreements. These are training-size and shot-budget learning
curves for closed-form fits, not fabricated gradient-optimizer histories.

The curve points reuse nested subsets of one completed measurement pool; their
costs must not be summed as additional experiments. The complete three-workload
run still charges 40,704 distinct preparations per seed and 203,520 across five
seeds. The observed curves need not be monotone under finite-shot noise; early
points are retained even when they perform better than the larger-budget fit.
