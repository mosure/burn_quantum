# Magic injection and 15-to-1 distillation

`photonic::qec` exposes a bounded implementation of T-state consumption and the
standard fifteen-qubit Reed–Muller distillation code. The raw resource convention
is `|A> = T|+>`, with `T = diag(1, exp(i*pi/4))`. A `MagicState` has separate
conditional Z-error probability and heralded availability. This notation follows
[Bravyi and Haah, Sec. V and Fig. 2](https://arxiv.org/abs/1209.2426). Their
ideal-Clifford assumption is explicit in this implementation.

## Native injection instrument

`MagicState::inject(input, target, limits)` appends the conditional resource to a
native Burn density matrix, applies CNOT from data to ancilla, projects the ancilla
onto each Z outcome, applies S to the data for outcome one, then traces out the
ancilla. Missing-resource, zero, and one records each retain a subnormalized data
state. This supports arbitrary entangled reference qubits and native autodiff.
`completed()` sums the two gate branches; its trace includes availability.

For resource phase error p and yield y, the completed map is exactly

```
y [(1-p) T rho T† + p Z T rho T† Z].
```

The missing branch is `(1-y) rho`. Each completed record has probability `y/2`
for normalized inputs. No measurement outcome is silently renormalized or
replaced by a claimed ideal gate. An unavailable resource executes no gate. One
source is attempted regardless of availability; a supplied resource consumes one
CNOT and one Z measurement, with an S correction on half the supplied attempts.

## Executable code and exact noise calculation

`ReedMuller15::x_checks()` returns four binary rows. Column q contains the
four-bit representation of q+1, for q from 0 through 14. Each row has weight
eight, pair intersections have weight four, and triple intersections have weight
two. Logical X is the all-one row. Bit q in a returned mask denotes physical wire
q; this is independent of the big-endian storage index of a state vector.

`encoding_cnots()` completes the logical row and four even rows to an invertible
binary basis, then synthesizes that transformation with CNOT elimination. Input
wire zero holds the logical state; wires 1 through 4 begin in |+>; the remaining
ten begin in |0>. Both encoder and inverse are ordinary executable Clifford
gates, with no hidden projection into a desired output state.

The zero logical codeword is uniform over the sixteen words in the even-row span.
Their weights are zero or eight. Its complementary logical-one coset has weights
seven or fifteen. Thus transversal T† acts as logical T: it supplies phase one on
the zero coset and `exp(i*pi/4)` on the one coset. Each physical T† can consume a
raw T resource through the injection instrument followed by S†.

`fault_circuit(mask, decode)` constructs this actual encoder, transversal gate,
physical Z pattern, and optional decoder. Following decoding, wires 1 through 4
carry the four X-stabilizer bits in the Z basis; wire zero carries the logical
magic state with Z error equal to the physical fault parity.

`distill(config)` enumerates **all 32,768** independent physical phase-fault
patterns. For each it evaluates the four X checks and logical parity, followed by
all classical readout-error records. Since each specified fault produces an
eigenstate of all four X checks, this enumeration is exactly equivalent to the
projective measurement protocol for the declared noise model. It is a bounded
host-F64 calculation; the selected-device pure circuit is an independent native
validation path, not a claim that the host enumeration ran on a GPU.

The code accepts only a declared all-zero X syndrome. Four independently supplied
readout-flip probabilities retain false acceptance of nonzero true syndromes and
the resulting logical error. The ten Z checks are already satisfied by the ideal
encoder and the declared phase-only errors. No arbitrary Clifford gate noise,
correlated phase faults, or unheralded resource loss is implicitly modeled.

## Resource and rejection accounting

All fifteen sources are attempted once before the code circuit starts. A missing
source aborts that block. The report retains both source readiness and absolute
acceptance, the true and observed syndrome distributions, false heralds, and the
conditional output error. Zero acceptance has no conditional output error and no
finite expected cost per output.

For a ready block, this explicit implementation prepares fifteen code carriers
(including the initial logical |+>) plus four syndrome ancillas. Fifteen additional
raw magic resources are consumed by injection. Each weight-eight X check can be
measured with a |+> syndrome ancilla, eight controlled-X gates, and an X readout.
The counts are therefore:

- Fifteen attempted raw sources, including source-aborted blocks; emitted raw
  states that are later discarded remain in the expected emitted-resource count.
- Fifteen code carriers and four syndrome ancillas prepared after source readiness.
  The ancillary-only field counts fourteen code ancillas plus four syndrome
  ancillas; the code-carrier field explicitly includes the logical carrier.
- Fifteen injection plus four syndrome measurements per ready block.
- Encoder CNOTs, fifteen injection CNOTs, thirty-two syndrome CNOTs, and decoder
  CNOTs only on accepted blocks.
- Five initial Hadamards, eight syndrome preparation/readout Hadamards, four
  decoder Hadamards only after acceptance, fifteen S† gates and mean 7.5 S
  injection corrections per ready block.

The report weights these counts by the actual readiness and acceptance. Expected
raw source attempts per accepted output are `15 / acceptance` for independent
repeated blocks. These are logical carrier/gate counts for the declared protocol,
not measured hardware energy, a photonic factory schedule, or an optimized
fault-tolerant footprint. A dual-rail implementation must separately account for
its fusion fabric and source construction; the calibration/factory interfaces can
carry that accounting without altering the distillation probability.

## Validation and example

`photonic_magic` verifies native injection on an entangled input, including all
records and reverse-mode gradients against a scalar derivative. A native
fifteen-qubit circuit executes the real encoder, transversal T†, and decoder for
no fault, a detected single fault, and a malignant three-fault pattern. Logical
states and all four syndrome wires are checked independently.

A separate explicit codeword-amplitude projector checks every one of the 32,768
fault masks. Its accepted weight enumerator is
`[1,0,0,35,105,168,280,435,435,280,168,105,35,0,0,1]`.
The enumerated iid result is compared with the exact acceptance/error expressions
in [Bravyi and Kitaev, Eqs. 35–36](https://arxiv.org/abs/quant-ph/0403025), including
p=0, p=1/2, and p=1. The implementation itself never substitutes those formulas
for the actual code calculation. A deterministic single fault with imperfect
readout separately checks false-herald probability and its erroneous output.

`photonic_magic_distillation` is a single-file example. It reports source and gate
costs for fifteen raw states with one-percent phase error, 0.995 individual source
yield, and 0.0001 syndrome readout error. It executes the code oracle on the
selected device, then consumes the distilled resource in an entangled T-gate
experiment. The reported conditional gate fidelity and absolute success are
checked against the independently computed distillation report.

```sh
cargo test -p burn_quantum --test photonic_magic
cargo run -p burn_quantum_example --bin photonic_magic_distillation
BURN_QUANTUM_DEVICE=wgpu cargo test -p burn_quantum --features wgpu,fusion --test photonic_magic
```

The native density injection and pure code execution follow the requested device;
there is no device fallback. Qualification claims refer to the executed matrix in
the repository validation record. The finite code model can be composed with
other simulations; it does not itself establish a hardware accuracy threshold.
