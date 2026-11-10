# Matrix-product states and distributed entanglement

`tensor_network::MpsState` represents an open chain of finite qudits with native
Burn split-complex site tensors `[left bond, physical dimension, right bond]`.
It supports arbitrary bounded dense local gates, ordered nonadjacent targets,
selective observables, computational measurements and local CPTP trajectories.
`MpsSimulator` implements the existing `QuantumBackend` contract with independent
MPS chains per batch. It accepts the existing `Circuit`, affine parameter/input
bindings, rotations, controlled-X and phase-flip operations. General dense
operators still have explicit local size limits. It is a quantum engine, not a
new Burn classical backend.

The chain layout does not multiply all local dimensions. An 80-qubit state can
therefore be represented without an overflowing dense `Register`. Storage is
`sum_i chi_(i-1) d_i chi_i`; low-entanglement circuits can remain small while
highly entangled circuits can still require exponential Schmidt rank. Dense
export and probability vectors are separately bounded and may fail even when
local operations and expectations are feasible. No general classical shortcut
for arbitrary quantum computation is claimed.

Contractions execute on the explicitly selected CPU/WGPU/CUDA device. Complex
SVD uses bounded host `nalgebra` work with explicit readback and a convergence
limit. Burn 0.22's own real SVD also has a host reference implementation and no
reverse-mode rule; this engine does not pretend SVD is an accelerated or
differentiable operation. Gradient-requiring circuit bindings are rejected.
Parameter-shift or SPSA evaluators can call the engine explicitly. Constant
native tensors remain native tensors; there is no implicit device fallback.

## Canonicalization and approximation

The state maintains one mixed-canonical center. Exact SVD gauge transfers move
it before a multi-site operation. A gate contracts its contiguous support,
applies its physical matrix, and splits it back into sites. Nonadjacent gates
use physical SWAPs (including heterogeneous local dimensions) and undo the
routing order. Exact-rank mode refuses an insufficient bond budget instead of
silently dropping entanglement. A failed `MpsState::apply` leaves the original
state unchanged.

For an allowed truncation with discarded normalized Schmidt weight `w`, the
normalized retained pure state has trace distance `sqrt(w)`. The implementation
adds these distances under subsequent unitaries and caps the bound at one.
It reports the sum of discarded weights separately; that sum is **not** claimed
to be total infidelity. A selective outcome with represented probability `p`
conservatively amplifies the bound to `min(1, 2 D / p)`. Operations reject a bound
above the configured allowance. Sampling describes the represented distribution
and does not require every extremely rare conditioned branch to satisfy the
initial approximation budget.

This accounting excludes floating-point error. The largest observed center norm
drift is reported separately, and native CPU/GPU dense amplitude tests enforce
numerical parity. The exact-budget comparison has a `1e-12` floating-point floor.
Asymmetric dual-rail loss is handled by its complete Kraus channel, including the
state-dependent no-jump filter; it is not replaced by balanced erasure.

Limits bound site counts, bond ranks, tensor entries, persistent entries and SVD
iterations. They are allocation preflights, not a total process-memory guarantee:
SVD workspaces, clones and backend buffers also consume memory. Checkpoint
restoration validates finite values, shapes, every bond, canonical Gram matrices,
center norm, schema and error report. A bounded streaming identity contraction
also checks the global norm: individually acceptable local gauge errors can
accumulate along the chain. No dense Hilbert-space state is built for this check.
Persisted approximation histories must be
authenticated by the caller; mathematical validation cannot recover forged
historical error data.

The representation follows [Vidal's MPS simulation](https://arxiv.org/abs/quant-ph/0301063).
The use of controlled bond truncation in quantum algorithms is also studied by
[Banuls et al.](https://arxiv.org/abs/quant-ph/0503174). The implementation and its
error accounting are independently exercised against this crate's dense engine.

## Distributed ownership

`tensor_network::distributed` assigns disjoint contiguous site ranges to real
workers. The coordinator retains shapes, provenance, generation and at most the
sites involved in a current gate. It transfers two tensors for a center move or
adjacent gate. Their Schmidt indices preserve entanglement across worker cuts;
there is no time-window trace, independent-boundary approximation or circuit
cutting substitution. Product observables and basis amplitudes stream a bounded
bond environment through the owners. Only the explicitly named `export` method
gathers the complete chain.

A single-writer two-phase transaction updates every owner's generation, including
owners with no changed tensors. Prepare verifies shape, ownership, gauge and
budgets. The coordinator persists its prepare journal before contacting owners,
then records and persists its commit decision before committing. Workers persist a complete
candidate journal before acknowledging a mutation. Duplicate matching operations
are idempotent; conflicting retries and stale reads are rejected. If any response
is uncertain, further reads are blocked. Recovery aborts pre-commit prepares or
finishes a commit decision. Recovery successfully persists that decision again
before any commit RPC, including after a write-then-error storage response.
Persistence callbacks must atomically replace complete checkpoints to avoid torn
files. It never guesses that a missing worker lost
its update. This is blocking two-phase recovery, not consensus or automatic
failover; durable recovery requires caller-supplied persistence callbacks and
all relevant journals.

Attaching or restoring a distributed state streams the same global normalization
check through the owners. A restored pending journal performs it after resolving
the transaction. Failed validation continues to block both reads and mutations,
even when the transaction journal itself has been cleared. This validation pass
holds one site and a bond environment at a time; normal gate transactions do not
repeat a full-chain scan.

The bounded versioned JSON/TCP transport shares absolute-deadline framing with
the existing shot workers. Capabilities authorize a configured local service;
use a trusted network or authenticated tunnel. Plain TCP is not encrypted.
Transport timeouts do not forcibly interrupt a local numerical kernel. Persisted
coordinator records omit credentials. Network statistics count successfully
acknowledged encoded exchanges, not failed TCP attempts or link-layer overhead.

`distributed_mps` starts one and three **actual child-process** ownership layouts,
executes the same entangled non-Clifford workload, and compares every amplitude
with a native dense oracle. A deliberately failed commit write blocks reads,
survives coordinator serialization/restoration and reconciles exactly once. Worker
unit tests also cover prepared-state restart, abort, stale epochs, conflicting
retries and corrupted checkpoints. This qualifies local multi-process execution;
it is not a measurement of multi-host network performance.

## Executable examples

```sh
cargo run --locked -p burn_quantum_example --bin mps_qaoa
cargo run --locked -p burn_quantum_example --bin distributed_mps
BURN_QUANTUM_DEVICE=wgpu cargo test --locked -p burn_quantum --features wgpu --test tensor_network
```

`mps_qaoa` searches a bounded p=1 parameter grid, then runs the chosen circuit on
an 80-qubit ring. It checks separated and wrap-around correlations against the
independent analytic law `cut = 1/2 + sin(4 beta) sin(2 gamma)/4`, including the
0.75 optimum and 0.5 random control. Its explicit accumulated truncation allowance
is `1e-4`, distinct from the `2e-5` observed-law acceptance tolerance. This is a
scaling and variational-circuit reference, not a claim of quantum advantage.
