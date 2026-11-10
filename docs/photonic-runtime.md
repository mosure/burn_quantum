# Physical scheduling, events and process workers

The local execution stack has three independent ownership boundaries:

* `FusionProgram::compile_finite` validates explicit fusion ticks against a directed
  fabric, finite packet buffers, link reservations, switch configuration time,
  source births, feed-forward latency and analyzer reset. `FabricConstraints`
  bounds the space-time search and retained reservations. Scheduling is greedy;
  a rejected placement is not a proof that every alternative schedule is impossible.
* `runtime::events::EventMachine<S,E>` owns a complete application state and a
  bounded causal event frontier. A transition receives a candidate state and a
  stable event seed, then emits future operations. Backward time, callback failure
  and backpressure roll back that transition. Prior committed events remain.
* `runtime::network` shards **independent shots** over explicitly selected TCP
  workers. The versioned, length-prefixed JSON protocol bounds frames, batches,
  retained payloads, request deadlines and retries. Workers run a configured
  callback, not instructions to execute arbitrary code. Every reply must match
  the exact immutable job and requested shot IDs before its wave is committed.

Finite routes include consumed photons' source storage, switch waiting and final
analyzer input occupancy. Unconsumed outputs occupy source buffers through the
schedule finish. The compiler charges pre-release, link/switch waiting, analyzer
waiting and output storage once each. A zero-slot junction can be traversed
without waiting; nonzero switch configuration requires storage there. Whole
packets retain synchronized rails. Independently switched rails and coherent
interference are optical-model concerns, not inferred from transport links.

`CorrelatedFactoryBank` provides a finite calibrated latent-mode mixture. One
mode can represent a common source outage or brightness drift; conditioned on
that mode, individual bounded factories are independent. Its joint yield is the
mixture of products, not the product of marginal yields. All failed source
attempts retain photon consumption and timing. This is classical availability
correlation, not a model of coherent resource-state preparation defects.

Event checkpoints include full caller state, pending events, causal parents,
watermark, complete audit history, budget, job identity and shot ID. Restoration
checks the frontier and calls the application's state validator. Byte admission
limits apply before decoding; checkpoint encoding stops at its byte limit.
The application must bound its live state and include every entangled mode,
Pauli frame, source queue and pending decoder/controller state. Advancing a time
window never implicitly traces or deletes quantum systems. This API does not
claim distributed ownership of a quantum state or a proven causal tensor cut.

Network retries reuse global shot IDs and component seeds; they do not resample
a changed experiment. A failed wave commits no partial replies; completed prior
waves can be checkpointed and resumed. Cancellation is cooperative. The transport
uses plain TCP, intended for a trusted local network or an authenticated encrypted
tunnel. A configured capability is not TLS. Fingerprints are caller-supplied and
must cover the actual recipe, parameters, compiler, calibration and readout.
There is no implicit remote-device or GPU fallback.

## Executable studies

`fbqc_remote` starts two independent local worker processes, simulates lossy Bell
swaps, compares every serialized shot with the single-worker executor, restores
its ledger and reports communication payload bytes and wall time. A native Burn
dense calculation on the selected device supplies its physical oracle.

`fbqc_adaptive` combines shared-outage sources, finite buffers, actual fusion
records, two-tick controller feedback and at most three fresh-resource attempts.
Destructive failure discards both entire resources; each retry creates new ones.
It preserves a pending measurement through a checkpoint, compares windowed and
uninterrupted traces, and checks unconditional success and photon cost against
exact finite enumeration. It models Bell distribution, not a retry that somehow
preserves an arbitrary unknown input after destructive fusion failure.

```sh
cargo run --locked -p burn_quantum_example --bin fbqc_remote
cargo run --locked -p burn_quantum_example --bin fbqc_adaptive
cargo test --locked -p burn_quantum --test network_runtime --test event_runtime --test factory_bank
```

Loopback TCP tests require an environment that permits local socket listeners.
No public service, provider credential or hardware connection is required.

## Immutable asynchronous local jobs

`runtime::jobs::LocalSimulationJobs` implements asynchronous submit, status,
result, cancellation and terminal-result release around a real local evaluator.
A fixed worker pool consumes a bounded FIFO. Queue length, retained jobs,
per-job attempted shots and exact retained encoded-record bytes have separate
limits. No callback runs while the service's state lock is held. Duplicate job
IDs are idempotent only for identical requests; changed content is rejected.

Requests identify the program, compiler, physical model, executor and calibration,
plus finite bindings, ordered readouts, precision, approximation allowance, global
seed, attempted-shot count and timeout. Capability checks reject unsupported
programs, readouts and precision before queueing. Calibration validity is checked
at submission, before each shot and after each completed callback using a monotonic
lease; a final callback crossing expiry cannot finish as a valid completed job.
The caller supplies a trustworthy wall-clock timestamp at admission. A local
adapter always reports `execution_kind = "local_simulator"`; it cannot silently
claim hardware execution or select a provider fallback.

Every completed attempt retains raw detector outcomes, acceptance, numerical
readouts, consumed photons and physical latency ticks. Unaccepted attempts remain
in the result and consume budget. Cancellation and deadlines act at shot
boundaries; a callback that crosses the deadline retains its actually completed
record and terminates as timed out. Cancellation observed during the final
callback likewise retains that completed record and ends as cancelled rather
than completed; the terminal transition is synchronized with cancellation. Errors/panics preserve earlier completed
attempt IDs. These are host scheduling deadlines, distinct from modeled photon
arrival/controller timing. Callbacks must cooperate for bounded shutdown; the
service does not kill a running numerical kernel. Result snapshots are bounded
copies and can temporarily add memory beyond retained storage.

`photonic_jobs` evaluates two immutable lossy Mach-Zehnder distributions on the
selected native device, then runs 8,192 actual simulated detector attempts under
one- and three-worker layouts. Raw outcomes replay exactly, including missed
photons. Native probabilities are checked against an analytic law and finite-shot
outputs against a conservative sampling tolerance. The service is a working
local asynchronous adapter, not a mock PsiQuantum QPU.

## Entangled-state ownership

The independent-shot and event APIs above remain distinct from
`tensor_network::distributed`. That new engine distributes actual MPS tensors
across child-process owners, preserves Schmidt indices, and reconciles two-phase
boundary transactions before allowing further reads. It shares the bounded
framing/absolute-deadline transport with shot workers. See the
[tensor-network guide](tensor-network.md) for the canonical-cut argument,
checkpoint requirements, communication costs and one-/three-owner validation.
Local process qualification does not establish multi-host throughput or QPU access.
