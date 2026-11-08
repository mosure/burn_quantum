# Boolean SAT with Grover search

The `sat` module solves small **classical Boolean CNF** problems by simulating
Grover search on Burn CPU/GPU devices. It includes a reversible clause compiler,
strict DIMACS parsing, explicit iteration experiments, a seeded search for an
unknown number of solutions, and classical witness verification. It does not
solve the distinct quantum constraint problem usually called QSAT.

This feature is included in `burn_quantum` 0.2.0.

## Use the solver

```rust
use burn::tensor::Device;
use burn_quantum::{ResourceLimits, sat::{Cnf, GroverSat, SearchConfig, SatOutcome}};

fn main() -> burn_quantum::Result<()> {
    // (x1 OR x2) AND (!x1 OR x2) AND (!x2 OR x3) AND (x1 OR !x3).
    let formula = Cnf::from_dimacs(
        "p cnf 3 4\n1 2 0\n-1 2 0\n-2 3 0\n1 -3 0\n"
    )?;
    let solver = GroverSat::new(formula, ResourceLimits::default())?;
    let result = solver.search(SearchConfig {
        seed: 7, max_trials: 64, max_oracle_calls: 256,
    }, &Device::flex())?;
    match result.outcome {
        SatOutcome::Satisfiable { assignment } => {
            assert!(solver.formula().evaluate(&assignment)?);
            println!("model: {assignment:?}");
        }
        SatOutcome::Unsatisfiable => println!("explicit empty-clause contradiction"),
        SatOutcome::Inconclusive => println!("no witness found within the budget"),
    }
    Ok(())
}
```

`Cnf::new(n, clauses)` accepts signed, one-based `i32` literals. Variable 1
corresponds to assignment element/wire zero, the most significant bit. Zero is
invalid in this API. In DIMACS text, zero terminates a clause. The parser requires
one `p cnf n m` header, checks the original clause count, permits comment lines
and clauses spanning lines, and rejects incomplete/out-of-range input.

Normalization removes duplicate literals, duplicate clauses and tautologies.
It retains unused declared variables. An empty clause is false; an empty list
of clauses is true, including when there are zero declared variables. JSON
deserialization repeats validation and normalization.

Run the single-file example from the repository root:

```sh
cargo run --locked -p burn_quantum_example --bin grover_sat
cargo run --locked -p burn_quantum_example --bin grover_sat -- problem.cnf
BURN_QUANTUM_DEVICE=wgpu cargo run --locked -p burn_quantum_example --features wgpu --bin grover_sat
BURN_QUANTUM_DEVICE=cuda cargo run --locked -p burn_quantum_example --features cuda,fusion --bin grover_sat
```

With no filename it verifies a known three-variable problem and prints the same
JSON report format as the QML suite. With a filename it prints backend metadata
and a `search` object containing `status`, a Boolean `assignment` if found,
`trials`, `oracle_calls`, and `seed`. A valid inconclusive result exits normally;
invalid inputs, unavailable devices, resource limits and numerical errors fail.
Use the Rust API to choose search/allocation budgets.

## Reversible oracle and reusable primitives

For n variables and m normalized clauses, variable wires are `0..n` and clause
ancillas are `n..n+m`. Each clause ancilla starts at zero. An unconditional X
followed by an X controlled on every literal being false computes that clause's
truth value. A phase flip controlled on all clause ancillas being one marks
satisfying assignments. Reversing the clause operations restores every ancilla:

`|x>|0...0> -> (-1)^CNF(x) |x>|0...0>`.

Construction inspects literals only; it never enumerates satisfying assignments
or counts solutions. The circuit requires initially clean ancillas for this
contract. Work qubits are explicitly uncomputed before diffusion and measurement.

The core adds `Circuit::controlled_x` and `Circuit::phase_flip`, with controls
specified as distinct `(wire, level)` pairs. Controls can be qudits; controlled X
requires a qubit target distinct from all controls. Empty controls mean ordinary
X or a global minus sign. The dense engine implements them with O(N) permutation
indices or diagonal signs, rather than O(N²) local matrices. Native tensor
operations preserve batch isolation, dtype, device and autodiff.

`Circuit::run_from` executes a circuit on an existing state, checking the register,
bindings, tensor contexts and budgets. This allows reuse of the same oracle and
diffusion circuits without allocating a circuit proportional to iteration count.
Circuits with structured controls serialize as schema v2; existing circuits
retain v1. The reader accepts both versions and rejects v2 operations in v1 data.

## Amplification and stopping

`GroverSat::initial_state` prepares uniform variables and zero work qubits.
`amplify(r, device)` applies exactly r oracle/diffusion pairs. Diffusion is
`2|s><s| - I` on variable wires and identity on work wires. If M of the N=2^n
assignments satisfy the formula, its ideal success probability is
`sin²((2r+1) asin(sqrt(M/N)))`. More iterations can overshoot the optimum.

`search` uses a bounded BBHT-style schedule. It draws an integer j below m,
starts from the uniform state, performs j iterations, samples one candidate and
checks it against the CNF. After failure, m grows by 6/5 up to sqrt(N). It needs
no supplied solution count. The unbounded quantum algorithm has expected
O(sqrt(N/M)) oracle-query cost for M>0; the configured finite budgets do not
provide a failure-confidence certificate. See [Boyer et al.](https://arxiv.org/abs/quant-ph/9605034).

`max_trials` bounds measurements/classical candidate checks; `max_oracle_calls`
bounds the total executed Grover iterations. A trial whose selected iteration
count would exceed the remaining budget is not executed. A zero oracle budget
still allows zero-iteration sampling. Random choices use a local ChaCha8 stream;
the host evaluates only measured candidates, with no exhaustive fallback.

`Satisfiable` always contains a checked witness. `Unsatisfiable` is reserved for
an explicit empty clause. All other exhausted searches return `Inconclusive`,
including ordinary contradictory formulas with no empty clause. An empty
conjunction returns the all-false model without sampling. Classical CDCL/proof
integration and confidence-controlled termination are future extensions.

The reversible construction is also illustrated in
[PsiQuantum's SAT search tutorial](https://docs.construct.psiquantum.com/e2e-tutorials/satsolver_1_circuit.html).

## Resource boundaries and validation

The reference compiler uses one work qubit per normalized clause, so the dense
state has 2^(n+m) complex amplitudes, or 8 * 2^(n+m) bytes of bare F32 amplitude
storage. A zero-wire constant uses a dummy qubit. Intermediates, index plans and
autodiff graphs cost additional memory. The current limits reject excessive
states/plans and integer overflows before allocation; use ordinary devices
without autodiff for search unless a retained analytic graph is needed.

This is not a competitive classical SAT solver, a quantum speedup on a GPU,
or a polynomial-time algorithm for SAT. Quantum query complexity assumes a
quantum oracle; every simulated gate still processes an exponential state.
Ancilla reduction, optimized compiled control kernels and hardware decomposition
are possible extensions. No QPU execution or noisy-SAT guarantee is implemented.

The new tests cover all assignments of small signed CNFs, complex phases and
ancilla cleanup, mixed qudit controls, batched permutations, reflection against
an independent complex scalar reference, analytic amplification curves for
zero/one/many/all solutions, reproducible witnesses, bounded failure, malformed
DIMACS, serialization, resource rejection, continuation and native gradients.

```sh
cargo test --locked -p burn_quantum --test sat
BURN_QUANTUM_DEVICE=wgpu cargo test --locked -p burn_quantum --features wgpu --test sat -- --test-threads=1
BURN_QUANTUM_DEVICE=cuda cargo test --locked -p burn_quantum --features cuda,fusion --test sat -- --test-threads=1
```

The standard workspace CPU checks and hardware validation scripts include the
SAT tests and example. On **2026-11-08**, the full CPU validation passed **63 tests
and one doctest**. Targeted hardware validation passed **21 checks per
configuration**: 11 SAT tests, nine existing backend regression tests and the new
example's test, followed by its executable. Parser/metadata cases run on the host,
and the explicit F64 continuation reference always uses Flex. The seven older
QML executables were not rerun on GPU for this change.

The GPU was an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, driver
610.43.02. Both WGPU/Vulkan and CUDA passed with fusion off and on. In all five
configurations, the demo measured success probability **0.9453119636** after two
iterations versus the analytic **0.9453125**. The independent unknown-count
search found `[true, true, true]` after four sampled candidates and one total
oracle call. Existing F32 parity tolerances were retained.

The [machine-readable record](sat-results.json) includes reports, comparisons
and source hashes. Qualification covers these small F32 problems on this GPU;
it is not a speedup measurement or qualification of other GPU platforms.
