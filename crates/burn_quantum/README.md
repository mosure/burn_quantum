# burn_quantum

Differentiable qubit and qudit simulation using Burn 0.22's native tensors,
devices, modules, parameters, optimizers and weight records. Rust 1.95+.

`QuantumBackend` describes quantum execution; `DenseSimulator` implements it
using supported Burn devices. The public API accepts native `Tensor<D>` and
`Device` values. It does not require a separate tensor backend or a Burn fork.

## Quick start

```toml
[dependencies]
burn_quantum = "0.2.0"
burn = { version = "~0.22.1", default-features = false, features = ["std", "flex"] }
```

```rust
use burn::tensor::Device;
use burn_quantum::{Circuit, DenseSimulator, QuantumBackend, Register, Unitary};

fn main() -> burn_quantum::Result<()> {
    let mut circuit = Circuit::new(Register::qubits(2)?, 0, 0);
    circuit.unitary(vec![0], Unitary::h())?;
    circuit.unitary(vec![0, 1], Unitary::cnot())?;
    let state = DenseSimulator::default()
        .execute(&circuit, None, None, 1, &Device::flex())?;
    state.validate(1e-5)?;
    // [0.5, 0.0, 0.0, 0.5] in |00>, |01>, |10>, |11> order.
    println!("{}", state.probabilities());
    Ok(())
}
```

## Capabilities

- Batched qubit, qudit and mixed-dimensional pure states and density matrices.
- Validated local unitaries, differentiable rotations, observables, partial
  traces, Kraus channels and seeded terminal measurement.
- Serializable circuits, native autodiff, qudit-aware parameter-shift
  references, and `QuantumLayer` integration with Burn modules and records.
- Reusable encodings, circuit templates, probability losses and fidelity kernels.
- Quantum natural SPSA with a regularized Fubini–Study metric and checkpoints.
- Boolean CNF/DIMACS SAT search with reversible oracles and verified witnesses.

Default features are `flex` and `autodiff`. Optional `wgpu`, `cuda` and `fusion`
features select accelerator capabilities; callers supply the execution device.
F32/F64 availability depends on the device. GPU qualification uses F32.

Dense simulation uses exponential state storage: for batch B and Hilbert
dimension N, pure states store 2BN real scalars and density matrices store 2BN².
Allocation limits are preflight checks, not a total memory cap. SAT search may
return `Inconclusive` when its budget is exhausted. QN-SPSA's dense host metric
uses O(d²) storage and an O(d³) solve for d parameters. These implementations do
not establish quantum advantage or efficient simulation of arbitrary circuits.

## Package and repository

This package owns `src/`, integration `tests/`, simulator `examples/`, and the
executable-only `support/device.rs` helper. From a repository checkout:

```sh
cargo run --locked -p burn_quantum --example bell
cargo test --locked -p burn_quantum
cargo check --locked -p burn_quantum --no-default-features
cargo run --locked -p burn_quantum --features wgpu --example wgpu_smoke
```

The sibling `burn_quantum_example` package contains nine single-file QML/search
applications, including VQE, QAOA, classifiers, a Born machine, hybrid
regression, Grover SAT and QN-SPSA VQE. It is run from a repository checkout:

```sh
cargo run --locked -p burn_quantum_example --bin qnspsa_vqe
```

See the [API documentation](https://docs.rs/burn_quantum),
[example suite](https://github.com/mosure/burn_quantum/tree/main/crates/burn_quantum_example),
[guides and validation records](https://github.com/mosure/burn_quantum/tree/main/docs),
and [changelog](https://github.com/mosure/burn_quantum/blob/main/CHANGELOG.md).

## License

MIT OR Apache-2.0. Both license texts are included in this package.
