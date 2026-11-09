//! Entanglement and seeded computational-basis measurement.
#[path = "../support/device.rs"]
mod runtime;
use burn_quantum::{
    Circuit, DenseSimulator, Observable, QuantumBackend, Register, Samples, Unitary,
};
fn main() -> burn_quantum::Result<()> {
    let mut circuit = Circuit::new(Register::qubits(2)?, 0, 0);
    circuit
        .unitary(vec![0], Unitary::h())?
        .unitary(vec![0, 1], Unitary::cnot())?;
    let engine = DenseSimulator::default();
    let state = engine.execute(
        &circuit,
        None,
        None,
        1,
        &runtime::execution()
            .map_err(burn_quantum::Error::Unsupported)?
            .device,
    )?;
    state.validate(1e-5)?;
    println!("Bell probabilities: {}", state.probabilities());
    println!("<Z0>: {}", engine.expectation(&state, &Observable::z(0))?);
    println!(
        "Samples: {:?}",
        Samples::from_state(&state, 16, 42, 100)?.digits(0, state.register())?
    );
    Ok(())
}
