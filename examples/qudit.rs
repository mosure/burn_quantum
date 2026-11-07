//! Maximally entangled qutrits and their reduced state.
#[path = "../support/device.rs"]
mod runtime;
use burn_quantum::{Circuit, DenseSimulator, QuantumBackend, Register, Unitary};
fn main() -> burn_quantum::Result<()> {
    let mut circuit = Circuit::new(Register::new(vec![3, 3])?, 0, 0);
    circuit
        .unitary(vec![0], Unitary::fourier(3)?)?
        .unitary(vec![0, 1], Unitary::sum(3)?)?;
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
    println!("Qutrit Bell probabilities: {}", state.probabilities());
    println!(
        "Reduced qutrit purity: {}",
        state.reduced_density(&[0], engine.limits)?.purity()
    );
    Ok(())
}
