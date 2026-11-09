//! Reproducible bounded device profiling, including synchronization and norm checks.
#[path = "../support/device.rs"]
mod runtime;
use burn_quantum::{Circuit, DenseSimulator, QuantumBackend, Register, Unitary};
use std::time::Instant;
fn main() -> burn_quantum::Result<()> {
    let device = runtime::execution()
        .map_err(burn_quantum::Error::Unsupported)?
        .device;
    let engine = DenseSimulator::default();
    println!("qubits,dimension,gates,state_bytes,median_ms,min_ms,max_ms");
    for n in [4, 8, 12] {
        let mut circuit = Circuit::new(Register::qubits(n)?, 0, 0);
        for w in 0..n {
            circuit.unitary(vec![w], Unitary::h())?;
        }
        for w in 1..n {
            circuit.unitary(vec![w - 1, w], Unitary::cnot())?;
        }
        // Warm up allocation and kernels, then include readback/validation in
        // every sample so asynchronous work cannot escape the timing interval.
        engine
            .execute(&circuit, None, None, 1, &device)?
            .validate(1e-4)?;
        let mut elapsed = Vec::with_capacity(7);
        for _ in 0..7 {
            let start = Instant::now();
            let state = engine.execute(&circuit, None, None, 1, &device)?;
            state.validate(1e-4)?;
            elapsed.push(start.elapsed().as_secs_f64() * 1000.);
        }
        elapsed.sort_by(f64::total_cmp);
        println!(
            "{n},{},{},{},{:.3},{:.3},{:.3}",
            circuit.register().size(),
            circuit.operations().len(),
            8 * circuit.register().size(),
            elapsed[3],
            elapsed[0],
            elapsed[6]
        );
    }
    Ok(())
}
