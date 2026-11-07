//! Optimize a variational circuit using Burn's Adam optimizer.
#[path = "../support/device.rs"]
mod runtime;
use burn::optim::{AdamConfig, GradientsParams};
use burn_quantum::{
    Angle, Circuit, DenseSimulator, Observable, QuantumLayer, Register, Rotation, gate::Axis,
};
fn main() -> burn_quantum::Result<()> {
    let device = runtime::execution()
        .map_err(burn_quantum::Error::Unsupported)?
        .device
        .autodiff();
    let mut circuit = Circuit::new(Register::qubits(1)?, 1, 0);
    circuit.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))?;
    let mut layer = QuantumLayer::new(
        circuit,
        vec![Observable::z(0)],
        vec![0.3],
        DenseSimulator::default(),
        &device,
    )?;
    let mut optimizer = AdamConfig::new().init();
    for step in 0..100 {
        let expectation = layer.forward(None, 1)?;
        let loss = (expectation + 1.).square().mean();
        if step % 20 == 0 {
            println!("step {step}, loss {}", loss.clone().into_scalar::<f32>());
        }
        let gradients = GradientsParams::from_grads(loss.backward(), &layer);
        layer = optimizer.step(0.08, layer, gradients);
    }
    let z = layer.forward(None, 1)?.into_scalar::<f32>();
    println!("Final <Z> = {z:.6}; optimum = -1");
    assert!(z < -0.98, "training did not converge");
    Ok(())
}
