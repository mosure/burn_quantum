//! Build a custom Burn circuit and inspect its actual simulated trajectory.
use burn::tensor::Device;
use burn_quantum::{Angle, Circuit, Register, Rotation, gate::Axis};
use burn_quantum_visualization::{
    app::run_document,
    model::VisualizationDocument,
    simulation::{TraceLimits, simulate},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(Register::qubits(1)?, 0, 0);
    circuit.rotate(0, Rotation::qubit(Axis::Y, Angle::Constant(0.8)))?;
    circuit.rotate(0, Rotation::qubit(Axis::Z, Angle::Constant(1.2)))?;
    let operation_labels = vec!["Ry(0.8)".into(), "Rz(1.2)".into()];
    let trace = pollster::block_on(simulate(
        &circuit,
        &[],
        &[],
        &operation_labels,
        &Device::flex(),
        TraceLimits::default(),
    ))?;
    run_document(VisualizationDocument {
        schema_version: 1,
        title: "My two rotations".into(),
        description: "A custom circuit passed directly to the reusable Bevy explorer.".into(),
        circuit,
        parameters: vec![],
        inputs: vec![],
        operation_labels,
        trace,
        fabric: None,
        optimization: vec![],
        protocol: None,
    })
}
