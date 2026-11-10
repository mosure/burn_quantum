//! Reproducible scaling probe; reports timing, stored data and bounded plot size.
use burn::tensor::Device;
use burn_quantum::{Circuit, Register, Unitary};
use burn_quantum_visualization::{
    model::{VisualizationDocument, aggregate_probabilities},
    simulation::{TraceLimits, simulate},
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let export = std::env::args().nth(1);
    let mut rows = Vec::new();
    for wires in [4, 8, 12] {
        let mut circuit = Circuit::new(Register::qubits(wires)?, 0, 0);
        for wire in 0..wires {
            circuit.unitary(vec![wire], Unitary::h())?;
        }
        let labels = (0..wires).map(|w| format!("H q{w}")).collect::<Vec<_>>();
        let begin = Instant::now();
        let trace = pollster::block_on(simulate(
            &circuit,
            &[],
            &[],
            &labels,
            &Device::flex(),
            TraceLimits::default(),
        ))?;
        let simulation_seconds = begin.elapsed().as_secs_f64();
        let bins = aggregate_probabilities(&trace.frames.last().unwrap().probabilities, 64)?;
        let mass: f64 = bins.iter().map(|b| b.probability).sum();
        assert!((mass - 1.).abs() < 5e-4);
        rows.push(serde_json::json!({"qubits": wires, "dimension": 1usize << wires, "frames": trace.frames.len(), "simulation_seconds": simulation_seconds, "serialized_trace_bytes": serde_json::to_vec(&trace)?.len(), "probability_bins": bins.len(), "retained_probability_mass": mass}));
        if wires == 12
            && let Some(path) = &export
        {
            let document = VisualizationDocument {
                schema_version: 1,
                title: "12-qubit bounded trace".into(),
                description: "Twelve Hadamards produce 4096 equally probable basis outcomes. Display bins retain all probability mass.".into(),
                circuit,
                parameters: Vec::new(),
                inputs: Vec::new(),
                operation_labels: labels,
                trace,
                fabric: None,
                optimization: Vec::new(),
                protocol: None,
        study: None,
            };
            std::fs::write(
                path,
                burn_quantum_visualization::io::document_json(&document)?,
            )?;
        }
    }
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}
