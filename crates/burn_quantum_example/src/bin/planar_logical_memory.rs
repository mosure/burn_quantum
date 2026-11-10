//! Open-boundary logical memory with actual encoded quantum states, physical losses
//! and explicit circuit-reference costs. Native Burn validates a 14-qubit encoded
//! Bell state; host tableaus run d3/5/7 memories. Perfect check gates are assumed.
use burn_quantum::photonic::{
    compiler::{PhysicalCircuit, PhysicalPlacement},
    qec::*,
    stabilizer::*,
    *,
};
use burn_quantum::{Circuit, DenseSimulator, Matrix, Observable, Register, Unitary};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use std::time::Instant;
fn run() -> ExampleResult<Report> {
    let code = PlanarCode::new(3, 13)?;
    let n = code.qubits();
    let mut circuit = Circuit::new(Register::qubits(n + 1)?, 0, 0);
    circuit.unitary(vec![n], Unitary::h())?;
    circuit.unitary(vec![n, code.input_site()], Unitary::cnot())?;
    for &q in code.encoding_plus_sites() {
        circuit.unitary(vec![q], Unitary::h())?;
    }
    for &(a, b) in code.encoding_cnots() {
        circuit.unitary(vec![a, b], Unitary::cnot())?;
    }
    // The same encoder is lowered through the physical compiler for routing and
    // all-success path costs. Its 14-input density channel is not executed here.
    let fabric = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: RailTransmission::IDEAL,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: RailTransmission::IDEAL,
            },
        ],
        vec![FabricLink {
            from: NodeId(0),
            to: NodeId(1),
            ticks: 1,
            transmission: RailTransmission::IDEAL,
        }],
    )?;
    let physical = PhysicalCircuit::compile(
        &circuit,
        &fabric,
        PhysicalPlacement {
            inputs: vec![NodeId(0); n + 1],
            resource: NodeId(0),
            analyzer: NodeId(1),
            feedback_ticks: 1,
        },
        CompileOptions::default(),
    )?;
    let engine = DenseSimulator::default();
    let state = circuit.run(&engine, None, None, 1, &execution_device()?)?;
    state.validate(2e-4)?;
    let mut correlations = Vec::new();
    for (p, basis) in [
        (code.logical_x(0)?, Pauli::X),
        (code.logical_z(0)?, Pauli::Z),
    ] {
        let mut wires: Vec<_> = p.terms().iter().map(|&(q, _)| q).collect();
        wires.push(n);
        let dim = 1 << wires.len();
        let mut matrix = vec![0.; dim * dim];
        for i in 0usize..dim {
            if basis == Pauli::X {
                matrix[(i ^ (dim - 1)) * dim + i] = 1.;
            } else {
                matrix[i * dim + i] = if i.count_ones() % 2 == 0 { 1. } else { -1. };
            }
        }
        correlations.push(
            values(
                Observable::new(wires, Matrix::real(dim, matrix)?)?.expectation(&engine, &state)?,
            )?[0],
        );
    }
    ensure(
        correlations.iter().all(|&v| v > 1. - 2e-4),
        "native planar encoded Bell oracle",
    )?;
    let mut metrics = vec![
        (
            "encoder_scheduled_resource_photons",
            physical.resource_photons() as f64,
        ),
        (
            "encoder_all_success_finish_tick",
            physical.finish_tick() as f64,
        ),
        ("native_encoded_xx", correlations[0]),
        ("native_encoded_zz", correlations[1]),
        ("host_attempts_per_distance", 64.),
        ("rounds_per_attempt", 2.),
    ];
    let labels = [
        [
            "d3_corrected",
            "d3_logical_errors",
            "d3_ambiguous",
            "d3_budget",
            "d3_failure_ci_low",
            "d3_failure_ci_high",
            "d3_replacements",
            "d3_cnots",
            "d3_preparations",
            "d3_host_seconds",
        ],
        [
            "d5_corrected",
            "d5_logical_errors",
            "d5_ambiguous",
            "d5_budget",
            "d5_failure_ci_low",
            "d5_failure_ci_high",
            "d5_replacements",
            "d5_cnots",
            "d5_preparations",
            "d5_host_seconds",
        ],
        [
            "d7_corrected",
            "d7_logical_errors",
            "d7_ambiguous",
            "d7_budget",
            "d7_failure_ci_low",
            "d7_failure_ci_high",
            "d7_replacements",
            "d7_cnots",
            "d7_preparations",
            "d7_host_seconds",
        ],
    ];
    for (distance, labels) in [3, 5, 7].into_iter().zip(labels) {
        let start = Instant::now();
        let code = PlanarCode::new(distance, 100)?;
        let result = run_planar_memory(
            &code,
            2,
            PlanarMemoryNoise::storage(
                RailTransmission::new(0.998, 0.998)?,
                2,
                PauliNoise::new(0.002, 0.002, 0.002)?,
            )?,
            0,
            64,
            987,
            PlanarDecodeBudget {
                max_weight: (distance - 1) / 2,
                max_candidates: 200_000,
            },
        )?;
        let counts = &result.outcomes;
        ensure(
            counts.attempts
                == counts.corrected
                    + counts.logical_errors
                    + counts.erased_logicals
                    + counts.decoder_budget,
            "planar attempted inputs were omitted",
        )?;
        let interval = counts.unsuccessful_interval(1.95996398454)?;
        for (label, value) in labels.into_iter().zip([
            counts.corrected as f64,
            counts.logical_errors as f64,
            counts.erased_logicals as f64,
            counts.decoder_budget as f64,
            interval.lower,
            interval.upper,
            result.replacements as f64,
            result.cost.cnots as f64,
            result.cost.preparations as f64,
            start.elapsed().as_secs_f64(),
        ]) {
            metrics.push((label, value));
        }
    }
    Report::new("planar_logical_memory", 987, 0, &metrics)
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn physical_loss_memory_and_native_encoded_oracle() {
        super::run().unwrap();
    }
}
