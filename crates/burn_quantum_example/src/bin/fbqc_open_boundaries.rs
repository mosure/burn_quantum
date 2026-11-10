//! Actual finite six-ring fusion strips with terminal boundary measurements.
//! Odd strips carry signed H Choi correlations; even strips carry identity Choi
//! correlations. A correlated resource-loss event and local Pauli faults retain
//! every failure. These small strips do not claim growing error-code distance.
use burn_quantum::photonic::{qec::*, stabilizer::*, *};
use burn_quantum::{Circuit, DenseSimulator, Matrix, Observable, Register, Unitary};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
fn run() -> ExampleResult<Report> {
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
    let patch = SixRingPatch::bell_strip(1, NodeId(0), NodeId(1), 6)?;
    let plan = patch
        .program()
        .compile(&fabric, CompileOptions::default())?;
    let relations = patch.relations(&plan, 1 << 20)?;
    let correlations = relations.output_relations(&[0, 3], 1 << 20)?;
    let mut circuit = Circuit::new(Register::qubits(6)?, 0, 0);
    for q in 0..6 {
        circuit.unitary(vec![q], Unitary::h())?;
    }
    for q in 0..6 {
        circuit.unitary(vec![(q + 1) % 6], Unitary::h())?;
        circuit.unitary(vec![q, (q + 1) % 6], Unitary::cnot())?;
        circuit.unitary(vec![(q + 1) % 6], Unitary::h())?;
    }
    let engine = DenseSimulator::default();
    let state = circuit.run(&engine, None, None, 1, &execution_device()?)?;
    let mut minimum = 1f64;
    for (target, relation) in correlations {
        let mut operator = target;
        let mut sign = relation.negative;
        for m in relation.terms {
            let (negative, p) = operator.multiply_commuting(&relations.observables()[m])?;
            operator = p;
            sign ^= negative;
        }
        let terms = operator.terms();
        let dim = 1usize << terms.len();
        let mut matrix = vec![0.; dim * dim];
        for i in 0..dim {
            let mut j = i;
            let mut negative = false;
            for (bit, &(_, pauli)) in terms.iter().enumerate() {
                let mask = 1 << (terms.len() - 1 - bit);
                match pauli {
                    Pauli::X => j ^= mask,
                    Pauli::Z => negative ^= i & mask != 0,
                    Pauli::Y => return Err("unexpected Y in real boundary oracle".into()),
                }
            }
            matrix[j * dim + i] = if negative { -1. } else { 1. };
        }
        let value = values(
            Observable::new(
                terms.iter().map(|&(q, _)| q).collect(),
                Matrix::real(dim, matrix)?,
            )?
            .expectation(&engine, &state)?,
        )?[0];
        let signed = if sign { -value } else { value };
        minimum = minimum.min(signed);
        ensure(signed > 1. - 2e-5, "native signed boundary correlation")?;
    }
    let mut metrics = vec![
        ("native_boundary_correlation_min", minimum),
        ("host_attempts_per_strip", 256.),
    ];
    let labels = [
        [
            "length1_correct",
            "length1_wrong",
            "length1_erased",
            "length1_inconsistent",
            "length1_failure_ci_low",
            "length1_failure_ci_high",
            "length1_photons",
            "length1_fusions",
            "length1_boundary_measurements",
        ],
        [
            "length2_correct",
            "length2_wrong",
            "length2_erased",
            "length2_inconsistent",
            "length2_failure_ci_low",
            "length2_failure_ci_high",
            "length2_photons",
            "length2_fusions",
            "length2_boundary_measurements",
        ],
        [
            "length3_correct",
            "length3_wrong",
            "length3_erased",
            "length3_inconsistent",
            "length3_failure_ci_low",
            "length3_failure_ci_high",
            "length3_photons",
            "length3_fusions",
            "length3_boundary_measurements",
        ],
    ];
    for (length, labels) in [1, 2, 3].into_iter().zip(labels) {
        let patch = SixRingPatch::bell_strip(length, NodeId(0), NodeId(1), 32)?;
        let plan = patch
            .program()
            .compile(&fabric, CompileOptions::default())?;
        let output: Vec<_> = patch.outputs().iter().map(|q| q.0).collect();
        let targets: Vec<_> = patch
            .relations(&plan, 1 << 20)?
            .output_relations(&output, 1 << 20)?
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        let middle = length / 2;
        let burst =
            PhotonLossBurst::new((6 * middle..6 * (middle + 1)).map(PhotonId).collect(), 0.1)?;
        let shots = patch.sample(
            &plan,
            0,
            256,
            884,
            PauliNoise::new(0.002, 0.002, 0.002)?,
            &[burst],
            &targets,
            1 << 20,
        )?;
        let mut counts = [0usize; 4];
        for shot in &shots {
            let category=if shot.inferred.contains(&ParityInference::Inconsistent) {3}else if shot.inferred.iter().any(|p|!matches!(p,ParityInference::Known(_))) {2}else if shot.inferred.iter().zip(&shot.expectations).all(|(p,&actual)|matches!(p,ParityInference::Known(parity) if actual==if *parity==Parity::Odd {-1}else {1})) {0}else {1};
            counts[category] += 1;
        }
        ensure(
            counts.iter().sum::<usize>() == 256,
            "open-boundary attempt accounting",
        )?;
        let ci = WilsonInterval::new(256 - counts[0], 256, 1.95996398454)?;
        for (name, value) in labels.into_iter().zip([
            counts[0] as f64,
            counts[1] as f64,
            counts[2] as f64,
            counts[3] as f64,
            ci.lower,
            ci.upper,
            patch.program().num_photons() as f64,
            patch.program().num_fusions() as f64,
            patch.boundary_measurements().len() as f64,
        ]) {
            metrics.push((name, value));
        }
    }
    Report::new("fbqc_open_boundaries", 884, 0, &metrics)
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn actual_boundary_instruments_and_native_oracle() {
        super::run().unwrap();
    }
}
