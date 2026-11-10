//! Physical gate-teleportation channels, compositional loss, and native gradients.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Tensor};
use burn_quantum::{
    gate::Axis,
    photonic::{compiler::*, *},
    *,
};
use common::{close, complex_values, values};
use num_complex::Complex64;
fn device() -> burn::tensor::Device {
    runtime::execution().unwrap().device
}
fn fabric(a: RailTransmission, b: RailTransmission, storage: RailTransmission) -> Fabric {
    Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: RailTransmission::IDEAL,
            },
            FabricNode {
                kind: NodeKind::Source,
                storage,
            },
            FabricNode {
                kind: NodeKind::Switch,
                storage: RailTransmission::IDEAL,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: RailTransmission::IDEAL,
            },
        ],
        vec![
            FabricLink {
                from: NodeId(0),
                to: NodeId(2),
                ticks: 1,
                transmission: a,
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: b,
            },
            FabricLink {
                from: NodeId(2),
                to: NodeId(3),
                ticks: 2,
                transmission: RailTransmission::IDEAL,
            },
        ],
    )
    .unwrap()
}
fn compile(circuit: &Circuit, fabric: &Fabric) -> PhysicalCircuit {
    PhysicalCircuit::compile(
        circuit,
        fabric,
        PhysicalPlacement {
            inputs: vec![NodeId(0); circuit.register().wires()],
            resource: NodeId(1),
            analyzer: NodeId(3),
            feedback_ticks: 0,
        },
        CompileOptions::default(),
    )
    .unwrap()
}
fn choi(n: usize) -> StateVector {
    let dev = device();
    let engine = DenseSimulator::default();
    let mut prep = Circuit::new(Register::qubits(2 * n).unwrap(), 0, 0);
    for i in 0..n {
        prep.unitary(vec![i], Unitary::h()).unwrap();
        prep.unitary(vec![i, n + i], Unitary::cnot()).unwrap();
    }
    prep.run_from(
        &engine,
        StateVector::zero(prep.register().clone(), 1, &dev, DType::F32, engine.limits).unwrap(),
        None,
        None,
    )
    .unwrap()
}
fn compare_encoded_density(actual: &DensityMatrix, expected: &StateVector, weight: f64) {
    let vector = complex_values(expected);
    let encoded = actual.register();
    let mut oracle = vec![Complex64::new(0., 0.); encoded.size() * encoded.size()];
    for row in 0..vector.len() {
        for col in 0..vector.len() {
            let r = encoded
                .encode(&expected.register().decode(row).unwrap())
                .unwrap();
            let c = encoded
                .encode(&expected.register().decode(col).unwrap())
                .unwrap();
            oracle[r * encoded.size() + c] = vector[row] * vector[col].conj() * weight;
        }
    }
    close(
        &values(actual.entries().real()),
        &oracle.iter().map(|v| v.re).collect::<Vec<_>>(),
        4e-6,
    );
    close(
        &values(actual.entries().imag()),
        &oracle.iter().map(|v| v.im).collect::<Vec<_>>(),
        4e-6,
    );
}
#[test]
fn physical_two_qubit_gadgets_match_complete_entangled_reference_channels() {
    let engine = DenseSimulator::default();
    let f = fabric(
        RailTransmission::IDEAL,
        RailTransmission::IDEAL,
        RailTransmission::IDEAL,
    );
    for gate in [
        Unitary::cnot(),
        Unitary::z().controlled(2, 1).unwrap(),
        Unitary::swap(2).unwrap(),
    ] {
        for wires in [vec![0, 1], vec![1, 0]] {
            let mut circuit = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
            circuit.unitary(wires, gate.clone()).unwrap();
            let input = choi(2);
            let expected = circuit
                .run_from(&engine, input.clone(), None, None)
                .unwrap();
            let plan = compile(&circuit, &f);
            let run = plan.execute(&engine, input, None, None).unwrap();
            close(&values(run.herald_probability()), &[0.25], 3e-6);
            close(&values(run.abort_probability()), &[0.75], 3e-6);
            assert_eq!(run.outcomes().len(), 25);
            assert_eq!(plan.resource_photons(), 4);
            for record in run.outcomes().iter().filter(|r| {
                r.outcomes
                    .iter()
                    .all(|o| matches!(o, FusionOutcome::Success { .. }))
            }) {
                close(&values(record.probability.clone()), &[1. / 16.], 2e-6);
            }
            compare_encoded_density(run.state(), &expected, 0.25);
        }
    }
}
#[test]
fn composed_symbolic_resources_preserve_gradients_and_abort_cost() {
    let dev = device().autodiff();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(Register::qubits(2).unwrap(), 1, 1);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    circuit
        .rotate(1, Rotation::qubit(Axis::Y, Angle::input(0)))
        .unwrap();
    circuit.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    let plan = compile(
        &circuit,
        &fabric(
            RailTransmission::IDEAL,
            RailTransmission::IDEAL,
            RailTransmission::IDEAL,
        ),
    );
    let theta = Tensor::from_data([0.7f32], &dev).require_grad();
    let x = Tensor::from_data([[0.4f32]], &dev).require_grad();
    let initial = StateVector::zero(
        Register::qubits(2).unwrap(),
        1,
        &dev,
        DType::F32,
        engine.limits,
    )
    .unwrap();
    let run = plan
        .execute(&engine, initial, Some(theta.clone()), Some(x.clone()))
        .unwrap();
    close(&values(run.herald_probability()), &[1. / 16.], 3e-6);
    close(&values(run.abort_probability()), &[15. / 16.], 3e-6);
    close(&values(run.expected_resource_photons()), &[4.], 3e-6);
    close(&values(run.expected_fusions()), &[2.], 3e-6);
    let z = run.state().expectation(&engine, &DualRail::z(1)).unwrap() / run.herald_probability();
    close(&values(z.clone()), &[0.7f64.cos() * 0.4f64.cos()], 3e-5);
    let grad = z.sum().backward();
    close(
        &values(theta.grad(&grad).unwrap()),
        &[-0.7f64.sin() * 0.4f64.cos()],
        5e-5,
    );
    close(
        &values(x.grad(&grad).unwrap()),
        &[-0.7f64.cos() * 0.4f64.sin()],
        5e-5,
    );
}
#[test]
fn compressed_channel_matches_explicit_resource_fusion_under_asymmetric_loss() {
    let engine = DenseSimulator::default();
    let f = fabric(
        RailTransmission::new(0.8, 0.2).unwrap(),
        RailTransmission::new(0.9, 0.7).unwrap(),
        RailTransmission::new(0.99, 0.98).unwrap(),
    );
    for gate in [Unitary::h(), Unitary::t()] {
        let mut circuit = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
        circuit.unitary(vec![0], gate.clone()).unwrap();
        let plan = compile(&circuit, &f);
        let run = plan.execute(&engine, choi(1), None, None).unwrap();
        let resource = ResourceState::bell_pair().circuit().unwrap();
        let encoded = resource
            .run_from(
                &engine,
                StateVector::zero(
                    DualRail::register(2).unwrap(),
                    1,
                    &device(),
                    DType::F32,
                    engine.limits,
                )
                .unwrap(),
                None,
                None,
            )
            .unwrap();
        let input = DensityMatrix::from_pure(&encoded, engine.limits).unwrap();
        let explicit = HeraldedGatePlan::compile(
            gate,
            &f,
            TeleportationPorts {
                input: NodeId(0),
                resource: NodeId(1),
                analyzer: NodeId(3),
            },
            CompileOptions::default(),
        )
        .unwrap()
        .execute(&engine, &input, 0)
        .unwrap();
        let re = explicit[0].unnormalized_state().unwrap().entries().real()
            + explicit[1].unnormalized_state().unwrap().entries().real();
        let im = explicit[0].unnormalized_state().unwrap().entries().imag()
            + explicit[1].unnormalized_state().unwrap().entries().imag();
        close(&values(run.state().entries().real()), &values(re), 4e-6);
        close(&values(run.state().entries().imag()), &values(im), 4e-6);
        for (observed, expected) in run.outcomes().iter().zip(&explicit) {
            close(
                &values(observed.probability.clone()),
                &values(expected.probability()),
                3e-6,
            );
        }
        assert!(
            values(run.logical_survival_probability().unwrap())[0]
                < values(run.herald_probability())[0]
        );
    }
}
#[test]
fn latent_output_loss_is_not_used_as_a_control_or_discarded() {
    let f = fabric(
        RailTransmission::IDEAL,
        RailTransmission::IDEAL,
        RailTransmission::uniform(0.).unwrap(),
    );
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 0, 0);
    circuit.unitary(vec![0], Unitary::h()).unwrap();
    let engine = DenseSimulator::default();
    let zero = || {
        StateVector::zero(
            Register::qubits(1).unwrap(),
            1,
            &device(),
            DType::F32,
            engine.limits,
        )
        .unwrap()
    };
    // With zero storage the first input waits at the ideal analyzer, the resource
    // measured half departs immediately, and only the unmeasured output is erased.
    let run = compile(&circuit, &f)
        .execute(&engine, zero(), None, None)
        .unwrap();
    close(&values(run.herald_probability()), &[0.5], 3e-6);
    close(
        &values(run.logical_survival_probability().unwrap()),
        &[0.],
        1e-7,
    );
    circuit.unitary(vec![0], Unitary::x()).unwrap();
    let run = compile(&circuit, &f)
        .execute(&engine, zero(), None, None)
        .unwrap();
    close(&values(run.herald_probability()), &[0.], 1e-7);
    close(&values(run.abort_probability()), &[1.], 3e-6);
    close(&values(run.expected_resource_photons()), &[3.], 3e-6);
    assert!(
        run.outcomes()
            .iter()
            .filter(|r| r.gate == 1)
            .any(|r| matches!(r.outcomes[0], FusionOutcome::Loss)
                && values(r.probability.clone())[0] > 0.49)
    );
}
#[test]
fn physical_universal_synthesis_and_budgets_are_explicit() {
    let f = fabric(
        RailTransmission::IDEAL,
        RailTransmission::IDEAL,
        RailTransmission::IDEAL,
    );
    let mut circuit = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    circuit
        .unitary(vec![0, 1], Unitary::t().controlled(2, 1).unwrap())
        .unwrap();
    let placement = PhysicalPlacement {
        inputs: vec![NodeId(0); 2],
        resource: NodeId(1),
        analyzer: NodeId(3),
        feedback_ticks: 5,
    };
    assert!(
        PhysicalCircuit::compile(&circuit, &f, placement.clone(), CompileOptions::default())
            .is_ok()
    );
    let mut circuit = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    circuit.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    assert!(
        PhysicalCircuit::compile(
            &circuit,
            &f,
            placement.clone(),
            CompileOptions {
                max_photons: 5,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        PhysicalCircuit::compile(
            &circuit,
            &f,
            placement,
            CompileOptions {
                max_ticks: 5,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn joint_failure_and_loss_records_match_scalar_effects_on_nonuniform_input() {
    let a = RailTransmission::new(0.8, 0.2).unwrap();
    let b = RailTransmission::new(0.9, 0.7).unwrap();
    let f = fabric(a, b, RailTransmission::IDEAL);
    let mut circuit = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    circuit.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    let engine = DenseSimulator::default();
    let input = StateVector::basis(
        Register::qubits(2).unwrap(),
        &[0, 1],
        1,
        &device(),
        DType::F32,
        engine.limits,
    )
    .unwrap();
    let run = compile(&circuit, &f)
        .execute(&engine, input, None, None)
        .unwrap();
    let first = [
        0.8 * 0.7 / 4.,
        0.8 * 0.7 / 4.,
        0.8 * 0.9 / 2.,
        0.,
        1. - 0.8 * (0.9 + 0.7) / 2.,
    ];
    let second = [
        0.2 * 0.9 / 4.,
        0.2 * 0.9 / 4.,
        0.,
        0.2 * 0.7 / 2.,
        1. - 0.2 * (0.9 + 0.7) / 2.,
    ];
    for (i, record) in run.outcomes().iter().enumerate() {
        close(
            &values(record.probability.clone()),
            &[first[i / 5] * second[i % 5]],
            3e-6,
        );
    }
    close(
        &values(run.herald_probability() + run.abort_probability()),
        &[1.],
        3e-6,
    );
}
