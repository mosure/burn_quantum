//! Independent channel and gradient tests for causal measurement-pattern lowering.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Tensor};
use burn_quantum::gate::Axis;
use burn_quantum::{photonic::compiler::*, *};
use common::{close, values};
use num_complex::Complex64;

fn device() -> burn::tensor::Device {
    runtime::execution().unwrap().device
}
fn zero(n: usize) -> StateVector {
    StateVector::zero(
        Register::qubits(n).unwrap(),
        1,
        &device(),
        DType::F32,
        ResourceLimits::default(),
    )
    .unwrap()
}
fn choi_input(n: usize) -> StateVector {
    let mut circuit = Circuit::new(Register::qubits(2 * n).unwrap(), 0, 0);
    for i in 0..n {
        circuit.unitary(vec![i], Unitary::h()).unwrap();
        circuit.unitary(vec![i, i + n], Unitary::cnot()).unwrap();
    }
    circuit
        .run_from(&DenseSimulator::default(), zero(2 * n), None, None)
        .unwrap()
}
fn all_channel_branches(circuit: &Circuit, input: StateVector) {
    let engine = DenseSimulator::default();
    let expected = circuit
        .run_from(&engine, input.clone(), None, None)
        .unwrap();
    let pattern = MeasurementPattern::compile(circuit, PatternCompileLimits::default()).unwrap();
    let branches = PatternSimulator::default()
        .execute_all(&pattern, input, None, None)
        .unwrap();
    assert_eq!(branches.len(), 1 << pattern.measurements());
    let p = 1. / branches.len() as f64;
    let mut total = 0.;
    for branch in branches {
        close(&values(branch.probability()), &[p], 2e-6);
        total += values(branch.probability())[0];
        let actual = branch.conditional_state(1e-12).unwrap();
        close(&values(actual.fidelity(&expected).unwrap()), &[1.], 3e-5);
    }
    assert!((total - 1.).abs() < 2e-5);
}
#[test]
fn teleportation_and_entangling_patterns_match_complete_choi_channels_in_every_branch() {
    let mut h = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    h.unitary(vec![0], Unitary::h()).unwrap();
    all_channel_branches(&h, choi_input(1));
    let mut cnot = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
    cnot.unitary(vec![1, 0], Unitary::cnot()).unwrap();
    all_channel_branches(&cnot, choi_input(2));
    let mut cz = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
    cz.phase_flip(vec![(0, 1), (1, 1)]).unwrap();
    all_channel_branches(&cz, choi_input(2));
    let mut swap = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
    swap.unitary(vec![0, 1], Unitary::swap(2).unwrap()).unwrap();
    all_channel_branches(&swap, choi_input(2));
}
#[test]
fn symbolic_bindings_feed_forward_and_gradients_survive_lowering() {
    let dev = device().autodiff();
    let engine = DenseSimulator::default();
    let mut prep = Circuit::new(Register::qubits(1).unwrap(), 0, 0);
    prep.unitary(vec![0], Unitary::h()).unwrap();
    let initial = prep
        .run_from(
            &engine,
            StateVector::zero(
                Register::qubits(1).unwrap(),
                1,
                &dev,
                DType::F32,
                engine.limits,
            )
            .unwrap(),
            None,
            None,
        )
        .unwrap();
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 1);
    circuit
        .rotate(0, Rotation::qubit(Axis::Z, Angle::input(0)))
        .unwrap();
    circuit
        .rotate(
            0,
            Rotation::qubit(
                Axis::X,
                Angle::Parameter {
                    index: 0,
                    scale: 2.,
                    offset: 0.1,
                },
            ),
        )
        .unwrap();
    let pattern = MeasurementPattern::compile(&circuit, PatternCompileLimits::default()).unwrap();
    assert_eq!(pattern.source(), &circuit);
    assert_eq!(pattern.measurements(), 4);
    assert_eq!(pattern.sites(), 5);
    assert_eq!(pattern.outputs(), &[PatternSite(4)]);
    let p = Tensor::from_data([0.2f32], &dev).require_grad();
    let x = Tensor::from_data([[0.3f32]], &dev).require_grad();
    let branch = PatternSimulator::default()
        .execute_branch(
            &pattern,
            initial,
            Some(p.clone()),
            Some(x.clone()),
            &[true, false, true, true],
        )
        .unwrap();
    close(&values(branch.probability()), &[1. / 16.], 2e-6);
    assert_eq!(branch.outcomes(), &[true, false, true, true]);
    let z = engine
        .expectation(&branch.conditional_state(1e-8).unwrap(), &Observable::z(0))
        .unwrap();
    close(&values(z.clone()), &[0.5f64.sin() * 0.3f64.sin()], 3e-6);
    let grads = z.sum().backward();
    close(
        &values(p.grad(&grads).unwrap()),
        &[2. * 0.5f64.cos() * 0.3f64.sin()],
        3e-5,
    );
    close(
        &values(x.grad(&grads).unwrap()),
        &[0.5f64.sin() * 0.3f64.cos()],
        3e-5,
    );
}
#[test]
fn arbitrary_local_unitaries_reversed_generators_and_control_polarities_preserve_channels() {
    let a = Complex64::new(0.3, 0.4);
    let b = Complex64::from_polar(0.75f64.sqrt(), 0.37);
    let unitary = Unitary::new(Matrix::new(2, vec![a, b, -b.conj(), a.conj()]).unwrap()).unwrap();
    let mut recipes = vec![];
    let mut arbitrary = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    arbitrary.unitary(vec![0], unitary).unwrap();
    recipes.push(arbitrary);
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        let mut circuit = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
        circuit
            .rotate(
                0,
                Rotation {
                    axis,
                    levels: [1, 0],
                    angle: Angle::Constant(0.73),
                },
            )
            .unwrap();
        recipes.push(circuit);
    }
    let mut controlled = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
    controlled.controlled_x(vec![(1, 0)], 0).unwrap();
    recipes.push(controlled);
    for circuit in recipes {
        let input = choi_input(circuit.register().wires() / 2);
        let expected = circuit
            .run_from(&DenseSimulator::default(), input.clone(), None, None)
            .unwrap();
        let pattern =
            MeasurementPattern::compile(&circuit, PatternCompileLimits::default()).unwrap();
        for offset in 0..3 {
            let record: Vec<_> = (0..pattern.measurements())
                .map(|i| match offset {
                    0 => false,
                    1 => true,
                    _ => i % 2 == 1,
                })
                .collect();
            let branch = PatternSimulator::default()
                .execute_branch(&pattern, input.clone(), None, None, &record)
                .unwrap();
            let state = branch.conditional_state(1e-8).unwrap();
            close(&values(state.fidelity(&expected).unwrap()), &[1.], 5e-5);
        }
    }
}
#[test]
fn compiler_and_executor_reject_unsupported_semantics_and_expansion_before_work() {
    let mut circuit = Circuit::new(Register::qubits(3).unwrap(), 0, 0);
    circuit.controlled_x(vec![(0, 1), (1, 1)], 2).unwrap();
    assert!(MeasurementPattern::compile(&circuit, PatternCompileLimits::default()).is_ok());
    assert!(
        MeasurementPattern::compile(
            &circuit,
            PatternCompileLimits {
                max_commands: 2,
                ..Default::default()
            }
        )
        .is_err()
    );
    let qutrit = Circuit::new(Register::new(vec![3]).unwrap(), 0, 0);
    assert!(MeasurementPattern::compile(&qutrit, PatternCompileLimits::default()).is_err());
    let mut h = Circuit::new(Register::qubits(1).unwrap(), 0, 0);
    h.unitary(vec![0], Unitary::h()).unwrap();
    assert!(
        MeasurementPattern::compile(
            &h,
            PatternCompileLimits {
                max_sites: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        MeasurementPattern::compile(
            &h,
            PatternCompileLimits {
                max_commands: 3,
                ..Default::default()
            }
        )
        .is_err()
    );
    let pattern = MeasurementPattern::compile(&h, PatternCompileLimits::default()).unwrap();
    let simulator = PatternSimulator {
        limits: PatternSimulationLimits {
            max_branches: 1,
            ..Default::default()
        },
    };
    assert!(
        simulator
            .execute_all(&pattern, zero(1), None, None)
            .is_err()
    );
    assert!(
        simulator
            .execute_branch(&pattern, zero(1), None, None, &[])
            .is_err()
    );
    assert!(
        simulator
            .execute_branch(&pattern, zero(2), None, None, &[false])
            .is_err()
    );
    // A selected transcript must not silently become zero-probability through
    // F32 underflow even when exponential enumeration is bypassed.
    let mut long = Circuit::new(Register::qubits(1).unwrap(), 0, 0);
    for _ in 0..120 {
        long.unitary(vec![0], Unitary::h()).unwrap();
    }
    let long = MeasurementPattern::compile(&long, PatternCompileLimits::default()).unwrap();
    assert!(
        simulator
            .execute_branch(&long, zero(1), None, None, &[false; 120])
            .is_err()
    );
    let encoded = serde_json::to_string(&pattern).unwrap();
    assert!(encoded.contains("CorrectX") && encoded.contains("Measure"));
}

fn routed_fabric(eta_input: f64, eta_resource: f64) -> burn_quantum::photonic::Fabric {
    use burn_quantum::photonic::*;
    let ideal = RailTransmission::IDEAL;
    Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Source,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Switch,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: ideal,
            },
        ],
        vec![
            FabricLink {
                from: NodeId(0),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::uniform(eta_input).unwrap(),
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::uniform(eta_resource).unwrap(),
            },
            FabricLink {
                from: NodeId(2),
                to: NodeId(3),
                ticks: 2,
                transmission: ideal,
            },
        ],
    )
    .unwrap()
}
fn gate_plan(gate: Unitary, eta_input: f64, eta_resource: f64) -> HeraldedGatePlan {
    use burn_quantum::photonic::*;
    HeraldedGatePlan::compile(
        gate,
        &routed_fabric(eta_input, eta_resource),
        TeleportationPorts {
            input: NodeId(0),
            resource: NodeId(1),
            analyzer: NodeId(3),
        },
        CompileOptions::default(),
    )
    .unwrap()
}
fn encoded_choi() -> DensityMatrix {
    use burn_quantum::photonic::*;
    let engine = DenseSimulator::default();
    let state = ResourceState::bell_pair()
        .circuit()
        .unwrap()
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
    DensityMatrix::from_pure(&state, engine.limits).unwrap()
}
#[test]
fn routed_heralded_gates_preserve_full_entangled_input_channels_and_failure_records() {
    use burn_quantum::photonic::*;
    let engine = DenseSimulator::default();
    let input = encoded_choi();
    for gate in [Unitary::h(), Unitary::t(), Unitary::y()] {
        for target in [0, 1] {
            let plan = gate_plan(gate.clone(), 1., 1.);
            assert_eq!(plan.routes()[0].arrival, 3);
            assert_eq!(plan.routes()[1].arrival, 5);
            assert_eq!(plan.finish_tick(), 6);
            let expected = input
                .apply(&engine, &[target], &DualRail::unitary(&gate).unwrap())
                .unwrap();
            let branches = plan.execute(&engine, &input, target).unwrap();
            close(
                &branches
                    .iter()
                    .map(|b| values(b.probability())[0])
                    .collect::<Vec<_>>(),
                &[0.25, 0.25, 0.25, 0.25, 0.],
                3e-6,
            );
            for branch in &branches[..2] {
                let state = branch.conditional_state(1e-8).unwrap().unwrap();
                close(
                    &values(state.entries().real()),
                    &values(expected.entries().real()),
                    3e-6,
                );
                close(
                    &values(state.entries().imag()),
                    &values(expected.entries().imag()),
                    3e-6,
                );
            }
            assert!(matches!(
                branches[2].outcomes()[0],
                FusionOutcome::Failure { .. }
            ));
            assert_eq!(branches[4].outcomes()[0], FusionOutcome::Loss);
        }
    }
}
#[test]
fn heralded_gate_loss_and_preflight_are_explicit() {
    let input = encoded_choi();
    let plan = gate_plan(Unitary::h(), 0.8, 0.9);
    let branches = plan.execute(&DenseSimulator::default(), &input, 0).unwrap();
    close(
        &branches
            .iter()
            .map(|b| values(b.probability())[0])
            .collect::<Vec<_>>(),
        &[0.18, 0.18, 0.18, 0.18, 0.28],
        3e-6,
    );
    let limited = DenseSimulator::new(ResourceLimits {
        max_state_elements: 100,
        ..Default::default()
    });
    assert!(plan.execute(&limited, &input, 0).is_err());
    assert!(plan.execute(&DenseSimulator::default(), &input, 2).is_err());
    use burn_quantum::photonic::*;
    assert!(
        HeraldedGatePlan::compile(
            Unitary::cnot(),
            &routed_fabric(1., 1.),
            TeleportationPorts {
                input: NodeId(0),
                resource: NodeId(1),
                analyzer: NodeId(3)
            },
            CompileOptions::default()
        )
        .is_err()
    );
    assert!(
        HeraldedGatePlan::compile(
            Unitary::h(),
            &routed_fabric(1., 1.),
            TeleportationPorts {
                input: NodeId(0),
                resource: NodeId(1),
                analyzer: NodeId(3)
            },
            CompileOptions {
                max_ticks: 5,
                ..Default::default()
            }
        )
        .is_err()
    );
}
#[test]
fn injected_tensor_gradients_survive_physical_gate_teleportation() {
    use burn_quantum::photonic::*;
    let dev = device().autodiff();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(DualRail::register(1).unwrap(), 1, 0);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let theta = Tensor::from_data([0.7f32], &dev).require_grad();
    let state = circuit
        .run(&engine, Some(theta.clone()), None, 1, &dev)
        .unwrap();
    let input = DensityMatrix::from_pure(&state, engine.limits).unwrap();
    let plan = gate_plan(Unitary::h(), 0.8, 0.9);
    let branch = plan.execute(&engine, &input, 0).unwrap().remove(0);
    let z = branch
        .conditional_state(1e-8)
        .unwrap()
        .unwrap()
        .expectation(&engine, &DualRail::z(0))
        .unwrap();
    close(&values(z.clone()), &[0.7f64.sin()], 3e-6);
    let grads = z.sum().backward();
    close(&values(theta.grad(&grads).unwrap()), &[0.7f64.cos()], 3e-5);
}
