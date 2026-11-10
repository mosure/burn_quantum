//! Unknown-input channel preservation under known-Z fusion failures and bounded retries.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::DType;
use burn_quantum::{
    photonic::{compiler::*, *},
    *,
};
use common::{close, complex_values, values};
fn plan(n: usize, gate: Unitary, transmission: f64) -> ParityRetryPlan {
    let f = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: RailTransmission::IDEAL,
            },
            FabricNode {
                kind: NodeKind::Source,
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
                transmission: RailTransmission::uniform(transmission).unwrap(),
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::IDEAL,
            },
        ],
    )
    .unwrap();
    ParityRetryPlan::compile(
        gate,
        n,
        &f,
        TeleportationPorts {
            input: NodeId(0),
            resource: NodeId(1),
            analyzer: NodeId(2),
        },
        2,
        CompileOptions::default(),
    )
    .unwrap()
}
fn bell() -> StateVector {
    let e = DenseSimulator::default();
    let d = runtime::execution().unwrap().device;
    let mut c = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    c.unitary(vec![0], Unitary::h()).unwrap();
    c.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    c.run_from(
        &e,
        StateVector::zero(c.register().clone(), 1, &d, DType::F32, e.limits).unwrap(),
        None,
        None,
    )
    .unwrap()
}
#[test]
fn every_successful_retry_preserves_complete_unknown_input_channel() {
    let e = DenseSimulator::default();
    for gate in [Unitary::h(), Unitary::t()] {
        let p = plan(2, gate.clone(), 1.);
        let input = bell();
        let mut ideal = Circuit::new(input.register().clone(), 0, 0);
        ideal.unitary(vec![0], gate).unwrap();
        let target = complex_values(&ideal.run_from(&e, input.clone(), None, None).unwrap());
        let run = p.execute_with_supplied_encoding(&e, input).unwrap();
        let mut total = 0.;
        let mut retry_success = 0.;
        for branch in run.branches() {
            let weight = values(branch.probability.clone())[0];
            total += weight;
            if branch.success {
                if branch.outcomes.len() == 2 {
                    retry_success += weight;
                }
                let state = branch.state.as_ref().unwrap();
                let real = values(state.entries().real());
                let imag = values(state.entries().imag());
                for row in 0..4 {
                    for col in 0..4 {
                        let r = state.register().encode(&[row / 2, row % 2]).unwrap();
                        let c = state.register().encode(&[col / 2, col % 2]).unwrap();
                        let expected = target[row] * target[col].conj() * weight;
                        assert!((real[r * 9 + c] - expected.re).abs() < 3e-6);
                        assert!((imag[r * 9 + c] - expected.im).abs() < 3e-6);
                    }
                }
            }
        }
        assert!((total - 1.).abs() < 3e-6);
        assert!((retry_success - 0.25).abs() < 3e-6);
        close(&values(run.herald_probability()), &[0.75], 3e-6);
        close(&values(run.expected_fusions()), &[1.5], 3e-6);
        close(&values(run.expected_resource_photons()), &[3.], 3e-6);
    }
}
#[test]
fn loss_aborts_without_reusing_destroyed_information() {
    let e = DenseSimulator::default();
    let r = plan(2, Unitary::h(), 0.)
        .execute_with_supplied_encoding(&e, bell())
        .unwrap();
    close(&values(r.herald_probability()), &[0.], 1e-7);
    close(&values(r.expected_fusions()), &[1.], 1e-6);
    let loss = r
        .branches()
        .iter()
        .filter(|b| matches!(b.outcomes.last(), Some(FusionOutcome::Loss)))
        .map(|b| values(b.probability.clone())[0])
        .sum::<f64>();
    assert!((loss - 1.).abs() < 1e-6);
    for b in r.branches() {
        if matches!(b.outcomes.first(), Some(FusionOutcome::Loss)) {
            assert_eq!(b.outcomes.len(), 1);
        }
    }
}
#[test]
fn three_port_retry_has_bounded_geometric_yield_and_cost() {
    let e = DenseSimulator::default();
    let d = runtime::execution().unwrap().device;
    let input =
        StateVector::zero(Register::qubits(1).unwrap(), 1, &d, DType::F32, e.limits).unwrap();
    let r = plan(3, Unitary::t(), 1.)
        .execute_with_supplied_encoding(&e, input)
        .unwrap();
    close(&values(r.herald_probability()), &[0.875], 5e-6);
    close(&values(r.expected_resource_photons()), &[3.5], 5e-6);
    assert!(r.branches().iter().all(|b| b.outcomes.len() <= 3));
}
#[test]
fn feedback_storage_precedes_all_outcome_dependent_output_corrections() {
    let eta = RailTransmission::new(0.8, 0.4).unwrap();
    let f = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: RailTransmission::IDEAL,
            },
            FabricNode {
                kind: NodeKind::Source,
                storage: eta,
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
                transmission: RailTransmission::IDEAL,
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::IDEAL,
            },
        ],
    )
    .unwrap();
    let plan = ParityRetryPlan::compile(
        Unitary::h(),
        2,
        &f,
        TeleportationPorts {
            input: NodeId(0),
            resource: NodeId(1),
            analyzer: NodeId(2),
        },
        2,
        CompileOptions::default(),
    )
    .unwrap();
    let e = DenseSimulator::default();
    let run = plan.execute_with_supplied_encoding(&e, bell()).unwrap();
    let survival = run
        .branches()
        .iter()
        .filter(|b| b.success)
        .map(|b| {
            values(
                b.state
                    .as_ref()
                    .unwrap()
                    .expectation(&e, &DualRail::survival(0))
                    .unwrap(),
            )[0]
        })
        .sum::<f64>();
    let oracle = 0.75 * (0.8f64.powi(5) + 0.4f64.powi(5)) / 2.;
    assert!(
        (survival - oracle).abs() < 4e-6,
        "survival={survival},oracle={oracle}"
    );
}
