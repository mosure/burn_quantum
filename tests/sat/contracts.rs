//! Contracts conformance using independent references.
use super::*;

#[test]
fn schema_versions_and_control_validation_preserve_invariants() {
    let mut circuit = Circuit::new(Register::new(vec![2, 3, 2]).unwrap(), 0, 0);
    assert_eq!(serde_json::to_value(&circuit).unwrap()["version"], 1);
    for (controls, target) in [
        (vec![(0, 0), (0, 1)], 2),
        (vec![(0, 2)], 2),
        (vec![(9, 0)], 2),
        (vec![(2, 1)], 2),
        (vec![], 1),
        (vec![], 3),
    ] {
        assert!(circuit.controlled_x(controls, target).is_err());
        assert!(circuit.operations().is_empty());
    }
    assert!(circuit.phase_flip(vec![(1, 3)]).is_err());
    circuit
        .controlled_x(vec![(1, 2)], 0)
        .unwrap()
        .phase_flip(vec![(2, 0)])
        .unwrap();
    let mut json = serde_json::to_value(&circuit).unwrap();
    assert_eq!(json["version"], 2);
    assert_eq!(
        serde_json::from_value::<Circuit>(json.clone()).unwrap(),
        circuit
    );
    json["version"] = 1.into();
    assert!(serde_json::from_value::<Circuit>(json).is_err());
    let input = StateVector::zero(
        Register::qubits(3).unwrap(),
        1,
        &device(),
        DType::F32,
        ResourceLimits::default(),
    )
    .unwrap();
    assert!(
        circuit
            .run_from(&DenseSimulator::default(), input, None, None)
            .is_err()
    );
}

#[test]
fn limits_reject_oversized_oracles_and_plans_before_state_allocation() {
    let formula = Cnf::new(3, vec![vec![1], vec![2], vec![3]]).unwrap();
    for limits in [
        ResourceLimits {
            max_state_elements: 63,
            ..Default::default()
        },
        ResourceLimits {
            max_plan_indices: 127,
            ..Default::default()
        },
        ResourceLimits {
            max_matrix_elements: 3,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            GroverSat::new(formula.clone(), limits),
            Err(Error::ResourceLimit { .. })
        ));
    }
    assert!(matches!(
        GroverSat::new(
            Cnf::new(usize::BITS as usize, vec![]).unwrap(),
            ResourceLimits::default()
        ),
        Err(Error::Overflow)
    ));
    let huge_limits = ResourceLimits {
        max_state_elements: usize::MAX,
        max_plan_indices: usize::MAX,
        ..Default::default()
    };
    assert!(matches!(
        GroverSat::new(Cnf::new(31, vec![]).unwrap(), huge_limits),
        Err(Error::Unsupported(_))
    ));
    let sat = solver(2, vec![vec![1, 2]]);
    let state = sat.initial_state(&device()).unwrap();
    let limited = DenseSimulator::new(ResourceLimits {
        max_plan_indices: 0,
        ..Default::default()
    });
    assert!(limited.controlled_x(&state, &[], 0).is_err());
    assert!(limited.phase_flip(&state, &[]).is_err());
}

#[cfg(feature = "autodiff")]
#[test]
fn structured_gates_and_continuations_retain_native_gradients() {
    use burn_quantum::{gate::Axis, gradient::parameter_shift};
    let device = device().autodiff();
    let mut circuit = Circuit::new(Register::qubits(3).unwrap(), 1, 0);
    circuit.unitary(vec![0], Unitary::h()).unwrap();
    circuit
        .rotate(1, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    circuit.controlled_x(vec![(0, 1), (1, 0)], 2).unwrap();
    circuit.phase_flip(vec![(0, 0), (1, 1)]).unwrap();
    let engine = DenseSimulator::default();
    let parameter = Tensor::from_floats([0.4], &device).require_grad();
    let initial = StateVector::zero(
        circuit.register().clone(),
        1,
        &device,
        DType::F32,
        engine.limits,
    )
    .unwrap();
    let output = circuit
        .run_from(&engine, initial, Some(parameter.clone()), None)
        .unwrap();
    let expectation = Observable::z(2).expectation(&engine, &output).unwrap();
    let gradients = expectation.sum().backward();
    close(
        &values(parameter.grad(&gradients).unwrap()),
        &[0.5 * 0.4f64.sin()],
        5e-5,
    );
    close(
        &values(
            parameter_shift(
                &engine,
                &circuit,
                parameter,
                None,
                1,
                &device,
                &Observable::z(2),
            )
            .unwrap(),
        ),
        &[0.5 * 0.4f64.sin()],
        5e-5,
    );
}

#[test]
fn continuation_preserves_f64_precision_and_rejects_mismatched_bindings() {
    use burn_quantum::gate::Axis;
    let device = Device::flex();
    let register = Register::qubits(1).unwrap();
    let initial = StateVector::basis(
        register.clone(),
        &[1],
        1,
        &device,
        DType::F64,
        ResourceLimits::default(),
    )
    .unwrap();
    let mut fixed = Circuit::new(register.clone(), 0, 0);
    fixed
        .rotate(0, Rotation::qubit(Axis::Y, Angle::Constant(0.7)))
        .unwrap();
    let output = fixed
        .run_from(&DenseSimulator::default(), initial.clone(), None, None)
        .unwrap();
    assert_eq!(output.amplitudes().dtype(), DType::F64);
    close(
        &values(output.probabilities()),
        &[(0.35f64).sin().powi(2), (0.35f64).cos().powi(2)],
        1e-12,
    );
    let mut bound = Circuit::new(register, 1, 0);
    bound
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    assert!(matches!(
        bound.run_from(
            &DenseSimulator::default(),
            initial,
            Some(Tensor::from_floats([0.7], &device)),
            None
        ),
        Err(Error::TensorContext)
    ));
}
