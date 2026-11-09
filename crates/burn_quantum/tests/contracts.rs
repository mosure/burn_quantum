//! Invalid-input, serialization, representation, and resource-boundary checks.
mod common;
use burn::tensor::{DType, Device, Tensor};
use burn_quantum::{gate::Axis, *};
use common::*;

#[test]
fn indexing_and_overflow_are_explicit() {
    let register = Register::new(vec![2, 3, 4]).unwrap();
    assert_eq!(register.size(), 24);
    assert_eq!(register.encode(&[1, 2, 3]).unwrap(), 23);
    for index in 0..24 {
        assert_eq!(
            register.encode(&register.decode(index).unwrap()).unwrap(),
            index
        );
    }
    assert!(register.encode(&[2, 0, 0]).is_err());
    assert!(register.encode(&[0]).is_err());
    assert!(register.decode(24).is_err());
    assert!(Register::new(vec![]).is_err());
    assert!(Register::new(vec![1]).is_err());
    assert_eq!(
        Register::new(vec![usize::MAX, 2]).unwrap_err(),
        Error::Overflow
    );
    assert_eq!(Register::qubits(usize::MAX).unwrap_err(), Error::Overflow);
}

#[test]
fn resources_reject_before_allocation_and_support_large_local_registers() {
    let limits = ResourceLimits {
        max_state_elements: 16,
        max_matrix_elements: 4,
        max_plan_indices: 16,
    };
    let device = Device::flex();
    let engine = DenseSimulator::new(limits);
    assert!(
        StateVector::zero(
            Register::qubits(30).unwrap(),
            1,
            &device,
            DType::F32,
            limits
        )
        .is_err()
    );
    let state =
        StateVector::zero(Register::qubits(3).unwrap(), 1, &device, DType::F32, limits).unwrap();
    assert!(engine.apply(&state, &[0, 2], &Unitary::cnot()).is_err());
    assert!(DensityMatrix::from_pure(&state, limits).is_err());
    assert!(Unitary::fourier(usize::MAX).is_err());
    assert!(limits.check_state(usize::MAX, 2).is_err());
    // The register itself doesn't impose the dense engine's budget.
    assert_eq!(Register::qubits(40).unwrap().size(), 1usize << 40);
}

#[test]
fn invalid_operators_and_targets_fail_without_mutating_circuit() {
    assert!(Matrix::new(2, vec![1.0.into()]).is_err());
    assert!(Matrix::real(2, vec![1., 0., 0., f64::NAN]).is_err());
    assert!(Unitary::new(Matrix::real(2, vec![1., 0., 0., 2.]).unwrap()).is_err());
    assert!(Observable::new(vec![0], Matrix::real(2, vec![0., 1., 0., 0.]).unwrap()).is_err());
    let mut circuit = Circuit::new(Register::new(vec![2, 3]).unwrap(), 1, 1);
    assert!(circuit.unitary(vec![0, 0], Unitary::cnot()).is_err());
    assert!(circuit.unitary(vec![1], Unitary::h()).is_err());
    assert!(circuit.unitary(vec![2], Unitary::h()).is_err());
    assert!(
        circuit
            .rotate(0, Rotation::qubit(Axis::X, Angle::parameter(1)))
            .is_err()
    );
    assert!(
        circuit
            .rotate(
                1,
                Rotation {
                    axis: Axis::Y,
                    levels: [1, 1],
                    angle: Angle::Constant(0.)
                }
            )
            .is_err()
    );
    assert!(circuit.operations().is_empty());
    for p in [-0.1, 1.1, f64::NAN] {
        assert!(KrausChannel::amplitude_damping(p).is_err());
    }
}

#[test]
fn circuit_serialization_roundtrip_and_invariant_validation() {
    let mut circuit = Circuit::new(Register::new(vec![2, 3]).unwrap(), 1, 1);
    circuit
        .unitary(vec![1], Unitary::fourier(3).unwrap())
        .unwrap();
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let json = serde_json::to_string(&circuit).unwrap();
    let restored: Circuit = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, circuit);
    let mut object: serde_json::Value = serde_json::from_str(&json).unwrap();
    object["version"] = 99.into();
    assert!(serde_json::from_value::<Circuit>(object).is_err());
    let mut object: serde_json::Value = serde_json::from_str(&json).unwrap();
    object["register"] = serde_json::json!([2, 0]);
    assert!(serde_json::from_value::<Circuit>(object).is_err());
    let mut object: serde_json::Value = serde_json::from_str(&json).unwrap();
    object["operations"][0]["Unitary"]["wires"] = serde_json::json!([9]);
    assert!(serde_json::from_value::<Circuit>(object).is_err());
    let serialized = serde_json::to_string(&Unitary::x()).unwrap();
    let mut object: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    object["entries"][0] = serde_json::json!([2., 0.]);
    assert!(serde_json::from_value::<Unitary>(object).is_err());
}

#[test]
fn tensor_metadata_bindings_and_sampling_reject_invalid_input() {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    assert!(
        ComplexTensor::new(
            Tensor::<2>::zeros([1, 2], &device),
            Tensor::zeros([2, 2], &device)
        )
        .is_err()
    );
    assert!(
        ComplexTensor::new(
            Tensor::<2>::zeros([1, 2], (&device, DType::F32)),
            Tensor::zeros([1, 2], (&device, DType::F64))
        )
        .is_err()
    );
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 1);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    assert!(circuit.run(&engine, None, None, 1, &device).is_err());
    assert!(
        circuit
            .run(
                &engine,
                Some(Tensor::from_floats([0.], &device)),
                Some(Tensor::zeros([2, 1], &device)),
                1,
                &device
            )
            .is_err()
    );
    assert!(circuit.run(&engine, None, None, 0, &device).is_err());
    assert!(
        Samples::from_probabilities(Tensor::from_floats([[0.2, 0.2]], &device), 1, 0, 1).is_err()
    );
    assert!(
        Samples::from_probabilities(Tensor::from_floats([[-0.1, 1.1]], &device), 1, 0, 1).is_err()
    );
    assert!(
        Samples::from_probabilities(Tensor::from_floats([[0.5, 0.5]], &device), 100, 0, 10)
            .is_err()
    );
}

#[test]
fn supported_capabilities_are_honest() {
    let capabilities = DenseSimulator::default().capabilities();
    assert!(capabilities.mixed_qudits);
    assert!(capabilities.density_matrices);
    assert!(!capabilities.dynamic_control);
    let device = Device::flex();
    let state = StateVector::zero(
        Register::qubits(1).unwrap(),
        1,
        &device,
        DType::F64,
        ResourceLimits::default(),
    )
    .unwrap();
    let invalid = StateVector::from_amplitudes(
        state.register().clone(),
        ComplexTensor::new(state.amplitudes().real() * 2., state.amplitudes().imag()).unwrap(),
        ResourceLimits::default(),
    )
    .unwrap();
    assert!(invalid.validate(1e-8).is_err());
    close(&values(state.probabilities()), &[1., 0.], 1e-12);
}

#[test]
fn conflicting_autodiff_contexts_are_rejected_across_complex_components() {
    let plain = Device::flex();
    let ordinary = plain.clone().autodiff();
    let checkpointed = plain.clone().autodiff().gradient_checkpointing();
    assert!(
        ComplexTensor::new(
            Tensor::<2>::zeros([1, 2], &ordinary),
            Tensor::zeros([1, 2], &checkpointed)
        )
        .is_err()
    );
    let amplitudes = ComplexTensor::new(
        Tensor::<2>::zeros([1, 2], &plain),
        Tensor::zeros([1, 2], &checkpointed),
    )
    .unwrap();
    assert_eq!(
        amplitudes.device().gradient_checkpointing_strategy(),
        checkpointed.gradient_checkpointing_strategy()
    );
    let state = StateVector::from_amplitudes(
        Register::qubits(1).unwrap(),
        amplitudes,
        ResourceLimits::default(),
    )
    .unwrap();
    let operator = ComplexTensor::new(
        Tensor::<3>::zeros([1, 2, 2], &ordinary),
        Tensor::zeros([1, 2, 2], &plain),
    )
    .unwrap();
    assert!(matches!(
        DenseSimulator::default().apply_tensor(&state, &[0], &operator),
        Err(Error::TensorContext)
    ));
    assert!(Unitary::x().matrix().to_tensor(&plain, DType::I32).is_err());
}
