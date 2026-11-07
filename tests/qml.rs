//! Independent numerical and API-contract checks for reusable QML operations.
mod common;
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{gate::Axis, qml::*, *};
use common::*;
use num_complex::Complex64;

#[test]
fn circuit_templates_preserve_order_and_fail_atomically() {
    let mut actual = Circuit::new(Register::qubits(3).unwrap(), 3, 3);
    angle_embedding(&mut actual, &[2, 0], Axis::X, 1).unwrap();
    ry_cnot_layer(&mut actual, &[2, 0], 1).unwrap();
    let mut expected = Circuit::new(Register::qubits(3).unwrap(), 3, 3);
    for (wire, index) in [(2, 1), (0, 2)] {
        expected
            .rotate(wire, Rotation::qubit(Axis::X, Angle::input(index)))
            .unwrap();
    }
    for (wire, index) in [(2, 1), (0, 2)] {
        expected
            .rotate(wire, Rotation::qubit(Axis::Y, Angle::parameter(index)))
            .unwrap();
    }
    expected.unitary(vec![2, 0], Unitary::cnot()).unwrap();
    assert_eq!(actual, expected);
    for wires in [vec![], vec![0, 0], vec![9]] {
        assert!(ry_cnot_layer(&mut actual, &wires, 0).is_err());
        assert_eq!(actual, expected);
    }
    assert!(angle_embedding(&mut actual, &[0, 1], Axis::Y, 2).is_err());
    assert!(ry_cnot_layer(&mut actual, &[0], usize::MAX).is_err());
    assert_eq!(actual, expected);
    let mut qudit = Circuit::new(Register::new(vec![3]).unwrap(), 1, 1);
    assert!(angle_embedding(&mut qudit, &[0], Axis::Y, 0).is_err());
    assert!(qudit.operations().is_empty());
}

#[test]
fn z_readout_matches_born_rule_and_gradients() {
    let device = Device::flex().autodiff();
    let z = Tensor::<1>::from_data(
        TensorData::new(vec![-1.0f64, 0., 1.], [3]),
        (&device, DType::F64),
    )
    .require_grad();
    let probabilities = pauli_z_probabilities(z.clone());
    close(
        &values(probabilities.clone()),
        &[0., 1., 0.5, 0.5, 1., 0.],
        1e-12,
    );
    let gradients = probabilities.slice([0..3, 1..2]).sum().backward();
    close(&values(z.grad(&gradients).unwrap()), &[-0.5; 3], 1e-12);
}

#[test]
fn probability_cross_entropy_matches_analytic_loss_and_gradient() {
    let device = Device::flex().autodiff();
    let p =
        Tensor::<2>::from_data([[0.25f64, 0.75], [0.6, 0.4]], (&device, DType::F64)).require_grad();
    let target = Tensor::<2>::from_data([[1.0f64, 0.], [1., 0.]], (&device, DType::F64));
    let loss = distribution_cross_entropy(p.clone(), target, 1e-9).unwrap();
    close(
        &values(loss.clone()),
        &[-(0.25f64.ln() + 0.6f64.ln()) / 2.],
        1e-12,
    );
    let gradients = loss.backward();
    close(
        &values(p.grad(&gradients).unwrap()),
        &[-2., 0., -1. / 1.2, 0.],
        1e-12,
    );
    let soft = Tensor::<2>::from_data([[0.5f64, 0.5]], (&device, DType::F64));
    close(
        &values(distribution_cross_entropy(soft.clone(), soft, 1e-9).unwrap()),
        &[2f64.ln()],
        1e-12,
    );
}

#[test]
fn probability_loss_rejects_bad_metadata_and_handles_zero_support() {
    let device = Device::flex();
    let p = Tensor::<2>::from_floats([[0., 1.]], &device);
    let target = Tensor::<2>::from_floats([[1., 0.]], &device);
    close(
        &values(distribution_cross_entropy(p.clone(), target.clone(), 1e-5).unwrap()),
        &[-1e-5f64.ln()],
        1e-6,
    );
    for epsilon in [0., -0.1, 1., f64::NAN, 1e-100] {
        assert!(distribution_cross_entropy(p.clone(), target.clone(), epsilon).is_err());
    }
    assert!(distribution_cross_entropy(p.clone(), Tensor::zeros([2, 2], &device), 1e-5).is_err());
    assert!(
        distribution_cross_entropy(p, Tensor::zeros([1, 2], (&device, DType::F64)), 1e-5).is_err()
    );
}

#[test]
fn complex_fidelities_match_scalar_reference_and_gram_psd() {
    let device = Device::flex();
    let register = Register::qubits(2).unwrap();
    let a: Vec<_> = (0..3).flat_map(|i| normalized(4, i + 1)).collect();
    let b: Vec<_> = (0..2).flat_map(|i| normalized(4, i + 11)).collect();
    let left = state(register.clone(), &a, 3, &device);
    let right = state(register, &b, 2, &device);
    let expected: Vec<_> = a
        .chunks_exact(4)
        .flat_map(|x| {
            b.chunks_exact(4).map(move |y| {
                x.iter()
                    .zip(y)
                    .map(|(a, b)| a.conj() * b)
                    .sum::<Complex64>()
                    .norm_sqr()
            })
        })
        .collect();
    close(
        &values(fidelity_kernel(&left, &right, ResourceLimits::default()).unwrap()),
        &expected,
        1e-12,
    );
    close(&values(left.fidelity(&left).unwrap()), &[1.; 3], 1e-12);
    let gram = values(fidelity_kernel(&left, &left, ResourceLimits::default()).unwrap());
    for i in 0..3 {
        for j in 0..3 {
            assert!((gram[i * 3 + j] - gram[j * 3 + i]).abs() < 1e-12);
        }
    }
    for v in [[1., -2., 3.], [-3., 1., 0.5], [0., 1., -1.]] {
        let mut quadratic = 0.;
        for i in 0..3 {
            for j in 0..3 {
                quadratic += v[i] * gram[i * 3 + j] * v[j];
            }
        }
        assert!(quadratic >= -1e-12);
    }
}

#[test]
fn fidelity_is_global_phase_invariant_and_phase_sensitive() {
    let device = Device::flex();
    let register = Register::qubits(1).unwrap();
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let plus = state(
        register.clone(),
        &[Complex64::new(s, 0.), Complex64::new(0., s)],
        1,
        &device,
    );
    let phase = state(
        register.clone(),
        &[Complex64::new(0., s), Complex64::new(-s, 0.)],
        1,
        &device,
    );
    let minus = state(
        register,
        &[Complex64::new(s, 0.), Complex64::new(0., -s)],
        1,
        &device,
    );
    close(&values(plus.fidelity(&phase).unwrap()), &[1.], 1e-12);
    close(&values(plus.fidelity(&minus).unwrap()), &[0.], 1e-12);
}

#[test]
fn fidelity_and_kernel_retain_angle_gradients() {
    let device = Device::flex().autodiff();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let angle = Tensor::<1>::from_data([0.7f64], (&device, DType::F64)).require_grad();
    let a = circuit
        .run(&engine, Some(angle.clone()), None, 1, &device)
        .unwrap();
    let b = StateVector::zero(
        Register::qubits(1).unwrap(),
        1,
        &device,
        DType::F64,
        engine.limits,
    )
    .unwrap();
    let loss =
        a.fidelity(&b).unwrap() + fidelity_kernel(&a, &b, engine.limits).unwrap().reshape([1]);
    let gradients = loss.backward();
    close(
        &values(angle.grad(&gradients).unwrap()),
        &[-0.7f64.sin()],
        1e-12,
    );
}

#[test]
fn kernel_rejects_register_context_and_budget_mismatches() {
    let device = Device::flex();
    let limits = ResourceLimits::default();
    let a =
        StateVector::zero(Register::qubits(2).unwrap(), 1, &device, DType::F32, limits).unwrap();
    let b = StateVector::zero(
        Register::new(vec![4]).unwrap(),
        1,
        &device,
        DType::F32,
        limits,
    )
    .unwrap();
    assert!(a.fidelity(&b).is_err());
    assert!(fidelity_kernel(&a, &b, limits).is_err());
    let b =
        StateVector::zero(Register::qubits(2).unwrap(), 1, &device, DType::F64, limits).unwrap();
    assert!(matches!(
        fidelity_kernel(&a, &b, limits),
        Err(Error::TensorContext)
    ));
    let many =
        StateVector::zero(Register::qubits(1).unwrap(), 3, &device, DType::F32, limits).unwrap();
    assert!(matches!(
        fidelity_kernel(
            &many,
            &many,
            ResourceLimits {
                max_state_elements: 8,
                ..limits
            }
        ),
        Err(Error::ResourceLimit { .. })
    ));
}

#[test]
fn state_based_quantum_layer_does_not_require_dummy_observables() {
    let device = Device::flex().autodiff();
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    ry_cnot_layer(&mut circuit, &[0], 0).unwrap();
    let layer = QuantumLayerConfig::new(circuit, vec![], vec![0.7])
        .init(&device)
        .unwrap();
    let probabilities = layer.state(None, 1).unwrap().probabilities();
    close(
        &values(probabilities.clone()),
        &[(0.35f64).cos().powi(2), (0.35f64).sin().powi(2)],
        1e-6,
    );
    let gradients = probabilities.slice([0..1, 0..1]).sum().backward();
    close(
        &values(layer.parameters.val().grad(&gradients).unwrap()),
        &[-0.7f64.sin() / 2.],
        1e-6,
    );
    assert!(matches!(layer.forward(None, 1), Err(Error::Shape(_))));
}
