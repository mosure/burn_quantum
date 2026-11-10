//! Exact dense oracles, approximation bounds, large chains and checkpoint validation.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::DType;
use burn_quantum::tensor_network::{MpsLimits, MpsState};
use burn_quantum::{DenseSimulator, Matrix, Register, StateVector, Unitary};
use common::{close, values};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn heterogeneous_nonadjacent_nonclifford_gates_match_dense_amplitudes() {
    let device = runtime::execution().unwrap().device;
    let dims = vec![2, 3, 2, 4];
    let digits = vec![0, 1, 0, 2];
    let mut mps =
        MpsState::basis(&dims, &digits, &device, DType::F32, MpsLimits::default()).unwrap();
    let engine = DenseSimulator::default();
    let mut dense = StateVector::basis(
        Register::new(dims).unwrap(),
        &digits,
        1,
        &device,
        DType::F32,
        engine.limits,
    )
    .unwrap();
    let gates = vec![
        (vec![0], Unitary::h()),
        (vec![1], Unitary::fourier(3).unwrap()),
        (vec![3, 0], Unitary::fourier(8).unwrap()),
        (vec![2, 0], Unitary::cnot()),
        (vec![0], Unitary::t()),
        (vec![3, 1], Unitary::fourier(12).unwrap()),
        (vec![2, 1, 0], Unitary::fourier(12).unwrap()),
    ];
    for (w, g) in gates {
        mps.apply(&w, &g).unwrap();
        dense = engine.apply(&dense, &w, &g).unwrap();
        let export = mps.to_dense(engine.limits).unwrap();
        close(
            &values(export.amplitudes().real()),
            &values(dense.amplitudes().real()),
            5e-6,
        );
        close(
            &values(export.amplitudes().imag()),
            &values(dense.amplitudes().imag()),
            5e-6,
        );
        let cp = mps.checkpoint().unwrap();
        MpsState::restore(&cp, &device, DType::F32, MpsLimits::default()).unwrap();
    }
    let op = Unitary::clock(3).unwrap().matrix().clone();
    let acted = engine
        .apply(&dense, &[1], &Unitary::clock(3).unwrap())
        .unwrap();
    let a = values(dense.amplitudes().real());
    let ai = values(dense.amplitudes().imag());
    let b = values(acted.amplitudes().real());
    let bi = values(acted.amplitudes().imag());
    let re: f64 = (0..a.len()).map(|i| a[i] * b[i] + ai[i] * bi[i]).sum();
    let im: f64 = (0..a.len()).map(|i| a[i] * bi[i] - ai[i] * b[i]).sum();
    let got = mps.expectation_product(&[(1, op)]).unwrap();
    assert!((got.re - re).abs() < 4e-6);
    assert!((got.im - im).abs() < 4e-6);
}
#[test]
fn truncation_bound_and_budget_rollback_are_physical() {
    let device = runtime::execution().unwrap().device;
    let limits = MpsLimits {
        max_bond: 1,
        ..Default::default()
    };
    let mut s = MpsState::basis(&[2, 2], &[0, 0], &device, DType::F32, limits).unwrap();
    s.apply(&[0], &Unitary::h()).unwrap();
    let before = s.checkpoint().unwrap();
    assert!(s.apply(&[0, 1], &Unitary::cnot()).is_err());
    assert_eq!(s.checkpoint().unwrap(), before);
    let mut s = MpsState::restore(
        &before,
        &device,
        DType::F32,
        MpsLimits {
            max_truncation_distance: 0.8,
            ..limits
        },
    )
    .unwrap();
    s.apply(&[0, 1], &Unitary::cnot()).unwrap();
    assert!((s.report().discarded_weight - 0.5).abs() < 1e-6);
    assert!((s.report().trace_distance_bound - 0.5f64.sqrt()).abs() < 1e-6);
    let v = s.to_dense(Default::default()).unwrap();
    let re = values(v.amplitudes().real());
    let im = values(v.amplitudes().imag());
    let fidelity = ((re[0] + re[3]).powi(2) + (im[0] + im[3]).powi(2)) / 2.;
    assert!((1. - fidelity).sqrt() <= s.report().trace_distance_bound + 1e-6);
    let mut corrupt = s.checkpoint().unwrap();
    corrupt.sites[0].values[0] *= 2.;
    assert!(
        MpsState::restore(
            &corrupt,
            &device,
            DType::F32,
            MpsLimits {
                max_truncation_distance: 0.8,
                ..limits
            }
        )
        .is_err()
    );
}
#[test]
fn long_entangled_chain_stays_linear_and_samples_correlations() {
    let device = runtime::execution().unwrap().device;
    let n = 80;
    let limits = MpsLimits {
        max_bond: 2,
        ..Default::default()
    };
    let mut s = MpsState::basis(&vec![2; n], &vec![0; n], &device, DType::F32, limits).unwrap();
    s.apply(&[0], &Unitary::h()).unwrap();
    for i in 0..n - 1 {
        s.apply(&[i, i + 1], &Unitary::cnot()).unwrap();
    }
    s.apply(&[n - 1], &Unitary::t()).unwrap();
    assert!(s.storage_elements() <= 8 * n);
    assert!(s.to_dense(Default::default()).is_err());
    assert!((s.expectation_product(&[]).unwrap().re - 1.).abs() < 2e-5);
    assert!(
        (s.expectation_product(&[
            (0, Unitary::z().matrix().clone()),
            (n - 1, Unitary::z().matrix().clone()),
        ])
        .unwrap()
        .re - 1.)
            .abs()
            < 2e-5
    );
    let x: Vec<_> = (0..n).map(|i| (i, Unitary::x().matrix().clone())).collect();
    assert!((s.expectation_product(&x).unwrap().re - std::f64::consts::FRAC_1_SQRT_2).abs() < 2e-5);
    let mut rng = ChaCha8Rng::seed_from_u64(987);
    for _ in 0..3 {
        let sample = s.sample(&mut rng).unwrap();
        assert!(sample.iter().all(|v| *v == sample[0]));
    }
}
#[test]
fn conditional_measurement_has_correct_joint_probabilities() {
    let device = runtime::execution().unwrap().device;
    let mut s = MpsState::basis(
        &[2, 3, 2],
        &[0, 0, 0],
        &device,
        DType::F32,
        Default::default(),
    )
    .unwrap();
    s.apply(&[0], &Unitary::h()).unwrap();
    s.apply(&[0, 2], &Unitary::cnot()).unwrap();
    let p = s.condition(2, 1).unwrap();
    assert!((p - 0.5).abs() < 2e-6);
    assert!(
        (s.expectation_product(&[(0, Matrix::real(2, vec![1., 0., 0., -1.]).unwrap())])
            .unwrap()
            .re
            + 1.)
            .abs()
            < 2e-6
    );
    let before = s.checkpoint().unwrap();
    assert!(s.condition(2, 0).is_err());
    assert_eq!(s.checkpoint().unwrap(), before);
}

#[test]
fn quantum_backend_bindings_and_sparse_observables_match_dense() {
    use burn::tensor::Tensor;
    use burn_quantum::{
        Angle, Circuit, Observable, QuantumBackend, Rotation, gate::Axis,
        tensor_network::MpsSimulator,
    };
    let device = runtime::execution().unwrap().device;
    let mut c = Circuit::new(Register::qubits(4).unwrap(), 1, 1);
    c.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    c.rotate(3, Rotation::qubit(Axis::X, Angle::input(0)))
        .unwrap();
    c.unitary(vec![0, 2], Unitary::cnot()).unwrap();
    c.controlled_x(vec![(0, 1), (2, 0)], 3).unwrap();
    c.phase_flip(vec![(0, 1), (3, 1)]).unwrap();
    let p = Tensor::from_data([0.73f32], &device);
    let x = Tensor::from_data([[0.31f32], [-0.61]], &device);
    let dense = DenseSimulator::default()
        .execute(&c, Some(p.clone()), Some(x.clone()), 2, &device)
        .unwrap();
    let engine = MpsSimulator::default();
    let m = engine.execute(&c, Some(p), Some(x), 2, &device).unwrap();
    close(
        &values(engine.probabilities(&m).unwrap()),
        &values(dense.probabilities()),
        4e-6,
    );
    let o = Observable::new(
        vec![3, 0],
        Matrix::real(
            4,
            vec![
                1., 0., 0., 0., 0., -1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1.,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    close(
        &values(engine.expectation(&m, &o).unwrap()),
        &values(o.expectation(&DenseSimulator::default(), &dense).unwrap()),
        4e-6,
    );
}
#[test]
fn asymmetric_photon_loss_trajectories_match_density_oracle() {
    use burn_quantum::{DensityMatrix, KrausChannel};
    let device = runtime::execution().unwrap().device;
    let engine = DenseSimulator::default();
    let channel = KrausChannel::new(vec![
        Matrix::real(
            3,
            vec![0.8f64.sqrt(), 0., 0., 0., 0.3f64.sqrt(), 0., 0., 0., 1.],
        )
        .unwrap(),
        Matrix::real(3, vec![0., 0., 0., 0., 0., 0., 0.2f64.sqrt(), 0., 0.]).unwrap(),
        Matrix::real(3, vec![0., 0., 0., 0., 0., 0., 0., 0.7f64.sqrt(), 0.]).unwrap(),
    ])
    .unwrap();
    let mut initial =
        MpsState::basis(&[3, 2], &[0, 0], &device, DType::F32, Default::default()).unwrap();
    initial.apply(&[0], &Unitary::fourier(3).unwrap()).unwrap();
    let dense = DensityMatrix::from_pure(&initial.to_dense(engine.limits).unwrap(), engine.limits)
        .unwrap()
        .channel(&engine, &[0], &channel)
        .unwrap();
    let expected = values(dense.probabilities());
    let mut mean = vec![0.; 6];
    let mut rng = ChaCha8Rng::seed_from_u64(433);
    for _ in 0..256 {
        let mut shot = initial.clone();
        let (_, p) = shot.trajectory(0, &channel, &mut rng).unwrap();
        assert!((0.0..=1.000001).contains(&p));
        let probabilities = values(shot.to_dense(engine.limits).unwrap().probabilities());
        for (a, b) in mean.iter_mut().zip(probabilities) {
            *a += b / 256.;
        }
    }
    close(&mean, &expected, 0.06);
}
#[test]
fn public_backend_batches_reject_mixed_metadata_before_readout() {
    use burn_quantum::{Error, Observable, QuantumBackend, tensor_network::MpsSimulator};
    let cpu = burn::tensor::Device::flex();
    let engine = MpsSimulator::default();
    let state = |dimensions: &[usize], dtype, device: &burn::tensor::Device| {
        MpsState::basis(
            dimensions,
            &vec![0; dimensions.len()],
            device,
            dtype,
            MpsLimits::default(),
        )
        .unwrap()
    };
    let a = state(&[2, 3], DType::F32, &cpu);
    let b = state(&[3, 2], DType::F32, &cpu);
    let z = Observable::new(vec![0], Unitary::z().matrix().clone()).unwrap();
    assert!(matches!(
        engine.probabilities(&vec![a.clone(), b.clone()]),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        engine.expectation(&vec![a.clone(), b], &z),
        Err(Error::Shape(_))
    ));
    let wide = state(&[2, 3], DType::F64, &cpu);
    assert!(matches!(
        engine.probabilities(&vec![a.clone(), wide.clone()]),
        Err(Error::TensorContext)
    ));
    assert!(matches!(
        engine.expectation(&vec![a.clone(), wide], &z),
        Err(Error::TensorContext)
    ));
    let selected = runtime::execution().unwrap().device;
    if selected != cpu {
        let remote = state(&[2, 3], DType::F32, &selected);
        assert!(matches!(
            engine.probabilities(&vec![a.clone(), remote.clone()]),
            Err(Error::TensorContext)
        ));
        assert!(matches!(
            engine.expectation(&vec![a.clone(), remote], &z),
            Err(Error::TensorContext)
        ));
    }
    let limited = MpsSimulator {
        readout_limits: burn_quantum::ResourceLimits {
            max_state_elements: 9,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(
        limited.expectation(&vec![a.clone(), a], &z),
        Err(Error::ResourceLimit { .. })
    ));
}
#[test]
fn qudit_rotation_and_global_phase_preflight_dense_operator_budget() {
    use burn_quantum::{
        Angle, Circuit, Error, QuantumBackend, Rotation, gate::Axis, tensor_network::MpsSimulator,
    };
    let engine = MpsSimulator {
        limits: MpsLimits {
            max_tensor_elements: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let device = burn::tensor::Device::flex();
    let mut circuit = Circuit::new(Register::new(vec![4]).unwrap(), 0, 0);
    circuit
        .rotate(
            0,
            Rotation {
                axis: Axis::Y,
                levels: [0, 3],
                angle: Angle::Constant(0.3),
            },
        )
        .unwrap();
    assert!(matches!(
        engine.execute(&circuit, None, None, 1, &device),
        Err(Error::ResourceLimit {
            resource: "MPS local tensor",
            required: 16,
            limit: 8
        })
    ));
    let mut circuit = Circuit::new(Register::new(vec![4]).unwrap(), 0, 0);
    circuit.phase_flip(vec![]).unwrap();
    assert!(matches!(
        engine.execute(&circuit, None, None, 1, &device),
        Err(Error::ResourceLimit {
            resource: "MPS local tensor",
            required: 16,
            limit: 8
        })
    ));
}

#[test]
fn restored_error_counters_cannot_overflow_during_truncation() {
    use burn_quantum::Error;
    let device = runtime::execution().unwrap().device;
    let limits = MpsLimits {
        max_bond: 1,
        max_truncation_distance: 0.8,
        ..Default::default()
    };
    let mut state = MpsState::basis(&[2, 2], &[0, 0], &device, DType::F32, limits).unwrap();
    state.apply(&[0], &Unitary::h()).unwrap();
    let mut checkpoint = state.checkpoint().unwrap();
    checkpoint.report.truncations = usize::MAX;
    let mut restored = MpsState::restore(&checkpoint, &device, DType::F32, limits).unwrap();
    assert_eq!(
        restored.apply(&[0, 1], &Unitary::cnot()),
        Err(Error::Overflow)
    );
    assert_eq!(restored.checkpoint().unwrap(), checkpoint);
}

#[test]
fn checkpoint_restore_checks_global_norm_without_dense_hilbert_expansion() {
    let device = runtime::execution().unwrap().device;
    let limits = MpsLimits {
        max_sites: 64,
        max_bond: 1,
        max_tensor_elements: 2,
        max_state_elements: 128,
        ..Default::default()
    };
    let original = MpsState::basis(&[2; 64], &[0; 64], &device, DType::F32, limits).unwrap();
    let checkpoint = original.checkpoint().unwrap();
    let restored = MpsState::restore(&checkpoint, &device, DType::F32, limits).unwrap();
    assert_eq!(restored.storage_elements(), 128);
    assert!(restored.to_dense(Default::default()).is_err());
    let mut corrupt = checkpoint;
    // Every local gauge is within 2e-5, but their global product is not normalized.
    for site in corrupt.sites.iter_mut().skip(1) {
        site.values[0] *= 1.000009;
    }
    assert!(
        matches!(MpsState::restore(&corrupt,&device,DType::F32,limits),Err(burn_quantum::Error::InvalidValue(message)) if message.contains("global norm"))
    );
}

#[test]
fn complex_gate_preserves_entangled_center() {
    use num_complex::Complex64;
    let device = runtime::execution().unwrap().device;
    let mut s = MpsState::basis(&[2, 2], &[0, 0], &device, DType::F32, Default::default()).unwrap();
    s.apply(&[0], &Unitary::h()).unwrap();
    s.apply(&[0, 1], &Unitary::cnot()).unwrap();
    // The final canonical center is a noncontiguous [2,2,1] view. CUDA fusion
    // previously deleted its |11> branch during scalar normalization.
    s.apply(&[1], &Unitary::t()).unwrap();
    let expected = [
        Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0.5, 0.5),
    ];
    let before = s.checkpoint().unwrap();
    // Independently contract the stored tensors on the host before any native
    // observable or dense export can materialize additional operations.
    let [left, right] = before.sites.as_slice() else {
        panic!("two sites")
    };
    let bond = left.shape[2];
    for (basis, target) in expected.iter().enumerate() {
        let actual: Complex64 = (0..bond)
            .map(|b| left.values[(basis / 2) * bond + b] * right.values[b * 2 + basis % 2])
            .sum();
        assert!(
            (actual - target).norm() < 2e-6,
            "basis {basis}: {actual} != {target}"
        );
    }
    let dense = s.to_dense(Default::default()).unwrap();
    close(
        &values(dense.amplitudes().real()),
        &expected.map(|z| z.re),
        2e-6,
    );
    close(
        &values(dense.amplitudes().imag()),
        &expected.map(|z| z.im),
        2e-6,
    );
    for (operator, target) in [
        (None, 1.),
        (Some(Unitary::z()), 1.),
        (Some(Unitary::x()), std::f64::consts::FRAC_1_SQRT_2),
    ] {
        let product = operator
            .map(|u| vec![(0, u.matrix().clone()), (1, u.matrix().clone())])
            .unwrap_or_default();
        assert!(
            (s.expectation_product(&product).unwrap() - Complex64::new(target, 0.)).norm() < 2e-6
        );
    }
    assert_eq!(s.checkpoint().unwrap(), before);
    assert_eq!(s.report().discarded_weight, 0.);
    assert_eq!(s.report().trace_distance_bound, 0.);
}
