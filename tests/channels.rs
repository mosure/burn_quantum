//! Mixed-state evolution, complete positivity, and complex partial traces.
mod common;
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::*;
use common::*;
use num_complex::Complex64;

#[test]
fn analytic_amplitude_damping_and_depolarization() {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    let one = StateVector::basis(
        Register::qubits(1).unwrap(),
        &[1],
        1,
        &device,
        DType::F64,
        engine.limits,
    )
    .unwrap();
    let rho = DensityMatrix::from_pure(&one, engine.limits).unwrap();
    for p in [0., 0.25, 1.] {
        let damped = rho
            .channel(&engine, &[0], &KrausChannel::amplitude_damping(p).unwrap())
            .unwrap();
        damped.validate(1e-9).unwrap();
        close(&values(damped.probabilities()), &[p, 1. - p], 1e-10);
        close(
            &values(damped.expectation(&engine, &Observable::z(0)).unwrap()),
            &[2. * p - 1.],
            1e-10,
        );
        let depolarized = rho
            .channel(&engine, &[0], &KrausChannel::depolarizing(p).unwrap())
            .unwrap();
        depolarized.validate(1e-9).unwrap();
        close(
            &values(depolarized.probabilities()),
            &[p / 2., 1. - p / 2.],
            1e-10,
        );
    }
}

#[test]
fn phase_flip_removes_coherence_at_half_probability() {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    let zero = StateVector::zero(
        Register::qubits(1).unwrap(),
        1,
        &device,
        DType::F64,
        engine.limits,
    )
    .unwrap();
    let plus = engine.apply(&zero, &[0], &Unitary::h()).unwrap();
    let rho = DensityMatrix::from_pure(&plus, engine.limits).unwrap();
    let mixed = rho
        .channel(&engine, &[0], &KrausChannel::phase_flip(0.5).unwrap())
        .unwrap();
    close(&values(mixed.entries().real()), &[0.5, 0., 0., 0.5], 1e-10);
    close(&values(mixed.purity()), &[0.5], 1e-10);
    mixed.validate(1e-9).unwrap();
}

#[test]
fn bell_reduced_states_are_maximally_mixed() {
    let engine = DenseSimulator::default();
    let device = Device::flex();
    for d in [2, 3, 4] {
        let zero = StateVector::zero(
            Register::new(vec![d, d]).unwrap(),
            2,
            &device,
            DType::F64,
            engine.limits,
        )
        .unwrap();
        let state = engine
            .apply(
                &engine
                    .apply(&zero, &[0], &Unitary::fourier(d).unwrap())
                    .unwrap(),
                &[0, 1],
                &Unitary::sum(d).unwrap(),
            )
            .unwrap();
        let full = DensityMatrix::from_pure(&state, engine.limits).unwrap();
        for w in 0..2 {
            let reduced = state.reduced_density(&[w], engine.limits).unwrap();
            reduced.validate(1e-9).unwrap();
            close(
                &values(reduced.probabilities()),
                &vec![1. / d as f64; 2 * d],
                1e-10,
            );
            close(&values(reduced.purity()), &[1. / d as f64; 2], 1e-10);
            let traced = full.reduced(&[w], engine.limits).unwrap();
            close(
                &values(reduced.entries().real()),
                &values(traced.entries().real()),
                1e-10,
            );
            close(
                &values(reduced.entries().imag()),
                &values(traced.entries().imag()),
                1e-10,
            );
        }
    }
}

#[test]
fn density_unitary_matches_pure_and_complex_partial_trace() {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    let register = Register::new(vec![2, 3, 2]).unwrap();
    let initial: Vec<_> = normalized(12, 3)
        .into_iter()
        .chain(normalized(12, 19))
        .collect();
    let pure = state(register, &initial, 2, &device);
    let rho = DensityMatrix::from_pure(&pure, engine.limits).unwrap();
    let gate = Unitary::fourier(4).unwrap();
    let pure = engine.apply(&pure, &[2, 0], &gate).unwrap();
    let rho = rho.apply(&engine, &[2, 0], &gate).unwrap();
    let expected = DensityMatrix::from_pure(&pure, engine.limits).unwrap();
    close(
        &values(rho.entries().real()),
        &values(expected.entries().real()),
        1e-10,
    );
    close(
        &values(rho.entries().imag()),
        &values(expected.entries().imag()),
        1e-10,
    );
    rho.validate(1e-9).unwrap();
    for wires in [vec![0], vec![1], vec![2, 0], vec![2, 1, 0]] {
        let a = pure.reduced_density(&wires, engine.limits).unwrap();
        let b = rho.reduced(&wires, engine.limits).unwrap();
        close(
            &values(a.entries().real()),
            &values(b.entries().real()),
            1e-10,
        );
        close(
            &values(a.entries().imag()),
            &values(b.entries().imag()),
            1e-10,
        );
    }
    close(
        &values(rho.expectation(&engine, &Observable::y(2)).unwrap()),
        &values(Observable::y(2).expectation(&engine, &pure).unwrap()),
        1e-10,
    );
}

#[test]
fn general_qutrit_kraus_channel_and_trace() {
    let engine = DenseSimulator::default();
    let device = Device::flex();
    let projectors: Vec<_> = (0..3)
        .map(|level| {
            let mut entries = vec![0.; 9];
            entries[level * 3 + level] = 1.;
            Matrix::real(3, entries).unwrap()
        })
        .collect();
    let channel = KrausChannel::new(projectors).unwrap();
    let zero = StateVector::zero(
        Register::new(vec![3]).unwrap(),
        1,
        &device,
        DType::F64,
        engine.limits,
    )
    .unwrap();
    let plus = engine
        .apply(&zero, &[0], &Unitary::fourier(3).unwrap())
        .unwrap();
    let mixed = DensityMatrix::from_pure(&plus, engine.limits)
        .unwrap()
        .channel(&engine, &[0], &channel)
        .unwrap();
    mixed.validate(1e-9).unwrap();
    close(&values(mixed.trace()), &[1.], 1e-10);
    close(&values(mixed.purity()), &[1. / 3.], 1e-10);
}

#[test]
fn rejects_unphysical_density_even_with_valid_trace() {
    let device = Device::flex();
    let limits = ResourceLimits::default();
    for data in [
        vec![1.1, 0., 0., -0.1],
        vec![0.5, 0.8, 0.8, 0.5],
        vec![0.5, 0.5, 0., 0.5],
        vec![1., 0., 0., 1.],
    ] {
        let entries = ComplexTensor::new(
            Tensor::from_data(TensorData::new(data, [1, 2, 2]), (&device, DType::F64)),
            Tensor::zeros([1, 2, 2], (&device, DType::F64)),
        )
        .unwrap();
        let rho =
            DensityMatrix::from_entries(Register::qubits(1).unwrap(), entries, limits).unwrap();
        assert!(rho.validate(1e-9).is_err());
    }
    assert!(
        KrausChannel::new(vec![
            Matrix::new(2, vec![Complex64::new(0., 0.); 4]).unwrap()
        ])
        .is_err()
    );
}
