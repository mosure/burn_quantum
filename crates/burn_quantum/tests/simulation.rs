//! Analytic and independent scalar-reference simulator checks.
mod common;
use burn::tensor::{DType, Device};
use burn_quantum::*;
use common::*;

#[test]
fn bell_ghz_and_sampling() {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    for n in [2, 3, 5] {
        let mut circuit = Circuit::new(Register::qubits(n).unwrap(), 0, 0);
        circuit.unitary(vec![0], Unitary::h()).unwrap();
        for w in 1..n {
            circuit.unitary(vec![0, w], Unitary::cnot()).unwrap();
        }
        let state = engine.execute(&circuit, None, None, 2, &device).unwrap();
        state.validate(1e-6).unwrap();
        let mut expected = vec![0.; 2 * (1 << n)];
        for b in 0..2 {
            expected[b * (1 << n)] = 0.5;
            expected[(b + 1) * (1 << n) - 1] = 0.5;
        }
        close(&values(state.probabilities()), &expected, 1e-6);
        let samples = Samples::from_state(&state, 1000, 123, 2000).unwrap();
        assert_eq!(
            samples,
            Samples::from_state(&state, 1000, 123, 2000).unwrap()
        );
        assert!(
            samples
                .outcomes
                .iter()
                .flatten()
                .all(|&x| x == 0 || x == (1 << n) - 1)
        );
        let zeros = samples.outcomes[0].iter().filter(|&&x| x == 0).count();
        assert!((400..600).contains(&zeros));
        close(
            &values(Observable::z(0).expectation(&engine, &state).unwrap()),
            &[0., 0.],
            1e-6,
        );
    }
}

#[test]
fn qudit_entanglement_and_weyl_relation() {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    for d in [3, 4, 5] {
        let mut circuit = Circuit::new(Register::new(vec![d, d]).unwrap(), 0, 0);
        circuit
            .unitary(vec![0], Unitary::fourier(d).unwrap())
            .unwrap()
            .unitary(vec![0, 1], Unitary::sum(d).unwrap())
            .unwrap();
        let state = engine.execute(&circuit, None, None, 1, &device).unwrap();
        state.validate(1e-6).unwrap();
        let expected: Vec<_> = (0..d * d)
            .map(|i| if i / d == i % d { 1. / d as f64 } else { 0. })
            .collect();
        close(&values(state.probabilities()), &expected, 1e-6);
        let register = Register::new(vec![d]).unwrap();
        let initial = normalized(d, 7);
        let state = state_fn(register, &initial, &device);
        let x = Unitary::shift(d, 1).unwrap();
        let z = Unitary::clock(d).unwrap();
        let zx = engine
            .apply(&engine.apply(&state, &[0], &x).unwrap(), &[0], &z)
            .unwrap();
        let xz = engine
            .apply(&engine.apply(&state, &[0], &z).unwrap(), &[0], &x)
            .unwrap();
        let phase = num_complex::Complex64::from_polar(1., std::f64::consts::TAU / d as f64);
        complex_close(
            &complex_values(&zx),
            &complex_values(&xz)
                .iter()
                .map(|c| *c * phase)
                .collect::<Vec<_>>(),
            1e-10,
        );
    }
}
fn state_fn(
    register: Register,
    amplitudes: &[num_complex::Complex64],
    device: &Device,
) -> StateVector {
    common::state(register, amplitudes, 1, device)
}

#[test]
fn mixed_radix_reversed_nonadjacent_targets_match_scalar_oracle() {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    for dimensions in [vec![2, 3, 2], vec![3, 2, 4], vec![2, 2, 2, 3]] {
        let register = Register::new(dimensions.clone()).unwrap();
        let n = register.size();
        let initial: Vec<_> = normalized(n, 3)
            .into_iter()
            .chain(normalized(n, 17))
            .collect();
        let state = state(register, &initial, 2, &device);
        for wires in [
            vec![0],
            vec![dimensions.len() - 1],
            vec![dimensions.len() - 1, 0],
            vec![1, 0],
        ] {
            let size: usize = wires.iter().map(|&w| dimensions[w]).product();
            let gate = Unitary::fourier(size).unwrap();
            let expected = apply_reference(&initial, &dimensions, &wires, gate.matrix());
            let actual = engine.apply(&state, &wires, &gate).unwrap();
            complex_close(&complex_values(&actual), &expected, 1e-10);
            let restored = engine.apply(&actual, &wires, &gate.adjoint()).unwrap();
            complex_close(&complex_values(&restored), &initial, 1e-10);
        }
    }
}

#[test]
fn controlled_gate_and_phase_sensitive_observables() {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    let register = Register::new(vec![3, 2]).unwrap();
    let gate = Unitary::y().controlled(3, 2).unwrap();
    for c in 0..3 {
        let initial = StateVector::basis(
            register.clone(),
            &[c, 0],
            1,
            &device,
            DType::F64,
            engine.limits,
        )
        .unwrap();
        let out = engine.apply(&initial, &[0, 1], &gate).unwrap();
        let mut expected = vec![num_complex::Complex64::new(0., 0.); 6];
        expected[c * 2 + usize::from(c == 2)] = if c == 2 {
            num_complex::Complex64::new(0., 1.)
        } else {
            1.0.into()
        };
        complex_close(&complex_values(&out), &expected, 1e-10);
    }
    let state = StateVector::zero(
        Register::qubits(1).unwrap(),
        1,
        &device,
        DType::F64,
        engine.limits,
    )
    .unwrap();
    let state = engine
        .apply(
            &engine.apply(&state, &[0], &Unitary::h()).unwrap(),
            &[0],
            &Unitary::s(),
        )
        .unwrap();
    close(
        &values(Observable::y(0).expectation(&engine, &state).unwrap()),
        &[1.],
        1e-10,
    );
    close(
        &values(Observable::x(0).variance(&engine, &state).unwrap()),
        &[1.],
        1e-10,
    );
    close(
        &values(Observable::y(0).variance(&engine, &state).unwrap()),
        &[0.],
        1e-10,
    );
}

#[test]
fn marginal_target_order_and_batch_isolation() {
    let register = Register::new(vec![2, 3, 2]).unwrap();
    let device = Device::flex();
    let mut initial = vec![0.0.into(); 24];
    initial[register.encode(&[1, 2, 0]).unwrap()] = 1.0.into();
    initial[12 + register.encode(&[0, 1, 1]).unwrap()] = 1.0.into();
    let state = state(register, &initial, 2, &device);
    let marginal = values(state.marginal(&[2, 0], ResourceLimits::default()).unwrap());
    close(&marginal, &[0., 1., 0., 0., 0., 0., 1., 0.], 1e-12);
}

#[test]
fn hamiltonian_is_sum_of_local_terms() {
    let engine = DenseSimulator::default();
    let device = Device::flex();
    let state = StateVector::basis(
        Register::qubits(2).unwrap(),
        &[0, 1],
        1,
        &device,
        DType::F64,
        engine.limits,
    )
    .unwrap();
    let h = Hamiltonian::new(vec![(2., Observable::z(0)), (-3., Observable::z(1))]).unwrap();
    close(
        &values(h.expectation(&engine, &state).unwrap()),
        &[5.],
        1e-12,
    );
    close(
        &values(
            Hamiltonian::new(vec![])
                .unwrap()
                .expectation(&engine, &state)
                .unwrap(),
        ),
        &[0.],
        1e-12,
    );
}
