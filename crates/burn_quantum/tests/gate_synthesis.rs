//! Complete-reference unitary synthesis, structured predicates, and expansion limits.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::DType;
use burn_quantum::{photonic::compiler::*, *};
use common::complex_values;
fn compare(circuit: Circuit) {
    let synthesized = GateSynthesis::compile(&circuit, SynthesisOptions::default()).unwrap();
    let n = circuit.register().wires();
    let mut a = Circuit::new(Register::qubits(2 * n).unwrap(), 0, 0);
    for i in 0..n {
        a.unitary(vec![i], Unitary::h()).unwrap();
        a.unitary(vec![i, n + i], Unitary::cnot()).unwrap();
    }
    let mut b = a.clone();
    for op in circuit.operations() {
        a.push(op.clone()).unwrap();
    }
    for op in synthesized.circuit().operations() {
        b.push(op.clone()).unwrap();
    }
    let e = DenseSimulator::default();
    let d = runtime::execution().unwrap().device;
    let zero = || StateVector::zero(a.register().clone(), 1, &d, DType::F32, e.limits).unwrap();
    let va = complex_values(&a.run_from(&e, zero(), None, None).unwrap());
    let vb = complex_values(&b.run_from(&e, zero(), None, None).unwrap());
    let overlap: num_complex::Complex64 = va.iter().zip(&vb).map(|(a, b)| a.conj() * b).sum();
    assert!(
        (overlap.norm_sqr() - 1.).abs() < 2e-4,
        "overlap={overlap}; gates={}",
        b.operations().len()
    );
    assert!(synthesized.numerical_error() < 1e-7);
}
#[test]
fn arbitrary_two_qubit_unitaries_preserve_complete_reference_channel() {
    for gate in [
        Unitary::fourier(4).unwrap(),
        Unitary::t().controlled(2, 1).unwrap(),
        Unitary::y().controlled(2, 0).unwrap(),
    ] {
        for wires in [vec![0, 1], vec![1, 0]] {
            let mut c = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
            c.unitary(wires, gate.clone()).unwrap();
            compare(c);
        }
    }
}
#[test]
fn multicontrols_and_dense_three_qubit_unitary_have_exact_channels() {
    for controls in [vec![(0, 0), (2, 1)], vec![(0, 1), (2, 0)]] {
        let mut c = Circuit::new(Register::qubits(3).unwrap(), 0, 0);
        c.push(Operation::ControlledX {
            controls: controls.clone(),
            target: 1,
        })
        .unwrap();
        c.push(Operation::PhaseFlip { controls }).unwrap();
        compare(c);
    }
    // QR residual is computed against the whole dense operator before acceptance.
    let mut c = Circuit::new(Register::qubits(3).unwrap(), 0, 0);
    c.unitary(vec![2, 0, 1], Unitary::fourier(8).unwrap())
        .unwrap();
    let s = GateSynthesis::compile(&c, SynthesisOptions::default()).unwrap();
    assert!(s.numerical_error() < 1e-7);
}
#[test]
fn invalid_or_exponential_synthesis_is_bounded_before_execution() {
    let mut c = Circuit::new(Register::qubits(3).unwrap(), 0, 0);
    c.push(Operation::ControlledX {
        controls: vec![(0, 1), (1, 1)],
        target: 2,
    })
    .unwrap();
    assert!(
        GateSynthesis::compile(
            &c,
            SynthesisOptions {
                max_native_gates: 2,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        GateSynthesis::compile(
            &c,
            SynthesisOptions {
                max_predicate_qubits: 2,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        GateSynthesis::compile(
            &c,
            SynthesisOptions {
                max_numerical_error: 1e-20,
                ..Default::default()
            }
        )
        .is_err()
    );
}
#[test]
fn existing_sat_oracle_and_diffusion_lower_without_an_unsupported_frontend() {
    use burn_quantum::sat::{Cnf, GroverSat};
    let solver = GroverSat::new(
        Cnf::new(3, vec![vec![1, -2, 3]]).unwrap(),
        ResourceLimits::default(),
    )
    .unwrap();
    for source in [solver.oracle(), solver.diffusion()] {
        let lowered = GateSynthesis::compile(source, SynthesisOptions::default()).unwrap();
        assert!(
            lowered
                .circuit()
                .operations()
                .iter()
                .all(|op| matches!(op,Operation::Unitary{wires,..} if wires.len()<=2))
        );
        assert!(MeasurementPattern::compile(source, PatternCompileLimits::default()).is_ok());
    }
}
