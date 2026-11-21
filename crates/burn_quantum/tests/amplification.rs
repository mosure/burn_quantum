//! Independent projector, Chebyshev and search-budget references on the selected device.
mod common;
#[path = "../support/device.rs"]
mod runtime;

use burn::tensor::{Device, Tensor, TensorData};
use burn_quantum::{algorithms::amplification::*, gate::Axis, *};
use common::{close, complex_close, complex_values, values};
use num_complex::Complex64;

fn device() -> Device {
    runtime::execution()
        .expect("requested device unavailable")
        .device
}

#[test]
fn custom_phases_match_full_complex_projectors_with_mixed_radices_and_shared_bindings() {
    let device = device();
    let engine = DenseSimulator::default();
    let mut a = Circuit::new(Register::new(vec![3, 2]).unwrap(), 1, 1);
    a.unitary(vec![0], Unitary::fourier(3).unwrap()).unwrap();
    a.rotate(
        0,
        Rotation {
            axis: Axis::Y,
            levels: [1, 2],
            angle: Angle::Parameter {
                index: 0,
                scale: -0.7,
                offset: 0.2,
            },
        },
    )
    .unwrap();
    a.rotate(
        1,
        Rotation::qubit(
            Axis::X,
            Angle::Input {
                index: 0,
                scale: 1.3,
                offset: -0.4,
            },
        ),
    )
    .unwrap();
    a.controlled_x(vec![(0, 2)], 1).unwrap();
    a.phase_flip(vec![(0, 1), (1, 1)]).unwrap();
    a.rotate(1, Rotation::qubit(Axis::Z, Angle::parameter(0)))
        .unwrap();
    a.unitary(vec![1], Unitary::s()).unwrap();
    let bindings = Some(Tensor::from_data([0.87_f32], &device));
    let inputs = Some(Tensor::from_data(
        TensorData::new(vec![-0.81_f32, 1.24], [2, 1]),
        &device,
    ));
    let run = |c: &Circuit| {
        c.run(&engine, bindings.clone(), inputs.clone(), 2, &device)
            .unwrap()
    };
    let sources = complex_values(&run(&a));
    let marked = [1, 4];
    let amp = AmplitudeAmplifier::from_marked(a.clone(), &marked).unwrap();
    let phases = [
        ReflectionPhases {
            good: 0.63,
            source: -1.12,
        },
        ReflectionPhases {
            good: -2.2,
            source: 1.91,
        },
    ];
    let c = amp.compile(phases.into_iter(), 200).unwrap();
    let restored: Circuit = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    let actual = complex_values(&run(&restored));
    // A valid predicate computation may change its data after writing the flag.
    // Uncomputing these temporary phases requires the true adjoint.
    let mut predicate = Circuit::new(amp.oracle().register().clone(), 2, 2);
    for op in amp.oracle().operations() {
        predicate.push(op.clone()).unwrap();
    }
    predicate.unitary(vec![1], Unitary::s()).unwrap();
    predicate
        .rotate(1, Rotation::qubit(Axis::X, Angle::parameter(1)))
        .unwrap();
    predicate
        .rotate(1, Rotation::qubit(Axis::Y, Angle::input(1)))
        .unwrap();
    predicate
        .unitary(vec![0], Unitary::fourier(3).unwrap())
        .unwrap();
    let general = AmplitudeAmplifier::new(a.clone(), predicate).unwrap();
    let computed = general.compile(phases.into_iter(), 200).unwrap();
    assert_eq!((computed.num_parameters(), computed.num_inputs()), (2, 2));
    let output = computed
        .run(
            &engine,
            Some(Tensor::from_data([0.87_f32, -0.6], &device)),
            Some(Tensor::from_data(
                [[-0.81_f32, 0.35], [1.24, -1.71]],
                &device,
            )),
            2,
            &device,
        )
        .unwrap();
    complex_close(&complex_values(&output), &actual, 4e-6);
    for (row, source) in sources.chunks_exact(6).enumerate() {
        let mut psi = source.to_vec();
        for pair in phases {
            for &x in &marked {
                psi[x] *= Complex64::from_polar(1., pair.good);
            }
            let inner: Complex64 = source.iter().zip(&psi).map(|(s, v)| s.conj() * v).sum();
            let coefficient = (Complex64::from_polar(1., pair.source) - 1.) * inner;
            for (v, s) in psi.iter_mut().zip(source) {
                *v += coefficient * s;
            }
        }
        let expected: Vec<_> = psi
            .into_iter()
            .flat_map(|v| [v, Complex64::default()])
            .collect();
        complex_close(&actual[row * 12..(row + 1) * 12], &expected, 3e-6);
    }
    assert_eq!(a.adjoint().adjoint(), a);
}

fn prepare_mass(mass: f64) -> AmplitudeAmplifier {
    let mut a = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    a.rotate(
        0,
        Rotation::qubit(Axis::Y, Angle::Constant(2. * mass.sqrt().asin())),
    )
    .unwrap();
    a.unitary(vec![1], Unitary::h()).unwrap();
    a.unitary(vec![1], Unitary::s()).unwrap();
    AmplitudeAmplifier::from_marked(a, &[2, 3]).unwrap()
}

fn measure(amp: &AmplitudeAmplifier, schedule: AmplificationSchedule) -> Vec<f64> {
    let c = amp.compile(schedule.phases(), 2000).unwrap();
    let state = c
        .run(&DenseSimulator::default(), None, None, 1, &device())
        .unwrap();
    state.validate(3e-5).unwrap();
    values(state.probabilities())
}

#[test]
fn ordinary_amplification_matches_rotation_including_overshoot_and_extreme_masses() {
    for p in [0_f64, 0.017, 0.25, 0.5, 0.79, 1.] {
        let amp = prepare_mass(p);
        let theta = p.sqrt().asin();
        for rounds in [0, 1, 2, 4] {
            let actual = measure(&amp, AmplificationSchedule::standard(rounds));
            let good = ((2 * rounds + 1) as f64 * theta).sin().powi(2);
            close(
                &actual,
                &[
                    (1. - good) / 2.,
                    0.,
                    (1. - good) / 2.,
                    0.,
                    good / 2.,
                    0.,
                    good / 2.,
                    0.,
                ],
                1e-5,
            );
        }
    }
    assert_eq!(
        AmplificationSchedule::known_success(0.25).unwrap().rounds(),
        1
    );
    assert_eq!(
        AmplificationSchedule::known_success(0.5).unwrap().rounds(),
        0
    );
    assert_eq!(
        AmplificationSchedule::known_success(1.).unwrap().rounds(),
        0
    );
}

#[test]
fn fixed_point_matches_independent_chebyshev_recurrence_and_promised_interval() {
    for (minimum, failure) in [
        (0.015_f64, 0.1_f64),
        (0.08, 0.01),
        (0.34, 0.001),
        (0.97, 0.05),
    ] {
        let plan = AmplificationSchedule::fixed_point(minimum, failure).unwrap();
        let degree = 2 * plan.rounds() + 1;
        let k = (1. / failure.sqrt()).acosh();
        // Minimal qualifying odd degree, including L=1 when preparation suffices.
        assert!((k / degree as f64).tanh().powi(2) <= minimum + 1e-14);
        assert!(degree == 1 || (k / (degree - 2) as f64).tanh().powi(2) > minimum);
        for p in [0., minimum / 3., minimum, 0.271, 0.5, 0.82, 0.997, 1.] {
            let x = (1. - p).sqrt() * (k / degree as f64).cosh();
            let (mut previous, mut current) = (1., x);
            for _ in 1..degree {
                (previous, current) = (current, 2. * x * current - previous);
            }
            let expected = 1. - failure * current * current;
            let actual = measure(&prepare_mass(p), plan);
            let good = actual[4] + actual[6];
            close(
                &[good, plan.predicted_success(p).unwrap()],
                &[expected, expected],
                2e-5,
            );
            if p >= minimum {
                assert!(good >= 1. - failure - 2e-5);
            }
            assert!(actual.iter().skip(1).step_by(2).sum::<f64>() < 1e-8);
        }
    }
}

#[test]
fn rare_event_fixed_point_predictions_are_resolved_without_expanding_the_circuit() {
    let promise = 1e-16_f64;
    let failure = 0.01_f64;
    let plan = AmplificationSchedule::fixed_point(promise, failure).unwrap();
    let k = (1. / failure.sqrt()).acosh();
    let degree = 2. * plan.rounds() as f64 + 1.;
    // In the small-width limit L*acos(x) approaches k at twice the promised mass.
    close(
        &[plan.predicted_success(2. * promise).unwrap()],
        &[1. - failure * k.cos().powi(2)],
        1e-8,
    );
    let slope_at_zero = degree * k.tanh() / (k / degree).tanh();
    let tiny = 1e-30;
    assert!((plan.predicted_success(tiny).unwrap() / (slope_at_zero * tiny) - 1.).abs() < 1e-10);
    assert_eq!(
        AmplificationSchedule::fixed_point(0.75, 0.25)
            .unwrap()
            .rounds(),
        0
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn amplification_preserves_the_gradient_through_preparation_and_its_adjoint() {
    let device = device().autodiff();
    let mut a = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    a.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let amp = AmplitudeAmplifier::from_marked(a, &[1]).unwrap();
    let circuit = amp
        .compile(AmplificationSchedule::standard(2).phases(), 64)
        .unwrap();
    let theta = 0.47_f64;
    let parameter = Tensor::from_floats([theta as f32], &device).require_grad();
    let state = circuit
        .run(
            &DenseSimulator::default(),
            Some(parameter.clone()),
            None,
            1,
            &device,
        )
        .unwrap();
    // Z on the data wire has expectation cos(5 theta).
    let loss = Observable::z(0)
        .expectation(&DenseSimulator::default(), &state)
        .unwrap()
        .sum();
    let gradients = loss.backward();
    close(
        &values(parameter.grad(&gradients).unwrap()),
        &[-5. * (5. * theta).sin()],
        5e-5,
    );
}

#[test]
fn explicit_oracles_support_no_marks_all_marks_and_reject_invalid_constructions() {
    let a = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    for marks in [&[][..], &[0, 1, 2, 3]] {
        let amp = AmplitudeAmplifier::from_marked(a.clone(), marks).unwrap();
        let p = measure(&amp, AmplificationSchedule::fixed_point(0.1, 0.03).unwrap());
        close(&[p[0]], &[1.], 1e-5);
        let found = amp
            .search(
                SearchConfig {
                    max_trials: 8,
                    max_oracle_calls: 12,
                    seed: 19,
                },
                1000,
                |c| c.run(&DenseSimulator::default(), None, None, 1, &device()),
                |x| Ok(marks.contains(x)),
            )
            .unwrap();
        assert_eq!(found.witness, marks.first().copied());
    }
    assert!(AmplitudeAmplifier::from_marked(a.clone(), &[1, 1]).is_err());
    assert!(AmplitudeAmplifier::from_marked(a.clone(), &[4]).is_err());
    assert!(AmplitudeAmplifier::new(a.clone(), a.clone()).is_err());
    assert!(
        AmplitudeAmplifier::new(
            a.clone(),
            Circuit::new(Register::new(vec![2, 2, 3]).unwrap(), 0, 0)
        )
        .is_err()
    );
    let amp = AmplitudeAmplifier::from_marked(a, &[1]).unwrap();
    assert!(matches!(
        amp.compile(AmplificationSchedule::standard(usize::MAX).phases(), 12),
        Err(Error::Overflow)
    ));
    let count = amp.operation_count(2).unwrap();
    assert!(matches!(
        amp.compile(AmplificationSchedule::standard(2).phases(), count - 1),
        Err(Error::ResourceLimit { .. })
    ));
    assert_eq!(
        amp.compile(AmplificationSchedule::standard(2).phases(), count)
            .unwrap()
            .operations()
            .len(),
        count
    );
    assert!(
        amp.compile(
            [ReflectionPhases {
                good: f64::NAN,
                source: 0.
            }]
            .into_iter(),
            32
        )
        .is_err()
    );
    for bad in [0., -0.5, 1.1, f64::NAN, f64::INFINITY] {
        assert!(AmplificationSchedule::known_success(bad).is_err());
        assert!(AmplificationSchedule::fixed_point(bad, 0.1).is_err());
        assert!(AmplificationSchedule::fixed_point(0.1, bad).is_err());
    }
    assert!(AmplificationSchedule::fixed_point(0.2, 1.).is_err());
}

#[test]
fn exponential_search_replays_samples_grows_windows_and_respects_both_budgets() {
    let config = SearchConfig {
        max_trials: 30,
        max_oracle_calls: 200,
        seed: 11,
    };
    let replay = || {
        let mut draws = Vec::new();
        let result = exponential_search(
            Some(1. / 64.),
            config,
            |j, seed| {
                draws.push((j, seed));
                Ok(seed)
            },
            |_| Ok(false),
        )
        .unwrap();
        assert_eq!(result.witness, None);
        assert_eq!(result.trials, draws.len());
        assert_eq!(
            result.oracle_calls,
            draws.iter().map(|d| d.0).sum::<usize>()
        );
        let mut window = 1_f64;
        for (j, _) in &draws {
            assert!((*j as f64) < window);
            window = (window * 1.2).min(8.);
        }
        assert!(draws.iter().any(|d| d.0 >= 3));
        draws
    };
    assert_eq!(replay(), replay());
    for (trials, calls) in [(0, 10), (10, 0), (2, 3), (20, 4)] {
        let result = exponential_search(
            Some(1. / 64.),
            SearchConfig {
                max_trials: trials,
                max_oracle_calls: calls,
                ..config
            },
            |_, _| Ok(42),
            |_| Ok(false),
        )
        .unwrap();
        assert!(
            result.trials <= trials && result.oracle_calls <= calls && result.witness.is_none()
        );
    }
    assert!(exponential_search(Some(0.), config, |_, _| Ok(0), |_| Ok(true)).is_err());
    let found = exponential_search(
        Some(1.),
        config,
        |j, _| {
            assert_eq!(j, 0);
            Ok(17)
        },
        |x| Ok(*x == 17),
    )
    .unwrap();
    assert_eq!(
        (found.witness, found.trials, found.oracle_calls),
        (Some(17), 1, 0)
    );
}

#[test]
fn adaptive_search_can_grow_past_the_uniform_cap_for_a_rare_single_qubit_event() {
    let mut a = Circuit::new(Register::qubits(1).unwrap(), 0, 0);
    let p = 0.001_f64;
    a.rotate(
        0,
        Rotation::qubit(Axis::Y, Angle::Constant(2. * p.sqrt().asin())),
    )
    .unwrap();
    let amp = AmplitudeAmplifier::from_marked(a, &[1]).unwrap();
    let engine = DenseSimulator::default();
    let device = device();
    let mut largest = 0;
    let found = amp
        .search(
            SearchConfig {
                seed: 7,
                ..Default::default()
            },
            8192,
            |c| {
                largest = largest.max(c.operations().len());
                c.run(&engine, None, None, 1, &device)
            },
            |x| Ok(*x == 1),
        )
        .unwrap();
    assert_eq!(found.witness, Some(1));
    assert!(largest > amp.operation_count(2).unwrap());
}

#[test]
fn parity_readout_distinguishes_flag_leakage_from_roundoff_in_the_state_norm() {
    let model = OddParityExperiment::new(1, 0.04).unwrap();
    let probabilities = [0.25, 0.05, 0.5, 0.2].map(|p| p * 0.9999);
    let (odd, clean) = model.readout(&probabilities).unwrap();
    close(&[odd, clean], &[0.7, 0.75], 1e-14);
    assert!(model.readout(&[0.2; 4]).is_err());
    assert!(model.readout(&[1., f64::NAN, 0., 0.]).is_err());
    assert!(model.readout(&[1.]).is_err());
}
