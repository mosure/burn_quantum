//! Actual six-ring resources, complete entangled projector channels, and herald scaling.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_quantum::{
    photonic::{
        qec::{fusion_surgery::*, *},
        stabilizer::*,
    },
    *,
};
use common::{close, values};

fn p(terms: &[(usize, Pauli)]) -> PauliProduct {
    PauliProduct::new(terms.to_vec()).unwrap()
}
fn entangled(k: usize) -> Tableau {
    let mut state = Tableau::zero(2 * k + 7, Default::default()).unwrap();
    for q in 0..k {
        state.h(q).unwrap();
        state.cnot(q, k + q).unwrap();
    }
    state
}
fn scratch(start: usize) -> [usize; 7] {
    std::array::from_fn(|i| start + i)
}

#[test]
fn six_ring_trimmed_resource_matches_full_native_cnot_choi() {
    let device = runtime::execution().unwrap().device;
    let engine = DenseSimulator::default();
    let mut ring = Circuit::new(Register::qubits(6).unwrap(), 0, 0);
    for q in 0..6 {
        ring.unitary(vec![q], Unitary::h()).unwrap();
    }
    for q in 0..6 {
        ring.unitary(vec![q, (q + 1) % 6], Unitary::z().controlled(2, 1).unwrap())
            .unwrap();
    }
    let prepared = engine.execute(&ring, None, None, 1, &device).unwrap();
    let mut target = vec![0f32; 16];
    for a in 0..2 {
        for b in 0..2 {
            target[8 * a + 4 * a + 2 * (a ^ b) + b] = 0.5;
        }
    }
    let ideal = StateVector::from_amplitudes(
        Register::qubits(4).unwrap(),
        ComplexTensor::new(
            Tensor::from_data(TensorData::new(target, [1, 16]), &device),
            Tensor::zeros([1, 16], (&device, DType::F32)),
        )
        .unwrap(),
        engine.limits,
    )
    .unwrap();
    for trim in 0..4 {
        let indices = Tensor::<1, Int>::from_data(
            TensorData::new(
                (0..16).map(|i| (4 * i + trim) as i64).collect::<Vec<_>>(),
                [16],
            ),
            &device,
        );
        let amps = ComplexTensor::new(
            prepared.amplitudes().real().select(1, indices.clone()) * 2.,
            prepared.amplitudes().imag().select(1, indices) * 2.,
        )
        .unwrap();
        let branch =
            StateVector::from_amplitudes(Register::qubits(4).unwrap(), amps, engine.limits)
                .unwrap();
        let mut correction = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
        if trim & 2 != 0 {
            correction.unitary(vec![3], Unitary::z()).unwrap();
        }
        if trim & 1 != 0 {
            correction.unitary(vec![0], Unitary::z()).unwrap();
        }
        correction
            .unitary(vec![0], Unitary::h())
            .unwrap()
            .unitary(vec![2], Unitary::h())
            .unwrap();
        let actual = correction.run_from(&engine, branch, None, None).unwrap();
        actual.validate(3e-6).unwrap();
        close(&values(actual.fidelity(&ideal).unwrap()), &[1.], 3e-6);
    }
}
#[test]
fn accepted_six_ring_checks_are_complete_entangled_projector_channels() {
    for k in [1, 2, 3, 4, 6] {
        for pattern in 0..3 {
            let observable = PauliProduct::new(
                (0..k)
                    .map(|q| {
                        (
                            q,
                            match pattern {
                                0 => Pauli::X,
                                1 => Pauli::Z,
                                _ => [Pauli::X, Pauli::Y, Pauli::Z][q % 3],
                            },
                        )
                    })
                    .collect(),
            )
            .unwrap();
            let initial = entangled(k);
            let original: Vec<_> = (0..k)
                .flat_map(|q| {
                    [
                        p(&[(q, Pauli::X), (q + k, Pauli::X)]),
                        p(&[(q, Pauli::Z), (q + k, Pauli::Z)]),
                    ]
                })
                .collect();
            let anti: Vec<_> = original
                .iter()
                .filter(|g| g.anticommutes(&observable))
                .collect();
            let mut witnesses: Vec<_> = original
                .iter()
                .filter(|g| !g.anticommutes(&observable))
                .cloned()
                .collect();
            for g in anti.iter().skip(1) {
                witnesses.push(anti[0].multiply_commuting(g).unwrap().1);
            }
            witnesses.push(observable.clone());
            assert_eq!(witnesses.len(), 2 * k);
            let mut signs = [false; 2];
            for seed in 0..16 {
                let mut actual = initial.clone();
                let result = SixRingCheckInstrument::IDEAL
                    .measure(
                        &mut actual,
                        &observable,
                        scratch(2 * k),
                        SixRingSampling::ConditionedReference,
                        seed,
                    )
                    .unwrap();
                let measurement = result.measurement.unwrap();
                signs[usize::from(measurement.negative)] = true;
                let mut expected = initial.clone();
                expected.measure(&observable, measurement.negative).unwrap();
                for witness in &witnesses {
                    assert_eq!(
                        actual.expectation(witness).unwrap(),
                        expected.expectation(witness).unwrap(),
                        "k={k}, pattern={pattern}, seed={seed}, witness={witness:?}"
                    );
                }
                assert_eq!(result.cost.rings_emitted, k);
                assert_eq!(result.cost.resource_photons, 6 * k);
                assert_eq!(result.cost.fusions, 2 * k);
                assert_eq!(result.cost.trim_measurements, 2 * k);
                assert_eq!(result.cost.syndrome_photons, 1);
                assert_eq!(result.cost.syndrome_measurements, 1);
                assert_eq!(result.log_herald_weight, Some(-(k as f64) * 4f64.ln()));
            }
            assert_eq!(signs, [true, true]);
        }
    }
}
#[test]
fn physical_sampling_retains_failures_losses_and_complete_costs() {
    let observable = p(&[(0, Pauli::Z)]);
    let mut accepted = 0;
    let mut emitted = 0;
    let mut rejected = 0;
    for seed in 0..2048 {
        let mut state = entangled(1);
        let result = SixRingCheckInstrument::IDEAL
            .measure(
                &mut state,
                &observable,
                scratch(2),
                SixRingSampling::Physical,
                seed,
            )
            .unwrap();
        emitted += result.cost.resource_photons;
        assert_eq!(result.log_herald_weight, None);
        if result.measurement.is_some() {
            accepted += 1;
        } else {
            rejected += 1;
            assert_eq!(result.failure, Some(OpticalCheckFailure::FusionRejected));
        }
        assert!((1..=2).contains(&result.cost.fusions));
    }
    assert_eq!(accepted + rejected, 2048);
    assert_eq!(emitted, 2048 * 6);
    assert!((accepted as f64 / 2048. - 0.25).abs() < 0.04);
    for (model, failure, photons, fusions) in [
        (
            SixRingCheckInstrument::new(0., 1., 1., PauliNoise::NONE).unwrap(),
            OpticalCheckFailure::MissingResource,
            0,
            0,
        ),
        (
            SixRingCheckInstrument::new(1., 0., 1., PauliNoise::NONE).unwrap(),
            OpticalCheckFailure::MissingTrim,
            6,
            0,
        ),
        (
            SixRingCheckInstrument::new(1., 1., 0., PauliNoise::NONE).unwrap(),
            OpticalCheckFailure::FusionLoss,
            6,
            1,
        ),
    ] {
        let result = model
            .measure(
                &mut entangled(1),
                &observable,
                scratch(2),
                SixRingSampling::Physical,
                1,
            )
            .unwrap();
        assert!(result.measurement.is_none());
        assert_eq!(result.failure, Some(failure));
        assert_eq!(result.cost.ring_attempts, 1);
        assert_eq!(result.cost.resource_photons, photons);
        assert_eq!(result.cost.fusions, fusions);
        assert!(
            model
                .measure(
                    &mut entangled(1),
                    &observable,
                    scratch(2),
                    SixRingSampling::ConditionedReference,
                    1
                )
                .is_err()
        );
    }
    let model = SixRingCheckInstrument::new(0.91, 0.87, 0.83, PauliNoise::NONE).unwrap();
    let expected = 0.91 * 0.87f64.powi(5) * 0.83f64.powi(2) * 0.25;
    assert!((model.herald_probability(1).unwrap() - expected).abs() < 1e-15);
    let count = (0..4096)
        .filter(|&seed| {
            model
                .measure(
                    &mut entangled(1),
                    &observable,
                    scratch(2),
                    SixRingSampling::Physical,
                    seed,
                )
                .unwrap()
                .measurement
                .is_some()
        })
        .count();
    assert!((count as f64 / 4096. - expected).abs() < 0.025);
}
#[test]
fn source_faults_remain_false_heralds_instead_of_being_declared_ideal() {
    // Z on every ring site becomes X(data) Z(syndrome) after the graph-to-Choi
    // basis changes and CNOT propagation. Trim-site Z phases are unobservable.
    let model =
        SixRingCheckInstrument::new(1., 1., 1., PauliNoise::new(0., 0., 1.).unwrap()).unwrap();
    let observable = p(&[(0, Pauli::Z)]);
    let mut accepted = 0;
    for seed in 0..256 {
        let mut state = entangled(1);
        let attempt = model
            .measure(
                &mut state,
                &observable,
                scratch(2),
                SixRingSampling::Physical,
                seed,
            )
            .unwrap();
        if let Some(measurement) = attempt.measurement {
            accepted += 1;
            assert_eq!(
                state
                    .expectation(&p(&[(0, Pauli::Z), (1, Pauli::Z)]))
                    .unwrap(),
                -1
            );
            assert_eq!(
                state.expectation(&observable).unwrap(),
                if measurement.negative { 1 } else { -1 }
            );
            assert_eq!(attempt.failure, None);
        }
    }
    assert!(accepted > 30);
    assert_eq!(model.herald_probability(1).unwrap(), 0.25);
}

#[test]
fn growing_distance_surgery_executes_photonic_checks_and_exposes_negative_herald_scaling() {
    let mut previous = 0;
    for d in [3, 5, 7] {
        let surgery = PlanarSurgery::new(d, MergeBasis::ZZ, Default::default()).unwrap();
        let compiled = SixRingSurgery::compile(&surgery, Default::default()).unwrap();
        assert!(compiled.scheduled_rings() > previous);
        previous = compiled.scheduled_rings();
        assert!(compiled.ideal_log_herald_probability() < -100.);
        // Every distance executes all resource preparation, trimming and fusions
        // in the explicitly conditioned channel reference; no probability is hidden.
        let n = surgery.code().qubits();
        let mut state = Tableau::zero(2 * n + 9, Default::default()).unwrap();
        for block in 0..2 {
            let reference = 2 * n + block;
            state.h(reference).unwrap();
            state
                .cnot(reference, block * n + surgery.code().input_site())
                .unwrap();
            surgery.code().encode(&mut state, block * n).unwrap();
        }
        let run = compiled
            .execute(
                &mut state,
                [0, n],
                scratch(2 * n + 2),
                SixRingCheckInstrument::IDEAL,
                SixRingSampling::ConditionedReference,
                41,
            )
            .unwrap();
        assert!(
            matches!(run.logical, Ok(SurgeryAttempt::Complete(_))),
            "{:?}",
            run.logical
        );
        let parity = match &run.logical {
            Ok(SurgeryAttempt::Complete(record)) => record.parity,
            _ => unreachable!(),
        };
        let za = surgery.code().logical_z(0).unwrap();
        let zb = surgery.code().logical_z(n).unwrap();
        let xa = surgery.code().logical_x(0).unwrap();
        let xb = surgery.code().logical_x(n).unwrap();
        for (logical, reference) in [(&za, 2 * n), (&zb, 2 * n + 1)] {
            let witness = logical
                .multiply_commuting(&p(&[(reference, Pauli::Z)]))
                .unwrap()
                .1;
            assert_eq!(state.expectation(&witness).unwrap(), 1);
        }
        let joint_x = xa
            .multiply_commuting(&xb)
            .unwrap()
            .1
            .multiply_commuting(&p(&[(2 * n, Pauli::X), (2 * n + 1, Pauli::X)]))
            .unwrap()
            .1;
        assert_eq!(state.expectation(&joint_x).unwrap(), 1);
        let joint_z = za.multiply_commuting(&zb).unwrap().1;
        assert_eq!(
            state.expectation(&joint_z).unwrap(),
            if parity { -1 } else { 1 }
        );
        for check in surgery.split_checks() {
            assert_eq!(state.expectation(check).unwrap(), 1);
        }
        assert_eq!(run.cost.rings_emitted, compiled.scheduled_rings());
        assert_eq!(run.checks.len(), compiled.scheduled_checks());
        assert_eq!(
            run.log_herald_weight,
            Some(compiled.ideal_log_herald_probability())
        );
        let mut unavailable = Tableau::zero(2 * n + 7, Default::default()).unwrap();
        for block in 0..2 {
            surgery.code().encode(&mut unavailable, block * n).unwrap();
        }
        let rejected = compiled
            .execute(
                &mut unavailable,
                [0, n],
                scratch(2 * n),
                SixRingCheckInstrument::new(0., 1., 1., PauliNoise::NONE).unwrap(),
                SixRingSampling::Physical,
                41,
            )
            .unwrap();
        assert!(matches!(rejected.logical, Ok(SurgeryAttempt::Aborted(_))));
        assert_eq!(rejected.checks.len(), 1);
        assert_eq!(rejected.cost.ring_attempts, 1);
    }
}
#[test]
fn optical_surgery_rejects_invalid_capabilities_and_budgets_before_mutation() {
    assert!(SixRingCheckInstrument::new(f64::NAN, 1., 1., PauliNoise::NONE).is_err());
    assert!(SixRingCheckInstrument::new(1., 1.01, 1., PauliNoise::NONE).is_err());
    let mut state = entangled(1);
    let observable = p(&[(0, Pauli::Z)]);
    let before = state
        .expectation(&p(&[(0, Pauli::X), (1, Pauli::X)]))
        .unwrap();
    assert!(
        SixRingCheckInstrument::IDEAL
            .measure(
                &mut state,
                &observable,
                [0, 2, 3, 4, 5, 6, 7],
                SixRingSampling::Physical,
                1
            )
            .is_err()
    );
    assert_eq!(
        state
            .expectation(&p(&[(0, Pauli::X), (1, Pauli::X)]))
            .unwrap(),
        before
    );
    assert!(SixRingCheckInstrument::IDEAL.herald_probability(0).is_err());
    assert!(SixRingCheckInstrument::IDEAL.herald_probability(7).is_err());
    let s = PlanarSurgery::new(3, MergeBasis::XX, Default::default()).unwrap();
    assert!(
        SixRingSurgery::compile(
            &s,
            SixRingSurgeryLimits {
                max_rings: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        SixRingSurgery::compile(
            &s,
            SixRingSurgeryLimits {
                max_checks: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}
