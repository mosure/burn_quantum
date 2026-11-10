//! Physical phase/splitter/PNR tomography, with independent mixed-state formulas.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Tensor, TensorData};
use burn_quantum::{photonic::optics::*, *};
use common::{close, values};
fn mixed(phase: f64, eta: f64, radius: f64) -> FockDensity {
    let device = runtime::execution().unwrap().device;
    let re = Tensor::from_data(
        TensorData::new(
            vec![
                1. - eta,
                0.,
                0.,
                0.,
                eta * (1. - radius) / 2.,
                0.,
                0.,
                0.,
                eta * (1. + radius) / 2.,
            ],
            [1, 3, 3],
        ),
        (&device, DType::F32),
    );
    FockDensity::from_entries(
        FockBasis::new(2, PhotonSector::UpTo(1), 3).unwrap(),
        ComplexTensor::new(re.clone(), re.zeros_like()).unwrap(),
        Default::default(),
    )
    .unwrap()
    .apply(
        &Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)
            .unwrap(),
        Default::default(),
    )
    .unwrap()
    .phase_shift(
        ModeId(0),
        Tensor::from_data([phase as f32], &device),
        Default::default(),
    )
    .unwrap()
}
#[test]
fn tomography_reconstructs_complex_mixed_states_with_every_no_click_counted() {
    let rho = mixed(0.6, 0.8, 0.6);
    let protocol = OpticalTomography::new(8192, 2, 24576, 0.01).unwrap();
    let a = protocol.reconstruct(&rho, 37, Default::default()).unwrap();
    let b = protocol.reconstruct(&rho, 37, Default::default()).unwrap();
    assert_eq!(a.histograms, b.histograms);
    assert_eq!(a.attempts, 24576);
    assert_eq!(a.settings.len(), 3);
    for h in &a.histograms {
        assert_eq!(h.iter().sum::<usize>(), 8192);
        assert!((h[0] as f64 / 8192. - 0.2).abs() < 0.02);
    }
    close(
        &values(a.state.entries().real()),
        &values(rho.entries().real()),
        0.025,
    );
    close(
        &values(a.state.entries().imag()),
        &values(rho.entries().imag()),
        0.025,
    );
    a.state.validate(3e-6).unwrap();
    assert!(a.trace_distance_bound > 0. && a.trace_distance_bound < 0.2);
    assert!(
        OpticalTomography::new(8192, 2, 24575, 0.01)
            .unwrap()
            .reconstruct(&rho, 37, Default::default())
            .is_err()
    );
}
#[test]
fn finite_shot_overlap_matches_lossy_bloch_formula_across_independent_seeds() {
    let eta = 0.8f64;
    let r = 0.6f64;
    let delta = 1.1f64;
    let reference = (1. - eta + eta * (1. - r * r * (delta / 2.).sin().powi(2)).sqrt()).powi(2);
    let left = mixed(0.6, eta, r);
    let right = mixed(0.6 + delta, eta, r);
    let protocol = OpticalTomography::new(4096, 2, 24576, 0.01).unwrap();
    for seed in [7, 11, 19, 23] {
        let result = protocol
            .overlap(&left, &right, seed, Default::default())
            .unwrap();
        assert!(
            (result.squared_fidelity - reference).abs() < 0.025,
            "{} != {reference}",
            result.squared_fidelity
        );
        assert_eq!(result.attempts, 24576);
        assert_eq!(result.failure_probability, 0.02);
        assert!((result.squared_fidelity - reference).abs() <= result.absolute_error_bound);
    }
}
#[test]
fn passive_protocol_rejects_unidentifiable_vacuum_coherence_and_multiphoton_sectors() {
    let device = runtime::execution().unwrap().device;
    let re = Tensor::from_data(
        [[
            std::f32::consts::FRAC_1_SQRT_2,
            0.,
            std::f32::consts::FRAC_1_SQRT_2,
        ]],
        &device,
    );
    let rho = FockState::from_amplitudes(
        FockBasis::new(2, PhotonSector::UpTo(1), 3).unwrap(),
        ComplexTensor::new(re.clone(), re.zeros_like()).unwrap(),
        Default::default(),
    )
    .unwrap()
    .density(Default::default())
    .unwrap();
    let protocol = OpticalTomography::new(8, 2, 48, 0.05).unwrap();
    assert!(protocol.reconstruct(&rho, 1, Default::default()).is_err());
    let two = FockState::number(
        FockBasis::new(2, PhotonSector::Fixed(2), 3).unwrap(),
        &[1, 1],
        1,
        &device,
        DType::F32,
        Default::default(),
    )
    .unwrap()
    .density(Default::default())
    .unwrap();
    assert!(protocol.reconstruct(&two, 1, Default::default()).is_err());
}
#[test]
fn qnspsa_consumes_measured_mixed_state_overlaps_and_reports_all_probe_costs() {
    use burn_quantum::optim::QnSpsaConfig;
    use std::cell::Cell;
    let device = runtime::execution().unwrap().device;
    let protocol = OpticalTomography::new(4096, 2, 24576, 0.01).unwrap();
    let attempts = Cell::new(0usize);
    let identity = Cell::new(0u64);
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)
        .unwrap();
    let mut loss = |theta: Tensor<1>| -> burn_quantum::Result<f64> {
        let angle = theta.into_scalar::<f64>();
        let readout = mixed(angle, 0.8, 0.6)
            .apply(&bs.adjoint(), Default::default())?
            .detect(&[Detector::ideal(1)?; 2], 4, Default::default())?;
        let values: Vec<_> = readout
            .records()
            .iter()
            .map(|r| if r == &[1, 0] { -1. } else { 0. })
            .collect();
        let seed = identity.get();
        identity.set(seed + 1);
        attempts.set(attempts.get() + 512);
        Ok(readout.sample_readout(&values, 512, seed + 71, 512)?[0].mean)
    };
    let mut fidelity = |a: Tensor<1>, b: Tensor<1>| -> burn_quantum::Result<f64> {
        let seed = identity.get();
        identity.set(seed + 1);
        let result = protocol.overlap(
            &mixed(a.into_scalar::<f64>(), 0.8, 0.6),
            &mixed(b.into_scalar::<f64>(), 0.8, 0.6),
            seed + 251,
            Default::default(),
        )?;
        attempts.set(attempts.get() + result.attempts);
        Ok(result.squared_fidelity)
    };
    let mut optimizer = QnSpsaConfig::new()
        .with_seed(7)
        .with_learning_rate(0.08)
        .with_learning_rate_decay(0.)
        .with_perturbation(0.3)
        .with_perturbation_decay(0.)
        .with_initial_metric_scale(0.072)
        .with_regularization(0.05)
        .init(1)
        .unwrap();
    let mut parameters = Tensor::from_data([1.2f32], &device);
    for _ in 0..2 {
        let update = optimizer
            .step(parameters, &mut loss, &mut fidelity)
            .unwrap();
        assert_eq!(update.report.loss_evaluations, 2);
        assert_eq!(update.report.fidelity_evaluations, 4);
        parameters = update.parameters;
    }
    let phase = parameters.into_scalar::<f64>();
    assert!(
        phase < 1.2 && phase > 0.,
        "finite-shot natural update moved to {phase}"
    );
    assert!(0.8 * (1. + 0.6 * phase.cos()) / 2. > 0.8 * (1. + 0.6 * 1.2f64.cos()) / 2. + 0.001);
    assert_eq!(attempts.get(), 2 * (2 * 512 + 4 * 24576));
    let model = OpticalCostModel {
        mean_photons_per_attempt: 1.,
        seconds_per_attempt: 1e-6,
        joules_per_attempt: 1e-12,
        parallel_lanes: 4,
    };
    let cost = model.cost(attempts.get()).unwrap();
    assert_eq!(cost.expected_photons, attempts.get() as f64);
    assert!((cost.serial_seconds / 4. - cost.ideal_parallel_seconds).abs() < 1e-12);
}
#[test]
fn minibatch_shots_track_completed_work_uncertainty_and_hard_caps() {
    let policy = ShotAllocationPolicy {
        minimum_per_item: 10,
        maximum_per_item: 200,
        round_budget: 180,
        total_budget: 200,
        variance_floor: 1e-6,
    };
    let items = [
        ShotAllocationItem {
            completed: 10,
            variance: 0.04,
            coefficient: 1.,
        },
        ShotAllocationItem {
            completed: 10,
            variance: 0.16,
            coefficient: 1.,
        },
    ];
    let plan = policy.allocate(&items).unwrap();
    assert_eq!(plan.reserved_attempts, 180);
    assert_eq!(plan.completed, [10, 10]);
    assert_eq!(plan.additional.iter().sum::<usize>(), 180);
    assert!((plan.planned_totals[1] as f64 / plan.planned_totals[0] as f64 - 2.).abs() < 0.04);
    let variance = 0.04 / plan.planned_totals[0] as f64 + 0.16 / plan.planned_totals[1] as f64;
    assert!((plan.estimated_standard_error.powi(2) - variance).abs() < 1e-12);
    assert!(
        ShotAllocationPolicy {
            round_budget: 1,
            ..policy
        }
        .allocate(&[
            ShotAllocationItem {
                completed: 0,
                ..items[0]
            },
            items[1]
        ])
        .is_err()
    );
    let capped = ShotAllocationPolicy {
        maximum_per_item: 100,
        ..policy
    }
    .allocate(&items)
    .unwrap();
    assert_eq!(capped.planned_totals, [100, 100]);
    let mut progress = ShotAccumulator::default();
    assert!(progress.estimate().is_none());
    progress
        .merge(ShotEstimate {
            mean: 0.25,
            variance: 0.25,
            standard_error: 0.25,
            attempts: 4,
        })
        .unwrap();
    progress
        .merge(ShotEstimate {
            mean: 0.75,
            variance: 0.25,
            standard_error: 0.25,
            attempts: 4,
        })
        .unwrap();
    let estimate = progress.estimate().unwrap();
    close(&[estimate.mean, estimate.variance], &[0.5, 2. / 7.], 1e-12);
    assert_eq!(estimate.attempts, 8);
    let before = serde_json::to_string(&progress).unwrap();
    assert!(
        progress
            .merge(ShotEstimate {
                mean: f64::NAN,
                variance: 0.,
                standard_error: 0.,
                attempts: 2
            })
            .is_err()
    );
    assert_eq!(before, serde_json::to_string(&progress).unwrap());
}
