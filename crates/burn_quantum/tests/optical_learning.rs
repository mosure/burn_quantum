//! Independent analytic and finite-shot references for physical photonic learning.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Tensor, TensorData};
use burn_quantum::{ComplexTensor, ResourceLimits, photonic::optics::*};
use common::{close, values};
fn device() -> burn::tensor::Device {
    runtime::execution().unwrap().device
}
fn preparation(batch: usize) -> FockState {
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)
        .unwrap();
    FockState::number(
        FockBasis::new(2, PhotonSector::UpTo(1), 3).unwrap(),
        &[1, 0],
        batch,
        &device(),
        DType::F32,
        Default::default(),
    )
    .unwrap()
    .apply(&bs, Default::default())
    .unwrap()
}
fn experiment(batch: usize, efficiency: f64) -> SinglePhotonPhaseExperiment {
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)
        .unwrap();
    SinglePhotonPhaseExperiment::new(
        preparation(batch),
        ModeId(0),
        bs.adjoint(),
        vec![Detector::threshold(efficiency, 0.).unwrap(); 2],
        4,
        Default::default(),
    )
    .unwrap()
}
#[test]
fn finite_shot_phase_probes_have_correct_derivative_error_and_physical_cost() {
    let phase_experiment = experiment(3, 0.8);
    let x = Tensor::from_data([0.3f32, 0.8, 1.4], &device());
    let exact = phase_experiment.evaluate(x.clone()).unwrap();
    let readout: Vec<_> = exact
        .records()
        .iter()
        .map(|r| f64::from(r == &[1, 0]))
        .collect();
    let gradients = phase_experiment
        .gradient(x.clone(), &readout, 20_000, 151, 120_000)
        .unwrap();
    let again = phase_experiment
        .gradient(x.clone(), &readout, 20_000, 151, 120_000)
        .unwrap();
    for ((g, r), angle) in gradients.iter().zip(again).zip([0.3f64, 0.8, 1.4]) {
        assert_eq!(g.gradient, r.gradient);
        assert_eq!(g.attempts, 40_000);
        assert!((g.gradient + 0.4 * angle.sin()).abs() < 7. * g.standard_error + 1e-6);
    }
    assert!(
        phase_experiment
            .gradient(x, &readout, 20_000, 151, 119_999)
            .is_err()
    );
    let histograms = exact.sample_counts(20_000, 17, 60_000).unwrap();
    let no_click = exact.records().iter().position(|r| r == &[0, 0]).unwrap();
    for h in histograms {
        assert_eq!(h.counts.iter().sum::<usize>(), h.attempts);
        assert!((h.counts[no_click] as f64 / 20_000. - 0.2).abs() < 0.02);
    }
    let zero = experiment(1, 0.)
        .evaluate(Tensor::from_data([0.7f32], &device()))
        .unwrap();
    let estimate = zero.sample_readout(&[0., 1., 1., 2.], 16, 1, 16).unwrap()[0];
    assert_eq!(
        (estimate.mean, estimate.variance, estimate.attempts),
        (0., 0., 16)
    );
    let two = FockState::number(
        FockBasis::new(2, PhotonSector::Fixed(2), 3).unwrap(),
        &[1, 1],
        1,
        &device(),
        DType::F32,
        Default::default(),
    )
    .unwrap();
    assert!(
        SinglePhotonPhaseExperiment::new(
            two,
            ModeId(0),
            Interferometer::phase(2, ModeId(0), 0.).unwrap(),
            vec![Detector::ideal(2).unwrap(); 2],
            9,
            Default::default()
        )
        .is_err()
    );
}
#[test]
fn finite_shot_classical_fisher_matches_lossy_mz_reference() {
    // P=(1-eta, eta sin²(theta/2), eta cos²(theta/2), 0) gives I=eta.
    let experiment = experiment(1, 0.8);
    let theta = Tensor::from_data([0.7f32], &device());
    let metric = experiment
        .classical_fisher(theta.clone(), 100_000, 1e-4, 97, 300_000)
        .unwrap()[0];
    assert!((metric.value - 0.8).abs() < 0.03, "{}", metric.value);
    assert_eq!(metric.attempts, 300_000);
    assert_eq!(metric.regularized_outcomes, 1); // structurally absent double click
    assert!(
        experiment
            .classical_fisher(theta, 100_000, 1e-4, 97, 299_999)
            .is_err()
    );
    assert!(
        experiment
            .classical_fisher(Tensor::from_data([0.7f32], &device()), 2, 1e-320, 97, 6)
            .is_err()
    );
}
fn feature(phases: &[f32], eta: f64) -> FockDensity {
    let phase = Tensor::from_data(TensorData::new(phases.to_vec(), [phases.len()]), &device());
    preparation(phases.len())
        .phase_shift(ModeId(0), phase, Default::default())
        .unwrap()
        .density(Default::default())
        .unwrap()
        .lose(ModeId(0), eta, Default::default())
        .unwrap()
        .lose(ModeId(1), eta, Default::default())
        .unwrap()
}
#[test]
fn native_mixed_kernel_matches_scalar_feature_map_and_retains_loss_mass() {
    let a = [-0.9f32, 0.2, 1.1];
    let b = [-0.2f32, 0.8];
    let eta = 0.7f64;
    let kernel = feature(&a, eta)
        .hilbert_schmidt_kernel(&feature(&b, eta), ResourceLimits::default())
        .unwrap();
    let oracle: Vec<_> = a
        .iter()
        .flat_map(|&x| {
            b.iter().map(move |&y| {
                (1. - eta).powi(2) + eta.powi(2) * ((x as f64 - y as f64) / 2.).cos().powi(2)
            })
        })
        .collect();
    close(&values(kernel), &oracle, 3e-6);
    let self_kernel = values(
        feature(&a, eta)
            .hilbert_schmidt_kernel(&feature(&a, eta), ResourceLimits::default())
            .unwrap(),
    );
    // Purity is below one: the vacuum and surviving-photon mixture was not normalized away.
    close(
        &[self_kernel[0], self_kernel[4], self_kernel[8]],
        &[0.58; 3],
        3e-6,
    );
}
fn bloch(radius: f64, angle: f64) -> FockDensity {
    let re = vec![
        (1. + radius * angle.cos()) / 2.,
        radius * angle.sin() / 2.,
        radius * angle.sin() / 2.,
        (1. - radius * angle.cos()) / 2.,
    ];
    let tensor = Tensor::from_data(TensorData::new(re, [1, 2, 2]), (&device(), DType::F32));
    FockDensity::from_entries(
        FockBasis::new(2, PhotonSector::Fixed(1), 2).unwrap(),
        ComplexTensor::new(tensor.clone(), tensor.zeros_like()).unwrap(),
        Default::default(),
    )
    .unwrap()
}
#[test]
fn squared_uhlmann_and_bures_match_independent_mixed_qubit_formula() {
    // A rotating Bloch vector of fixed radius r has F=1-r² sin²(delta/2), g_B=r²/4.
    for radius in [0., 0.4, 0.9, 1.] {
        let delta = 0.1f64;
        let center = bloch(radius, 0.3);
        let plus = center
            .uhlmann_fidelity(&bloch(radius, 0.3 + delta), 2, 2e-6)
            .unwrap()[0];
        let minus = center
            .uhlmann_fidelity(&bloch(radius, 0.3 - delta), 2, 2e-6)
            .unwrap()[0];
        let expected = 1. - radius.powi(2) * (delta / 2.).sin().powi(2);
        close(&[plus, minus], &[expected; 2], 2e-6);
        let metric = bures_metric_from_fidelities(plus, minus, delta).unwrap();
        assert!(
            (metric - radius.powi(2) / 4.).abs() < 5e-4,
            "radius={radius} metric={metric}"
        );
    }
    let a = bloch(0.4, 0.);
    let b = bloch(0.8, 0.);
    let commuting = (0.7f64.sqrt() * 0.9f64.sqrt() + 0.3f64.sqrt() * 0.1f64.sqrt()).powi(2);
    close(
        &a.uhlmann_fidelity(&b, 2, 2e-6).unwrap(),
        &[commuting],
        2e-6,
    );
    assert!(a.uhlmann_fidelity(&b, 1, 2e-6).is_err());
    assert!(a.uhlmann_fidelity(&bloch(1.2, 0.), 2, 2e-6).is_err());
    assert!(bures_metric_from_fidelities(1.1, 0.9, 0.1).is_err());
    assert!(bures_metric_from_fidelities(0.9, 0.9, 1e200).is_err());
    assert!(bures_metric_from_fidelities(0.9, 0.9, 1e-200).is_err());
}
