//! Explicit environments and independent calibrated instrument references.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{Tensor, TensorData};
use burn_quantum::{
    photonic::{optics::*, *},
    *,
};
use common::{close, values};
use num_complex::Complex64;
fn density(psi: Vec<Complex64>) -> DensityMatrix {
    let device = runtime::execution().unwrap().device;
    let re = Tensor::from_data(
        TensorData::new(psi.iter().map(|z| z.re as f32).collect(), [1, 9]),
        &device,
    );
    let im = Tensor::from_data(
        TensorData::new(psi.iter().map(|z| z.im as f32).collect(), [1, 9]),
        &device,
    );
    DensityMatrix::from_pure(
        &StateVector::from_amplitudes(
            DualRail::register(2).unwrap(),
            ComplexTensor::new(re, im).unwrap(),
            Default::default(),
        )
        .unwrap(),
        Default::default(),
    )
    .unwrap()
}
fn bell(sign: f64) -> DensityMatrix {
    let mut p = vec![Complex64::new(0., 0.); 9];
    p[1] = std::f64::consts::FRAC_1_SQRT_2.into();
    p[3] = (sign * std::f64::consts::FRAC_1_SQRT_2).into();
    density(p)
}
#[test]
fn explicit_environment_grice_matches_old_ideal_and_balanced_loss_effects() {
    for eta in [1., 0.73] {
        let old = OpticalBellInstrument::grice(GriceConfig {
            signal_transmissions: [eta; 2],
            ancilla_transmissions: [eta; 2],
            detector_efficiencies: [0.91; 8],
            ..Default::default()
        })
        .unwrap();
        let new = OpticalBellInstrument::grice_calibrated(GriceNoiseConfig {
            rail_transmissions: [[eta; 2]; 4],
            detectors: [Detector::new(0.91, 4, 0.).unwrap(); 8],
            ..Default::default()
        })
        .unwrap();
        for (a, b) in old.effects().iter().zip(new.effects()) {
            assert_eq!(a.record(), b.record());
            common::complex_close(a.matrix().entries(), b.matrix().entries(), 2e-12);
        }
    }
    assert!(
        OpticalBellInstrument::grice_calibrated(GriceNoiseConfig {
            max_terms: 1,
            ..Default::default()
        })
        .is_err()
    );
    assert!(
        OpticalBellInstrument::grice_calibrated(GriceNoiseConfig {
            max_work: 1,
            ..Default::default()
        })
        .is_err()
    );
}
#[test]
fn asymmetric_signal_rails_preserve_exact_filter_coherences() {
    let ideal = OpticalBellInstrument::grice(GriceConfig::default()).unwrap();
    let eta = [[0.7f64, 0.3], [0.4, 0.9], [1., 1.], [1., 1.]];
    let phases = [[0.2, -0.3], [0.5, 0.7], [0., 0.], [0., 0.]];
    let actual = OpticalBellInstrument::grice_calibrated(GriceNoiseConfig {
        rail_transmissions: eta,
        rail_phases: phases,
        ..Default::default()
    })
    .unwrap();
    for outcome in 0..5 {
        for a in 0..9 {
            for b in 0..9 {
                let filter = |i: usize| {
                    if i / 3 < 2 && i % 3 < 2 {
                        Complex64::from_polar(
                            (eta[0][i / 3] * eta[1][i % 3]).sqrt(),
                            phases[0][i / 3] + phases[1][i % 3],
                        )
                    } else {
                        Complex64::new(0., 0.)
                    }
                };
                let expected = ideal.effects()[outcome].matrix().entries()[a * 9 + b]
                    * filter(a).conj()
                    * filter(b);
                assert!(
                    (actual.effects()[outcome].matrix().entries()[a * 9 + b] - expected).norm()
                        < 2e-12
                );
            }
        }
    }
}
#[test]
fn distinguishable_boosted_photons_generate_false_parity_declarations() {
    let packet = |i: usize| {
        (0..4)
            .map(|j| Complex64::new(f64::from(i == j), 0.))
            .collect()
    };
    let instrument = OpticalBellInstrument::grice_calibrated(GriceNoiseConfig {
        wavepackets: std::array::from_fn(packet),
        ..Default::default()
    })
    .unwrap();
    let probabilities = |sign| {
        instrument
            .branches(&DenseSimulator::default(), &bell(sign), [0, 1])
            .unwrap()
            .into_iter()
            .map(|b| values(b.probability())[0])
            .collect::<Vec<_>>()
    };
    let plus = probabilities(1.);
    let minus = probabilities(-1.);
    close(&plus, &minus, 3e-6);
    assert!(plus[0] > 0.01 && plus[1] > 0.01);
    close(&[plus.iter().sum()], &[1.], 3e-6);
}
#[test]
fn boosted_vacuum_false_heralds_match_pinned_slos_and_independent_dark_convolution() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("optical_noise_perceval_v1_3.json")).unwrap();
    common::pinned_perceval_metadata(&fixture);
    assert_eq!(
        fixture["grice"]["bell_order"],
        serde_json::json!(["psi+", "psi-", "phi+", "phi-"])
    );
    let patterns: Vec<Vec<usize>> =
        serde_json::from_value(fixture["grice"]["basis"].clone()).unwrap();
    let bell_prob: Vec<Vec<f64>> =
        serde_json::from_value(fixture["grice"]["probabilities"].clone()).unwrap();
    let basis: Vec<Vec<usize>> =
        serde_json::from_value(fixture["grice_vacuum_signals"]["basis"].clone()).unwrap();
    let probs: Vec<f64> =
        serde_json::from_value(fixture["grice_vacuum_signals"]["probabilities"].clone()).unwrap();
    assert_eq!(
        patterns,
        FockBasis::new(8, PhotonSector::Fixed(4), 330)
            .unwrap()
            .occupations()
    );
    assert_eq!(bell_prob.len(), 4);
    assert!(bell_prob.iter().all(|row| row.len() == 330));
    assert_eq!(
        basis,
        FockBasis::new(8, PhotonSector::Fixed(2), 36)
            .unwrap()
            .occupations()
    );
    assert_eq!(probs.len(), 36);
    assert!(probs.iter().all(|&p| p.is_finite() && p >= 0.));
    close(&[probs.iter().sum()], &[1.], 1e-12);
    let dark = 0.07f64;
    let mut expected = [0.; 6];
    for (occupation, p) in basis.iter().zip(probs) {
        for mask in 0u32..256 {
            let clicks = mask.count_ones() as i32;
            let weight = p * dark.powi(clicks) * (1. - dark).powi(8 - clicks);
            let measured: Vec<_> = occupation
                .iter()
                .enumerate()
                .map(|(i, n)| n + usize::from(mask & (1 << i) != 0))
                .collect();
            let label = patterns
                .iter()
                .position(|r| r == &measured)
                .map(|index| {
                    let support: Vec<_> = (0..4).filter(|&b| bell_prob[b][index] > 1e-12).collect();
                    match support.as_slice() {
                        [only] => *only,
                        [2, 3] => 4,
                        _ => 5,
                    }
                })
                .unwrap_or(5);
            expected[label] += weight;
        }
    }
    let instrument = OpticalBellInstrument::grice_calibrated(GriceNoiseConfig {
        detectors: [Detector::new(1., 4, dark).unwrap(); 8],
        ..Default::default()
    })
    .unwrap();
    let mut vacuum = vec![Complex64::new(0., 0.); 9];
    vacuum[8] = 1.0.into();
    let actual = instrument
        .branches(&DenseSimulator::default(), &density(vacuum), [0, 1])
        .unwrap()
        .into_iter()
        .map(|b| values(b.probability())[0])
        .collect::<Vec<_>>();
    close(&actual, &expected, 3e-6);
    assert!(actual[..4].iter().sum::<f64>() > 0.001);
}

#[test]
fn joint_source_and_correlated_phase_loss_are_not_independent_marginal_models() {
    let device = runtime::execution().unwrap().device;
    let source = JointPhotonSource::new(
        2,
        vec![(vec![0, 0], 0.3), (vec![1, 1], 0.7)],
        "pair-source".into(),
    )
    .unwrap();
    let basis = FockBasis::new(2, PhotonSector::UpTo(2), 6).unwrap();
    let rho = source
        .prepare(
            basis.clone(),
            1,
            &device,
            burn::tensor::DType::F32,
            Default::default(),
        )
        .unwrap();
    let probs = values(rho.probabilities());
    close(
        &[
            probs[basis.index(&[0, 0]).unwrap()],
            probs[basis.index(&[1, 1]).unwrap()],
        ],
        &[0.3, 0.7],
        2e-6,
    );
    assert_eq!(probs[basis.index(&[1, 0]).unwrap()], 0.);
    assert!(
        source
            .prepare(
                FockBasis::new(2, PhotonSector::UpTo(1), 3).unwrap(),
                1,
                &device,
                burn::tensor::DType::F32,
                Default::default()
            )
            .is_err()
    );
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)
        .unwrap();
    let state = FockState::number(
        FockBasis::new(2, PhotonSector::UpTo(1), 3).unwrap(),
        &[1, 0],
        1,
        &device,
        burn::tensor::DType::F32,
        Default::default(),
    )
    .unwrap()
    .apply(&bs, Default::default())
    .unwrap()
    .density(Default::default())
    .unwrap();
    let noise = CalibratedNoiseEnsemble::new(
        vec![
            OpticalNoiseBranch {
                probability: 0.5,
                transmissions: vec![0.9; 2],
                phases: vec![0.4, 0.],
            },
            OpticalNoiseBranch {
                probability: 0.5,
                transmissions: vec![0.3; 2],
                phases: vec![-0.7, 0.],
            },
        ],
        "shared-latent".into(),
    )
    .unwrap();
    let actual = noise.apply(&state, Default::default()).unwrap();
    let re = values(actual.entries().real());
    let im = values(actual.entries().imag());
    let coherence = (Complex64::from_polar(0.9, 0.4) + Complex64::from_polar(0.3, -0.7)) / 4.;
    close(&[re[0], re[4], re[8]], &[0.4, 0.3, 0.3], 3e-6);
    close(&[re[7], im[7]], &[coherence.re, coherence.im], 3e-6);
    actual.validate(3e-6).unwrap();
    let factorized = 0.6 * (Complex64::from_polar(1., 0.4) + Complex64::from_polar(1., -0.7)) / 4.;
    assert!((coherence - factorized).norm() > 0.05);
}
#[test]
fn fringe_calibration_identifies_only_effective_response_and_rejects_singular_designs() {
    let eta = 0.72f64;
    let v = 0.83;
    let offset = 0.37;
    let probes: Vec<_> = [
        0.,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        3. * std::f64::consts::FRAC_PI_2,
    ]
    .into_iter()
    .map(|phase| {
        let left = (100_000. * eta * (1. + v * (phase + offset).cos()) / 2.).round() as usize;
        CalibrationProbe {
            phase,
            counts: [28_000, left, 72_000 - left],
        }
    })
    .collect();
    let fit = MachZehnderCalibration::fit(&probes, 0.05).unwrap();
    close(
        &[fit.survival, fit.visibility, fit.phase_offset],
        &[eta, v, offset],
        4e-5,
    );
    assert_eq!(fit.attempts, 400_000);
    for phase in [0.2f64, 0.9, 2.1, 4.4] {
        close(
            &fit.probabilities(phase).unwrap(),
            &[
                1. - eta,
                eta * (1. + v * (phase + offset).cos()) / 2.,
                eta * (1. - v * (phase + offset).cos()) / 2.,
            ],
            3e-5,
        );
    }
    assert!(MachZehnderCalibration::fit(&[probes[0], probes[0]], 0.05).is_err());
    assert!(
        MachZehnderCalibration::fit(
            &[
                CalibrationProbe {
                    phase: 0.,
                    counts: [20, 40, 40]
                },
                CalibrationProbe {
                    phase: std::f64::consts::FRAC_PI_2,
                    counts: [20, 40, 40]
                }
            ],
            0.05
        )
        .is_err()
    );
}
