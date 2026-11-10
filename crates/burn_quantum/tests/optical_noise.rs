//! Physical optical noise, distinguishability, source and detector-event references.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Tensor, TensorData};
use burn_quantum::{
    photonic::{optics::*, *},
    *,
};
use common::{close, values};
use num_complex::Complex64;
fn device() -> burn::tensor::Device {
    runtime::execution().unwrap().device
}
fn density(data: Vec<Complex64>, sites: usize) -> DensityMatrix {
    let dev = device();
    let n = data.len();
    let state = StateVector::from_amplitudes(
        DualRail::register(sites).unwrap(),
        ComplexTensor::new(
            Tensor::from_data(
                TensorData::new(data.iter().map(|z| z.re as f32).collect(), [1, n]),
                &dev,
            ),
            Tensor::from_data(
                TensorData::new(data.iter().map(|z| z.im as f32).collect(), [1, n]),
                &dev,
            ),
        )
        .unwrap(),
        Default::default(),
    )
    .unwrap();
    DensityMatrix::from_pure(&state, Default::default()).unwrap()
}
#[test]
fn internal_modes_give_the_hom_overlap_law_without_postselection() {
    let layout = WavepacketLayout::new(2, 2).unwrap();
    let basis = FockBasis::new(4, PhotonSector::Fixed(2), 10).unwrap();
    let bs = layout
        .lift(
            &Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)
                .unwrap(),
        )
        .unwrap();
    for visibility in [0f64, 0.3, 0.8, 1.] {
        let a = PhotonPacket::new(0, vec![1.0.into(), 0.0.into()]).unwrap();
        let b = PhotonPacket::new(
            1,
            vec![visibility.sqrt().into(), (1. - visibility).sqrt().into()],
        )
        .unwrap();
        assert!((a.overlap(&b).unwrap().norm_sqr() - visibility).abs() < 1e-12);
        let state = FockState::wavepackets(
            basis.clone(),
            layout,
            &[a, b],
            1,
            &device(),
            DType::F32,
            Default::default(),
        )
        .unwrap()
        .apply(&bs, Default::default())
        .unwrap();
        let readout = state
            .detect_spatial(
                layout,
                &[Detector::ideal(2).unwrap(); 2],
                9,
                Default::default(),
            )
            .unwrap();
        let index = readout.records().iter().position(|x| x == &[1, 1]).unwrap();
        assert!((values(readout.probabilities())[index] - (1. - visibility) / 2.).abs() < 3e-6);
    }
}
#[test]
fn calibrated_source_moments_mixture_and_support_guards() {
    let source = CalibratedSource::vacuum_single_pair(0.6, 0.1, "source-2026-11".into()).unwrap();
    close(source.probabilities(), &[0.418, 0.564, 0.018], 1e-12);
    assert!((source.g2().unwrap() - 0.1).abs() < 1e-12);
    let basis = FockBasis::new(2, PhotonSector::UpTo(4), 15).unwrap();
    let rho = FockDensity::source_product(
        basis.clone(),
        &[source.clone(), source.clone()],
        1,
        &device(),
        DType::F32,
        Default::default(),
    )
    .unwrap();
    let expected: Vec<_> = basis
        .occupations()
        .iter()
        .map(|o| {
            source.probabilities().get(o[0]).unwrap_or(&0.)
                * source.probabilities().get(o[1]).unwrap_or(&0.)
        })
        .collect();
    close(&values(rho.probabilities()), &expected, 2e-6);
    rho.validate(2e-6).unwrap();
    assert!(
        FockDensity::source_product(
            FockBasis::new(2, PhotonSector::UpTo(1), 3).unwrap(),
            &[source.clone(), source],
            1,
            &device(),
            DType::F32,
            Default::default()
        )
        .is_err()
    );
    assert!(CalibratedSource::vacuum_single_pair(0.9, 3., "invalid".into()).is_err());
    assert!(
        serde_json::from_str::<CalibratedSource>(
            r#"{"probabilities":[1.1,-0.1],"calibration":"bad"}"#
        )
        .is_err()
    );
}
#[test]
fn nonparalyzable_dead_time_is_causal_atomic_and_restartable() {
    let mut bank = DetectorBank::new(vec![Detector::ideal(2).unwrap()], vec![5], 4).unwrap();
    assert_eq!(bank.observe(2, &[1], 4, 8, 0).unwrap().counts, [1]);
    let blind = bank.observe(3, &[2], 4, 8, 1).unwrap();
    assert_eq!(blind.counts, [0]);
    assert_eq!(blind.blind, [true]);
    assert_eq!(bank.available_at(), [7]);
    let before = serde_json::to_string(&bank).unwrap();
    assert!(bank.observe(3, &[1], 4, 8, 2).is_err());
    assert_eq!(serde_json::to_string(&bank).unwrap(), before);
    let mut restored: DetectorBank = serde_json::from_str(&before).unwrap();
    assert_eq!(
        bank.observe(7, &[2], 4, 8, 2).unwrap(),
        restored.observe(7, &[2], 4, 8, 2).unwrap()
    );
    let mut overflow = DetectorBank::new(vec![Detector::ideal(1).unwrap()], vec![2], 1).unwrap();
    assert!(overflow.observe(u64::MAX, &[1], 0, 0, 0).is_err());
    assert_eq!(overflow.events(), 0);
}
#[test]
fn optical_standard_instrument_matches_encoded_reference_including_remote_coherences() {
    let instrument =
        OpticalBellInstrument::standard(1., [Detector::ideal(2).unwrap(); 4], "ideal-v1".into())
            .unwrap();
    let rho = density(common::normalized(81, 31), 4);
    let engine = DenseSimulator::default();
    let actual = instrument.branches(&engine, &rho, [2, 0]).unwrap();
    let expected = StandardFusion::branches(&engine, &rho, [2, 0]).unwrap();
    for (a, b) in actual.iter().zip(expected) {
        close(&values(a.probability()), &values(b.probability()), 3e-6);
        close(
            &values(a.unnormalized_state().unwrap().entries().real()),
            &values(b.unnormalized_state().unwrap().entries().real()),
            3e-6,
        );
        close(
            &values(a.unnormalized_state().unwrap().entries().imag()),
            &values(b.unnormalized_state().unwrap().entries().imag()),
            3e-6,
        );
    }
    let mut psi = vec![Complex64::new(0., 0.); 9];
    psi[1] = std::f64::consts::FRAC_1_SQRT_2.into();
    psi[3] = std::f64::consts::FRAC_1_SQRT_2.into();
    let distinguishable =
        OpticalBellInstrument::standard(0., [Detector::ideal(2).unwrap(); 4], "orthogonal".into())
            .unwrap()
            .branches(&engine, &density(psi, 2), [0, 1])
            .unwrap();
    close(
        &[
            values(distinguishable[0].probability())[0],
            values(distinguishable[1].probability())[0],
        ],
        &[0.5, 0.5],
        3e-6,
    );
}
#[test]
fn grice_optical_recipe_resolves_three_quarters_and_charges_both_ancillas() {
    let ideal = OpticalBellInstrument::grice(GriceConfig::default()).unwrap();
    let engine = DenseSimulator::default();
    assert_eq!(ideal.supplied_ancillas(), 2);
    let lossy = OpticalBellInstrument::grice(GriceConfig {
        signal_transmissions: [0.8, 0.9],
        ancilla_transmissions: [0.7, 0.6],
        detector_efficiencies: [0.95; 8],
        calibration: "lossy-grice".into(),
    })
    .unwrap();
    let survival = 0.8 * 0.9 * 0.7 * 0.6 * 0.95f64.powi(4);
    let mut average = 0.;
    for (pair, sign, expected, xx, zz) in [
        ([1, 3], 1., 1., Parity::Even, Parity::Odd),
        ([1, 3], -1., 1., Parity::Odd, Parity::Odd),
        ([0, 4], 1., 0.5, Parity::Even, Parity::Even),
        ([0, 4], -1., 0.5, Parity::Odd, Parity::Even),
    ] {
        let mut psi = vec![Complex64::new(0., 0.); 9];
        psi[pair[0]] = std::f64::consts::FRAC_1_SQRT_2.into();
        psi[pair[1]] = (sign * std::f64::consts::FRAC_1_SQRT_2).into();
        let rho = density(psi, 2);
        let branches = ideal.branches(&engine, &rho, [0, 1]).unwrap();
        let success = branches
            .iter()
            .filter(|b| matches!(b.record(), BellRecord::Resolved { .. }))
            .map(|b| values(b.probability())[0])
            .sum::<f64>();
        assert!((success - expected).abs() < 3e-6);
        for branch in &branches {
            if let BellRecord::Resolved {
                xx: reported_xx,
                zz: reported_zz,
            } = branch.record()
            {
                let mass = if reported_xx == xx && reported_zz == zz {
                    expected
                } else {
                    0.
                };
                close(&values(branch.probability()), &[mass], 3e-6);
            }
        }
        average += success / 4.;
        let branches = lossy.branches(&engine, &rho, [0, 1]).unwrap();
        let success = branches
            .iter()
            .filter(|b| matches!(b.record(), BellRecord::Resolved { .. }))
            .map(|b| values(b.probability())[0])
            .sum::<f64>();
        assert!((success - expected * survival).abs() < 3e-6);
        close(
            &values(branches.last().unwrap().probability()),
            &[1. - survival],
            3e-6,
        );
    }
    assert!((average - 0.75).abs() < 3e-6);
}
#[test]
fn vacuum_dark_counts_generate_false_heralds_without_relabeling_them_as_true_parities() {
    let p = 0.13f64;
    let detector = Detector::new(0.8, 2, p).unwrap();
    let instrument = OpticalBellInstrument::standard(0.6, [detector; 4], "dark-v1".into()).unwrap();
    let mut vacuum = vec![Complex64::new(0., 0.); 9];
    vacuum[8] = 1.0.into();
    let branches = instrument
        .branches(&DenseSimulator::default(), &density(vacuum, 2), [0, 1])
        .unwrap();
    let two_dark = p.powi(2) * (1. - p).powi(2);
    close(
        &branches
            .iter()
            .map(|b| values(b.probability())[0])
            .collect::<Vec<_>>(),
        &[
            2. * two_dark,
            2. * two_dark,
            two_dark,
            two_dark,
            1. - 6. * two_dark,
        ],
        3e-6,
    );
}
#[test]
fn pinned_perceval_grice_and_internal_mode_probabilities_match_native_execution() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("optical_noise_perceval_v1_3.json")).unwrap();
    common::pinned_perceval_metadata(&fixture);
    assert_eq!(
        fixture["grice"]["bell_order"],
        serde_json::json!(["psi+", "psi-", "phi+", "phi-"])
    );
    let resolved: Vec<f64> =
        serde_json::from_value(fixture["grice"]["resolved_probabilities"].clone()).unwrap();
    close(&resolved, &[1., 1., 0.5, 0.5], 1e-12);
    let basis = FockBasis::new(8, PhotonSector::Fixed(4), 330).unwrap();
    let occupations: Vec<Vec<usize>> =
        serde_json::from_value(fixture["grice"]["basis"].clone()).unwrap();
    assert_eq!(basis.occupations(), occupations);
    let mut input = vec![0f32; 4 * basis.size()];
    for (bell, (pair, sign)) in [([1, 2], 1.), ([1, 2], -1.), ([0, 3], 1.), ([0, 3], -1.)]
        .into_iter()
        .enumerate()
    {
        for (term, &computational) in pair.iter().enumerate() {
            let a = computational / 2;
            let b = computational % 2;
            for rail in 0..2 {
                let mut occupation = vec![0; 8];
                for mode in [a, 2 + b, 4 + rail, 6 + rail] {
                    occupation[mode] += 1;
                }
                input[bell * basis.size() + basis.index(&occupation).unwrap()] =
                    if term == 0 { 0.5 } else { 0.5 * sign };
            }
        }
    }
    let re = Tensor::from_data(TensorData::new(input, [4, 330]), &device());
    let state = FockState::from_amplitudes(
        basis,
        ComplexTensor::new(re.clone(), re.zeros_like()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let output = state
        .apply(
            &OpticalBellInstrument::grice_interferometer().unwrap(),
            Default::default(),
        )
        .unwrap();
    let expected: Vec<Vec<f64>> =
        serde_json::from_value(fixture["grice"]["probabilities"].clone()).unwrap();
    assert_eq!(expected.len(), 4);
    assert!(expected.iter().all(|row| row.len() == 330));
    close(
        &values(output.probabilities()),
        &expected.into_iter().flatten().collect::<Vec<_>>(),
        3e-6,
    );
    let cases = fixture["partial_hom"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for (case, visibility) in cases.iter().zip([0., 0.3, 0.8, 1.]) {
        let v = case["visibility"].as_f64().unwrap();
        assert_eq!(v, visibility);
        let layout = WavepacketLayout::new(2, 2).unwrap();
        let state = FockState::wavepackets(
            FockBasis::new(4, PhotonSector::Fixed(2), 10).unwrap(),
            layout,
            &[
                PhotonPacket::new(0, vec![1.0.into(), 0.0.into()]).unwrap(),
                PhotonPacket::new(1, vec![v.sqrt().into(), (1. - v).sqrt().into()]).unwrap(),
            ],
            1,
            &device(),
            DType::F32,
            Default::default(),
        )
        .unwrap()
        .apply(
            &layout
                .lift(
                    &Interferometer::beam_splitter(
                        2,
                        ModeId(0),
                        ModeId(1),
                        std::f64::consts::FRAC_PI_2,
                    )
                    .unwrap(),
                )
                .unwrap(),
            Default::default(),
        )
        .unwrap();
        let detection = state
            .detect_spatial(
                layout,
                &[Detector::ideal(2).unwrap(); 2],
                9,
                Default::default(),
            )
            .unwrap();
        let probabilities = values(detection.probabilities());
        let selected: Vec<_> = [[0, 2], [1, 1], [2, 0]]
            .iter()
            .map(|r| probabilities[detection.records().iter().position(|o| o == r).unwrap()])
            .collect();
        let expected: Vec<f64> =
            serde_json::from_value(case["probabilities_02_11_20"].clone()).unwrap();
        close(&selected, &expected, 3e-6);
    }
}
