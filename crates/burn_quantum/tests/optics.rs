//! Independent bosonic and detector oracles, executed on the selected Burn device.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Tensor, TensorData};
use burn_quantum::{photonic::optics::*, *};
use common::{close, complex_close, values};
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2};
fn device() -> burn::tensor::Device {
    runtime::execution().unwrap().device
}
fn basis(m: usize, n: usize) -> FockBasis {
    FockBasis::new(m, PhotonSector::UpTo(n), 1024).unwrap()
}
fn factorial(n: usize) -> f64 {
    (1..=n).fold(1., |x, i| x * i as f64)
}
fn permanent(matrix: &[Complex64], n: usize) -> Complex64 {
    fn visit(
        matrix: &[Complex64],
        n: usize,
        row: usize,
        used: &mut [bool],
        value: Complex64,
        total: &mut Complex64,
    ) {
        if row == n {
            *total += value;
            return;
        }
        for col in 0..n {
            if !used[col] {
                used[col] = true;
                visit(
                    matrix,
                    n,
                    row + 1,
                    used,
                    value * matrix[row * n + col],
                    total,
                );
                used[col] = false;
            }
        }
    }
    let mut total = Complex64::new(0., 0.);
    visit(matrix, n, 0, &mut vec![false; n], 1.0.into(), &mut total);
    total
}
#[test]
fn combinatorial_basis_preflights_and_vacuum_are_exact() {
    assert_eq!(
        FockBasis::new(3, PhotonSector::Fixed(3), 10)
            .unwrap()
            .size(),
        10
    );
    assert_eq!(
        FockBasis::new(3, PhotonSector::UpTo(3), 20).unwrap().size(),
        20
    );
    assert!(FockBasis::new(3, PhotonSector::UpTo(3), 19).is_err());
    assert!(FockBasis::new(256, PhotonSector::UpTo(64), 4096).is_err());
    assert!(FockBasis::new(0, PhotonSector::UpTo(0), 10).is_err());
    assert!(FockBasis::new(2, PhotonSector::UpTo(0), 0).is_err());
    let b = FockBasis::new(2, PhotonSector::Fixed(0), 1).unwrap();
    let vacuum =
        FockState::number(b, &[0, 0], 1, &device(), DType::F32, Default::default()).unwrap();
    let out = vacuum
        .apply(
            &Interferometer::beam_splitter(2, ModeId(0), ModeId(1), 1.2).unwrap(),
            Default::default(),
        )
        .unwrap();
    close(&values(out.probabilities()), &[1.], 1e-6);
    out.density(Default::default())
        .unwrap()
        .lose(ModeId(0), 0., Default::default())
        .unwrap()
        .validate(1e-6)
        .unwrap();
}
#[test]
fn lifted_three_mode_fourier_matches_independent_permanents() {
    let b = FockBasis::new(3, PhotonSector::Fixed(3), 10).unwrap();
    let u = Unitary::fourier(3).unwrap();
    let lifted = Interferometer::new(u.matrix().clone())
        .unwrap()
        .lift(&b, Default::default())
        .unwrap();
    let mut oracle = vec![];
    for output in b.occupations() {
        for input in b.occupations() {
            let rows: Vec<_> = output
                .iter()
                .enumerate()
                .flat_map(|(i, &n)| std::iter::repeat_n(i, n))
                .collect();
            let cols: Vec<_> = input
                .iter()
                .enumerate()
                .flat_map(|(i, &n)| std::iter::repeat_n(i, n))
                .collect();
            let mut sub = vec![];
            for &r in &rows {
                for &c in &cols {
                    sub.push(u.matrix().entries()[r * 3 + c]);
                }
            }
            let norm = (output
                .iter()
                .chain(input)
                .map(|&n| factorial(n))
                .product::<f64>())
            .sqrt();
            oracle.push(permanent(&sub, 3) / norm);
        }
    }
    complex_close(lifted.entries(), &oracle, 1e-12);
    // Execute that same operator on the selected device, not only its host lifting.
    let state = FockState::number(
        b.clone(),
        &[1, 1, 1],
        1,
        &device(),
        DType::F32,
        Default::default(),
    )
    .unwrap();
    let out = state
        .apply(
            &Interferometer::new(u.matrix().clone()).unwrap(),
            Default::default(),
        )
        .unwrap();
    let actual: Vec<_> = values(out.amplitudes().real())
        .into_iter()
        .zip(values(out.amplitudes().imag()))
        .map(|(re, im)| Complex64::new(re, im))
        .collect();
    let column = b.index(&[1, 1, 1]).unwrap();
    let expected: Vec<_> = (0..b.size())
        .map(|row| oracle[row * b.size() + column])
        .collect();
    complex_close(&actual, &expected, 2e-6);
}
#[test]
fn hong_ou_mandel_bunching_inverse_permutation_and_number_conservation() {
    let b = basis(2, 3);
    let limits = ResourceLimits::default();
    let state = FockState::number(b.clone(), &[1, 1], 2, &device(), DType::F32, limits).unwrap();
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), FRAC_PI_2).unwrap();
    let out = state.apply(&bs, limits).unwrap();
    let probabilities = values(out.probabilities());
    for batch in 0..2 {
        for (i, occupation) in b.occupations().iter().enumerate() {
            let expected = if occupation == &[2, 0] || occupation == &[0, 2] {
                0.5
            } else {
                0.
            };
            assert!((probabilities[batch * b.size() + i] - expected).abs() < 2e-6);
        }
    }
    let restored = out.apply(&bs.adjoint(), limits).unwrap();
    close(
        &values(restored.amplitudes().real()),
        &values(state.amplitudes().real()),
        2e-6,
    );
    close(
        &values(restored.amplitudes().imag()),
        &values(state.amplitudes().imag()),
        2e-6,
    );
    let input = FockState::number(b.clone(), &[2, 1], 1, &device(), DType::F32, limits).unwrap();
    let swapped = input
        .apply(&Interferometer::permutation(&[1, 0]).unwrap(), limits)
        .unwrap();
    assert!((values(swapped.probabilities())[b.index(&[1, 2]).unwrap()] - 1.).abs() < 2e-6);
    assert!(Interferometer::permutation(&[0, 0]).is_err());
}
#[test]
fn pure_loss_matches_binomial_and_preserves_surviving_coherence() {
    let limits = ResourceLimits::default();
    let dev = device();
    let input = FockState::number(basis(1, 3), &[3], 1, &dev, DType::F32, limits).unwrap();
    let out = input
        .density(limits)
        .unwrap()
        .lose(ModeId(0), 0.6, limits)
        .unwrap();
    close(
        &values(out.probabilities()),
        &[0.064, 0.288, 0.432, 0.216],
        2e-6,
    );
    out.validate(2e-6).unwrap();
    let b = basis(2, 1);
    let amplitudes = ComplexTensor::new(
        Tensor::from_data(
            TensorData::new(vec![0f32, 0., FRAC_1_SQRT_2 as f32], [1, 3]),
            &dev,
        ),
        Tensor::from_data(
            TensorData::new(vec![0f32, FRAC_1_SQRT_2 as f32, 0.], [1, 3]),
            &dev,
        ),
    )
    .unwrap();
    let rho = FockState::from_amplitudes(b, amplitudes, limits)
        .unwrap()
        .density(limits)
        .unwrap();
    let attenuated = rho.lose(ModeId(0), 0.25, limits).unwrap();
    close(
        &values(attenuated.probabilities()),
        &[0.375, 0.5, 0.125],
        2e-6,
    );
    assert!((values(attenuated.entries().imag())[7] + 0.25).abs() < 2e-6);
    let composed = attenuated.lose(ModeId(0), 0.36, limits).unwrap();
    let direct = rho.lose(ModeId(0), 0.09, limits).unwrap();
    close(
        &values(composed.entries().real()),
        &values(direct.entries().real()),
        2e-6,
    );
    close(
        &values(composed.entries().imag()),
        &values(direct.entries().imag()),
        2e-6,
    );
    let fixed = FockState::number(
        FockBasis::new(2, PhotonSector::Fixed(1), 2).unwrap(),
        &[1, 0],
        1,
        &dev,
        DType::F32,
        limits,
    )
    .unwrap();
    assert!(
        fixed
            .density(limits)
            .unwrap()
            .lose(ModeId(0), 0.5, limits)
            .is_err()
    );
}
#[test]
fn native_phase_gradients_survive_loss_and_detection() {
    let limits = ResourceLimits::default();
    let dev = device().autodiff();
    let b = basis(2, 1);
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), FRAC_PI_2).unwrap();
    let input = FockState::number(b.clone(), &[1, 0], 2, &dev, DType::F32, limits)
        .unwrap()
        .apply(&bs, limits)
        .unwrap();
    let angles = Tensor::from_data([0.7f32, 1.1], &dev).require_grad();
    let rho = input
        .density(limits)
        .unwrap()
        .phase_shift(ModeId(0), angles.clone(), limits)
        .unwrap()
        .apply(&bs.adjoint(), limits)
        .unwrap()
        .lose(ModeId(0), 0.8, limits)
        .unwrap();
    let detectors = [
        Detector::threshold(0.9, 0.).unwrap(),
        Detector::ideal(1).unwrap(),
    ];
    let out = rho.detect(&detectors, 4, limits).unwrap();
    let index = out.records().iter().position(|x| x == &[1, 0]).unwrap();
    let probability = out.probabilities().slice_dim(1, index..index + 1);
    close(
        &values(probability.clone()),
        &[
            0.72 * (0.7f64 / 2.).cos().powi(2),
            0.72 * (1.1f64 / 2.).cos().powi(2),
        ],
        3e-6,
    );
    let grads = probability.sum().backward();
    close(
        &values(angles.grad(&grads).unwrap()),
        &[-0.36 * 0.7f64.sin(), -0.36 * 1.1f64.sin()],
        3e-5,
    );
}
#[test]
fn detector_noise_saturation_and_budgets_retain_no_click_events() {
    close(
        &Detector::new(0.5, 2, 0.1).unwrap().response(2).unwrap(),
        &[0.225, 0.475, 0.3],
        1e-12,
    );
    close(
        &Detector::threshold(1., 0.).unwrap().response(8).unwrap(),
        &[0., 1.],
        1e-12,
    );
    close(
        &Detector::threshold(0., 0.2).unwrap().response(8).unwrap(),
        &[0.8, 0.2],
        1e-12,
    );
    assert!(Detector::new(1.1, 1, 0.).is_err());
    assert!(Detector::ideal(0).is_err());
    let state = FockState::number(
        basis(2, 1),
        &[1, 0],
        1,
        &device(),
        DType::F32,
        Default::default(),
    )
    .unwrap();
    assert!(
        state
            .detect(&[Detector::ideal(1).unwrap(); 2], 3, Default::default())
            .is_err()
    );
    let detected = state
        .detect(
            &[Detector::threshold(0., 0.).unwrap(); 2],
            4,
            Default::default(),
        )
        .unwrap();
    close(&values(detected.probabilities()), &[1., 0., 0., 0.], 2e-6);
}

#[test]
fn pinned_perceval_slos_amplitudes_and_unfiltered_loss_match() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("optics_perceval_v1_3.json")).unwrap();
    common::pinned_perceval_metadata(&fixture);
    assert_eq!(
        fixture["convention"],
        "single-photon matrix rows=output, columns=input; Ry splitter; unfiltered probabilities"
    );
    assert_eq!(
        fixture["hom_asymmetric_loss"]["transmissions"],
        serde_json::json!([0.8, 0.4])
    );
    let limits = ResourceLimits::default();
    let dev = device();
    let b = FockBasis::new(3, PhotonSector::Fixed(3), 10).unwrap();
    let fixture_basis: Vec<Vec<usize>> =
        serde_json::from_value(fixture["fourier_3_modes_3_photons"]["basis"].clone()).unwrap();
    assert_eq!(b.occupations(), fixture_basis);
    let entries: Vec<Vec<[f64; 2]>> = serde_json::from_value(
        fixture["fourier_3_modes_3_photons"]["amplitudes_output_input"].clone(),
    )
    .unwrap();
    assert_eq!(entries.len(), 10);
    assert!(entries.iter().all(|row| row.len() == 10));
    let mut identity = vec![0f32; 100];
    for i in 0..10 {
        identity[i * 10 + i] = 1.;
    }
    let re = Tensor::from_data(TensorData::new(identity, [10, 10]), &dev);
    let input = FockState::from_amplitudes(
        b,
        ComplexTensor::new(re.clone(), re.zeros_like()).unwrap(),
        limits,
    )
    .unwrap();
    let out = input
        .apply(
            &Interferometer::new(Unitary::fourier(3).unwrap().matrix().clone()).unwrap(),
            limits,
        )
        .unwrap();
    let actual: Vec<_> = values(out.amplitudes().real())
        .into_iter()
        .zip(values(out.amplitudes().imag()))
        .map(|(re, im)| Complex64::new(re, im))
        .collect();
    let mut expected = vec![];
    for column in 0..10 {
        for row in &entries {
            expected.push(Complex64::new(row[column][0], row[column][1]));
        }
    }
    complex_close(&actual, &expected, 2e-6);
    let b = basis(2, 2);
    let state = FockState::number(b.clone(), &[1, 1], 1, &dev, DType::F32, limits)
        .unwrap()
        .apply(
            &Interferometer::beam_splitter(2, ModeId(0), ModeId(1), FRAC_PI_2).unwrap(),
            limits,
        )
        .unwrap();
    let out = state
        .density(limits)
        .unwrap()
        .lose(ModeId(0), 0.8, limits)
        .unwrap()
        .lose(ModeId(1), 0.4, limits)
        .unwrap();
    let expected: Vec<f64> =
        serde_json::from_value(fixture["hom_asymmetric_loss"]["probabilities"].clone()).unwrap();
    let expected_basis: Vec<Vec<usize>> =
        serde_json::from_value(fixture["hom_asymmetric_loss"]["basis"].clone()).unwrap();
    assert_eq!(b.occupations(), expected_basis);
    close(&values(out.probabilities()), &expected, 2e-6);
}
