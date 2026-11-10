//! Explicit Clifford encoder, projector, and entangled-reference injection oracles.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Tensor};
use burn_quantum::{photonic::qec::*, *};
use common::{close, values};

#[test]
fn t_injection_preserves_entanglement_all_records_and_native_gradients() {
    let device = runtime::execution().unwrap().device;
    let limits = ResourceLimits::default();
    let real = Tensor::from_data([[0.6f32, 0., 0., 0.8]], &device)
        .autodiff()
        .require_grad();
    let psi = StateVector::from_amplitudes(
        Register::qubits(2).unwrap(),
        ComplexTensor::new(real.clone(), real.zeros_like()).unwrap(),
        limits,
    )
    .unwrap();
    let rho = DensityMatrix::from_pure(&psi, limits).unwrap();
    let p = 0.17;
    let yield_probability = 0.73;
    let injected = MagicState::new(p, yield_probability)
        .unwrap()
        .inject(&rho, 1, limits)
        .unwrap();
    assert_eq!(injected.raw_source_attempts, 1);
    assert_eq!(
        injected
            .branches
            .iter()
            .map(|b| b.record)
            .collect::<Vec<_>>(),
        vec![MagicRecord::Missing, MagicRecord::Zero, MagicRecord::One]
    );
    close(
        &injected
            .branches
            .iter()
            .flat_map(|b| values(b.state.trace()))
            .collect::<Vec<_>>(),
        &[0.27, 0.365, 0.365],
        3e-6,
    );
    let completed = injected.completed(limits).unwrap();
    let ideal = rho
        .apply(&DenseSimulator::default(), &[1], &Unitary::t())
        .unwrap()
        .channel(
            &DenseSimulator::default(),
            &[1],
            &KrausChannel::phase_flip(p).unwrap(),
        )
        .unwrap();
    close(
        &values(completed.entries().real()),
        &values(ideal.entries().real() * yield_probability),
        3e-6,
    );
    close(
        &values(completed.entries().imag()),
        &values(ideal.entries().imag() * yield_probability),
        3e-6,
    );
    let gradient = real
        .grad(&completed.entries().real().sum().backward())
        .unwrap();
    let gradient = values(gradient);
    let coherence = (1. - 2. * p) * std::f64::consts::FRAC_1_SQRT_2;
    close(
        &[gradient[0], gradient[3]],
        &[
            yield_probability * (1.2 + 1.6 * coherence),
            yield_probability * (1.6 + 1.2 * coherence),
        ],
        3e-6,
    );
    assert!(MagicState::new(f64::NAN, 1.).is_err());
    assert!(MagicState::new(0., 1.1).is_err());
    assert!(
        MagicState::new(0., 1.)
            .unwrap()
            .inject(&rho, 2, limits)
            .is_err()
    );
    assert!(
        MagicState::new(0., 1.)
            .unwrap()
            .inject(
                &rho,
                1,
                ResourceLimits {
                    max_state_elements: 63,
                    ..limits
                }
            )
            .is_err()
    );
}

#[test]
fn reed_muller_encoder_transversal_gate_and_decoder_execute_on_selected_device() {
    let device = runtime::execution().unwrap().device;
    let engine = DenseSimulator::default();
    for mask in [0u16, 1, 7] {
        let circuit = ReedMuller15::fault_circuit(mask, true).unwrap();
        let decoded = engine.execute(&circuit, None, None, 1, &device).unwrap();
        decoded.validate(3e-6).unwrap();
        let (syndrome, logical) = ReedMuller15::phase_fault(mask).unwrap();
        let density = decoded.reduced_density(&[0], engine.limits).unwrap();
        let reference = MagicState::new(f64::from(logical), 1.)
            .unwrap()
            .density(&device, DType::F32, engine.limits)
            .unwrap();
        close(
            &values(density.entries().real()),
            &values(reference.entries().real()),
            3e-6,
        );
        close(
            &values(density.entries().imag()),
            &values(reference.entries().imag()),
            3e-6,
        );
        for bit in 0..4 {
            let marginal = values(decoded.marginal(&[bit + 1], engine.limits).unwrap());
            close(
                &marginal,
                if syndrome & (1 << bit) == 0 {
                    &[1., 0.]
                } else {
                    &[0., 1.]
                },
                3e-6,
            );
        }
    }
}

#[test]
fn every_fault_mask_matches_independent_codespace_projector_and_weight_enumerator() {
    // |0_L> is uniform over the sixteen linear Boolean functions on the
    // nonzero four-bit inputs; |1_L> is their bitwise complement. These explicit
    // amplitudes compute the codespace projector, independently of syndrome rows.
    let words: Vec<u16> = (0u32..16)
        .map(|function| {
            (1u32..16).fold(0u16, |word, input| {
                word | ((((function & input).count_ones() % 2) as u16) << (input - 1))
            })
        })
        .collect();
    let mut accepted_by_weight = [0usize; 16];
    for mask in 0u16..1 << 15 {
        let zero = words
            .iter()
            .map(|w| {
                if (w & mask).count_ones() % 2 == 0 {
                    1.
                } else {
                    -1.
                }
            })
            .sum::<f64>()
            / 16.;
        let one = words
            .iter()
            .map(|w| {
                if ((w ^ 0x7fff) & mask).count_ones() % 2 == 0 {
                    1.
                } else {
                    -1.
                }
            })
            .sum::<f64>()
            / 16.;
        let norm = (zero * zero + one * one) / 2.;
        let (syndrome, logical) = ReedMuller15::phase_fault(mask).unwrap();
        assert_eq!(norm, if syndrome == 0 { 1. } else { 0. });
        if norm == 1. {
            assert_eq!(one / zero, if logical { -1. } else { 1. });
            accepted_by_weight[mask.count_ones() as usize] += 1;
        }
    }
    assert_eq!(
        accepted_by_weight,
        [
            1, 0, 0, 35, 105, 168, 280, 435, 435, 280, 168, 105, 35, 0, 0, 1
        ]
    );
}

#[test]
fn distillation_matches_published_exact_polynomial_and_counts_source_failures() {
    for p in [0., 0.01, 0.07, 0.2, 0.5, 1.] {
        let report = ReedMuller15::distill(&MagicDistillationConfig {
            phase_errors: [p; 15],
            source_availability: [0.97; 15],
            ..Default::default()
        })
        .unwrap();
        let t = 1. - 2. * p;
        let success = (1. + 15. * t.powi(8)) / 16.;
        let error =
            (1. - 15. * t.powi(7) + 15. * t.powi(8) - t.powi(15)) / (2. * (1. + 15. * t.powi(8)));
        close(
            &[report.acceptance, report.output_phase_error.unwrap()],
            &[0.97f64.powi(15) * success, error],
            2e-12,
        );
        close(
            &[
                report.observed_syndrome_probabilities.iter().sum(),
                report.true_syndrome_probabilities.iter().sum(),
            ],
            &[0.97f64.powi(15), 1.],
            2e-12,
        );
        assert_eq!(report.evaluated_fault_patterns, 32768);
        assert_eq!(report.false_herald_probability, 0.);
        assert_eq!(report.cost.raw_source_attempts, 15);
        close(
            &[
                report.cost.expected_measurements,
                report.cost.expected_raw_states_emitted,
            ],
            &[19. * 0.97f64.powi(15), 15. * 0.97],
            2e-12,
        );
    }
    let missing = ReedMuller15::distill(&MagicDistillationConfig {
        source_availability: [0.; 15],
        ..Default::default()
    })
    .unwrap();
    assert_eq!(missing.acceptance, 0.);
    assert_eq!(missing.output_phase_error, None);
    assert_eq!(missing.cost.raw_attempts_per_output, None);
    assert_eq!(missing.cost.expected_measurements, 0.);
    assert!(
        ReedMuller15::distill(&MagicDistillationConfig {
            max_fault_patterns: 32767,
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn noisy_syndrome_readout_keeps_false_acceptances_and_output_errors() {
    let mut phase_errors = [0.; 15];
    phase_errors[0] = 1.; // deterministic single Z, syndrome 0001.
    let report = ReedMuller15::distill(&MagicDistillationConfig {
        phase_errors,
        syndrome_readout_errors: [0.1, 0.2, 0.3, 0.4],
        ..Default::default()
    })
    .unwrap();
    let accepted = 0.1 * 0.8 * 0.7 * 0.6;
    close(
        &[
            report.acceptance,
            report.false_herald_probability,
            report.output_phase_error.unwrap(),
        ],
        &[accepted, accepted, 1.],
        1e-14,
    );
    let perfect = ReedMuller15::distill(&MagicDistillationConfig {
        phase_errors,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(perfect.acceptance, 0.);
}
