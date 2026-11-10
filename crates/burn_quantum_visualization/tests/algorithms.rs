//! Independent mathematical acceptance oracles for the reusable catalog algorithms.
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{
    DenseSimulator, QuantumBackend, StateVector, algorithms::*, protocol::BranchExecution,
};
use burn_quantum_visualization::{
    demo::{self, DemoConfig, DemoKind},
    simulation::{self, TraceLimits},
};
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 5e-5, "{a} != {b}");
}
async fn amplitudes(a: &AlgorithmCircuit) -> Vec<(f64, f64)> {
    let p = (!a.parameters.is_empty()).then(|| {
        Tensor::from_data(
            TensorData::new(a.parameters.clone(), [a.parameters.len()]),
            (&Device::flex(), DType::F32),
        )
    });
    let state = DenseSimulator::default()
        .execute(&a.circuit, p, None, 1, &Device::flex())
        .unwrap();
    simulation::read_tensor(state.amplitudes().real())
        .await
        .unwrap()
        .into_iter()
        .zip(
            simulation::read_tensor(state.amplitudes().imag())
                .await
                .unwrap(),
        )
        .collect()
}
#[test]
fn fourier_matches_independent_dft_and_inverse() {
    pollster::block_on(async {
        for n in 1..=5 {
            let size = 1usize << n;
            for input in [0, 1, size - 1] {
                for inverse in [false, true] {
                    let a = qft(n, input, inverse).unwrap();
                    let v = amplitudes(&a).await;
                    for (y, &(re, im)) in v.iter().enumerate() {
                        let angle = std::f64::consts::TAU * (input * y) as f64 / size as f64
                            * if inverse { -1. } else { 1. };
                        near(re, angle.cos() / (size as f64).sqrt());
                        near(im, angle.sin() / (size as f64).sqrt());
                    }
                }
                let mut a = qft(n, input, false).unwrap();
                append_qft(&mut a.circuit, &(0..n).collect::<Vec<_>>(), true).unwrap();
                for (i, (re, im)) in amplitudes(&a).await.into_iter().enumerate() {
                    near(re, if i == input { 1. } else { 0. });
                    near(im, 0.);
                }
            }
        }
    });
}
#[test]
fn dj_and_qpe_respect_big_endian_readout() {
    pollster::block_on(async {
        for n in 2..=6 {
            for mask in [0, 1, (1 << n) - 1] {
                for offset in [false, true] {
                    let v = amplitudes(&deutsch_jozsa(n, mask, offset).unwrap()).await;
                    for (i, p) in v.chunks(2).enumerate() {
                        near(
                            p.iter().map(|&(re, im)| re * re + im * im).sum(),
                            if i == mask { 1. } else { 0. },
                        );
                    }
                }
            }
            for phase in [0., 0.375, 1. / 3.] {
                let v = amplitudes(&phase_estimation(n, phase).unwrap()).await;
                let count = 1usize << n;
                for (k, p) in v.chunks(2).enumerate() {
                    let delta = phase - k as f64 / count as f64;
                    let expected = if delta.abs() < 1e-12 {
                        1.
                    } else {
                        (std::f64::consts::PI * count as f64 * delta).sin().powi(2)
                            / (count as f64 * (std::f64::consts::PI * delta).sin()).powi(2)
                    };
                    near(p.iter().map(|&(r, i)| r * r + i * i).sum(), expected);
                }
            }
        }
    });
}
#[test]
fn grover_matches_rotation_formula_and_toffoli_superposition() {
    pollster::block_on(async {
        for n in 2..=7 {
            let theta = (1. / ((1usize << n) as f64).sqrt()).asin();
            for rounds in [
                0,
                1,
                grover_iterations(n).unwrap(),
                grover_iterations(n).unwrap() + 1,
            ] {
                let target = (1 << n) - 3;
                let v = amplitudes(&grover(n, target, rounds).unwrap()).await;
                near(
                    v[target].0.powi(2) + v[target].1.powi(2),
                    ((2 * rounds + 1) as f64 * theta).sin().powi(2),
                );
            }
        }
        for controls in 2..=5 {
            let v = amplitudes(&controlled_x_demo(controls, true).unwrap()).await;
            for (i, (re, im)) in v.into_iter().enumerate() {
                let input = i >> 1;
                let target = i & 1;
                let expected = usize::from(input == (1 << controls) - 1);
                near(
                    re,
                    if target == expected {
                        1. / ((1usize << controls) as f64).sqrt()
                    } else {
                        0.
                    },
                );
                near(im, 0.);
            }
        }
    });
}
#[test]
fn every_teleportation_branch_preserves_the_input_and_likelihood() {
    pollster::block_on(async {
        let limits = TraceLimits::default();
        let engine = DenseSimulator::new(limits.resources());
        let device = Device::flex();
        for transfers in 1..=2 {
            let a = teleportation(transfers, 1.1, 0.7).unwrap();
            for branch in 0..1usize << (2 * transfers) {
                let initial = StateVector::zero(
                    a.circuit.register().clone(),
                    1,
                    &device,
                    DType::F32,
                    engine.limits,
                )
                .unwrap();
                let outcomes = (0..2 * transfers)
                    .map(|b| branch >> (2 * transfers - 1 - b) & 1)
                    .collect();
                let mut run = BranchExecution::new(
                    &a.circuit,
                    a.protocol.as_ref().unwrap(),
                    &engine,
                    initial,
                    outcomes,
                    None,
                    None,
                )
                .unwrap();
                while run.advance().await.unwrap() {}
                near(run.probability(), 1. / (1usize << (2 * transfers)) as f64);
                let frame = simulation::snapshot_pure(run.state(), 0, "".into(), limits)
                    .await
                    .unwrap();
                for j in 0..transfers {
                    let b = frame.subsystems[3 * j + 2].bloch.unwrap();
                    for (x, y) in b.into_iter().zip([
                        1.1f64.sin() * 0.7f64.cos(),
                        1.1f64.sin() * 0.7f64.sin(),
                        1.1f64.cos(),
                    ]) {
                        near(x, y);
                    }
                }
            }
        }
    });
}
#[test]
fn repetition_recovers_all_correctable_patterns_and_exposes_failure() {
    pollster::block_on(async {
        let limits = TraceLimits::default();
        let engine = DenseSimulator::new(limits.resources());
        for n in [3usize, 5, 7] {
            for pattern in 0usize..1 << n {
                if pattern.count_ones() as usize > n / 2 {
                    continue;
                }
                let errors = (0..n).filter(|w| pattern >> w & 1 == 1).collect::<Vec<_>>();
                let a = bit_flip_code(n, &errors, 1.1, 0.7).unwrap();
                let state = engine
                    .execute(&a.circuit, None, None, 1, &Device::flex())
                    .unwrap();
                let frame = simulation::snapshot_pure(&state, 0, "".into(), limits)
                    .await
                    .unwrap();
                for (x, y) in frame.subsystems[0].bloch.unwrap().into_iter().zip([
                    1.1f64.sin() * 0.7f64.cos(),
                    1.1f64.sin() * 0.7f64.sin(),
                    1.1f64.cos(),
                ]) {
                    near(x, y);
                }
            }
        }
        let a = bit_flip_code(3, &[0, 1], 1.1, 0.7).unwrap();
        let state = engine
            .execute(&a.circuit, None, None, 1, &Device::flex())
            .unwrap();
        let frame = simulation::snapshot_pure(&state, 0, "".into(), limits)
            .await
            .unwrap();
        near(frame.subsystems[0].bloch.unwrap()[2], -1.1f64.cos());
    });
}
#[test]
fn qaoa_matches_independent_dense_cost_and_mixer_evolution() {
    pollster::block_on(async {
        for n in 3..=6 {
            let edges = (0..n).map(|w| (w, (w + 1) % n)).collect::<Vec<_>>();
            let angles = [(0.7, 0.35), (0.2, -0.4)];
            let a = qaoa_maxcut(n, &edges, &angles).unwrap();
            let actual = amplitudes(&a).await;
            let size = 1usize << n;
            let mut reference = vec![(1. / (size as f64).sqrt(), 0.); size];
            for (gamma, beta) in angles {
                for (basis, (re, im)) in reference.iter_mut().enumerate() {
                    let cut = edges
                        .iter()
                        .filter(|&&(u, v)| {
                            ((basis >> (n - 1 - u)) ^ (basis >> (n - 1 - v))) & 1 == 1
                        })
                        .count();
                    let phase = -gamma * cut as f64;
                    (*re, *im) = (
                        *re * phase.cos() - *im * phase.sin(),
                        *re * phase.sin() + *im * phase.cos(),
                    );
                }
                for w in 0..n {
                    let old = reference.clone();
                    for i in 0..size {
                        let (ar, ai) = old[i];
                        let (br, bi) = old[i ^ (1 << (n - 1 - w))];
                        reference[i] = (
                            beta.cos() * ar + beta.sin() * bi,
                            beta.cos() * ai - beta.sin() * br,
                        );
                    }
                }
            }
            // Cost decomposition omits a known global phase; compare all complex pairwise overlaps through fidelity.
            let (re, im) = actual
                .iter()
                .zip(&reference)
                .fold((0., 0.), |(r, i), (&(ar, ai), &(br, bi))| {
                    (r + ar * br + ai * bi, i + ar * bi - ai * br)
                });
            near(re * re + im * im, 1.);
        }
    });
}
#[test]
fn catalog_preflight_roundtrip_and_protocol_tampering() {
    pollster::block_on(async {
        for &kind in DemoKind::all() {
            let doc = demo::build(kind, &Device::flex(), TraceLimits::default())
                .await
                .unwrap();
            doc.validate().unwrap();
            let text = burn_quantum_visualization::io::document_json(&doc).unwrap();
            burn_quantum_visualization::io::parse_document(&text).unwrap();
        }
        let mut config = DemoConfig::for_demo(DemoKind::Ghz);
        config.size = 16;
        assert!(demo::preview(DemoKind::Ghz, config, TraceLimits::default()).is_err());
        config.size = 12;
        let p = demo::preview(DemoKind::Ghz, config, TraceLimits::default())
            .unwrap()
            .unwrap();
        assert_eq!((p.wires, p.dimension), (12, 4096));
        let doc = demo::build(
            DemoKind::Teleportation,
            &Device::flex(),
            TraceLimits::default(),
        )
        .await
        .unwrap();
        let mut bad = doc.clone();
        bad.protocol.as_mut().unwrap().probabilities[7] = 1.;
        assert!(bad.validate().is_err());
        let mut bad = doc;
        bad.protocol.as_mut().unwrap().outcomes[0] = 2;
        assert!(bad.validate().is_err());
    });
}
