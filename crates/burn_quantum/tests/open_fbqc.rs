//! Finite six-ring boundary instruments, conditional output channels and loss correlations.
use burn_quantum::photonic::{qec::*, stabilizer::*, *};
fn compile(patch: &SixRingPatch, eta: f64) -> CompiledFusionNetwork {
    let fabric = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: RailTransmission::IDEAL,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: RailTransmission::IDEAL,
            },
        ],
        vec![FabricLink {
            from: NodeId(0),
            to: NodeId(1),
            ticks: 1,
            transmission: RailTransmission::new(eta, eta).unwrap(),
        }],
    )
    .unwrap();
    patch
        .program()
        .compile(&fabric, CompileOptions::default())
        .unwrap()
}
#[test]
fn finite_box_output_algebra_and_physical_records() {
    for dims in [[1, 1, 1], [1, 1, 3], [2, 2, 2]] {
        let patch = SixRingPatch::new(
            dims,
            [
                PatchBoundary::Measure(Pauli::X),
                PatchBoundary::Measure(Pauli::X),
                PatchBoundary::Output,
                PatchBoundary::Measure(Pauli::Z),
                PatchBoundary::Measure(Pauli::Z),
                PatchBoundary::Output,
            ],
            NodeId(0),
            NodeId(1),
            100,
        )
        .unwrap();
        let plan = compile(&patch, 1.);
        let relations = patch.relations(&plan, 1 << 20).unwrap();
        let output: Vec<_> = patch.outputs().iter().map(|q| q.0).collect();
        let correlations = relations.output_relations(&output, 1 << 20).unwrap();
        assert_eq!(correlations.len(), output.len());
        for (i, (a, _)) in correlations.iter().enumerate() {
            for (b, _) in &correlations[..i] {
                assert!(!a.anticommutes(b));
            }
        }
        let targets: Vec<_> = correlations.into_iter().map(|(p, _)| p).collect();
        let shots = patch
            .sample(&plan, 0, 256, 19, PauliNoise::NONE, &[], &targets, 1 << 20)
            .unwrap();
        let mut known = 0;
        for shot in &shots {
            for (inferred, &actual) in shot.inferred.iter().zip(&shot.expectations) {
                if let ParityInference::Known(p) = inferred {
                    known += 1;
                    assert_eq!(actual, if *p == Parity::Odd { -1 } else { 1 });
                }
            }
        }
        assert!(known > 0);
    }
}

fn graph_amplitudes(rings: usize) -> Vec<f64> {
    let n = 6 * rings;
    let norm = (1usize << n) as f64;
    (0..1usize << n)
        .map(|i| {
            let mut minus = false;
            for r in 0..rings {
                for q in 0..6 {
                    minus ^= (i >> (6 * r + q)) & 1 == 1 && (i >> (6 * r + (q + 1) % 6)) & 1 == 1;
                }
            }
            if minus {
                -1. / norm.sqrt()
            } else {
                1. / norm.sqrt()
            }
        })
        .collect()
}
fn apply_real_pauli(state: &[f64], terms: &[(usize, Pauli)]) -> Vec<f64> {
    let mut result = vec![0.; state.len()];
    for (i, &a) in state.iter().enumerate() {
        let mut j = i;
        let mut negative = false;
        for &(q, p) in terms {
            match p {
                Pauli::X => j ^= 1 << q,
                Pauli::Z => negative ^= i & (1 << q) != 0,
                Pauli::Y => panic!("oracle uses only real X/Z terms"),
            }
        }
        result[j] = if negative { -a } else { a };
    }
    result
}
fn project(state: &mut [f64], terms: &[(usize, Pauli)], negative: bool) -> f64 {
    let applied = apply_real_pauli(state, terms);
    let sign = if negative { -1. } else { 1. };
    for (a, p) in state.iter_mut().zip(applied) {
        *a = (*a + sign * p) * 0.5;
    }
    let weight = state.iter().map(|v| v * v).sum::<f64>();
    if weight > 1e-20 {
        for a in state {
            *a /= weight.sqrt();
        }
    }
    weight
}
#[test]
fn all_physical_two_ring_branches_match_independent_real_amplitude_oracle() {
    let patch = SixRingPatch::bell_strip(2, NodeId(0), NodeId(1), 12).unwrap();
    let plan = compile(&patch, 1.);
    let relations = patch.relations(&plan, 1 << 20).unwrap();
    let outputs = patch.outputs();
    assert_eq!(outputs, &[PhotonId(3), PhotonId(6)]);
    let targets: [PauliProduct; 2] = [Pauli::X, Pauli::Z]
        .map(|p| PauliProduct::new(outputs.iter().map(|q| (q.0, p)).collect()).unwrap());
    for p in &targets {
        assert!(relations.correlation(p).unwrap().is_some());
    }
    let initial = graph_amplitudes(2);
    let mut total = 0.;
    let mut known_mass = 0.;
    for kind in 0..4 {
        for bits in 0..1usize << 8 {
            let mut state = initial.clone();
            let success = kind < 2;
            let sign = kind % 2 == 1;
            let mut mass = project(&mut state, &[(0, Pauli::Z), (9, Pauli::Z)], success);
            let mut records = if success {
                mass *= project(&mut state, &[(0, Pauli::X), (9, Pauli::X)], sign);
                vec![Some(sign), Some(true)]
            } else {
                mass *= project(&mut state, &[(0, Pauli::Z)], sign);
                vec![None, Some(false)]
            };
            for (i, p) in patch.boundary_measurements().iter().enumerate() {
                let negative = bits & (1 << i) != 0;
                mass *= project(&mut state, p.terms(), negative);
                records.push(Some(negative));
            }
            total += mass;
            if mass < 1e-18 {
                continue;
            }
            let mut both = true;
            for p in &targets {
                let actual: f64 = state
                    .iter()
                    .zip(apply_real_pauli(&state, p.terms()))
                    .map(|(a, b)| a * b)
                    .sum();
                if let ParityInference::Known(parity) = relations.infer(&records, p).unwrap() {
                    assert!((actual - if parity == Parity::Odd { -1. } else { 1. }).abs() < 1e-10);
                } else {
                    both = false;
                }
            }
            if both {
                known_mass += mass;
            }
        }
    }
    assert!((total - 1.).abs() < 1e-10);
    assert!((known_mass - 0.5).abs() < 1e-10);
}
#[test]
fn correlated_middle_resource_loss_is_one_physical_event_and_shards_replay() {
    let patch = SixRingPatch::bell_strip(3, NodeId(0), NodeId(1), 18).unwrap();
    let plan = compile(&patch, 1.);
    let targets = patch
        .relations(&plan, 1 << 20)
        .unwrap()
        .output_relations(&[3, 12], 1 << 20)
        .unwrap()
        .into_iter()
        .map(|(p, _)| p)
        .collect::<Vec<_>>();
    let burst = PhotonLossBurst::new((6..12).map(PhotonId).collect(), 0.3).unwrap();
    let shots = patch
        .sample(
            &plan,
            20,
            1024,
            771,
            PauliNoise::NONE,
            std::slice::from_ref(&burst),
            &targets,
            1 << 20,
        )
        .unwrap();
    let mut split = patch
        .sample(
            &plan,
            20,
            19,
            771,
            PauliNoise::NONE,
            std::slice::from_ref(&burst),
            &targets,
            1 << 20,
        )
        .unwrap();
    split.extend(
        patch
            .sample(
                &plan,
                39,
                1005,
                771,
                PauliNoise::NONE,
                &[burst],
                &targets,
                1 << 20,
            )
            .unwrap(),
    );
    assert_eq!(shots, split);
    let mut losses = 0;
    for shot in &shots {
        let a = shot.fusion_outcomes[0] == FusionOutcome::Loss;
        let b = shot.fusion_outcomes[1] == FusionOutcome::Loss;
        assert_eq!(a, b);
        if a {
            losses += 1;
            assert_eq!(shot.records.iter().filter(|v| v.is_none()).count(), 8);
            assert!(shot.inferred.iter().all(|p| *p == ParityInference::Erased));
        }
        assert!(shot.output_present.iter().all(|&v| v));
        for (inferred, &actual) in shot.inferred.iter().zip(&shot.expectations) {
            if let ParityInference::Known(p) = inferred {
                assert_eq!(actual, if *p == Parity::Odd { -1 } else { 1 });
            }
        }
    }
    let ci = WilsonInterval::new(losses, shots.len(), 3.).unwrap();
    assert!(ci.lower < 0.3 && ci.upper > 0.3);
    let all = PhotonLossBurst::new((0..18).map(PhotonId).collect(), 1.).unwrap();
    for shot in patch
        .sample(
            &plan,
            0,
            4,
            771,
            PauliNoise::NONE,
            &[all],
            &targets,
            1 << 20,
        )
        .unwrap()
    {
        assert!(shot.output_present.iter().all(|&v| !v));
        assert!(shot.expectations.iter().all(|&v| v == 0));
    }
}
