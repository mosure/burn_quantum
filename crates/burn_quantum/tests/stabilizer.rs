//! Independent exact-amplitude and dense-instrument checks for the host Clifford engine.
use burn_quantum::Error;
use burn_quantum::photonic::{stabilizer::*, *};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
mod common;

fn pauli(terms: &[(usize, Pauli)]) -> PauliProduct {
    PauliProduct::new(terms.to_vec()).unwrap()
}
fn products(n: usize) -> Vec<PauliProduct> {
    (0..4usize.pow(n as u32))
        .map(|mut k| {
            let mut terms = vec![];
            for q in 0..n {
                let p = k % 4;
                k /= 4;
                if p != 0 {
                    terms.push((q, [Pauli::X, Pauli::Y, Pauli::Z][p - 1]));
                }
            }
            PauliProduct::new(terms).unwrap()
        })
        .collect()
}
fn apply_pauli(state: &[Complex64], n: usize, p: &PauliProduct) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0., 0.); state.len()];
    for (basis, &amplitude) in state.iter().enumerate() {
        let mut target = basis;
        let mut coefficient = Complex64::new(1., 0.);
        for &(q, factor) in p.terms() {
            let mask = 1 << (n - 1 - q);
            let one = basis & mask != 0;
            if factor != Pauli::Z {
                target ^= mask;
            }
            match factor {
                Pauli::Y => coefficient *= Complex64::new(0., if one { -1. } else { 1. }),
                Pauli::Z if one => coefficient = -coefficient,
                _ => {}
            }
        }
        out[target] += coefficient * amplitude;
    }
    out
}
fn expectation(state: &[Complex64], n: usize, p: &PauliProduct) -> f64 {
    state
        .iter()
        .zip(apply_pauli(state, n, p))
        .map(|(a, b)| (a.conj() * b).re)
        .sum()
}
#[test]
fn random_clifford_circuits_and_measurements_match_all_scalar_pauli_expectations() {
    let n = 4;
    let queries = products(n);
    let mut rng = ChaCha8Rng::seed_from_u64(3907);
    for _ in 0..12 {
        let mut tableau = Tableau::zero(n, TableauLimits::default()).unwrap();
        let mut vector = vec![Complex64::new(0., 0.); 1 << n];
        vector[0] = 1.0.into();
        for _ in 0..48 {
            let q = rng.random_range(0..n);
            let gate = rng.random_range(0..4);
            let (targets, matrix) = match gate {
                0 => {
                    tableau.h(q).unwrap();
                    (vec![q], burn_quantum::Unitary::h().matrix().clone())
                }
                1 => {
                    tableau.s(q).unwrap();
                    (vec![q], burn_quantum::Unitary::s().matrix().clone())
                }
                2 => {
                    tableau.x(q).unwrap();
                    (vec![q], burn_quantum::Unitary::x().matrix().clone())
                }
                _ => {
                    let t = (q + rng.random_range(1..n)) % n;
                    tableau.cnot(q, t).unwrap();
                    (vec![q, t], burn_quantum::Unitary::cnot().matrix().clone())
                }
            };
            vector = common::apply_reference(&vector, &vec![2; n], &targets, &matrix);
        }
        for _ in 0..8 {
            for query in &queries {
                assert!(
                    (tableau.expectation(query).unwrap() as f64 - expectation(&vector, n, query))
                        .abs()
                        < 1e-10
                );
            }
            let query = &queries[rng.random_range(1..queries.len())];
            let mean = expectation(&vector, n, query);
            let result = tableau.measure(query, rng.random()).unwrap();
            assert_eq!(result.random, mean.abs() < 0.5);
            let sign = if result.negative { -1. } else { 1. };
            let transformed = apply_pauli(&vector, n, query);
            let norm = (2. * (1. + sign * mean)).sqrt();
            for (a, b) in vector.iter_mut().zip(transformed) {
                *a = (*a + sign * b) / norm;
            }
        }
    }
}
#[test]
fn pauli_frames_match_explicit_corrections_through_clifford_gates() {
    let mut frame = PauliFrame::new(3, 3).unwrap();
    let mut reference = Tableau::zero(3, TableauLimits::default()).unwrap();
    reference.h(0).unwrap();
    reference.cnot(0, 1).unwrap();
    reference.s(2).unwrap();
    let mut corrected = reference.clone();
    corrected.x(1).unwrap();
    corrected.z(2).unwrap();
    frame.toggle(1, Pauli::X).unwrap();
    frame.toggle(2, Pauli::Z).unwrap();
    reference.h(1).unwrap();
    corrected.h(1).unwrap();
    frame.h(1).unwrap();
    reference.s(1).unwrap();
    corrected.s(1).unwrap();
    frame.s(1).unwrap();
    reference.cnot(1, 2).unwrap();
    corrected.cnot(1, 2).unwrap();
    frame.cnot(1, 2).unwrap();
    for p in products(3) {
        let sign = if frame.flips(&p).unwrap() { -1 } else { 1 };
        assert_eq!(
            corrected.expectation(&p).unwrap(),
            sign * reference.expectation(&p).unwrap()
        );
    }
    assert!(PauliProduct::new(vec![(0, Pauli::X), (0, Pauli::Z)]).is_err());
    assert!(reference.measure(&pauli(&[(3, Pauli::Z)]), false).is_err());
    assert!(frame.cnot(0, 0).is_err());
}
fn plan(eta: RailTransmission, reverse: bool) -> CompiledFusionNetwork {
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
            transmission: eta,
        }],
    )
    .unwrap();
    let mut program = FusionProgram::new();
    let a = program
        .add_resource(ResourceState::bell_pair(), NodeId(0))
        .unwrap();
    let b = program
        .add_resource(ResourceState::bell_pair(), NodeId(0))
        .unwrap();
    if reverse {
        program.fuse(b[0], a[1], NodeId(1)).unwrap();
    } else {
        program.fuse(a[1], b[0], NodeId(1)).unwrap();
    }
    program.compile(&fabric, CompileOptions::default()).unwrap()
}
fn branch(outcome: FusionOutcome) -> usize {
    match outcome {
        FusionOutcome::Success { xx: Parity::Even } => 0,
        FusionOutcome::Success { xx: Parity::Odd } => 1,
        FusionOutcome::Failure { bit: false } => 2,
        FusionOutcome::Failure { bit: true } => 3,
        FusionOutcome::Loss => 4,
        _ => panic!("unknown outcome"),
    }
}
#[test]
fn lossy_fusions_match_dense_joint_distribution_and_conditional_coherences() {
    for reverse in [false, true] {
        let plan = plan(RailTransmission::uniform(0.8).unwrap(), reverse);
        let exact = DenseFusionSimulator::default()
            .execute(
                &plan,
                1,
                &burn::tensor::Device::flex(),
                burn::tensor::DType::F64,
            )
            .unwrap();
        let engine = StabilizerSimulator::default();
        let shots = engine
            .sample(
                &plan,
                0,
                20000,
                792,
                &[pauli(&[(0, Pauli::X), (3, Pauli::X)])],
            )
            .unwrap();
        let mut counts = [[0usize; 9]; 5];
        for shot in &shots {
            let b = branch(shot.outcomes[0]);
            let index = shot.output_digits[0] * 3 + shot.output_digits[1];
            counts[b][index] += 1;
            if b < 2 {
                assert_eq!(shot.pauli_expectations[0], if b == 0 { 1 } else { -1 });
            }
        }
        for (b, exact_branch) in exact.branches().iter().enumerate() {
            let expected =
                common::values(exact_branch.unnormalized_state().unwrap().probabilities());
            for (i, p) in expected.iter().enumerate() {
                assert!(
                    (counts[b][i] as f64 / 20000. - p).abs() < 0.012,
                    "branch {b} basis {i}"
                );
            }
        }
        // Hidden measurements of the lost Bell partner must randomize the survivor.
        let loss_zero = shots
            .iter()
            .filter(|s| branch(s.outcomes[0]) == 4 && s.output_digits[0] == 0)
            .count();
        let loss_count = shots.iter().filter(|s| branch(s.outcomes[0]) == 4).count();
        assert!((loss_zero as f64 / loss_count as f64 - 0.5).abs() < 0.03);
    }
}
#[test]
fn shot_sharding_replays_identically_and_capabilities_fail_before_work() {
    let plan = plan(RailTransmission::uniform(0.73).unwrap(), false);
    let engine = StabilizerSimulator::default();
    let all = engine.sample(&plan, 10, 97, 81, &[]).unwrap();
    let mut parts = engine.sample(&plan, 10, 23, 81, &[]).unwrap();
    parts.extend(engine.sample(&plan, 33, 74, 81, &[]).unwrap());
    assert_eq!(all, parts);
    assert!(engine.sample(&plan, u64::MAX, 1, 0, &[]).is_err());
    assert!(
        engine
            .sample(&plan, 0, 1, 0, &[pauli(&[(1, Pauli::Z)])])
            .is_err()
    );
    let asymmetric = plan_for_asymmetry();
    assert!(matches!(
        engine.prepare(&asymmetric),
        Err(Error::Unsupported(_))
    ));
    let bounded = StabilizerSimulator {
        limits: StabilizerLimits {
            max_result_elements: 1,
            ..Default::default()
        },
    };
    assert!(bounded.sample(&plan, 0, 1, 0, &[]).is_err());
}
fn plan_for_asymmetry() -> CompiledFusionNetwork {
    plan(RailTransmission::new(0.8, 0.9).unwrap(), false)
}
#[test]
fn tableau_storage_is_polynomial_and_large_resources_do_not_allocate_dense_states() {
    let mut previous = 0;
    for n in [256, 512, 1024, 2048] {
        let start = std::time::Instant::now();
        let mut state = Tableau::zero(n, TableauLimits::default()).unwrap();
        state.h(0).unwrap();
        for q in 1..n {
            state.cnot(0, q).unwrap();
        }
        assert_eq!(
            state
                .expectation(&pauli(&[(0, Pauli::Z), (n - 1, Pauli::Z)]))
                .unwrap(),
            1
        );
        let bytes = state.allocated_bytes();
        assert!(bytes < 4 * n * n + 128 * n);
        if previous != 0 {
            assert!(bytes <= 4 * previous);
        }
        previous = bytes;
        eprintln!(
            "stabilizer qubits={n} allocated_bytes={bytes} preparation_ms={}",
            start.elapsed().as_millis()
        );
    }
    assert!(Tableau::zero(usize::MAX, TableauLimits::default()).is_err());
    // Execute an actual resource recipe far beyond the dense model's six-site limit.
    let fabric = Fabric::new(
        vec![FabricNode {
            kind: NodeKind::Source,
            storage: RailTransmission::IDEAL,
        }],
        vec![],
    )
    .unwrap();
    let mut program = FusionProgram::new();
    for _ in 0..100 {
        program
            .add_resource(ResourceState::bell_pair(), NodeId(0))
            .unwrap();
    }
    let compiled = program.compile(&fabric, CompileOptions::default()).unwrap();
    let records = StabilizerSimulator::default()
        .sample(&compiled, 0, 4, 93, &[])
        .unwrap();
    assert_eq!(compiled.outputs().len(), 200);
    for record in records {
        assert!(
            record
                .output_digits
                .chunks_exact(2)
                .all(|p| p[0] == p[1] && p[0] < 2)
        );
    }
    assert!(
        Tableau::zero(
            1024,
            TableauLimits {
                max_qubits: 2048,
                max_words: 1
            }
        )
        .is_err()
    );
}

#[test]
fn clifford_and_postmeasurement_states_match_pinned_stim_1_15_0() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        stim_version: String,
        source_archive_sha256: String,
        qubits: usize,
        cases: Vec<Case>,
    }
    #[derive(serde::Deserialize)]
    struct Case {
        gates: Vec<String>,
        postselections: Vec<(String, bool)>,
        expectations: Vec<(String, i8)>,
    }
    fn parse(text: &str) -> PauliProduct {
        PauliProduct::new(
            text.chars()
                .enumerate()
                .filter_map(|(q, c)| match c {
                    '_' => None,
                    'X' => Some((q, Pauli::X)),
                    'Y' => Some((q, Pauli::Y)),
                    'Z' => Some((q, Pauli::Z)),
                    _ => panic!("invalid fixture Pauli"),
                })
                .collect(),
        )
        .unwrap()
    }
    let fixture: Fixture =
        serde_json::from_str(include_str!("stabilizer/stim-1.15.0.json")).unwrap();
    assert_eq!(fixture.stim_version, "1.15.0");
    assert_eq!(
        fixture.source_archive_sha256,
        "95236006859d6754be99629d4fb44788e742e962ac8c59caad421ca088f7350e"
    );
    assert_eq!(fixture.qubits, 4);
    assert_eq!(fixture.cases.len(), 4);
    for (index, case) in fixture.cases.into_iter().enumerate() {
        assert_eq!(case.gates.len(), [6, 6, 7, 7][index]);
        assert_eq!(case.postselections.len(), [0, 3, 0, 3][index]);
        assert_eq!(case.expectations.len(), 256);
        assert_eq!(
            case.expectations
                .iter()
                .map(|(p, _)| p)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            256
        );
        assert!(
            case.expectations
                .iter()
                .all(|(p, v)| p.len() == 4 && (-1..=1).contains(v))
        );
        let mut tableau = Tableau::zero(fixture.qubits, TableauLimits::default()).unwrap();
        for gate in case.gates {
            let parts: Vec<_> = gate.split_whitespace().collect();
            let operands: Vec<usize> = parts[1..].iter().map(|q| q.parse().unwrap()).collect();
            match parts[0] {
                "H" => {
                    for q in operands {
                        tableau.h(q).unwrap();
                    }
                }
                "S" => {
                    for q in operands {
                        tableau.s(q).unwrap();
                    }
                }
                "CX" => tableau.cnot(operands[0], operands[1]).unwrap(),
                "CZ" => tableau.cz(operands[0], operands[1]).unwrap(),
                _ => panic!("unknown fixture gate"),
            }
        }
        for (observable, negative) in case.postselections {
            assert_eq!(
                tableau
                    .measure(&parse(&observable), negative)
                    .unwrap()
                    .negative,
                negative
            );
        }
        for (observable, expected) in case.expectations {
            assert_eq!(
                tableau.expectation(&parse(&observable)).unwrap(),
                expected,
                "{observable}"
            );
        }
    }
}
