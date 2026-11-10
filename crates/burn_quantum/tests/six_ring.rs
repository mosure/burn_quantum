//! Rotated six-ring geometric checks are verified against actual quantum resource operators.
use burn_quantum::photonic::{qec::*, stabilizer::*, *};
fn prepare(network: &SixRingTorus) -> Tableau {
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
            transmission: RailTransmission::IDEAL,
        }],
    )
    .unwrap();
    let plan = network
        .program()
        .compile(&fabric, CompileOptions::default())
        .unwrap();
    assert!(plan.outputs().is_empty());
    StabilizerSimulator::default().prepare(&plan).unwrap()
}
#[test]
fn every_twelve_term_vertex_check_is_a_positive_actual_six_ring_stabilizer() {
    for side in [2, 4, 6] {
        let network = SixRingTorus::new(side, NodeId(0), NodeId(1), 2000).unwrap();
        let initial = prepare(&network);
        let observables = network.observables().unwrap();
        assert_eq!(network.program().num_photons(), 6 * side.pow(3));
        assert_eq!(network.program().num_fusions(), 3 * side.pow(3));
        let mut used = vec![false; observables.len()];
        for color in 0..2 {
            let graph = network.graph(color).unwrap();
            let records = network.record_indices(color).unwrap();
            assert_eq!(graph.vertices(), side.pow(3) / 2);
            assert_eq!(graph.edges().len(), 3 * side.pow(3));
            for &r in records {
                assert!(!used[r]);
                used[r] = true;
            }
            for check in graph.checks(1 << 22).unwrap() {
                assert_eq!(check.iter().filter(|&&b| b).count(), 12);
                let mut product = PauliProduct::new(vec![]).unwrap();
                let mut sign = false;
                for (edge, &present) in check.iter().enumerate() {
                    if present {
                        let (negative, next) = product
                            .multiply_commuting(&observables[records[edge]])
                            .unwrap();
                        sign ^= negative;
                        product = next;
                    }
                }
                assert_eq!(
                    initial.expectation(&product).unwrap(),
                    if sign { -1 } else { 1 }
                );
            }
        }
        assert!(used.iter().all(|&b| b));
    }
}
#[test]
fn full_reference_records_obey_both_geometric_and_independently_eliminated_checks() {
    let network = SixRingTorus::new(2, NodeId(0), NodeId(1), 100).unwrap();
    let initial = prepare(&network);
    let obs = network.observables().unwrap();
    let relations = MeasurementRelations::derive(&initial, &obs, 1 << 20).unwrap();
    let sampled = ReferenceSampler::compile(&initial, &obs, 1 << 20)
        .unwrap()
        .sample(0, 256, 819, PauliNoise::NONE, 1 << 20)
        .unwrap();
    for shot in 0..sampled.shots() {
        for color in 0..2 {
            let record: Vec<_> = network
                .record_indices(color)
                .unwrap()
                .iter()
                .map(|&i| sampled.bit(i, shot).unwrap())
                .collect();
            assert!(
                network
                    .graph(color)
                    .unwrap()
                    .syndrome(&record)
                    .unwrap()
                    .iter()
                    .all(|&b| !b)
            );
        }
        for check in relations.checks() {
            assert_eq!(
                check
                    .terms
                    .iter()
                    .fold(false, |v, &m| v ^ sampled.bit(m, shot).unwrap()),
                check.negative
            );
        }
    }
    assert!(SixRingTorus::new(3, NodeId(0), NodeId(1), 1000).is_err());
    assert!(SixRingTorus::new(100, NodeId(0), NodeId(1), usize::MAX).is_err());
    assert!(network.graph(2).is_err());
}
#[test]
fn single_physical_parity_faults_are_corrected_on_the_rotated_side_four_geometry() {
    let network = SixRingTorus::new(4, NodeId(0), NodeId(1), 1000).unwrap();
    for color in 0..2 {
        let graph = network.graph(color).unwrap();
        let n = graph.edges().len();
        for edge in 0..n {
            let mut error = vec![false; n];
            error[edge] = true;
            let MatchingDecode::Correction(correction) = graph
                .decode(&graph.syndrome(&error).unwrap(), &vec![false; n], 2)
                .unwrap()
            else {
                panic!("single fault rejected")
            };
            let residual: Vec<_> = error.iter().zip(correction).map(|(&a, b)| a ^ b).collect();
            assert_eq!(graph.logical_parities(&residual).unwrap(), [false; 3]);
        }
        assert_eq!(graph.erasure_logicals(&vec![true; n]).unwrap(), [true; 3]);
    }
}
fn compiled_memory(model: &PhysicalToricMemory) -> CompiledFusionNetwork {
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
            transmission: RailTransmission::IDEAL,
        }],
    )
    .unwrap();
    model
        .program()
        .compile(&fabric, CompileOptions::default())
        .unwrap()
}
#[test]
fn physical_shor_fusions_drive_outer_four_star_and_six_ring_memory_checks() {
    let four = FourStarTorus::new(2, NodeId(0), NodeId(1), 1000).unwrap();
    let six = SixRingTorus::new(2, NodeId(0), NodeId(1), 1000).unwrap();
    for code in [
        ToricFusionCode::four_star(&four),
        ToricFusionCode::six_ring(&six),
    ] {
        let bare = PhysicalToricMemory::bare(code.clone());
        let encoded = PhysicalToricMemory::shor(code, 1000).unwrap();
        assert_eq!(
            encoded.program().num_photons(),
            4 * bare.program().num_photons()
        );
        assert_eq!(
            encoded.program().num_fusions(),
            4 * bare.program().num_fusions()
        );
        assert_eq!(encoded.encoded_fusions(), bare.program().num_fusions());
        for model in [&bare, &encoded] {
            let plan = compiled_memory(model);
            for loss in [0., 1.] {
                let result = model
                    .run(
                        &plan,
                        NetworkNoise::new(loss, PauliNoise::NONE, false).unwrap(),
                        0,
                        4,
                        76,
                        MemoryBudgets::default(),
                    )
                    .unwrap();
                assert_eq!(result.memory.corrected, if loss == 0. { 4 } else { 0 });
                assert_eq!(
                    result.memory.erased_logicals,
                    if loss == 1. { 4 } else { 0 }
                );
                assert_eq!(result.detected_inner_events, 0);
                assert_eq!(result.photons_per_attempt, model.program().num_photons());
            }
        }
    }
}
#[test]
fn actual_encoded_unboosted_loss_and_pauli_noise_replay_without_postselection() {
    let six = SixRingTorus::new(2, NodeId(0), NodeId(1), 1000).unwrap();
    let model = PhysicalToricMemory::shor(ToricFusionCode::six_ring(&six), 1000).unwrap();
    let plan = compiled_memory(&model);
    let noise =
        NetworkNoise::new(0.01, PauliNoise::new(0.003, 0.003, 0.003).unwrap(), true).unwrap();
    let all = model
        .run(&plan, noise, 0, 80, 73, MemoryBudgets::default())
        .unwrap();
    let a = model
        .run(&plan, noise, 0, 31, 73, MemoryBudgets::default())
        .unwrap();
    let b = model
        .run(&plan, noise, 31, 49, 73, MemoryBudgets::default())
        .unwrap();
    assert_eq!(all.memory.attempts, 80);
    assert_eq!(
        all.memory.corrected,
        a.memory.corrected + b.memory.corrected
    );
    assert_eq!(
        all.memory.erased_logicals,
        a.memory.erased_logicals + b.memory.erased_logicals
    );
    assert_eq!(
        all.memory.logical_errors,
        a.memory.logical_errors + b.memory.logical_errors
    );
    assert_eq!(
        all.memory.decoder_budget,
        a.memory.decoder_budget + b.memory.decoder_budget
    );
    assert_eq!(
        all.detected_inner_events,
        a.detected_inner_events + b.detected_inner_events
    );
    assert!(all.detected_inner_events > 0);
    assert_eq!(
        all.memory.corrected
            + all.memory.erased_logicals
            + all.memory.logical_errors
            + all.memory.decoder_budget,
        80
    );
}

#[test]
fn every_conditional_bell_branch_matches_independent_pinned_stim() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("stabilizer/six-ring-stim-1.15.0.json")).unwrap();
    assert_eq!(fixture["stim_version"], "1.15.0");
    assert_eq!(
        fixture["source_archive_sha256"],
        "95236006859d6754be99629d4fb44788e742e962ac8c59caad421ca088f7350e"
    );
    assert_eq!(fixture["side"], 2);
    let network = SixRingTorus::new(2, NodeId(0), NodeId(1), 100).unwrap();
    let initial = prepare(&network);
    let observables = network.observables().unwrap();
    assert_eq!(observables.len(), 48);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let records = case.as_array().unwrap();
        assert_eq!(records.len(), observables.len());
        let mut state = initial.clone();
        for (record, observable) in records.iter().zip(&observables) {
            let p = match record["basis"].as_str().unwrap() {
                "X" => Pauli::X,
                "Z" => Pauli::Z,
                invalid => panic!("invalid pinned Bell basis {invalid}"),
            };
            let expected = PauliProduct::new(
                record["sites"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|q| (q.as_u64().unwrap() as usize, p))
                    .collect(),
            )
            .unwrap();
            assert_eq!(*observable, expected);
            let prior = record["prior"].as_i64().unwrap();
            assert!((-1..=1).contains(&prior));
            let prior = prior as i8;
            assert_eq!(state.expectation(observable).unwrap(), prior);
            let negative = record["negative"].as_bool().unwrap();
            let actual = state.measure(observable, negative).unwrap();
            assert_eq!(actual.negative, negative);
            assert_eq!(actual.random, prior == 0);
        }
    }
}
