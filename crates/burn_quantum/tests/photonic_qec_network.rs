//! Physical four-star incidence, complete resource stabilizer checks, and toric decoder tests.
use burn_quantum::photonic::{qec::*, stabilizer::*, *};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
fn network(l: usize) -> (FourStarTorus, CompiledFusionNetwork) {
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
    let network = FourStarTorus::new(l, NodeId(0), NodeId(1), 4096).unwrap();
    let plan = network
        .program()
        .compile(&fabric, CompileOptions::default())
        .unwrap();
    (network, plan)
}
#[test]
fn four_star_checks_follow_actual_resource_stabilizers_and_full_fusion_records() {
    let (network, plan) = network(2);
    let initial = StabilizerSimulator::default().prepare(&plan).unwrap();
    let obs = network.observables().unwrap();
    assert_eq!(plan.program().num_photons(), 192);
    assert_eq!(plan.fusions().len(), 96);
    let derived = MeasurementRelations::derive(&initial, &obs, 1 << 20).unwrap();
    assert!(derived.checks().len() >= 14);
    let sampled = ReferenceSampler::compile(&initial, &obs, 1 << 20)
        .unwrap()
        .sample(0, 137, 72, PauliNoise::NONE, 1 << 20)
        .unwrap();
    for (graph, offset) in [(network.dual(), 0), (network.primal(), 1)] {
        assert_eq!(graph.vertices(), 8);
        assert_eq!(graph.edges().len(), 96);
        for check in graph.checks(1 << 20).unwrap() {
            assert_eq!(check.iter().filter(|&&b| b).count(), 24);
            let mut product = PauliProduct::new(vec![]).unwrap();
            let mut sign = false;
            for (i, &selected) in check.iter().enumerate() {
                if selected {
                    let (negative, next) =
                        product.multiply_commuting(&obs[2 * i + offset]).unwrap();
                    sign ^= negative;
                    product = next;
                }
            }
            assert_eq!(
                initial.expectation(&product).unwrap(),
                if sign { -1 } else { 1 }
            );
        }
        for shot in 0..sampled.shots() {
            let record: Vec<_> = (0..96)
                .map(|i| sampled.bit(2 * i + offset, shot).unwrap())
                .collect();
            assert!(graph.syndrome(&record).unwrap().iter().all(|&x| !x));
        }
    }
    for c in derived.checks() {
        for shot in 0..sampled.shots() {
            assert_eq!(
                c.terms
                    .iter()
                    .fold(false, |v, &i| v ^ sampled.bit(i, shot).unwrap()),
                c.negative
            );
        }
    }
    assert!(FourStarTorus::new(1000, NodeId(0), NodeId(1), usize::MAX).is_err());
}
#[test]
fn geometric_erasure_cycles_match_independent_binary_linear_algebra() {
    let (network, _) = network(2);
    let mut rng = ChaCha8Rng::seed_from_u64(71);
    for graph in [network.primal(), network.dual()] {
        let logicals = (0..3)
            .map(|j| graph.cuts().iter().map(|c| c[j]).collect())
            .collect();
        let code = ParityCode::new(
            graph.edges().len(),
            graph.checks(1 << 20).unwrap(),
            logicals,
        )
        .unwrap();
        for k in 0..200 {
            let p = (k % 20) as f64 / 20.;
            let erased: Vec<_> = (0..graph.edges().len())
                .map(|_| rng.random::<f64>() < p)
                .collect();
            let ambiguous = graph.erasure_logicals(&erased).unwrap();
            match code
                .decode_erasures(&vec![false; graph.vertices()], &erased)
                .unwrap()
            {
                ErasureDecode::Ambiguous { logicals } => assert_eq!(
                    (0..3).filter(|&i| ambiguous[i]).collect::<Vec<_>>(),
                    logicals
                ),
                ErasureDecode::Decoded { .. } => assert_eq!(ambiguous, [false; 3]),
                _ => panic!("homogeneous contradiction"),
            }
        }
    }
}
#[test]
fn matching_corrects_every_single_fault_and_rejects_excess_work() {
    let (network, _) = network(3);
    for graph in [network.primal(), network.dual()] {
        let n = graph.edges().len();
        for e in 0..n {
            let mut error = vec![false; n];
            error[e] = true;
            for erased in [vec![false; n], error.clone()] {
                let MatchingDecode::Correction(correction) = graph
                    .decode(&graph.syndrome(&error).unwrap(), &erased, 2)
                    .unwrap()
                else {
                    panic!("single error rejected")
                };
                let residual: Vec<_> = error.iter().zip(correction).map(|(&a, b)| a ^ b).collect();
                assert_eq!(graph.logical_parities(&residual).unwrap(), [false; 3]);
                assert!(graph.syndrome(&residual).unwrap().iter().all(|&b| !b));
            }
            assert_eq!(
                graph
                    .decode(&graph.syndrome(&error).unwrap(), &vec![false; n], 0)
                    .unwrap(),
                MatchingDecode::BudgetExceeded
            );
        }
        assert_eq!(graph.erasure_logicals(&vec![true; n]).unwrap(), [true; 3]);
        assert!(
            graph
                .decode(&vec![false; graph.vertices()], &vec![false; n], 21)
                .is_err()
        );
    }
}
#[test]
fn physical_memory_study_counts_all_attempts_and_replays_shards() {
    let (network, plan) = network(2);
    for loss in [0., 1.] {
        let result = run_four_star_memory(
            &network,
            &plan,
            NetworkNoise::new(loss, PauliNoise::NONE, false).unwrap(),
            0,
            64,
            73,
            16,
            1 << 22,
        )
        .unwrap();
        assert_eq!(result.attempts, 64);
        assert_eq!(result.corrected, if loss == 0. { 64 } else { 0 });
        assert_eq!(result.erased_logicals, if loss == 1. { 64 } else { 0 });
    }
    let noise =
        NetworkNoise::new(0.015, PauliNoise::new(0.002, 0.002, 0.002).unwrap(), false).unwrap();
    let all = run_four_star_memory(&network, &plan, noise, 0, 80, 99, 16, 1 << 22).unwrap();
    let (profiled, timings) =
        run_four_star_memory_profiled(&network, &plan, noise, 0, 80, 99, 16, 1 << 22).unwrap();
    assert_eq!(all, profiled);
    for seconds in [
        timings.setup_seconds,
        timings.sampling_seconds,
        timings.decoding_seconds,
    ] {
        assert!(seconds.is_finite() && seconds >= 0.);
    }
    let a = run_four_star_memory(&network, &plan, noise, 0, 31, 99, 16, 1 << 22).unwrap();
    let b = run_four_star_memory(&network, &plan, noise, 31, 49, 99, 16, 1 << 22).unwrap();
    assert_eq!(all.attempts, a.attempts + b.attempts);
    assert_eq!(all.corrected, a.corrected + b.corrected);
    assert_eq!(all.logical_errors, a.logical_errors + b.logical_errors);
    assert_eq!(all.erased_logicals, a.erased_logicals + b.erased_logicals);
    assert_eq!(all.decoder_budget, a.decoder_budget + b.decoder_budget);
    assert_eq!(
        all.attempts,
        all.corrected + all.logical_errors + all.erased_logicals + all.decoder_budget
    );
    let interval = WilsonInterval::new(0, 100, 1.95996398454).unwrap();
    assert!(interval.upper > 0.036 && interval.upper < 0.038);
    assert!(interval.lower < 1e-15);
}
#[test]
fn exact_matching_cost_agrees_with_exhaustive_cubic_subgraphs() {
    let (network, _) = network(2);
    let graph = network.primal();
    let mut unique = std::collections::BTreeMap::new();
    for (e, &(a, b)) in graph.edges().iter().enumerate() {
        unique.entry((a.min(b), a.max(b))).or_insert(e);
    }
    let coordinates: Vec<_> = unique.into_values().collect();
    assert_eq!(coordinates.len(), 12);
    for erase_some in [false, true] {
        let mut erased = vec![false; graph.edges().len()];
        if erase_some {
            for &e in coordinates.iter().take(4) {
                erased[e] = true;
            }
        }
        let mut minimum = std::collections::BTreeMap::new();
        for mask in 0..1usize << coordinates.len() {
            let mut fault = vec![false; graph.edges().len()];
            let mut cost = 0;
            for (j, &e) in coordinates.iter().enumerate() {
                if mask >> j & 1 != 0 {
                    fault[e] = true;
                    cost += usize::from(!erased[e]);
                }
            }
            let syndrome = graph.syndrome(&fault).unwrap();
            minimum
                .entry(syndrome)
                .and_modify(|c: &mut usize| *c = (*c).min(cost))
                .or_insert(cost);
        }
        assert_eq!(minimum.len(), 128);
        for (syndrome, cost) in minimum {
            let MatchingDecode::Correction(correction) =
                graph.decode(&syndrome, &erased, 8).unwrap()
            else {
                panic!("reachable syndrome rejected")
            };
            assert_eq!(graph.syndrome(&correction).unwrap(), syndrome);
            assert_eq!(
                correction
                    .iter()
                    .zip(&erased)
                    .filter(|&(e, l)| *e && !*l)
                    .count(),
                cost
            );
        }
    }
}
