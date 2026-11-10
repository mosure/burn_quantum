//! Repeated noisy measurement maps checked against actual planar tableaus.
use burn_quantum::photonic::{
    qec::{PlanarCode, threshold::*},
    stabilizer::Tableau,
};
fn memory(d: usize, r: usize) -> PhenomenologicalMemory {
    PhenomenologicalMemory::new(
        &PlanarCode::new(d, 2000).unwrap(),
        r,
        GraphLimits::default(),
    )
    .unwrap()
}
#[test]
fn all_low_weight_faults_decode_below_the_distance_bound() {
    for (d, rounds, max_weight) in [(3, 3, 1), (5, 2, 2)] {
        let m = memory(d, rounds);
        for channel in 0..2 {
            let g = m.channel(channel).unwrap().graph();
            let n = g.edges().len();
            let erased = vec![false; n];
            for a in 0..n {
                for b in a..if max_weight == 2 { n } else { a + 1 } {
                    let mut fault = vec![false; n];
                    fault[a] = true;
                    if b != a {
                        fault[b] = true;
                    }
                    let syndrome = g.syndrome(&fault).unwrap();
                    let out = g
                        .decode_union_find(&syndrome, &erased, DecoderBudget::default())
                        .unwrap();
                    assert_eq!(
                        out.status,
                        GraphStatus::Corrected,
                        "d{d} ch{channel} {a},{b}"
                    );
                    assert_eq!(g.syndrome(&out.correction).unwrap(), syndrome);
                    assert_eq!(
                        out.logical,
                        g.logical_flips(&fault).unwrap(),
                        "d{d} ch{channel} {a},{b}"
                    );
                }
            }
        }
    }
}
#[test]
fn detector_edges_match_actual_planar_tableau_histories_with_noisy_records() {
    let code = PlanarCode::new(3, 13).unwrap();
    let memory = memory(3, 3);
    for channel in 0..2 {
        let repeated = memory.channel(channel).unwrap();
        let graph = repeated.graph();
        let checks = if channel == 0 {
            code.z_checks(0).unwrap()
        } else {
            code.x_checks(0).unwrap()
        };
        for injected in 0..graph.edges().len() {
            let mut state = Tableau::zero(code.qubits(), Default::default()).unwrap();
            code.encode(&mut state, 0).unwrap();
            let mut history = Vec::new();
            for round in 0..repeated.rounds() {
                for q in 0..code.qubits() {
                    if repeated.data_edge(round, q).unwrap() == injected {
                        match channel {
                            0 => state.x(q).unwrap(),
                            _ => state.z(q).unwrap(),
                        }
                    }
                }
                let mut row: Vec<_> = checks
                    .iter()
                    .map(|p| state.expectation(p).unwrap() == -1)
                    .collect();
                for (c, v) in row.iter_mut().enumerate() {
                    *v ^= repeated.measurement_edge(round, c).unwrap() == injected;
                }
                history.push(row);
            }
            history.push(
                checks
                    .iter()
                    .map(|p| state.expectation(p).unwrap() == -1)
                    .collect(),
            );
            let mut fault = vec![false; graph.edges().len()];
            fault[injected] = true;
            assert_eq!(
                repeated.detectors_from_history(&history).unwrap(),
                graph.syndrome(&fault).unwrap()
            );
            let logical = if channel == 0 {
                code.logical_z(0).unwrap()
            } else {
                code.logical_x(0).unwrap()
            };
            // The Z observable is initialized known; the X logical is random in |0L>.
            if channel == 0 {
                assert_eq!(
                    state.expectation(&logical).unwrap() == -1,
                    graph.logical_flips(&fault).unwrap() != 0
                );
            }
        }
    }
}
#[test]
fn growing_distances_replay_fixed_shots_with_explicit_noisy_measurements() {
    for d in [3, 5, 7, 9, 11] {
        let m = memory(d, d);
        let mut noise = PhenomenologicalNoise::symmetric(0.015).unwrap();
        noise.data_erasure = 0.005;
        noise.measurement_erasure = 0.003;
        let full = m.run(noise, 20, 32, 987, DecoderBudget::default()).unwrap();
        let mut shards = m.run(noise, 20, 9, 987, DecoderBudget::default()).unwrap();
        shards
            .add(&m.run(noise, 29, 23, 987, DecoderBudget::default()).unwrap())
            .unwrap();
        assert_eq!(full, shards);
        assert_eq!(full.decoder_limits, 0);
        assert_eq!(full.inconsistent, 0);
        assert_eq!(full.logical_rank_histogram.iter().sum::<usize>(), 32);
        let ideal = m
            .run(
                PhenomenologicalNoise::symmetric(0.).unwrap(),
                0,
                3,
                1,
                DecoderBudget::default(),
            )
            .unwrap();
        assert_eq!(ideal.unsuccessful().unwrap(), 0);
    }
    let m = memory(3, 3);
    let budget = DecoderBudget {
        max_edge_visits: 0,
        max_growth_rounds: 0,
    };
    let limited = m
        .run(
            PhenomenologicalNoise::symmetric(0.).unwrap(),
            0,
            12,
            1,
            budget,
        )
        .unwrap();
    assert_eq!(limited.decoder_limits, 12);
    assert_eq!(limited.unsuccessful().unwrap(), 12);
}
#[test]
fn malformed_histories_and_hyperedges_are_rejected() {
    let m = memory(3, 3);
    assert!(m.channel(0).unwrap().detectors_from_history(&[]).is_err());
    let code =
        burn_quantum::photonic::qec::ParityCode::new(1, vec![vec![true]; 3], vec![]).unwrap();
    assert!(RepeatedMeasurements::new(&code, 3, GraphLimits::default()).is_err());
    let mut n = PhenomenologicalNoise::symmetric(0.1).unwrap();
    n.measurement = f64::NAN;
    assert!(m.run(n, 0, 1, 1, DecoderBudget::default()).is_err());
}
