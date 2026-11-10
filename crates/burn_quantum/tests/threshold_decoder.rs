//! Exhaustive independent binary-algebra oracles for sparse erasure homology.
use burn_quantum::photonic::{
    NodeId,
    qec::{SixRingTorus, threshold::*},
};
fn graph() -> DecoderGraph {
    // Two boundary paths, a triangle and a logical self-loop exercise relative
    // cycles, parallel edges and distinct (possibly dependent) logical labels.
    DecoderGraph::new(
        3,
        3,
        vec![
            DetectorEdge {
                a: 3,
                b: 0,
                logical: 1,
            },
            DetectorEdge {
                a: 0,
                b: 1,
                logical: 0,
            },
            DetectorEdge {
                a: 1,
                b: 3,
                logical: 0,
            },
            DetectorEdge {
                a: 1,
                b: 2,
                logical: 2,
            },
            DetectorEdge {
                a: 2,
                b: 0,
                logical: 0,
            },
            DetectorEdge {
                a: 2,
                b: 2,
                logical: 4,
            },
            DetectorEdge {
                a: 0,
                b: 1,
                logical: 3,
            },
        ],
        GraphLimits::default(),
    )
    .unwrap()
}
#[test]
fn every_erasure_mask_matches_enumerated_kernel_image_and_rank() {
    let g = graph();
    let n = g.edges().len();
    for mask in 0usize..1 << n {
        let erased: Vec<_> = (0..n).map(|i| mask >> i & 1 != 0).collect();
        let actual = g
            .analyze_erasures(&erased, DecoderBudget::default())
            .unwrap();
        let mut image = std::collections::BTreeSet::new();
        for fault in 0usize..1 << n {
            if fault & !mask != 0 {
                continue;
            }
            let bits: Vec<_> = (0..n).map(|i| fault >> i & 1 != 0).collect();
            if g.syndrome(&bits).unwrap().iter().all(|&v| !v) {
                image.insert(g.logical_flips(&bits).unwrap());
            }
        }
        assert_eq!(1 << actual.rank(), image.len(), "mask{mask}");
        let mut basis_image = std::collections::BTreeSet::new();
        for select in 0..1 << actual.rank() {
            basis_image.insert(
                actual
                    .logical_basis
                    .iter()
                    .enumerate()
                    .fold(0, |v, (i, &b)| v ^ if select >> i & 1 != 0 { b } else { 0 }),
            );
        }
        assert_eq!(basis_image, image, "mask{mask}");
        assert_eq!(
            actual.status,
            if image.len() == 1 {
                GraphStatus::Corrected
            } else {
                GraphStatus::Ambiguous
            }
        );
    }
}
#[test]
fn graph_bounds_and_work_exhaustion_are_explicit() {
    assert!(DecoderGraph::new(2, 65, vec![], GraphLimits::default()).is_err());
    assert!(
        DecoderGraph::new(
            2,
            1,
            vec![DetectorEdge {
                a: 0,
                b: 3,
                logical: 0
            }],
            GraphLimits::default()
        )
        .is_err()
    );
    assert!(
        DecoderGraph::new(
            2,
            1,
            vec![DetectorEdge {
                a: 0,
                b: 1,
                logical: 2
            }],
            GraphLimits::default()
        )
        .is_err()
    );
    assert!(
        DecoderGraph::new(
            3,
            0,
            vec![],
            GraphLimits {
                max_detectors: 2,
                max_edges: 0
            }
        )
        .is_err()
    );
    let g = graph();
    let out = g
        .analyze_erasures(
            &vec![true; g.edges().len()],
            DecoderBudget {
                max_edge_visits: 3,
                max_growth_rounds: 0,
            },
        )
        .unwrap();
    assert_eq!(out.status, GraphStatus::WorkLimit);
    assert_eq!(out.work.edge_visits, 3);
    assert!(out.logical_basis.is_empty());
}
#[test]
fn six_ring_exact_study_is_shard_replayable_and_counts_all_failures() {
    let network = SixRingTorus::new(4, NodeId(0), NodeId(1), 400).unwrap();
    let model = SixRingErasureModel::new(&network, GraphLimits::default()).unwrap();
    let budget = DecoderBudget::default();
    let all = model.run(0.12, 100, 500, 91, budget).unwrap();
    let mut parts = model.run(0.12, 100, 177, 91, budget).unwrap();
    parts
        .add(&model.run(0.12, 277, 323, 91, budget).unwrap())
        .unwrap();
    assert_eq!(all, parts);
    assert_eq!(all.logical_rank_histogram.iter().sum::<usize>(), 500);
    let zero = model.run(0., 0, 100, 10, budget).unwrap();
    assert_eq!(zero.logical_failures, 0);
    assert_eq!(zero.ambiguous_erasures, 0);
    let lost = model.run(1., 0, 2000, 10, budget).unwrap();
    assert_eq!(lost.logical_rank_histogram[6], 2000);
    assert!(lost.logical_failures > 1920 && lost.logical_failures < 1995);
    assert_eq!(lost.conditional_logical_failures(), 2000. * 63. / 64.);
    let limited = model
        .run(
            0.,
            0,
            10,
            10,
            DecoderBudget {
                max_edge_visits: 0,
                max_growth_rounds: 0,
            },
        )
        .unwrap();
    assert_eq!(limited.decoder_limits, 10);
    assert_eq!(limited.unsuccessful().unwrap(), 10);
    let serialized = serde_json::to_string(&all).unwrap();
    assert_eq!(
        serde_json::from_str::<ThresholdCounts>(&serialized).unwrap(),
        all
    );
}
#[test]
fn six_ring_record_masks_preserve_correlations_and_match_existing_homology() {
    let network = SixRingTorus::new(4, NodeId(0), NodeId(1), 400).unwrap();
    let model = SixRingErasureModel::new(&network, GraphLimits::default()).unwrap();
    for pattern in 0..100 {
        for color in 0..2 {
            let mask: Vec<_> = model
                .record_indices(color)
                .unwrap()
                .iter()
                .map(|r| ((r * 17 + pattern * 31) % 101) < pattern)
                .collect();
            let new = model
                .graph(color)
                .unwrap()
                .analyze_erasures(&mask, DecoderBudget::default())
                .unwrap();
            let old = network
                .graph(color)
                .unwrap()
                .erasure_logicals(&mask)
                .unwrap();
            for (i, &v) in old.iter().enumerate() {
                assert_eq!(new.ambiguous_mask() >> i & 1 != 0, v);
            }
        }
    }
    let full = model
        .run_with_masks(0, 40, 11, DecoderBudget::default(), |shot, _, mask| {
            mask.fill(shot % 2 == 0);
            Ok(())
        })
        .unwrap();
    assert_eq!(full.logical_rank_histogram[0], 20);
    assert_eq!(full.logical_rank_histogram[6], 20);
}
#[test]
fn peeling_matches_every_supported_syndrome_even_with_boundary_cycles() {
    let g = graph();
    let n = g.edges().len();
    for mask in 0usize..1 << n {
        let erased: Vec<_> = (0..n).map(|i| mask >> i & 1 != 0).collect();
        for fault in 0usize..1 << n {
            if fault & !mask != 0 {
                continue;
            }
            let errors: Vec<_> = (0..n).map(|i| fault >> i & 1 != 0).collect();
            let syndrome = g.syndrome(&errors).unwrap();
            let out = g
                .decode_erasures(&syndrome, &erased, DecoderBudget::default())
                .unwrap();
            assert!(matches!(
                out.status,
                GraphStatus::Corrected | GraphStatus::Ambiguous
            ));
            assert_eq!(g.syndrome(&out.correction).unwrap(), syndrome);
            assert!(out.correction.iter().zip(&erased).all(|(&c, &e)| !c || e));
            if out.status == GraphStatus::Corrected {
                assert_eq!(out.logical, g.logical_flips(&errors).unwrap());
            }
        }
    }
    let impossible = g
        .decode_erasures(
            &[true, false, false],
            &vec![false; n],
            DecoderBudget::default(),
        )
        .unwrap();
    assert_eq!(impossible.status, GraphStatus::Inconsistent);
}
#[test]
fn independent_fault_runner_replays_and_preserves_incomplete_attempts() {
    let g = graph();
    let budget = DecoderBudget::default();
    let whole = g.run_independent_faults(0.15, 10, 100, 93, budget).unwrap();
    let mut parts = g.run_independent_faults(0.15, 10, 41, 93, budget).unwrap();
    let other = g.run_independent_faults(0.15, 51, 59, 93, budget).unwrap();
    parts.counts.add(&other.counts).unwrap();
    for (a, b) in parts
        .logical_failures
        .iter_mut()
        .zip(other.logical_failures)
    {
        *a += b;
    }
    assert_eq!(parts, whole);
    let limited = g
        .run_independent_faults(
            0.1,
            0,
            20,
            93,
            DecoderBudget {
                max_edge_visits: 0,
                max_growth_rounds: 0,
            },
        )
        .unwrap();
    assert_eq!(limited.counts.unsuccessful().unwrap(), 20);
    assert_eq!(limited.counts.decoder_limits, 20);
    assert_eq!(limited.logical_failures, vec![0; 3]);
}
