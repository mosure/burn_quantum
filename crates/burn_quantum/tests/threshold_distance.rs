//! Exhaustive binary-fault oracle for shortest logical graph distance.
use burn_quantum::Error;
use burn_quantum::photonic::qec::threshold::{DecoderGraph, DetectorEdge, GraphLimits};

fn graph(detectors: usize, logicals: usize, edges: &[(usize, usize, u64)]) -> DecoderGraph {
    DecoderGraph::new(
        detectors,
        logicals,
        edges
            .iter()
            .map(|&(a, b, logical)| DetectorEdge { a, b, logical })
            .collect(),
        GraphLimits::default(),
    )
    .unwrap()
}

fn brute_force(graph: &DecoderGraph, logical: usize) -> Option<usize> {
    (1usize..1 << graph.edges().len())
        .filter_map(|bits| {
            let mask = (0..graph.edges().len())
                .map(|i| bits >> i & 1 != 0)
                .collect::<Vec<_>>();
            (graph.syndrome(&mask).unwrap().iter().all(|&v| !v)
                && graph.logical_flips(&mask).unwrap() >> logical & 1 != 0)
                .then_some(bits.count_ones() as usize)
        })
        .min()
}

fn compare(graph: &DecoderGraph) {
    for logical in 0..graph.logicals() {
        let expected = brute_force(graph, logical);
        let result = graph.minimum_logical_distance(logical, 1 << 20);
        if let Some(expected) = expected {
            let witness = result.unwrap();
            assert_eq!(witness.distance, expected, "{graph:?}, logical={logical}");
            assert_eq!(witness.edges.len(), expected);
            assert!(witness.edges.windows(2).all(|p| p[0] < p[1]));
            let mut faults = vec![false; graph.edges().len()];
            for &i in &witness.edges {
                faults[i] = true;
            }
            assert!(graph.syndrome(&faults).unwrap().iter().all(|&v| !v));
            assert_ne!(graph.logical_flips(&faults).unwrap() & (1 << logical), 0);
            assert!(witness.edge_visits <= 1 << 20);
            assert_eq!(
                graph
                    .minimum_logical_distance(logical, witness.edge_visits)
                    .unwrap(),
                witness
            );
            assert!(matches!(
                graph.minimum_logical_distance(logical, witness.edge_visits - 1),
                Err(Error::ResourceLimit { .. })
            ));
        } else {
            assert!(matches!(result, Err(Error::InvalidValue(_))), "{graph:?}");
        }
    }
}

#[test]
fn exact_distance_handles_boundary_parallel_edges_self_loops_and_disconnected_components() {
    for g in [
        graph(2, 2, &[(0, 2, 1), (0, 1, 0), (1, 2, 0)]),
        graph(2, 2, &[(0, 1, 1), (0, 1, 0)]),
        graph(2, 2, &[(0, 0, 1), (2, 2, 2)]),
        graph(
            5,
            2,
            &[(0, 1, 1), (1, 2, 0), (2, 0, 0), (3, 4, 0), (3, 4, 2)],
        ),
        // An odd cycle reached through a labelled bridge can generate a
        // doubled-stem BFS walk; the returned binary support cancels the stem.
        graph(4, 2, &[(0, 1, 1), (1, 2, 0), (2, 3, 1), (3, 1, 0)]),
        graph(3, 2, &[(0, 1, 1), (1, 2, 2)]),
        graph(1, 2, &[]),
    ] {
        compare(&g);
    }
}

#[test]
fn all_small_graph_masks_match_exhaustive_fault_enumeration() {
    // Every subgraph under several two-logical labelings of the three-detector
    // triangle plus its three virtual-boundary edges; every fault mask is tested.
    let topology = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
    for graph_bits in 0usize..1 << topology.len() {
        for labels in [0usize, 0b000001, 0b010101, 0b101010, 0b111111] {
            let edges = topology
                .iter()
                .enumerate()
                .filter_map(|(i, &(a, b))| {
                    (graph_bits >> i & 1 != 0).then_some((
                        a,
                        b,
                        ((labels >> i & 1) | ((labels.rotate_left(1) >> i & 1) << 1)) as u64,
                    ))
                })
                .collect::<Vec<_>>();
            compare(&graph(3, 2, &edges));
        }
    }
    // Deterministic multigraphs add multiple loops, disconnected pieces and
    // repeated endpoints, retaining each edge as a separate physical fault.
    let mut seed = 0x71a9_30cb_u64;
    for _ in 0..128 {
        let mut edges = vec![];
        for _ in 0..8 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            edges.push((
                (seed as usize) % 5,
                ((seed >> 16) as usize) % 5,
                (seed >> 40) & 7,
            ));
        }
        compare(&graph(4, 3, &edges));
    }
}

#[test]
fn logical_indices_budgets_and_isolated_vertices_are_checked() {
    let g = graph(2, 64, &[(0, 2, 1u64 << 63), (1, 2, 0), (0, 1, 0)]);
    assert_eq!(g.minimum_logical_distance(63, 1000).unwrap().distance, 3);
    assert!(matches!(
        g.minimum_logical_distance(64, 1000),
        Err(Error::InvalidValue(_))
    ));
    for budget in [0, 1, 5, 6] {
        assert!(matches!(
            g.minimum_logical_distance(63, budget),
            Err(Error::ResourceLimit { .. })
        ));
    }
    assert!(graph(1, 0, &[]).minimum_logical_distance(0, 1000).is_err());
    // The graph can represent many isolated detectors without allocating their
    // state. Distance search must retain that sparse-memory property.
    let huge = DecoderGraph::new(
        usize::MAX - 1,
        1,
        vec![DetectorEdge {
            a: usize::MAX - 1,
            b: usize::MAX - 1,
            logical: 1,
        }],
        GraphLimits {
            max_detectors: usize::MAX,
            max_edges: 1,
        },
    )
    .unwrap();
    let witness = huge.minimum_logical_distance(0, 3).unwrap();
    assert_eq!(witness.distance, 1);
    assert_eq!(witness.edges, vec![0]);
    assert_eq!(witness.edge_visits, 3);
}
