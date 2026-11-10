//! Local physical cube checks are proved complete against independent global
//! stabilizer elimination, then used without a dense state for larger distances.
use burn_quantum::photonic::{
    NodeId,
    qec::{ParityRelation, fusion_surgery::*, threshold::*},
    stabilizer::*,
};
fn initial(net: &SixRingSurfaceNetwork) -> Tableau {
    let mut s = Tableau::zero(
        net.patch().program().num_photons(),
        TableauLimits::default(),
    )
    .unwrap();
    for c in 0..net.resource_centers().len() {
        for q in 0..6 {
            s.h(6 * c + q).unwrap();
        }
        for q in 0..6 {
            s.cz(6 * c + q, 6 * c + (q + 1) % 6).unwrap();
        }
    }
    s
}
fn row(relation: &ParityRelation, width: usize) -> Vec<u64> {
    let mut v = vec![0; width.div_ceil(64)];
    for &q in &relation.terms {
        v[q / 64] ^= 1 << (q % 64);
    }
    v
}
fn insert(basis: &mut Vec<(usize, Vec<u64>)>, mut value: Vec<u64>) -> bool {
    for (p, row) in basis.iter() {
        if value[p / 64] >> (p % 64) & 1 != 0 {
            for (a, &b) in value.iter_mut().zip(row) {
                *a ^= b;
            }
        }
    }
    let Some(p) = value
        .iter()
        .enumerate()
        .find_map(|(i, &word)| (word != 0).then(|| 64 * i + word.trailing_zeros() as usize))
    else {
        return false;
    };
    basis.push((p, value));
    true
}
fn in_span(basis: &[(usize, Vec<u64>)], mut value: Vec<u64>) -> bool {
    for (p, row) in basis {
        if value[p / 64] >> (p % 64) & 1 != 0 {
            for (a, &b) in value.iter_mut().zip(row) {
                *a ^= b;
            }
        }
    }
    value.iter().all(|&v| v == 0)
}
#[test]
fn local_closed_physical_check_space_equals_full_resource_relation_space() {
    for (d, t, seam) in [
        (3, 3, None),
        (5, 5, None),
        (3, 5, Some([0, 0])),
        (3, 5, Some([1, 4])),
        (3, 5, Some([2, 3])),
        (5, 7, Some([1, 6])),
    ] {
        let net = match seam {
            None => SixRingSurfaceNetwork::memory(d, t, NodeId(0), NodeId(1), 16384),
            Some([0, 0]) => SixRingSurfaceNetwork::separated(d, t, NodeId(0), NodeId(1), 16384),
            Some(w) => SixRingSurfaceNetwork::with_window(d, t, w, NodeId(0), NodeId(1), 16384),
        }
        .unwrap();
        let local = CubicalRecordModel::new(&net, 1 << 20).unwrap();
        let raw = net.ideal_relations(1 << 27).unwrap();
        let full = raw.extended(local.closure(), 1 << 27).unwrap();
        assert_eq!(full.observables(), local.observables());
        let width = local.observables().len();
        let mut complete = Vec::new();
        for r in full.checks() {
            assert!(insert(&mut complete, row(r, width)));
        }
        let mut derived = Vec::new();
        let state = initial(&net);
        for check in local.checks() {
            assert!(in_span(&complete, row(check, width)));
            insert(&mut derived, row(check, width));
            let mut operator = PauliProduct::new(vec![]).unwrap();
            let mut phase = check.negative;
            for &i in &check.terms {
                let (sign, p) = operator
                    .multiply_commuting(&local.observables()[i])
                    .unwrap();
                phase ^= sign;
                operator = p;
            }
            assert_eq!(
                state.expectation(&operator).unwrap(),
                if phase { -1 } else { 1 }
            );
        }
        assert_eq!(
            derived.len(),
            complete.len(),
            "incomplete local check space d{d} t{t} seam{seam:?}: local{} full{}",
            derived.len(),
            complete.len()
        );
        let proof = net
            .certify(BoundaryCertificateLimits {
                max_relation_entries: 1 << 27,
                ..Default::default()
            })
            .unwrap();
        let targets: Vec<_> = proof
            .logical_relations
            .iter()
            .map(|(p, _)| p.clone())
            .collect();
        let graph = local
            .graph_from_relations(&full, &targets, GraphLimits::default())
            .unwrap();
        let distances: Vec<_> = (0..targets.len())
            .map(|i| graph.minimum_logical_distance(i, 1 << 28).unwrap().distance)
            .collect();
        eprintln!(
            "physical surface d{d} t{t} seam{seam:?}, raw={} closure={} checks={} distances={distances:?}",
            local.raw_records(),
            local.closure().len(),
            local.checks().len()
        );
        if seam == Some([2, 3]) {
            assert_eq!(*distances.iter().min().unwrap(), 1);
        } else {
            assert!(distances.iter().all(|&x| x >= d));
            assert_eq!(*distances.iter().min().unwrap(), d);
        }
        let mut trajectory = initial(&net);
        let records: Vec<_> = local
            .observables()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                trajectory
                    .measure(p, (i * 13 + 7) % 19 < 8)
                    .unwrap()
                    .negative
            })
            .collect();
        assert!(local.syndrome(&records).unwrap().iter().all(|&b| !b));
        assert!(records[local.raw_records()..].iter().any(|&b| b));
        let target_relations: Vec<_> = targets
            .iter()
            .map(|p| full.correlation(p).unwrap().unwrap())
            .collect();
        for q in 0..local.raw_records() {
            let mut noisy = records.clone();
            noisy[q] ^= true;
            let mut fault = vec![false; local.raw_records()];
            fault[q] = true;
            assert_eq!(
                local.syndrome(&noisy).unwrap(),
                graph.syndrome(&fault).unwrap()
            );
            if d == 3 && seam != Some([2, 3]) {
                // Recovery receives only actual observed detector parities.
                // The physical quantum output is an independent truth oracle.
                let decoded = graph
                    .decode_union_find(
                        &local.syndrome(&noisy).unwrap(),
                        &vec![false; local.raw_records()],
                        DecoderBudget::default(),
                    )
                    .unwrap();
                assert_eq!(decoded.status, GraphStatus::Corrected);
                for (r, corrected) in decoded.correction.iter().enumerate() {
                    noisy[r] ^= corrected;
                }
                assert!(local.syndrome(&noisy).unwrap().iter().all(|&b| !b));
                for (target, relation) in targets.iter().zip(&target_relations) {
                    let predicted = relation
                        .terms
                        .iter()
                        .fold(relation.negative, |value, &r| value ^ noisy[r]);
                    let actual = trajectory.expectation(target).unwrap();
                    assert_ne!(actual, 0);
                    assert_eq!(predicted, actual < 0);
                }
            }
        }
        if seam.is_none() || seam.is_some_and(|w| w != [0, 0]) {
            let cuts = if seam.is_none() {
                local.memory_graph(GraphLimits::default()).unwrap()
            } else {
                local.surgery_graph(GraphLimits::default()).unwrap()
            };
            let mut h = Vec::new();
            for check in local.checks() {
                let restricted = ParityRelation {
                    terms: check
                        .terms
                        .iter()
                        .copied()
                        .filter(|&r| r < local.raw_records())
                        .collect(),
                    negative: false,
                };
                insert(&mut h, row(&restricted, local.raw_records()));
            }
            let mut labels = Vec::new();
            for bit in 0..targets.len() {
                let actual: Vec<_> = graph
                    .edges()
                    .iter()
                    .map(|e| e.logical >> bit & 1 != 0)
                    .collect();
                let selected = (1..(1u64 << cuts.logicals()))
                    .find(|&mask| {
                        let mut delta = vec![0; local.raw_records().div_ceil(64)];
                        for (q, (edge, &a)) in cuts.edges().iter().zip(&actual).enumerate() {
                            let p = (edge.logical & mask).count_ones() % 2 != 0;
                            if p ^ a {
                                delta[q / 64] ^= 1 << (q % 64);
                            }
                        }
                        in_span(&h, delta)
                    })
                    .expect("geometric cuts must span actual quantum Choi labels");
                labels.push(selected);
            }
            let mut label_basis = Vec::new();
            for mask in labels {
                assert!(insert(&mut label_basis, vec![mask]));
            }
            assert_eq!(label_basis.len(), cuts.logicals());
        }
    }
}
#[test]
fn larger_actual_surface_memory_has_growing_raw_record_distance_without_dense_elimination() {
    for d in [3, 5, 7, 9] {
        let net = SixRingSurfaceNetwork::memory(d, d, NodeId(0), NodeId(1), 16384).unwrap();
        let local = CubicalRecordModel::new(&net, 1 << 20).unwrap();
        let graph = local.memory_graph(GraphLimits::default()).unwrap();
        assert_eq!(graph.edges().len(), local.raw_records());
        for logical in 0..2 {
            let distance = graph.minimum_logical_distance(logical, 1 << 29).unwrap();
            assert_eq!(distance.distance, d, "d{d} logical{logical}");
            let mut fault = vec![false; graph.edges().len()];
            for e in distance.edges {
                fault[e] ^= true;
            }
            assert!(graph.syndrome(&fault).unwrap().iter().all(|&b| !b));
            assert_ne!(graph.logical_flips(&fault).unwrap() & (1 << logical), 0);
        }
        assert!(CubicalRecordModel::new(&net, 1).is_err());
    }
}

#[test]
fn growing_physical_merge_cuts_and_short_window_controls() {
    for d in [3, 5, 7, 9] {
        for duration in [1, d] {
            let net = SixRingSurfaceNetwork::with_window(
                d,
                d + 2,
                [1, 1 + duration],
                NodeId(0),
                NodeId(1),
                16384,
            )
            .unwrap();
            let local = CubicalRecordModel::new(&net, 1 << 20).unwrap();
            let graph = local.surgery_graph(GraphLimits::default()).unwrap();
            let distances: Vec<_> = (0..4)
                .map(|logical| {
                    let witness = graph.minimum_logical_distance(logical, 1 << 29).unwrap();
                    let mut fault = vec![false; graph.edges().len()];
                    for &edge in &witness.edges {
                        fault[edge] ^= true;
                    }
                    assert!(graph.syndrome(&fault).unwrap().iter().all(|&b| !b));
                    assert_ne!(graph.logical_flips(&fault).unwrap() & (1 << logical), 0);
                    witness.distance
                })
                .collect();
            eprintln!("geometric surgery d{d}, duration{duration}, distances{distances:?}");
            assert_eq!(*distances.iter().min().unwrap(), duration);
            if duration == d {
                assert_eq!(distances, vec![d; 4]);
            }
        }
    }
}
