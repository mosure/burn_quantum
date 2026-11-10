//! Physical fusion-record distance is independently checked against actual
//! resource-state measurements, including explicitly measured perfect end checks.
use burn_quantum::photonic::{
    NodeId,
    qec::{ErasureDecode, fusion_surgery::*, threshold::*},
    stabilizer::*,
};
fn network(width: usize, depth: usize, window: Option<[usize; 2]>) -> SixRingBoundaryNetwork {
    match window {
        None => SixRingBoundaryNetwork::memory(width, depth, NodeId(0), NodeId(1), 4096),
        Some([0, 0]) => SixRingBoundaryNetwork::separated(width, depth, NodeId(0), NodeId(1), 4096),
        Some(w) => {
            SixRingBoundaryNetwork::parity_merge(width, depth, w, NodeId(0), NodeId(1), 4096)
        }
    }
    .unwrap()
}
fn initial(net: &SixRingBoundaryNetwork) -> Tableau {
    let mut state = Tableau::zero(
        net.patch().program().num_photons(),
        TableauLimits::default(),
    )
    .unwrap();
    for cell in 0..net.resource_centers().len() {
        for q in 0..6 {
            state.h(6 * cell + q).unwrap();
        }
        for q in 0..6 {
            state.cz(6 * cell + q, 6 * cell + (q + 1) % 6).unwrap();
        }
    }
    state
}
#[test]
fn raw_and_perfectly_closed_mixed_boundary_networks_retain_single_fusion_bottlenecks() {
    for (width, depth, window) in [
        (3, 3, None),
        (5, 3, None),
        (7, 3, None),
        (3, 10, Some([0, 0])),
        (3, 11, Some([1, 4])),
    ] {
        let net = network(width, depth, window);
        let relations = net.ideal_relations(1 << 24).unwrap();
        let proof = net.certify(Default::default()).unwrap();
        let targets: Vec<_> = proof
            .logical_relations
            .iter()
            .map(|(p, _)| p.clone())
            .collect();
        let original = relations.observables().len();
        let fusion_count = 2 * net.patch().program().num_fusions();
        let raw = RecordCode::new(
            &relations,
            &targets,
            (0..original).collect(),
            Default::default(),
        )
        .unwrap();
        for logical in 0..targets.len() {
            let witness = raw.minimum_distance(logical, 1, original).unwrap();
            assert_eq!(witness.distance, 1);
            assert!(witness.records[0] < fusion_count);
        }
        let closure: Vec<_> = proof
            .codes
            .iter()
            .flat_map(|c| c.x_checks.iter().chain(&c.z_checks).map(|(p, _)| p.clone()))
            .collect();
        let extended = relations.extended(&closure, 1 << 25).unwrap();
        let code = RecordCode::new(
            &extended,
            &targets,
            (0..original).collect(),
            Default::default(),
        )
        .unwrap();
        let witnesses: Vec<_> = (0..targets.len())
            .filter_map(
                |logical| match code.minimum_distance(logical, 1, original) {
                    Ok(w) => Some((logical, w)),
                    Err(burn_quantum::Error::ResourceLimit {
                        resource: "record distance search weight",
                        ..
                    }) => None,
                    Err(e) => panic!("unexpected distance failure {e}"),
                },
            )
            .collect();
        assert!(
            !witnesses.is_empty(),
            "expected documented physical bottleneck w{width} t{depth}"
        );
        let mut state = initial(&net);
        let measured: Vec<_> = extended
            .observables()
            .iter()
            .enumerate()
            .map(|(i, p)| state.measure(p, (i * 17 + 3) % 11 < 5).unwrap().negative)
            .collect();
        assert!(code.syndrome(&measured).unwrap().iter().all(|&v| !v));
        // Closure signs are actual measured outcomes, not substituted +1 values.
        assert!(measured[original..].iter().any(|&v| v));
        for (logical, witness) in witnesses {
            assert!(witness.records[0] < fusion_count);
            let mut wrong = measured.clone();
            for &r in &witness.records {
                wrong[r] ^= true;
            }
            assert!(code.syndrome(&wrong).unwrap().iter().all(|&v| !v));
            let relation = &code.logical_relations()[logical];
            let predicted = relation
                .terms
                .iter()
                .fold(relation.negative, |v, &r| v ^ wrong[r]);
            let actual = state.expectation(&targets[logical]).unwrap();
            assert_ne!(actual, 0);
            assert_ne!(
                predicted,
                actual < 0,
                "an undetected physical fusion-readout flip changes the inferred Choi sign"
            );
            let mut erased = vec![false; original];
            for &r in &witness.records {
                erased[r] = true;
            }
            assert!(
                matches!(code.parity_code().decode_erasures(&vec![false;code.checks().len()],&erased).unwrap(),ErasureDecode::Ambiguous{logicals}if logicals.contains(&logical))
            );
        }
    }
}
#[test]
fn signed_graph_conversion_matches_observed_faults_and_dense_erasure_oracle() {
    for width in [3, 5, 7] {
        let net = network(width, 3, None);
        let relations = net.ideal_relations(1 << 24).unwrap();
        let proof = net.certify(Default::default()).unwrap();
        let targets: Vec<_> = proof
            .logical_relations
            .iter()
            .map(|(p, _)| p.clone())
            .collect();
        let m = relations.observables().len();
        let code =
            RecordCode::new(&relations, &targets, (0..m).collect(), Default::default()).unwrap();
        let converted = code
            .try_graph(Default::default(), GraphLimits::default())
            .unwrap();
        let mut state = initial(&net);
        let clean: Vec<_> = relations
            .observables()
            .iter()
            .enumerate()
            .map(|(i, p)| state.measure(p, i % 3 == 1).unwrap().negative)
            .collect();
        assert!(converted.syndrome(&clean).unwrap().iter().all(|&v| !v));
        for q in 0..m {
            let mut raw = clean.clone();
            raw[q] ^= true;
            let mut error = vec![false; m];
            error[q] = true;
            assert_eq!(
                converted.syndrome(&raw).unwrap(),
                converted.graph().syndrome(&error).unwrap()
            );
        }
        for seed in 0..64 {
            let erased: Vec<_> = (0..m).map(|q| (q * 31 + seed * 23) % 101 < seed).collect();
            let sparse = converted
                .graph()
                .analyze_erasures(&erased, Default::default())
                .unwrap();
            let dense = code
                .parity_code()
                .decode_erasures(&vec![false; code.checks().len()], &erased)
                .unwrap();
            let mask = match dense {
                ErasureDecode::Decoded { .. } => 0,
                ErasureDecode::Ambiguous { logicals } => {
                    logicals.into_iter().fold(0, |v, i| v | (1 << i))
                }
                ErasureDecode::Inconsistent => panic!("homogeneous erasure system"),
            };
            assert_eq!(sparse.ambiguous_mask(), mask);
        }
        if code.checks().len() > 1 {
            assert!(
                code.try_graph(
                    RecordCodeLimits {
                        max_row_operations: 0,
                        ..Default::default()
                    },
                    GraphLimits::default()
                )
                .is_err()
            );
        }
        assert!(code.minimum_distance(0, 1, 0).is_err());
    }
}
#[test]
fn explicit_closure_rederivation_validates_operators_and_budgets() {
    let net = network(3, 3, None);
    let relations = net.ideal_relations(1 << 24).unwrap();
    let measured = relations.observables()[0].terms()[0];
    let incompatible = PauliProduct::new(vec![(
        measured.0,
        if measured.1 == Pauli::X {
            Pauli::Z
        } else {
            Pauli::X
        },
    )])
    .unwrap();
    assert!(relations.extended(&[incompatible], 1 << 24).is_err());
    assert!(relations.extended(&[], 0).is_err());
    let proof = net.certify(Default::default()).unwrap();
    let targets: Vec<_> = proof
        .logical_relations
        .iter()
        .map(|(p, _)| p.clone())
        .collect();
    assert!(RecordCode::new(&relations, &targets, vec![0, 0], Default::default()).is_err());
    assert!(
        RecordCode::new(
            &relations,
            &targets,
            vec![relations.observables().len()],
            Default::default()
        )
        .is_err()
    );
    assert!(
        RecordCode::new(
            &relations,
            &targets,
            vec![0],
            RecordCodeLimits {
                max_records: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}
