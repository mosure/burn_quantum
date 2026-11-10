//! Independent signed six-ring boundary geometry and finite code-distance certificates.
use burn_quantum::photonic::{
    CompileOptions, Fabric, FabricLink, FabricNode, FusionOutcome, NodeId, NodeKind, Parity,
    RailTransmission,
    qec::{MeasurementRelations, ParityInference, PhotonLossBurst, fusion_surgery::*},
    stabilizer::{Pauli, PauliNoise, PauliProduct, Tableau},
};
use serde_json::Value;
fn make(width: usize, depth: usize, window: Option<[usize; 2]>) -> SixRingBoundaryNetwork {
    match window {
        None => SixRingBoundaryNetwork::memory(width, depth, NodeId(0), NodeId(1), 4096),
        Some([0, 0]) => SixRingBoundaryNetwork::separated(width, depth, NodeId(0), NodeId(1), 4096),
        Some(window) => {
            SixRingBoundaryNetwork::parity_merge(width, depth, window, NodeId(0), NodeId(1), 4096)
        }
    }
    .unwrap()
}
fn terms(value: &Value) -> PauliProduct {
    PauliProduct::new(
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                (
                    v[0].as_u64().unwrap() as usize,
                    match v[1].as_str().unwrap() {
                        "X" => Pauli::X,
                        "Y" => Pauli::Y,
                        "Z" => Pauli::Z,
                        _ => panic!("unknown fixture Pauli"),
                    },
                )
            })
            .collect(),
    )
    .unwrap()
}
fn numbers(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}
fn validate_fixture_code_contract(c: &Value) {
    let all_outputs = c["ends"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(numbers)
        .collect::<Vec<_>>();
    let vector = |p: &PauliProduct| {
        let mut v = vec![false; 2 * all_outputs.len()];
        for &(q, p) in p.terms() {
            let i = all_outputs.iter().position(|&r| r == q).unwrap();
            v[2 * i] = p != Pauli::Z;
            v[2 * i + 1] = p != Pauli::X;
        }
        v
    };
    fn add(basis: &mut Vec<(usize, Vec<bool>)>, mut value: Vec<bool>) -> bool {
        for (pivot, row) in basis.iter() {
            if value[*pivot] {
                for (a, b) in value.iter_mut().zip(row) {
                    *a ^= *b;
                }
            }
        }
        let Some(pivot) = value.iter().position(|&b| b) else {
            return false;
        };
        let at = basis.partition_point(|(p, _)| *p < pivot);
        basis.insert(at, (pivot, value));
        true
    }
    let mut full = vec![];
    for r in c["output_relations"].as_array().unwrap() {
        assert!(add(&mut full, vector(&terms(&r["terms"]))));
    }
    assert_eq!(full.len(), all_outputs.len());
    for (key, groups) in [
        ("end_codes", 2),
        (
            "patch_codes",
            if c["merge_window"].is_null() { 2 } else { 4 },
        ),
    ] {
        let codes = c[key].as_array().unwrap();
        assert_eq!(codes.len(), groups);
        for (i, code) in codes.iter().enumerate() {
            let expected = if groups == 2 {
                numbers(&c["ends"][i])
            } else {
                numbers(&c["ends"][i / 2])
                    .into_iter()
                    .filter(|&q| {
                        (c["cells"][q / 6][0].as_u64().unwrap() <= c["width"].as_u64().unwrap())
                            == (i % 2 == 0)
                    })
                    .collect()
            };
            assert_eq!(numbers(&code["outputs"]), expected);
            let mut checks = vec![];
            let mut local = vec![];
            for (name, axis) in [("x_checks", Pauli::X), ("z_checks", Pauli::Z)] {
                for row in code[name].as_array().unwrap() {
                    let p = PauliProduct::new(
                        numbers(row)
                            .into_iter()
                            .map(|q| {
                                assert!(expected.contains(&q));
                                (q, axis)
                            })
                            .collect(),
                    )
                    .unwrap();
                    assert!(add(&mut local, vector(&p)));
                    assert!(!add(&mut full.clone(), vector(&p)));
                    checks.push(p);
                }
            }
            assert_eq!(local.len(), code["rank"].as_u64().unwrap() as usize);
            let k = expected.len() - local.len();
            assert_eq!(k, code["logical_qubits"].as_u64().unwrap() as usize);
            let reps = code["logical_representatives"].as_array().unwrap();
            if k <= 1 {
                assert_eq!(reps.len(), 2 * k);
            }
            if k == 1 {
                let x = terms(&reps[0]);
                let z = terms(&reps[1]);
                assert!(x.anticommutes(&z));
                for p in [&x, &z] {
                    assert!(p.terms().iter().all(|(q, _)| expected.contains(q)));
                    assert!(checks.iter().all(|r| !r.anticommutes(p)));
                }
            }
        }
    }
}
#[test]
fn exact_end_distances_grow_and_time_seam_projects_one_encoded_parity() {
    for (width, count, dx, dz) in [(3, 8, 2, 2), (5, 21, 3, 4), (7, 40, 4, 6)] {
        let network = make(width, 3, None);
        let proof = network.certify(Default::default()).unwrap();
        assert_eq!(proof.transmitted_logicals, 1);
        assert_eq!(proof.logical_relations.len(), 2);
        for code in &proof.codes {
            assert_eq!(code.photons.len(), count);
            assert_eq!(code.x_checks.len() + code.z_checks.len(), count - 1);
            // The opposite time end exchanges X/Z through a known logical Hadamard.
            assert_eq!(code.x_distance.distance.min(code.z_distance.distance), dx);
            assert_eq!(code.x_distance.distance.max(code.z_distance.distance), dz);
            assert!(code.logical_x.anticommutes(&code.logical_z));
            for (distance, pauli, opposite) in [
                (&code.x_distance, Pauli::X, &code.logical_z),
                (&code.z_distance, Pauli::Z, &code.logical_x),
            ] {
                let witness =
                    PauliProduct::new(distance.photons.iter().map(|&q| (q, pauli)).collect())
                        .unwrap();
                assert_eq!(witness.terms().len(), distance.distance);
                assert!(witness.anticommutes(opposite));
                assert!(
                    code.x_checks
                        .iter()
                        .chain(&code.z_checks)
                        .all(|(p, _)| !witness.anticommutes(p))
                );
            }
        }
    }
    for (window, expected) in [([0, 0], 2), ([1, 4], 1)] {
        let proof = make(3, 10, Some(window))
            .certify(Default::default())
            .unwrap();
        assert_eq!(proof.transmitted_logicals, expected);
        assert_eq!(proof.codes.len(), 4);
        assert_eq!(proof.logical_relations.len(), 4);
        let network = make(3, 10, Some(window));
        let relations = network.ideal_relations(1 << 24).unwrap();
        let patterns = if expected == 1 {
            vec!["XXII", "IIZZ", "XIZI", "ZZXX"]
        } else {
            vec!["XIZI", "ZIXI", "IXIZ", "IZIX"]
        };
        for pattern in patterns {
            let mut terms = vec![];
            for (i, axis) in pattern.bytes().enumerate() {
                let code = &proof.codes[i];
                match axis {
                    b'X' => terms.extend(code.logical_x.terms()),
                    b'Z' => terms.extend(code.logical_z.terms()),
                    _ => {}
                }
            }
            assert!(
                relations
                    .correlation(&PauliProduct::new(terms).unwrap())
                    .unwrap()
                    .is_some(),
                "missing {pattern}, window {window:?}"
            );
        }
    }
}
#[test]
fn independent_stim_geometry_and_all_signed_output_records_match() {
    let fixture: Value =
        serde_json::from_str(include_str!("stabilizer/boundary-stim-1.15.0.json")).unwrap();
    assert_eq!(fixture["stim_version"], "1.15.0");
    assert_eq!(
        fixture["generator"],
        "independent integer symplectic algebra and Stim circuit sampling"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for (case_index, c) in cases.iter().enumerate() {
        let width = c["width"].as_u64().unwrap() as usize;
        let depth = c["depth"].as_u64().unwrap() as usize;
        let window = c["merge_window"].as_array().map(|v| {
            [
                v[0].as_u64().unwrap() as usize,
                v[1].as_u64().unwrap() as usize,
            ]
        });
        assert_eq!(
            (width, depth, window),
            [
                (3, 3, None),
                (5, 3, None),
                (7, 3, None),
                (3, 10, Some([0, 0])),
                (3, 10, Some([1, 4])),
                (3, 10, Some([2, 4])),
                (3, 11, Some([1, 4])),
                (5, 13, Some([1, 6]))
            ][case_index]
        );
        assert_eq!(c["stim_shots"], 32);
        let hash = c["stim_circuit_sha256"].as_str().unwrap();
        assert_eq!(
            hash,
            [
                "25ce0dad8e1238a244a67f025e5adbce9db9d3d2ad621a343721f349515a95eb",
                "1b07848a368ec19c7272193fa5fc248762ac5dfd52aebf60dbc71ef35f5a325e",
                "0d49f8dad263769bb5b382fc03651d0f4773b8feb441580faa22f9b72c9f4308",
                "c6d81ff619624148a6db50e9abead8a0456d5a7a2c7e5d9ea4bbb77b56ebd371",
                "6ea743e1d0f93fa7f4db9da5e4f29a0bf496d5e48878e4a9726214145256f9c4",
                "95f846b6797f4a07543b30cc202d9a580aeef94c1bf1c23894892693842190a9",
                "530cb933fd08b4ff3661295ebf3b84f2cd62fcf2909e77fb64b7575640a8cd69",
                "a61941ad04ac6f92a2dd570cf26335791ee4cc94466750d385b8ad78fa1f83d6",
            ][case_index]
        );
        assert_eq!(
            c["stim_seed"],
            [613, 615, 617, 630, 631, 632, 640, 641][case_index]
        );
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|b| b.is_ascii_hexdigit()));
        validate_fixture_code_contract(c);
        if window == Some([2, 4]) {
            assert_eq!(c["transmitted_logicals"], 0);
            assert!(
                SixRingBoundaryNetwork::parity_merge(
                    width,
                    depth,
                    [2, 4],
                    NodeId(0),
                    NodeId(1),
                    4096
                )
                .is_err()
            );
            // Reconstruct the rejected timing graph independently from the pinned physical
            // fixture and verify that both end states are pure: no Bell information survives.
            let mut initial =
                Tableau::zero(6 * c["cells"].as_array().unwrap().len(), Default::default())
                    .unwrap();
            for cell in 0..c["cells"].as_array().unwrap().len() {
                for q in 0..6 {
                    initial.h(6 * cell + q).unwrap();
                }
                for q in 0..6 {
                    initial.cz(6 * cell + q, 6 * cell + (q + 1) % 6).unwrap();
                }
            }
            let mut observables = vec![];
            for pair in c["fusions"].as_array().unwrap() {
                for axis in [Pauli::X, Pauli::Z] {
                    observables.push(
                        PauliProduct::new(numbers(pair).into_iter().map(|q| (q, axis)).collect())
                            .unwrap(),
                    );
                }
            }
            for term in c["terminal"].as_array().unwrap() {
                observables.push(terms(&serde_json::json!([term])));
            }
            let relations = MeasurementRelations::derive(&initial, &observables, 1 << 24).unwrap();
            for end in c["ends"].as_array().unwrap() {
                let outputs = numbers(end);
                assert_eq!(
                    relations.output_relations(&outputs, 1 << 24).unwrap().len(),
                    outputs.len()
                );
            }
            let m = observables.len();
            let outputs = c["output_relations"].as_array().unwrap();
            assert_eq!(c["stim_records"].as_array().unwrap().len(), 32);
            for (i, r) in outputs.iter().enumerate() {
                let own = relations.correlation(&terms(&r["terms"])).unwrap().unwrap();
                for record in c["stim_records"].as_array().unwrap() {
                    let bits = record.as_str().unwrap().as_bytes();
                    assert_eq!(bits.len(), m + outputs.len());
                    assert_eq!(
                        own.terms
                            .iter()
                            .fold(own.negative, |p, &j| p ^ (bits[j] == b'1')),
                        bits[m + i] == b'1'
                    );
                }
            }
            continue;
        }
        let network = make(width, depth, window);
        assert_eq!(
            serde_json::to_value(network.resource_centers()).unwrap(),
            c["cells"]
        );
        let fusions = network
            .patch()
            .program()
            .fusions()
            .iter()
            .map(|f| f.photons.map(|p| p.0))
            .collect::<Vec<_>>();
        assert_eq!(serde_json::to_value(&fusions).unwrap(), c["fusions"]);
        let terminal = c["terminal"].as_array().unwrap();
        assert_eq!(
            terminal.len(),
            network.patch().boundary_measurements().len()
        );
        for (p, v) in network.patch().boundary_measurements().iter().zip(terminal) {
            assert_eq!(*p, terms(&serde_json::json!([v])));
        }
        for end in 0..2 {
            let mut expected = numbers(&c["ends"][end]);
            expected.sort_unstable();
            let mut actual = (0..network.blocks())
                .flat_map(|b| network.end(end, b).unwrap().iter().map(|p| p.0))
                .collect::<Vec<_>>();
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
        let relations = network.ideal_relations(1 << 24).unwrap();
        let m = relations.observables().len();
        assert_eq!(m, c["measurement_count"].as_u64().unwrap() as usize);
        let outputs = c["output_relations"].as_array().unwrap();
        assert_eq!(outputs.len(), network.patch().outputs().len());
        let checks = c["checks"].as_array().unwrap();
        assert_eq!(checks.len(), relations.checks().len());
        let records = c["stim_records"].as_array().unwrap();
        assert_eq!(records.len(), 32);
        // Solve each physical target once, then validate its signed parity on every independent trajectory.
        for (i, r) in outputs.iter().enumerate() {
            let target = terms(&r["terms"]);
            let ours = relations.correlation(&target).unwrap().unwrap();
            let independent = numbers(&r["measurement_indices"]);
            assert!(independent.iter().all(|&j| j < m));
            for record in records {
                let bits = record.as_str().unwrap().as_bytes();
                assert_eq!(bits.len(), m + outputs.len());
                assert!(bits.iter().all(|&b| b == b'0' || b == b'1'));
                let parity = |indices: &[usize], negative: bool| {
                    indices.iter().fold(negative, |p, &j| p ^ (bits[j] == b'1'))
                };
                let observed = bits[m + i] == b'1';
                assert_eq!(
                    parity(&independent, r["negative"].as_bool().unwrap()),
                    observed
                );
                assert_eq!(parity(&ours.terms, ours.negative), observed);
            }
            // One full inference exercises check elimination rather than merely the direct relation.
            let record = records[0].as_str().unwrap().as_bytes();
            let observed = record[m + i] == b'1';
            assert_eq!(
                relations
                    .infer(
                        &record[..m]
                            .iter()
                            .map(|&b| Some(b == b'1'))
                            .collect::<Vec<_>>(),
                        &target
                    )
                    .unwrap(),
                ParityInference::Known(if observed { Parity::Odd } else { Parity::Even })
            );
        }
    }
}
#[test]
fn invalid_timing_geometry_and_exhausted_proof_budgets_do_not_claim_a_code() {
    for window in [[0, 4], [2, 4], [1, 3], [5, 4], [1, 10]] {
        assert!(
            SixRingBoundaryNetwork::parity_merge(3, 10, window, NodeId(0), NodeId(1), 4096)
                .is_err()
        );
    }
    for (width, depth, budget) in [(2, 3, 4096), (3, 0, 4096), (3, 3, 95), (31, 63, 4096)] {
        assert!(
            SixRingBoundaryNetwork::memory(width, depth, NodeId(0), NodeId(1), budget).is_err()
        );
    }
    let network = make(3, 3, None);
    for limits in [
        BoundaryCertificateLimits {
            max_relation_entries: 1,
            ..Default::default()
        },
        BoundaryCertificateLimits {
            max_distance_candidates: 1,
            ..Default::default()
        },
        BoundaryCertificateLimits {
            max_distance_weight: 1,
            ..Default::default()
        },
    ] {
        assert!(network.certify(limits).is_err());
    }
}

#[test]
fn actual_boundary_network_sampling_retains_failed_fusions_and_output_loss() {
    let network = make(3, 3, None);
    let proof = network.certify(Default::default()).unwrap();
    let targets = proof
        .logical_relations
        .iter()
        .map(|(p, _)| p.clone())
        .collect::<Vec<_>>();
    let fabric = |eta| {
        Fabric::new(
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
        .unwrap()
    };
    let mut successes = 0;
    let mut failures = 0;
    for eta in [1., 0.] {
        let plan = network
            .patch()
            .program()
            .compile(&fabric(eta), CompileOptions::default())
            .unwrap();
        let shots = network
            .patch()
            .sample(&plan, 0, 64, 73, PauliNoise::NONE, &[], &targets, 1 << 24)
            .unwrap();
        assert_eq!(shots.len(), 64);
        for shot in shots {
            assert_eq!(
                shot.fusion_outcomes.len(),
                network.patch().program().fusions().len()
            );
            assert_eq!(
                shot.records.len(),
                network.patch().observables().unwrap().len()
            );
            for outcome in shot.fusion_outcomes {
                match outcome {
                    FusionOutcome::Success { .. } => successes += 1,
                    FusionOutcome::Failure { .. } => failures += 1,
                    FusionOutcome::Loss => assert_eq!(eta, 0.),
                    _ => panic!("unexpected physical standard analyzer outcome"),
                }
                if eta == 0. {
                    assert_eq!(outcome, FusionOutcome::Loss);
                }
            }
            for (inferred, &actual) in shot.inferred.iter().zip(&shot.expectations) {
                if let ParityInference::Known(parity) = inferred {
                    assert_eq!(actual, if *parity == Parity::Odd { -1 } else { 1 });
                }
            }
        }
    }
    assert!(successes > 0 && failures > 0);
    let plan = network
        .patch()
        .program()
        .compile(&fabric(1.), CompileOptions::default())
        .unwrap();
    let loss = PhotonLossBurst::new(network.patch().outputs().to_vec(), 1.).unwrap();
    let lost = network
        .patch()
        .sample(
            &plan,
            0,
            1,
            73,
            PauliNoise::NONE,
            &[loss],
            &targets,
            1 << 24,
        )
        .unwrap();
    assert!(lost[0].output_present.iter().all(|&b| !b));
    assert!(lost[0].expectations.iter().all(|&e| e == 0));
}

#[test]
fn growing_seam_preserves_the_complete_parity_channel() {
    for width in [5] {
        let network = make(width, width + 8, Some([1, width + 1]));
        let proof = network
            .certify(BoundaryCertificateLimits {
                max_relation_entries: 1 << 25,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(proof.transmitted_logicals, 1);
        assert_eq!(proof.codes.len(), 4);
        assert!(
            proof
                .codes
                .iter()
                .all(|c| c.x_distance.distance.min(c.z_distance.distance) >= width.div_ceil(2))
        );
        let relations = network.ideal_relations(1 << 25).unwrap();
        for pattern in ["XXII", "IIXX", "XIXI", "ZZZZ"] {
            let mut terms = vec![];
            for (i, axis) in pattern.bytes().enumerate() {
                match axis {
                    b'X' => terms.extend(proof.codes[i].logical_x.terms()),
                    b'Z' => terms.extend(proof.codes[i].logical_z.terms()),
                    _ => {}
                }
            }
            assert!(
                relations
                    .correlation(&PauliProduct::new(terms).unwrap())
                    .unwrap()
                    .is_some()
            );
        }
    }
}
