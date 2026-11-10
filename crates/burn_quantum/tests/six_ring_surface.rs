//! Color-preserving six-ring surface geometry, explicit local-H frames and physical channels.
use burn_quantum::photonic::{
    NodeId,
    qec::fusion_surgery::*,
    stabilizer::{Pauli, PauliProduct},
};
fn verify(proof: &BoundaryCertificate, distance: usize) {
    for code in &proof.codes {
        assert_eq!(code.photons.len(), distance * distance);
        assert_eq!(
            code.x_checks.len() + code.z_checks.len(),
            distance * distance - 1
        );
        assert_eq!(
            (code.x_distance.distance, code.z_distance.distance),
            (distance, distance)
        );
        assert!(code.logical_x.anticommutes(&code.logical_z));
        for (witness, opposite) in [
            (&code.x_distance, &code.logical_z),
            (&code.z_distance, &code.logical_x),
        ] {
            assert_eq!(witness.operator.terms().len(), distance);
            assert!(witness.operator.anticommutes(opposite));
            assert!(
                code.x_checks
                    .iter()
                    .chain(&code.z_checks)
                    .all(|(p, _)| !witness.operator.anticommutes(p))
            );
        }
        for (checks, axis) in [(&code.x_checks, Pauli::X), (&code.z_checks, Pauli::Z)] {
            for (p, _) in checks {
                assert!(p.terms().iter().all(|&(q, p)| p
                    == if code.hadamard_frame.contains(&q) {
                        if axis == Pauli::X { Pauli::Z } else { Pauli::X }
                    } else {
                        axis
                    }));
            }
        }
    }
}
#[test]
fn color_preserving_memory_has_growing_framed_surface_code_distance() {
    for d in [3, 5] {
        let network = SixRingSurfaceNetwork::memory(d, d, NodeId(0), NodeId(1), 4096).unwrap();
        let proof = network.certify(Default::default()).unwrap();
        verify(&proof, d);
        assert_eq!(proof.transmitted_logicals, 1);
        assert_eq!(proof.logical_relations.len(), 2);
        for p in proof.logical_observables {
            assert_eq!(p.terms().len(), 2);
        }
    }
}
#[test]
fn real_color_preserving_seam_projects_joint_parity_without_measuring_each_block() {
    for merge in [false, true] {
        let network = if merge {
            SixRingSurfaceNetwork::parity_merge(3, 1, 3, 1, NodeId(0), NodeId(1), 4096)
        } else {
            SixRingSurfaceNetwork::separated(3, 5, NodeId(0), NodeId(1), 4096)
        }
        .unwrap();
        let proof = network.certify(Default::default()).unwrap();
        verify(&proof, 3);
        assert_eq!(proof.codes.len(), 4);
        assert_eq!(proof.transmitted_logicals, if merge { 1 } else { 2 });
        assert_eq!(proof.logical_relations.len(), 4);
        if merge {
            let relations = network.ideal_relations(1 << 24).unwrap();
            for offset in [0, 2] {
                let mut joint = 0;
                for a in [Pauli::X, Pauli::Z] {
                    for b in [Pauli::X, Pauli::Z] {
                        let op = |i: usize, axis| {
                            if axis == Pauli::X {
                                &proof.codes[i].logical_x
                            } else {
                                &proof.codes[i].logical_z
                            }
                        };
                        let physical = PauliProduct::new(
                            op(offset, a)
                                .terms()
                                .iter()
                                .chain(op(offset + 1, b).terms())
                                .copied()
                                .collect(),
                        )
                        .unwrap();
                        joint += usize::from(relations.correlation(&physical).unwrap().is_some());
                    }
                }
                assert_eq!(
                    joint, 1,
                    "one joint parity, not independent single-block readouts"
                );
            }
        }
    }
}
#[test]
fn merge_duration_and_allocation_contracts_are_explicit() {
    assert!(SixRingSurfaceNetwork::parity_merge(3, 1, 2, 1, NodeId(0), NodeId(1), 4096).is_err());
    assert!(SixRingSurfaceNetwork::with_window(3, 5, [2, 3], NodeId(0), NodeId(1), 4096).is_ok());
    assert!(SixRingSurfaceNetwork::parity_merge(3, 0, 3, 1, NodeId(0), NodeId(1), 4096).is_err());
    assert!(SixRingSurfaceNetwork::parity_merge(3, 1, 3, 0, NodeId(0), NodeId(1), 4096).is_err());
    assert!(SixRingSurfaceNetwork::memory(2, 3, NodeId(0), NodeId(1), 4096).is_err());
    assert!(SixRingSurfaceNetwork::memory(3, 3, NodeId(0), NodeId(1), 161).is_err());
    let large =
        SixRingSurfaceNetwork::parity_merge(7, 1, 7, 1, NodeId(0), NodeId(1), 8192).unwrap();
    assert_eq!(large.patch().program().num_photons(), 5292);
    assert_eq!(large.merge_window(), Some([1, 8]));
    assert!(SixRingSurfaceNetwork::parity_merge(7, 1, 7, 1, NodeId(0), NodeId(1), 4096).is_err());
}

#[test]
fn pinned_stim_surface_geometry_signed_frames_and_every_trajectory_match() {
    use serde_json::Value;
    fn numbers(v: &Value) -> Vec<usize> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }
    fn pauli(v: &Value) -> PauliProduct {
        PauliProduct::new(
            v.as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    (
                        v[0].as_u64().unwrap() as usize,
                        match v[1].as_str().unwrap() {
                            "X" => Pauli::X,
                            "Y" => Pauli::Y,
                            "Z" => Pauli::Z,
                            _ => panic!("fixture Pauli"),
                        },
                    )
                })
                .collect(),
        )
        .unwrap()
    }
    fn add(basis: &mut Vec<(u32, u128)>, mut v: u128) -> bool {
        for &(pivot, row) in basis.iter() {
            if v & (1 << pivot) != 0 {
                v ^= row;
            }
        }
        if v == 0 {
            return false;
        }
        let pivot = v.trailing_zeros();
        let at = basis.partition_point(|(p, _)| *p < pivot);
        basis.insert(at, (pivot, v));
        true
    }
    let fixture: Value =
        serde_json::from_str(include_str!("stabilizer/surface-stim-1.15.0.json")).unwrap();
    assert_eq!(fixture["stim_version"], "1.15.0");
    assert_eq!(
        fixture["source_archive_sha256"],
        "95236006859d6754be99629d4fb44788e742e962ac8c59caad421ca088f7350e"
    );
    assert_eq!(
        fixture["generator"],
        "independent axis geometry and signed integer symplectic algebra with Stim sampling"
    );
    assert_eq!(
        fixture["record_order"],
        "fusion XX/ZZ pairs then terminal side measurements; output_relations then logical_relations then logical_product_relations physical masks appended"
    );
    assert_eq!(fixture["boundary_colors"], serde_json::json!({"x":1,"y":0}));
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    for (ci, c) in cases.iter().enumerate() {
        let d = c["distance_parameter"].as_u64().unwrap() as usize;
        let rounds = c["depth"].as_u64().unwrap() as usize;
        let window = c["merge_window"].as_array().map(|v| {
            [
                v[0].as_u64().unwrap() as usize,
                v[1].as_u64().unwrap() as usize,
            ]
        });
        assert_eq!(
            (d, rounds, window),
            [
                (3, 3, None),
                (5, 5, None),
                (3, 5, Some([0, 0])),
                (3, 5, Some([1, 4])),
                (3, 5, Some([2, 3]))
            ][ci]
        );
        assert_eq!(c["stim_seed"], 810 + ci);
        assert_eq!(c["transmitted_logicals"], if ci == 2 { 2 } else { 1 });
        assert_eq!(c["stim_shots"], 32);
        assert_eq!(
            c["stim_circuit_sha256"],
            [
                "f657e8725848154f9119e4a609b089832bcdd3f48d01444fd56e9e7128bcaa25",
                "e9693adc7066edab847f2d42c224ba1a0cbbda08cc009dcf6717c482a85789cb",
                "b33812b80e07ed5b9da1f81fcc001d7b07f8a7f4f86fed0e11981d183ec3437c",
                "c8e389396685582b0e2f110b686b4c72aaf4297bd0086e54e61199f98b357115",
                "fb3e14f09fe7f54065443cab79e51a3a32d6b3f6ba664a8e7b6ed411a6fd9f7c"
            ][ci]
        );
        let network = match window {
            None => SixRingSurfaceNetwork::memory(d, rounds, NodeId(0), NodeId(1), 4096),
            Some([0, 0]) => SixRingSurfaceNetwork::separated(d, rounds, NodeId(0), NodeId(1), 4096),
            Some(window) => {
                SixRingSurfaceNetwork::with_window(d, rounds, window, NodeId(0), NodeId(1), 4096)
            }
        }
        .unwrap();
        assert_eq!(
            serde_json::to_value(network.resource_centers()).unwrap(),
            c["cells"]
        );
        assert_eq!(
            serde_json::to_value(
                network
                    .patch()
                    .program()
                    .fusions()
                    .iter()
                    .map(|f| f.photons.map(|q| q.0))
                    .collect::<Vec<_>>()
            )
            .unwrap(),
            c["fusions"]
        );
        let terminal = c["terminal"].as_array().unwrap();
        assert_eq!(
            terminal.len(),
            network.patch().boundary_measurements().len()
        );
        for (p, v) in network.patch().boundary_measurements().iter().zip(terminal) {
            assert_eq!(*p, pauli(&serde_json::json!([v])));
        }
        let ends = c["end_blocks"].as_array().unwrap();
        assert_eq!(ends.len(), 2 * network.blocks());
        for (i, group) in ends.iter().enumerate() {
            assert_eq!(
                numbers(group),
                network
                    .end(i / network.blocks(), i % network.blocks())
                    .unwrap()
                    .iter()
                    .map(|q| q.0)
                    .collect::<Vec<_>>()
            );
        }
        let relations = network.ideal_relations(1 << 24).unwrap();
        let m = relations.observables().len();
        assert_eq!(m, c["measurement_count"].as_u64().unwrap() as usize);
        let outputs = c["output_relations"].as_array().unwrap();
        let logicals = c["logical_relations"].as_array().unwrap();
        let products = c["logical_product_relations"].as_array().unwrap();
        assert_eq!(products.len(), (1 << logicals.len()) - 1);
        assert_eq!(outputs.len(), network.patch().outputs().len());
        assert_eq!(logicals.len(), ends.len());
        let all_outputs = ends.iter().flat_map(numbers).collect::<Vec<_>>();
        let vector = |p: &PauliProduct| {
            p.terms().iter().fold(0u128, |v, &(q, p)| {
                let i = all_outputs.iter().position(|&r| r == q).unwrap();
                v | if p != Pauli::Z { 1 << (2 * i) } else { 0 }
                    | if p != Pauli::X { 1 << (2 * i + 1) } else { 0 }
            })
        };
        let mut full = vec![];
        for r in outputs {
            assert!(add(&mut full, vector(&pauli(&r["terms"]))));
        }
        let codes = c["end_codes"].as_array().unwrap();
        assert_eq!(codes.len(), ends.len());
        for (i, code) in codes.iter().enumerate() {
            let photons = numbers(&code["photons"]);
            assert_eq!(photons, numbers(&ends[i]));
            let h = numbers(&code["hadamards"]);
            assert_eq!(
                h,
                photons
                    .iter()
                    .filter(|q| network.hadamard_frame().contains(q))
                    .copied()
                    .collect::<Vec<_>>()
            );
            let mut checks = vec![];
            let mut local = vec![];
            for (key, axis) in [("css_x_checks", Pauli::X), ("css_z_checks", Pauli::Z)] {
                for row in code[key].as_array().unwrap() {
                    let p = PauliProduct::new(
                        numbers(row)
                            .into_iter()
                            .map(|q| {
                                assert!(photons.contains(&q));
                                (
                                    q,
                                    if h.contains(&q) {
                                        if axis == Pauli::X { Pauli::Z } else { Pauli::X }
                                    } else {
                                        axis
                                    },
                                )
                            })
                            .collect(),
                    )
                    .unwrap();
                    assert!(add(&mut local, vector(&p)));
                    assert!(!add(&mut full.clone(), vector(&p)));
                    checks.push(p);
                }
            }
            assert_eq!(local.len(), photons.len() - 1);
            assert_eq!(code["rank"], local.len());
            assert_eq!(code["logical_qubits"], 1);
            let x = pauli(&code["logical_x"]);
            let z = pauli(&code["logical_z"]);
            assert!(x.anticommutes(&z));
            assert!(
                checks
                    .iter()
                    .all(|p| !p.anticommutes(&x) && !p.anticommutes(&z))
            );
        }
        let records = c["stim_records"].as_array().unwrap();
        assert_eq!(records.len(), 32);
        let checks = c["checks"].as_array().unwrap();
        assert_eq!(checks.len(), relations.checks().len());
        for (i, r) in outputs.iter().chain(logicals).chain(products).enumerate() {
            let own = relations.correlation(&pauli(&r["terms"])).unwrap().unwrap();
            let independent = numbers(&r["measurement_indices"]);
            assert!(independent.iter().all(|&j| j < m));
            if i >= outputs.len() {
                // Reconstruct the entire encoded Pauli lift from the independently chosen
                // physical representatives, including i X_L Z_L for every encoded Y.
                let mut phase = 0u8;
                let mut physical = std::collections::BTreeMap::<usize, Pauli>::new();
                for &(index, axis) in pauli(&r["encoded_terms"]).terms() {
                    if axis == Pauli::Y {
                        phase = (phase + 1) % 4;
                    }
                    for key in match axis {
                        Pauli::X => vec!["logical_x"],
                        Pauli::Z => vec!["logical_z"],
                        Pauli::Y => vec!["logical_x", "logical_z"],
                    } {
                        for &(q, b) in pauli(&codes[index][key]).terms() {
                            if let Some(a) = physical.remove(&q) {
                                let (delta, p) = match (a, b) {
                                    (Pauli::X, Pauli::Y) => (1, Some(Pauli::Z)),
                                    (Pauli::Y, Pauli::Z) => (1, Some(Pauli::X)),
                                    (Pauli::Z, Pauli::X) => (1, Some(Pauli::Y)),
                                    (Pauli::Y, Pauli::X) => (3, Some(Pauli::Z)),
                                    (Pauli::Z, Pauli::Y) => (3, Some(Pauli::X)),
                                    (Pauli::X, Pauli::Z) => (3, Some(Pauli::Y)),
                                    _ => (0, None),
                                };
                                phase = (phase + delta) % 4;
                                if let Some(p) = p {
                                    physical.insert(q, p);
                                }
                            } else {
                                physical.insert(q, b);
                            }
                        }
                    }
                }
                assert!(phase == 0 || phase == 2);
                assert_eq!(r["lift_negative"].as_bool().unwrap(), phase == 2);
                assert_eq!(
                    PauliProduct::new(physical.into_iter().collect()).unwrap(),
                    pauli(&r["terms"])
                );
                assert_eq!(
                    r["encoded_negative"].as_bool().unwrap(),
                    r["negative"].as_bool().unwrap() ^ r["lift_negative"].as_bool().unwrap()
                );
            }
            for record in records {
                let bits = record.as_str().unwrap().as_bytes();
                assert_eq!(
                    bits.len(),
                    m + outputs.len() + logicals.len() + products.len()
                );
                assert!(bits.iter().all(|b| *b == b'0' || *b == b'1'));
                for (indices, sign) in [
                    (&own.terms, own.negative),
                    (&independent, r["negative"].as_bool().unwrap()),
                ] {
                    assert_eq!(
                        indices.iter().fold(sign, |v, &j| v ^ (bits[j] == b'1')),
                        bits[m + i] == b'1'
                    );
                }
            }
        }
        for record in records {
            let bits = record.as_str().unwrap().as_bytes();
            for check in relations.checks() {
                assert_eq!(
                    check
                        .terms
                        .iter()
                        .fold(false, |v, &j| v ^ (bits[j] == b'1')),
                    check.negative
                );
            }
        }
    }
}

#[test]
fn distance_seven_colored_surface_has_exact_physical_support_certificates() {
    let network = SixRingSurfaceNetwork::memory(7, 7, NodeId(0), NodeId(1), 4096).unwrap();
    let proof = network
        .certify(BoundaryCertificateLimits {
            max_relation_entries: 1 << 25,
            max_distance_candidates: 1 << 27,
            max_distance_weight: 7,
        })
        .unwrap();
    verify(&proof, 7);
    assert_eq!(proof.transmitted_logicals, 1);
}

#[test]
fn growing_color_preserving_merge_has_complete_signed_distance_five_and_seven_channels() {
    for d in [5, 7] {
        let network =
            SixRingSurfaceNetwork::parity_merge(d, 1, d, 1, NodeId(0), NodeId(1), 8192).unwrap();
        let proof = network
            .certify(BoundaryCertificateLimits {
                max_relation_entries: 1 << 27,
                max_distance_candidates: 1 << 27,
                max_distance_weight: 7,
            })
            .unwrap();
        verify(&proof, d);
        assert_eq!(proof.transmitted_logicals, 1);
        assert_eq!(proof.codes.len(), 4);
        assert_eq!(proof.logical_observables.len(), 4);
        let mut joint = [0; 2];
        for select in 1..16 {
            let mut value = 0u16;
            for (i, p) in proof.logical_observables.iter().enumerate() {
                if select & (1 << i) != 0 {
                    for &(q, p) in p.terms() {
                        if p != Pauli::Z {
                            value ^= 1 << (2 * q);
                        }
                        if p != Pauli::X {
                            value ^= 1 << (2 * q + 1);
                        }
                    }
                }
            }
            for (side, count) in joint.iter_mut().enumerate() {
                let local = (value >> (4 * side)) & 15;
                let other = (value >> (4 * (1 - side))) & 15;
                if other == 0 && local & 3 != 0 && local & 12 != 0 {
                    *count += 1;
                }
            }
        }
        assert_eq!(joint, [1, 1]);
    }
}

#[test]
fn physical_color_surface_keeps_all_standard_fusion_failures_and_route_losses() {
    use burn_quantum::photonic::{
        CompileOptions, Fabric, FabricLink, FabricNode, FusionOutcome, NodeKind, Parity,
        RailTransmission, qec::ParityInference, stabilizer::PauliNoise,
    };
    let network = SixRingSurfaceNetwork::memory(3, 3, NodeId(0), NodeId(1), 4096).unwrap();
    let proof = network.certify(Default::default()).unwrap();
    let targets = proof
        .codes
        .iter()
        .flat_map(|c| c.x_checks.iter().chain(&c.z_checks))
        .chain(&proof.logical_relations)
        .map(|(p, _)| p.clone())
        .collect::<Vec<_>>();
    let mut failures = 0;
    let mut losses = 0;
    let mut known = 0;
    for eta in [1., 0.8] {
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
        let plan = network
            .patch()
            .program()
            .compile(&fabric, CompileOptions::default())
            .unwrap();
        let shots = network
            .patch()
            .sample(&plan, 0, 64, 821, PauliNoise::NONE, &[], &targets, 1 << 24)
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
                    FusionOutcome::Failure { .. } => failures += 1,
                    FusionOutcome::Loss => {
                        assert!(eta < 1.);
                        losses += 1;
                    }
                    FusionOutcome::Success { .. } => {}
                    _ => panic!("standard analyzer"),
                }
            }
            for (inferred, &actual) in shot.inferred.iter().zip(&shot.expectations) {
                if let ParityInference::Known(parity) = inferred {
                    known += 1;
                    assert_eq!(actual, if *parity == Parity::Odd { -1 } else { 1 });
                }
            }
        }
    }
    assert!(failures > 0 && losses > 0 && known > 0);
}

#[test]
fn composed_css_and_logical_signed_relations_match_actual_measurement_branches() {
    use burn_quantum::photonic::stabilizer::Tableau;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    for merge in [false, true] {
        let network = if merge {
            SixRingSurfaceNetwork::parity_merge(3, 1, 3, 1, NodeId(0), NodeId(1), 4096)
        } else {
            SixRingSurfaceNetwork::memory(3, 3, NodeId(0), NodeId(1), 4096)
        }
        .unwrap();
        let proof = network.certify(Default::default()).unwrap();
        let mut prepared =
            Tableau::zero(network.patch().program().num_photons(), Default::default()).unwrap();
        for cell in 0..network.resource_centers().len() {
            for q in 0..6 {
                prepared.h(6 * cell + q).unwrap();
            }
            for q in 0..6 {
                prepared.cz(6 * cell + q, 6 * cell + (q + 1) % 6).unwrap();
            }
        }
        let observables = network.patch().observables().unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(901);
        let mut negative_values = 0;
        for _ in 0..16 {
            let mut state = prepared.clone();
            let records = observables
                .iter()
                .map(|p| state.measure(p, rng.random()).unwrap().negative)
                .collect::<Vec<_>>();
            for (p, relation) in proof
                .codes
                .iter()
                .flat_map(|c| c.x_checks.iter().chain(&c.z_checks))
                .chain(&proof.logical_relations)
            {
                let negative = relation
                    .terms
                    .iter()
                    .fold(relation.negative, |v, &j| v ^ records[j]);
                negative_values += usize::from(negative);
                assert_eq!(state.expectation(p).unwrap(), if negative { -1 } else { 1 });
            }
        }
        assert!(
            negative_values > 0,
            "signed check records must not be replaced by zeroes"
        );
    }
}
