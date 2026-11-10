//! Independent code-space, exact branch and complete erasure-pattern encoded fusion checks.
use burn_quantum::photonic::{qec::*, stabilizer::*, *};
use std::collections::BTreeMap;
fn p(terms: &[(usize, Pauli)]) -> PauliProduct {
    PauliProduct::new(terms.to_vec()).unwrap()
}
fn fabric(eta: f64) -> Fabric {
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
            transmission: RailTransmission::uniform(eta).unwrap(),
        }],
    )
    .unwrap()
}
fn parity(negative: bool) -> Parity {
    if negative { Parity::Odd } else { Parity::Even }
}
fn encoded_bell(xx: bool, zz: bool) -> Tableau {
    let mut state = Tableau::zero(8, TableauLimits::default()).unwrap();
    state.h(0).unwrap();
    state.cnot(0, 4).unwrap();
    if zz {
        state.x(4).unwrap();
    }
    if xx {
        state.z(4).unwrap();
    }
    // Independently specified Clifford encoding isometry, not the graph transformation.
    for a in [0, 4] {
        state.h(a + 1).unwrap();
        state.cnot(a + 1, a).unwrap();
        state.cnot(a, a + 2).unwrap();
        state.cnot(a + 1, a + 3).unwrap();
    }
    state
}
fn branches(state: &Tableau, observable: &PauliProduct) -> Vec<(Tableau, bool, f64)> {
    let random = state.expectation(observable).unwrap() == 0;
    (0..if random { 2 } else { 1 })
        .map(|b| {
            let mut next = state.clone();
            let outcome = next.measure(observable, b == 1).unwrap();
            (next, outcome.negative, if random { 0.5 } else { 1. })
        })
        .collect()
}
fn ideal_records(state: Tableau, i: usize, record: u8, weight: f64, out: &mut BTreeMap<u8, f64>) {
    if i == 8 {
        *out.entry(record).or_default() += weight;
        return;
    }
    let q = i / 2;
    let pauli = if i.is_multiple_of(2) {
        Pauli::X
    } else {
        Pauli::Z
    };
    for (next, bit, w) in branches(&state, &p(&[(q, pauli), (q + 4, pauli)])) {
        ideal_records(next, i + 1, record | ((bit as u8) << i), weight * w, out);
    }
}
fn physical_records(
    state: Tableau,
    i: usize,
    records: [FusionOutcome; 4],
    weight: f64,
    out: &mut Vec<([FusionOutcome; 4], f64)>,
) {
    if i == 4 {
        out.push((records, weight));
        return;
    }
    for (next, odd, w) in branches(&state, &p(&[(i, Pauli::Z), (i + 4, Pauli::Z)])) {
        let observable = if odd {
            p(&[(i, Pauli::X), (i + 4, Pauli::X)])
        } else {
            p(&[(i, Pauli::Z)])
        };
        for (after, bit, v) in branches(&next, &observable) {
            let mut record = records;
            record[i] = if odd {
                FusionOutcome::Success { xx: parity(bit) }
            } else {
                FusionOutcome::Failure { bit }
            };
            physical_records(after, i + 1, record, weight * w * v, out);
        }
    }
}
#[test]
fn encoded_six_ring_has_the_declared_code_space_logical_graph_and_distance_two() {
    let graph = ResourceState::six_ring();
    let encoded = ShorEncodedResource::new(&graph).unwrap();
    assert_eq!(encoded.logical_qubits(), 6);
    assert_eq!(encoded.resource().qubits(), 24);
    let mut program = FusionProgram::new();
    let blocks = encoded.add_to(&mut program, NodeId(0)).unwrap();
    let state = StabilizerSimulator::default()
        .prepare(
            &program
                .compile(&fabric(1.), CompileOptions::default())
                .unwrap(),
        )
        .unwrap();
    for (q, block) in blocks.iter().enumerate() {
        let stabilizers = block.stabilizers();
        for s in &stabilizers {
            assert_eq!(state.expectation(s).unwrap(), 1);
            assert!(!s.anticommutes(&block.logical_x()));
            assert!(!s.anticommutes(&block.logical_z()));
        }
        assert!(block.logical_x().anticommutes(&block.logical_z()));
        assert_eq!(block.logical_x().terms().len(), 2);
        assert_eq!(block.logical_z().terms().len(), 2);
        for photon in block.photons() {
            for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                assert!(
                    stabilizers
                        .iter()
                        .any(|s| s.anticommutes(&p(&[(photon.0, pauli)])))
                );
            }
        }
        let mut generator = block.logical_x();
        for &(a, b) in graph.edges() {
            if a == q || b == q {
                generator = generator
                    .multiply_commuting(&blocks[if a == q { b } else { a }].logical_z())
                    .unwrap()
                    .1;
            }
        }
        assert_eq!(state.expectation(&generator).unwrap(), 1);
    }
    assert!(ShorEncodedResource::new(&ResourceState::bell_pair()).is_ok());
    assert!(ShorEncodedResource::new(&ResourceState::graph(17, vec![]).unwrap()).is_err());
    assert!(ShorBlock::new([PhotonId(0); 4]).is_err());
}
#[test]
fn actual_unboosted_measurements_recover_correct_logical_bell_parities_with_exact_yield() {
    for xx in [false, true] {
        for zz in [false, true] {
            let mut records = vec![];
            physical_records(
                encoded_bell(xx, zz),
                0,
                [FusionOutcome::Loss; 4],
                1.,
                &mut records,
            );
            assert!((records.iter().map(|r| r.1).sum::<f64>() - 1.).abs() < 1e-12);
            let mut complete = 0.;
            for (record, weight) in records {
                let EncodedFusionRecovery::Consistent(logical) =
                    EncodedFusion::recover_outcomes(record)
                else {
                    panic!("physical branch contradicted the code")
                };
                assert_eq!(logical.zz, Some(parity(zz)));
                if let Some(actual) = logical.xx {
                    assert_eq!(actual, parity(xx));
                    complete += weight;
                }
            }
            assert!((complete - if zz { 1. } else { 0.5 }).abs() < 1e-12);
        }
    }
    // Available identical-Z bits also expose physical-code violations, even when ZZ parities agree.
    assert_eq!(
        EncodedFusion::recover_outcomes([
            FusionOutcome::Failure { bit: false },
            FusionOutcome::Loss,
            FusionOutcome::Failure { bit: true },
            FusionOutcome::Loss
        ]),
        EncodedFusionRecovery::Inconsistent
    );
}
#[test]
fn every_erased_record_is_classified_against_exact_quantum_branch_completions() {
    let mut valid = BTreeMap::new();
    for xx in [false, true] {
        for zz in [false, true] {
            let mut records = BTreeMap::new();
            ideal_records(encoded_bell(xx, zz), 0, 0, 1., &mut records);
            assert_eq!(records.len(), 8);
            for (record, weight) in records {
                assert!((weight - 0.125).abs() < 1e-12);
                assert!(valid.insert(record, (parity(xx), parity(zz))).is_none());
            }
        }
    }
    assert_eq!(valid.len(), 32);
    let convert = |record: u8, erased: u8| {
        std::array::from_fn(|i| FusionParities {
            xx: (erased & (1 << (2 * i)) == 0).then_some(parity(record & (1 << (2 * i)) != 0)),
            zz: (erased & (1 << (2 * i + 1)) == 0)
                .then_some(parity(record & (1 << (2 * i + 1)) != 0)),
        })
    };
    for record in 0..=255u8 {
        let recovered = EncodedFusion::recover_parities(convert(record, 0));
        assert_eq!(
            matches!(recovered, EncodedFusionRecovery::Consistent(_)),
            valid.contains_key(&record)
        );
    }
    for &record in valid.keys() {
        for erased in 0..=255u8 {
            let compatible: Vec<_> = valid
                .iter()
                .filter(|&(&candidate, _)| (candidate ^ record) & !erased == 0)
                .map(|(_, v)| *v)
                .collect();
            let infer = |axis: usize| {
                let values: Vec<_> = compatible
                    .iter()
                    .map(|v| if axis == 0 { v.0 } else { v.1 })
                    .collect();
                values.iter().all(|&v| v == values[0]).then_some(values[0])
            };
            assert_eq!(
                EncodedFusion::recover_parities(convert(record, erased)),
                EncodedFusionRecovery::Consistent(FusionParities {
                    xx: infer(0),
                    zz: infer(1)
                })
            );
        }
    }
}
#[test]
fn compiled_physical_loss_attempts_preserve_known_logical_output_correlations_and_cost() {
    let encoded =
        ShorEncodedResource::new(&ResourceState::graph(2, vec![(0, 1)]).unwrap()).unwrap();
    let mut program = FusionProgram::new();
    let a = encoded.add_to(&mut program, NodeId(0)).unwrap();
    let b = encoded.add_to(&mut program, NodeId(0)).unwrap();
    let fusion = EncodedFusion::append(&mut program, a[0], b[0], NodeId(1)).unwrap();
    assert_eq!(fusion.physical_fusions(), &[0, 1, 2, 3]);
    assert_eq!(fusion.photons_consumed(), 8);
    assert_eq!(program.num_photons(), 16);
    assert_eq!(program.num_fusions(), 4);
    let before = program.clone();
    assert!(EncodedFusion::append(&mut program, a[1], b[0], NodeId(1)).is_err());
    assert_eq!(program, before);
    let logical = fusion.logical_observables();
    assert!(logical.iter().all(|p| p.terms().len() == 4));
    let output_z = a[1]
        .logical_z()
        .multiply_commuting(&b[1].logical_z())
        .unwrap()
        .1;
    let output_x = a[1]
        .logical_x()
        .multiply_commuting(&b[1].logical_x())
        .unwrap()
        .1;
    for eta in [1., 0.8] {
        let plan = program
            .compile(&fabric(eta), CompileOptions::default())
            .unwrap();
        assert_eq!(plan.outputs().len(), 8);
        let shots = StabilizerSimulator::default()
            .sample(&plan, 0, 2048, 87, &[output_z.clone(), output_x.clone()])
            .unwrap();
        let mut known = 0;
        let mut lost = 0;
        for shot in shots {
            lost += usize::from(shot.outcomes.contains(&FusionOutcome::Loss));
            let EncodedFusionRecovery::Consistent(parities) =
                fusion.recover(&shot.outcomes).unwrap()
            else {
                panic!("valid lossy shot rejected")
            };
            for (parity, value) in [parities.xx, parities.zz]
                .into_iter()
                .zip(shot.pauli_expectations)
            {
                if let Some(parity) = parity {
                    assert_eq!(value, if parity == Parity::Even { 1 } else { -1 });
                    known += 1;
                }
            }
        }
        assert!(known > 100);
        assert_eq!(lost == 0, eta == 1.);
    }
    assert!(fusion.recover(&[FusionOutcome::Loss; 3]).is_err());
}
#[test]
fn logical_h_frames_and_all_mixed_orientation_fusions_preserve_canonical_paulis() {
    for resource in [
        ResourceState::bell_pair(),
        ResourceState::four_star(),
        ResourceState::four_star()
            .with_local_hadamards(&[0, 1, 2, 3])
            .unwrap(),
    ] {
        let encoded = ShorEncodedResource::new(&resource).unwrap();
        let mut program = FusionProgram::new();
        let blocks = encoded.add_to(&mut program, NodeId(0)).unwrap();
        let state = StabilizerSimulator::default()
            .prepare(
                &program
                    .compile(&fabric(1.), CompileOptions::default())
                    .unwrap(),
            )
            .unwrap();
        for (q, block) in blocks.iter().enumerate() {
            for s in block.stabilizers() {
                assert_eq!(state.expectation(&s).unwrap(), 1);
            }
            let operator = |site: usize, pauli: Pauli| {
                let pauli = if resource.hadamards().contains(&site) {
                    if pauli == Pauli::X {
                        Pauli::Z
                    } else {
                        Pauli::X
                    }
                } else {
                    pauli
                };
                if pauli == Pauli::X {
                    blocks[site].logical_x()
                } else {
                    blocks[site].logical_z()
                }
            };
            let mut generator = operator(q, Pauli::X);
            for &(a, b) in resource.edges() {
                if a == q || b == q {
                    generator = generator
                        .multiply_commuting(&operator(if a == q { b } else { a }, Pauli::Z))
                        .unwrap()
                        .1;
                }
            }
            assert_eq!(state.expectation(&generator).unwrap(), 1);
        }
    }
    for left in [ShorOrientation::Standard, ShorOrientation::Dual] {
        for right in [ShorOrientation::Standard, ShorOrientation::Dual] {
            let mut program = FusionProgram::new();
            program
                .add_resource(ResourceState::graph(8, vec![]).unwrap(), NodeId(0))
                .unwrap();
            let a = ShorBlock::with_orientation([0, 1, 2, 3].map(PhotonId), left).unwrap();
            let b = ShorBlock::with_orientation([4, 5, 6, 7].map(PhotonId), right).unwrap();
            let fusion = EncodedFusion::append(&mut program, a, b, NodeId(1)).unwrap();
            let obs: Vec<_> = program
                .fusions()
                .iter()
                .flat_map(|f| {
                    [Pauli::X, Pauli::Z].map(|s| p(&[(f.photons[0].0, s), (f.photons[1].0, s)]))
                })
                .collect();
            for xx in [false, true] {
                for zz in [false, true] {
                    let mut state = Tableau::zero(8, TableauLimits::default()).unwrap();
                    state.h(0).unwrap();
                    state.cnot(0, 4).unwrap();
                    if zz {
                        state.x(4).unwrap();
                    }
                    if xx {
                        state.z(4).unwrap();
                    }
                    for (q, orientation) in [(0, left), (4, right)] {
                        if orientation == ShorOrientation::Dual {
                            state.h(q).unwrap();
                        }
                        state.h(q + 1).unwrap();
                        state.cnot(q + 1, q).unwrap();
                        state.cnot(q, q + 2).unwrap();
                        state.cnot(q + 1, q + 3).unwrap();
                        if orientation == ShorOrientation::Dual {
                            for i in 0..4 {
                                state.h(q + i).unwrap();
                            }
                        }
                    }
                    for (observable, bit) in fusion.logical_observables().iter().zip([xx, zz]) {
                        assert_eq!(
                            state.expectation(observable).unwrap(),
                            if bit { -1 } else { 1 }
                        );
                    }
                    let samples = ReferenceSampler::compile(&state, &obs, 100)
                        .unwrap()
                        .sample(0, 128, 781, PauliNoise::NONE, 1 << 20)
                        .unwrap();
                    for shot in 0..128 {
                        let records = std::array::from_fn(|i| FusionParities {
                            xx: Some(parity(samples.bit(2 * i, shot).unwrap())),
                            zz: Some(parity(samples.bit(2 * i + 1, shot).unwrap())),
                        });
                        assert_eq!(
                            fusion.recover_local_parities(records),
                            EncodedFusionRecovery::Consistent(FusionParities {
                                xx: Some(parity(xx)),
                                zz: Some(parity(zz))
                            })
                        );
                    }
                }
            }
        }
    }
}
