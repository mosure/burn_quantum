//! Growing-distance local surgery channels, distance witnesses and physical aborts.
use burn_quantum::photonic::{qec::*, stabilizer::*};
fn product(terms: Vec<(usize, Pauli)>) -> PauliProduct {
    PauliProduct::new(terms).unwrap()
}
fn join(a: &PauliProduct, b: &PauliProduct) -> PauliProduct {
    a.multiply_commuting(b).unwrap().1
}
fn choi(plan: &PlanarSurgery) -> Tableau {
    let code = plan.code();
    let n = code.qubits();
    let mut state = Tableau::zero(2 * n + 2, TableauLimits::default()).unwrap();
    for patch in 0..2 {
        state.h(2 * n + patch).unwrap();
        state
            .cnot(2 * n + patch, patch * n + code.input_site())
            .unwrap();
        code.encode(&mut state, patch * n).unwrap();
    }
    state
}
fn assert_choi(plan: &PlanarSurgery, state: &Tableau, parity: bool) {
    let code = plan.code();
    let n = code.qubits();
    for p in plan.split_checks() {
        assert_eq!(state.expectation(p).unwrap(), 1);
    }
    let xa = code.logical_x(0).unwrap();
    let xb = code.logical_x(n).unwrap();
    let za = code.logical_z(0).unwrap();
    let zb = code.logical_z(n).unwrap();
    let xr = product(vec![(2 * n, Pauli::X), (2 * n + 1, Pauli::X)]);
    let zr = product(vec![(2 * n, Pauli::Z), (2 * n + 1, Pauli::Z)]);
    let checks = match plan.basis() {
        MergeBasis::ZZ => vec![
            (join(&za, &product(vec![(2 * n, Pauli::Z)])), 1),
            (join(&zb, &product(vec![(2 * n + 1, Pauli::Z)])), 1),
            (join(&join(&xa, &xb), &xr), 1),
            (join(&za, &zb), if parity { -1 } else { 1 }),
        ],
        MergeBasis::XX => vec![
            (join(&xa, &product(vec![(2 * n, Pauli::X)])), 1),
            (join(&xb, &product(vec![(2 * n + 1, Pauli::X)])), 1),
            (join(&join(&za, &zb), &zr), 1),
            (join(&xa, &xb), if parity { -1 } else { 1 }),
        ],
    };
    // Together with the2(n−1) code stabilizers, these four independent Choi
    // stabilizers determine the entire conditional state on2n+2 physical sites.
    for (p, sign) in checks {
        assert_eq!(state.expectation(&p).unwrap(), sign);
    }
}
#[test]
fn local_merge_split_complete_choi_branches_and_growing_distances() {
    for d in [3, 5, 7] {
        for basis in [MergeBasis::ZZ, MergeBasis::XX] {
            let plan = PlanarSurgery::new(d, basis, SurgeryLimits::default()).unwrap();
            let expected = if basis == MergeBasis::ZZ {
                (2 * d, d)
            } else {
                (d, 2 * d)
            };
            assert_eq!(
                (
                    plan.distance_certificate().x_distance,
                    plan.distance_certificate().z_distance
                ),
                expected
            );
            assert_eq!(plan.distance_certificate().temporal_repetitions, d);
            assert_eq!(plan.merged_checks().len(), 2 * plan.code().qubits() - 1);
            for p in plan.merged_checks() {
                assert!(p.terms().len() <= 6);
                for &(a, _) in p.terms() {
                    for &(b, _) in p.terms() {
                        let a = plan.coordinates()[a];
                        let b = plan.coordinates()[b];
                        assert!(a[0].abs_diff(b[0]) + a[1].abs_diff(b[1]) <= 6);
                    }
                }
            }
            let initial = choi(&plan);
            let random = 2 * d - 1;
            // Exhaustive d3 branches; atlargerd verify zero, allones and every
            // affine-outcome generator, plus deterministic pseudorandom combinations.
            let patterns: Vec<usize> = if d == 3 {
                (0..1usize << random).collect()
            } else {
                [0, (1 << random) - 1]
                    .into_iter()
                    .chain((0..random).map(|i| 1 << i))
                    .chain((0..16).map(|i| (i * 7919 + 17) & ((1 << random) - 1)))
                    .collect()
            };
            for pattern in patterns {
                let mut state = initial.clone();
                let mut used = 0;
                let record = plan
                    .execute(&mut state, [0, plan.code().qubits()], || {
                        let bit = pattern & (1 << used) != 0;
                        used += 1;
                        bit
                    })
                    .unwrap();
                assert_eq!(used, random);
                assert_eq!(record.random_measurements, random);
                assert_eq!(record.outcomes.len(), plan.measurements());
                assert_eq!(record.merged.len(), d);
                assert_eq!(record.split.len(), d);
                assert!(record.split[1..].iter().flatten().all(|v| !*v));
                assert_choi(&plan, &state, record.parity);
            }
        }
    }
}
#[test]
fn distance_three_shortest_paths_agree_with_independent_low_weight_enumeration() {
    for basis in [MergeBasis::ZZ, MergeBasis::XX] {
        let plan = PlanarSurgery::new(3, basis, Default::default()).unwrap();
        let n = plan.code().qubits();
        let mut lx = plan.code().logical_x(0).unwrap();
        let mut lz = plan.code().logical_z(0).unwrap();
        match basis {
            MergeBasis::ZZ => lx = join(&lx, &plan.code().logical_x(n).unwrap()),
            MergeBasis::XX => lz = join(&lz, &plan.code().logical_z(n).unwrap()),
        }
        for (p, logical, distance, witness) in [
            (
                Pauli::X,
                lz,
                plan.distance_certificate().x_distance,
                &plan.distance_certificate().x_witness,
            ),
            (
                Pauli::Z,
                lx,
                plan.distance_certificate().z_distance,
                &plan.distance_certificate().z_witness,
            ),
        ] {
            let checks: Vec<_> = plan
                .merged_checks()
                .iter()
                .filter(|c| c.terms()[0].1 != p)
                .map(|c| c.terms().iter().fold(0u64, |m, (q, _)| m | (1 << q)))
                .collect();
            let logical = logical.terms().iter().fold(0u64, |m, (q, _)| m | (1 << q));
            let valid = |mask: u64| {
                (mask & logical).count_ones() % 2 == 1
                    && checks
                        .iter()
                        .all(|c| (mask & c).count_ones().is_multiple_of(2))
            };
            assert!(valid(witness.iter().fold(0, |v, q| v ^ (1 << q))));
            fn visit(q: usize, left: usize, n: usize, mask: u64, test: &impl Fn(u64) -> bool) {
                if left == 0 {
                    assert!(!test(mask));
                    return;
                }
                for i in q..=n - left {
                    visit(i + 1, left - 1, n, mask | (1 << i), test);
                }
            }
            for weight in 0..distance {
                visit(0, weight, 2 * n, 0, &valid);
            }
        }
    }
}
#[test]
fn measurement_only_cnot_preserves_both_unknown_inputs_entangled_with_references() {
    for d in [3, 5, 7] {
        let code = PlanarCode::new(d, 2000).unwrap();
        let n = code.qubits();
        for pattern in [0u64, 1, 0x19937, 0xffff_ffff_ffff_ffff] {
            let mut state = Tableau::zero(3 * n + 2, TableauLimits::default()).unwrap();
            for (patch, reference) in [(0, 3 * n), (2 * n, 3 * n + 1)] {
                state.h(reference).unwrap();
                state.cnot(reference, patch + code.input_site()).unwrap();
                code.encode(&mut state, patch).unwrap();
            }
            state.h(n + code.input_site()).unwrap();
            code.encode(&mut state, n).unwrap();
            let mut bits = pattern;
            let record =
                PlanarSurgery::cnot(d, &mut state, [0, n, 2 * n], Default::default(), || {
                    let out = bits & 1 != 0;
                    bits = bits.rotate_right(1);
                    out
                })
                .unwrap();
            assert_eq!(record.surgeries[0].merged.len(), d);
            let xc = code.logical_x(0).unwrap();
            let xt = code.logical_x(2 * n).unwrap();
            let zc = code.logical_z(0).unwrap();
            let zt = code.logical_z(2 * n).unwrap();
            for p in [
                join(&join(&xc, &xt), &product(vec![(3 * n, Pauli::X)])),
                join(&zc, &product(vec![(3 * n, Pauli::Z)])),
                join(&xt, &product(vec![(3 * n + 1, Pauli::X)])),
                join(&join(&zc, &zt), &product(vec![(3 * n + 1, Pauli::Z)])),
            ] {
                assert_eq!(state.expectation(&p).unwrap(), 1);
            }
            for offset in [0, 2 * n] {
                for p in code
                    .x_checks(offset)
                    .unwrap()
                    .iter()
                    .chain(&code.z_checks(offset).unwrap())
                {
                    assert_eq!(state.expectation(p).unwrap(), 1);
                }
            }
        }
    }
}
#[test]
fn limits_invalid_codes_and_physical_abort_are_explicit() {
    assert!(PlanarSurgery::new(4, MergeBasis::XX, Default::default()).is_err());
    for limits in [
        SurgeryLimits {
            max_data_qubits: 25,
            ..Default::default()
        },
        SurgeryLimits {
            max_checks: 2,
            ..Default::default()
        },
        SurgeryLimits {
            max_measurements: 10,
            ..Default::default()
        },
        SurgeryLimits {
            max_distance_work: 1,
            ..Default::default()
        },
    ] {
        assert!(PlanarSurgery::new(3, MergeBasis::ZZ, limits).is_err());
    }
    let plan = PlanarSurgery::new(3, MergeBasis::ZZ, Default::default()).unwrap();
    let n = plan.code().qubits();
    let mut state = choi(&plan);
    let mut calls = 0;
    let run = plan
        .execute_with(&mut state, [0, n], |state, r| {
            calls += 1;
            if calls == 3 {
                return Ok(None);
            }
            Ok(Some(state.measure(&r.observable, false)?))
        })
        .unwrap();
    let SurgeryAttempt::Aborted(abort) = run else {
        panic!("failed instrument accepted")
    };
    assert_eq!(calls, 3);
    assert_eq!(abort.outcomes.len(), 2);
    assert_eq!(abort.at.check, 2);
    assert!(
        plan.split_correction(&vec![true; plan.split_checks().len()])
            .is_err()
    );
}

#[test]
fn growing_surgery_transcripts_match_independent_pinned_stim_115() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("stabilizer/surgery-stim-1.15.0.json")).unwrap();
    assert_eq!(fixture["stim_version"], "1.15.0");
    assert_eq!(
        fixture["source_archive_sha256"],
        "95236006859d6754be99629d4fb44788e742e962ac8c59caad421ca088f7350e"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 18);
    for case in cases {
        let d = case["distance"].as_u64().unwrap() as usize;
        let seed = case["seed"].as_u64().unwrap() as usize;
        let basis = match case["basis"].as_str().unwrap() {
            "ZZ" => MergeBasis::ZZ,
            "XX" => MergeBasis::XX,
            _ => panic!("unknown pinned basis"),
        };
        let plan = PlanarSurgery::new(d, basis, Default::default()).unwrap();
        let mut state = choi(&plan);
        let mut random = 0;
        let result = plan
            .execute(&mut state, [0, plan.code().qubits()], || {
                let bit = (random * 17 + seed * 13 + random / 3) % 7 < 3;
                random += 1;
                bit
            })
            .unwrap();
        let record_bits: String = result
            .outcomes
            .iter()
            .map(|(_, m)| if m.negative { '1' } else { '0' })
            .collect();
        assert_eq!(record_bits, case["record_bits"].as_str().unwrap());
        assert_eq!(
            result.random_measurements,
            case["random_measurements"].as_u64().unwrap() as usize
        );
        assert_eq!(result.parity, case["parity"].as_bool().unwrap());
        let mut correction = vec!['_'; 2 * plan.code().qubits()];
        for &(q, p) in result.correction.terms() {
            correction[q] = match p {
                Pauli::X => 'X',
                Pauli::Z => 'Z',
                Pauli::Y => 'Y',
            };
        }
        assert_eq!(
            correction.into_iter().collect::<String>(),
            case["correction"].as_str().unwrap()
        );
        assert_choi(&plan, &state, result.parity);
    }
}
