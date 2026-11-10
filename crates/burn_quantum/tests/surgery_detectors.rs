//! Quantum-trajectory oracles for every local merge/split detector fault location.
use burn_quantum::photonic::{qec::*, stabilizer::*};
fn initial(plan: &PlanarSurgery) -> Tableau {
    let code = plan.code();
    let n = code.qubits();
    let mut state = Tableau::zero(2 * n + 2, TableauLimits::default()).unwrap();
    for i in 0..2 {
        state.h(2 * n + i).unwrap();
        state.cnot(2 * n + i, i * n + code.input_site()).unwrap();
        code.encode(&mut state, i * n).unwrap();
    }
    state
}
fn baseline(model: &SurgeryDetectorModel) -> Vec<bool> {
    let mut state = initial(model.plan());
    model
        .records()
        .iter()
        .enumerate()
        .map(|(i, r)| {
            state
                .measure(&r.observable, (i * 17 + 3) % 11 < 5)
                .unwrap()
                .negative
        })
        .collect()
}
fn pauli(state: &mut Tableau, site: usize, p: Pauli) {
    match p {
        Pauli::X => state.x(site).unwrap(),
        Pauli::Z => state.z(site).unwrap(),
        Pauli::Y => unreachable!(),
    }
}
fn trajectory(
    model: &SurgeryDetectorModel,
    faults: &[usize],
    base: &[bool],
) -> (Tableau, Vec<bool>) {
    let mut state = initial(model.plan());
    let mut frame = vec![];
    let mut raw = vec![];
    let mut prior = None;
    for (i, r) in model.records().iter().enumerate() {
        if prior != Some(r.round) {
            for &event in faults {
                if let SurgeryFault::Data {
                    round,
                    site,
                    pauli: p,
                } = model.faults()[event]
                    && round == r.round
                {
                    pauli(&mut state, site, p);
                    frame.push(PauliProduct::new(vec![(site, p)]).unwrap());
                }
            }
            prior = Some(r.round);
        }
        let flip = frame
            .iter()
            .fold(false, |b, p| b ^ p.anticommutes(&r.observable));
        let mut bit = state
            .measure(&r.observable, base[i] ^ flip)
            .unwrap()
            .negative;
        for &event in faults {
            if model.faults()[event] == (SurgeryFault::Readout { record: i }) {
                bit ^= true;
            }
        }
        raw.push(bit);
    }
    (state, raw)
}
fn oracle(model: &SurgeryDetectorModel, faults: &[usize], base: &[bool]) -> (Vec<bool>, u64) {
    let (mut state, raw) = trajectory(model, faults, base);
    // Controller's intended split frame, based only on the first restored-A
    // boundary signs. Inconsistent syndromes are retained for graph decoding.
    let plan = model.plan();
    let d = plan.code().distance();
    let split_start = d * plan.merged_checks().len();
    let mut bit = false;
    for (i, &[a, _]) in plan.boundary_checks().iter().enumerate() {
        bit ^= raw[split_start + a];
        if bit {
            for &(q, p) in plan.seam_checks()[i + 1].terms() {
                pauli(&mut state, q, p);
            }
        }
    }
    let reported = raw[..d].iter().fold(false, |a, &b| a ^ b);
    let logical = score(model, &state, reported);
    (raw, logical)
}
fn score(model: &SurgeryDetectorModel, state: &Tableau, reported: bool) -> u64 {
    let plan = model.plan();
    let n = plan.code().qubits();
    let primary = if plan.basis() == MergeBasis::ZZ {
        Pauli::Z
    } else {
        Pauli::X
    };
    let dual = if primary == Pauli::Z {
        Pauli::X
    } else {
        Pauli::Z
    };
    let refs = [
        vec![(2 * n, primary)],
        vec![(2 * n + 1, primary)],
        vec![(2 * n, dual), (2 * n + 1, dual)],
        vec![],
    ];
    let mut logical = 0;
    for (i, (p, extra)) in model
        .logical_data_observables()
        .iter()
        .zip(refs)
        .enumerate()
    {
        let mut terms = p.terms().to_vec();
        terms.extend(extra);
        let sign = state
            .expectation(&PauliProduct::new(terms).unwrap())
            .unwrap();
        assert_ne!(sign, 0);
        let wrong = (sign < 0) ^ (i == 3 && reported);
        logical |= u64::from(wrong) << i;
    }
    logical
}
#[test]
fn every_declared_transition_fault_matches_actual_entangled_tableau_records_and_output_frame() {
    for basis in [MergeBasis::ZZ, MergeBasis::XX] {
        let model = SurgeryDetectorModel::new(
            PlanarSurgery::new(3, basis, Default::default()).unwrap(),
            Default::default(),
        )
        .unwrap();
        let base = baseline(&model);
        assert!(model.detector_values(&base).unwrap().iter().all(|v| !*v));
        let (_, logical) = oracle(&model, &[], &base);
        assert_eq!(logical, 0);
        for event in 0..model.faults().len() {
            let (raw, logical) = oracle(&model, &[event], &base);
            let mut faults = vec![false; model.faults().len()];
            faults[event] = true;
            assert_eq!(
                model.detector_values(&raw).unwrap(),
                model.graph().syndrome(&faults).unwrap(),
                "{basis:?} event{event} {:?}",
                model.faults()[event]
            );
            assert_eq!(
                logical,
                model.graph().logical_flips(&faults).unwrap(),
                "{basis:?} event{event} {:?}",
                model.faults()[event]
            );
            let mask = model.fault_records(event).unwrap();
            assert_eq!(
                raw.iter()
                    .zip(&base)
                    .map(|(a, b)| a ^ b)
                    .collect::<Vec<_>>(),
                mask
            );
        }
    }
}
#[test]
fn combined_surgery_space_time_distance_grows_and_correlated_fault_combinations_replay() {
    for d in [3, 5, 7, 9, 11] {
        for basis in [MergeBasis::ZZ, MergeBasis::XX] {
            let model = SurgeryDetectorModel::new(
                PlanarSurgery::new(d, basis, Default::default()).unwrap(),
                Default::default(),
            )
            .unwrap();
            assert!(model.logical_distances().iter().all(|&value| value >= d));
            assert_eq!(*model.logical_distances().iter().min().unwrap(), d);
            for (i, witness) in model.distance_witnesses().iter().enumerate() {
                let mut faults = vec![false; model.faults().len()];
                for &edge in witness {
                    faults[edge] ^= true;
                }
                assert!(model.graph().syndrome(&faults).unwrap().iter().all(|v| !*v));
                assert_eq!((model.graph().logical_flips(&faults).unwrap() >> i) & 1, 1);
            }
            if d <= 7 {
                let base = baseline(&model);
                for shot in 0..8 {
                    let events = vec![
                        (shot * 7919 + 3) % model.faults().len(),
                        (shot * 104729 + 17) % model.faults().len(),
                        (shot * 3167 + 5) % model.faults().len(),
                    ];
                    let (raw, logical) = oracle(&model, &events, &base);
                    let mut faults = vec![false; model.faults().len()];
                    for event in events {
                        faults[event] ^= true;
                    }
                    assert_eq!(
                        model.detector_values(&raw).unwrap(),
                        model.graph().syndrome(&faults).unwrap()
                    );
                    assert_eq!(logical, model.graph().logical_flips(&faults).unwrap());
                }
            }
        }
    }
}
#[test]
fn random_initial_seam_and_split_signs_are_not_assumed_zero_and_budgets_reject() {
    let plan = PlanarSurgery::new(3, MergeBasis::ZZ, Default::default()).unwrap();
    for limits in [
        SurgeryDetectorLimits {
            max_records: 1,
            ..Default::default()
        },
        SurgeryDetectorLimits {
            max_faults: 1,
            ..Default::default()
        },
        SurgeryDetectorLimits {
            max_signature_work: 1,
            ..Default::default()
        },
        SurgeryDetectorLimits {
            max_distance_work: 1,
            ..Default::default()
        },
    ] {
        assert!(SurgeryDetectorModel::new(plan.clone(), limits).is_err());
    }
    let model = SurgeryDetectorModel::new(plan, Default::default()).unwrap();
    for pattern in [0, 1, 0x19937, u64::MAX] {
        let mut state = initial(model.plan());
        let mut bits = pattern;
        let raw: Vec<_> = model
            .records()
            .iter()
            .map(|r| {
                let bit = bits & 1 != 0;
                bits = bits.rotate_right(1);
                state.measure(&r.observable, bit).unwrap().negative
            })
            .collect();
        assert!(model.detector_values(&raw).unwrap().iter().all(|v| !*v));
    }
}

#[test]
fn typed_recovery_restores_actual_unknown_inputs_and_rejects_ambiguous_or_exhausted_decoding() {
    use burn_quantum::photonic::qec::threshold::{DecoderBudget, GraphStatus};
    for basis in [MergeBasis::ZZ, MergeBasis::XX] {
        let model = SurgeryDetectorModel::new(
            PlanarSurgery::new(3, basis, Default::default()).unwrap(),
            Default::default(),
        )
        .unwrap();
        let base = baseline(&model);
        let clear = vec![false; model.faults().len()];
        for event in 0..model.faults().len() {
            let (mut state, raw) = trajectory(&model, &[event], &base);
            let recovery = model
                .recover(&raw, &clear, DecoderBudget::default())
                .unwrap();
            assert_eq!(
                recovery.decode.status,
                GraphStatus::Corrected,
                "fault{event}"
            );
            recovery
                .apply(&mut state, [0, model.plan().code().qubits()])
                .unwrap();
            for p in model.plan().split_checks() {
                assert_eq!(state.expectation(p).unwrap(), 1, "fault{event}");
            }
            assert_eq!(
                score(&model, &state, recovery.parity.unwrap()),
                0,
                "fault{event}"
            );
            assert!(
                model
                    .detector_values(recovery.corrected_records.as_ref().unwrap())
                    .unwrap()
                    .iter()
                    .all(|v| !*v)
            );
        }
        let mut state = initial(model.plan());
        let limited = model
            .recover(
                &base,
                &clear,
                DecoderBudget {
                    max_edge_visits: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(limited.decode.status, GraphStatus::WorkLimit);
        assert!(
            limited.correction.is_none()
                && limited.parity.is_none()
                && limited.corrected_records.is_none()
        );
        assert!(
            limited
                .apply(&mut state, [0, model.plan().code().qubits()])
                .is_err()
        );
        let ambiguous = model
            .recover(&base, &vec![true; model.faults().len()], Default::default())
            .unwrap();
        assert_eq!(ambiguous.decode.status, GraphStatus::Ambiguous);
        assert!(
            ambiguous.correction.is_none()
                && ambiguous
                    .apply(&mut state, [0, model.plan().code().qubits()])
                    .is_err()
        );
        assert!(
            model
                .recover(&base[..1], &clear, Default::default())
                .is_err()
        );
        assert!(
            model
                .recover(&base, &clear[..1], Default::default())
                .is_err()
        );
    }
}
