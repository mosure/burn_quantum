//! Independent noisy quantum trajectories use fresh measurement randomness, not
//! forced outcomes obtained from a clean trajectory or predicted Pauli frame.
use burn_quantum::photonic::{qec::threshold::*, qec::*, stabilizer::*};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
fn apply(state: &mut Tableau, site: usize, pauli: Pauli) {
    match pauli {
        Pauli::X => state.x(site).unwrap(),
        Pauli::Z => state.z(site).unwrap(),
        Pauli::Y => {
            state.x(site).unwrap();
            state.z(site).unwrap();
        }
    }
}
#[test]
fn independent_quantum_measurement_branches_match_decoded_instrument_error_events() {
    for basis in [MergeBasis::ZZ, MergeBasis::XX] {
        let model = SurgeryDetectorModel::new(
            PlanarSurgery::new(3, basis, Default::default()).unwrap(),
            Default::default(),
        )
        .unwrap();
        let plan = model.plan();
        let code = plan.code();
        let n = code.qubits();
        let graph = model.graph();
        for seed in 0..48 {
            let mut fault_rng = ChaCha8Rng::seed_from_u64(1000 + seed);
            let faults: Vec<_> = graph
                .edges()
                .iter()
                .map(|_| fault_rng.random::<f64>() < 0.025)
                .collect();
            let mut quantum_rng = ChaCha8Rng::seed_from_u64(9000 + seed);
            let mut state = Tableau::zero(2 * n + 2, TableauLimits::default()).unwrap();
            for block in 0..2 {
                state.h(2 * n + block).unwrap();
                state
                    .cnot(2 * n + block, block * n + code.input_site())
                    .unwrap();
                code.encode(&mut state, block * n).unwrap();
            }
            let mut previous = None;
            let mut raw = Vec::new();
            for (index, record) in model.records().iter().enumerate() {
                if previous != Some(record.round) {
                    for (&present, fault) in faults.iter().zip(model.faults()) {
                        if present
                            && let SurgeryFault::Data { round, site, pauli } = fault
                            && *round == record.round
                        {
                            apply(&mut state, *site, *pauli);
                        }
                    }
                    previous = Some(record.round);
                }
                // No predicted frame enters this quantum measurement choice.
                let mut bit = state
                    .measure(&record.observable, quantum_rng.random())
                    .unwrap()
                    .negative;
                for (&present, fault) in faults.iter().zip(model.faults()) {
                    if present && *fault == (SurgeryFault::Readout { record: index }) {
                        bit ^= true;
                    }
                }
                raw.push(bit);
            }
            let detector = model.detector_values(&raw).unwrap();
            assert_eq!(detector, graph.syndrome(&faults).unwrap());
            let recovered = graph
                .decode_union_find(
                    &detector,
                    &vec![false; faults.len()],
                    DecoderBudget::default(),
                )
                .unwrap();
            assert_eq!(recovered.status, GraphStatus::Corrected);
            let mut physically_recovered = state.clone();
            let proposal = model
                .recover(&raw, &vec![false; faults.len()], DecoderBudget::default())
                .unwrap();
            assert_eq!(proposal.decode, recovered);
            proposal.apply(&mut physically_recovered, [0, n]).unwrap();
            for check in plan.split_checks() {
                assert_eq!(physically_recovered.expectation(check).unwrap(), 1);
            }
            // Physical split gauge frame uses observed first-split records only.
            let split_start = code.distance() * plan.merged_checks().len();
            let mut frame = false;
            for (i, &[a, _]) in plan.boundary_checks().iter().enumerate() {
                frame ^= raw[split_start + a];
                if frame {
                    for &(q, p) in plan.seam_checks()[i + 1].terms() {
                        apply(&mut state, q, p);
                    }
                }
            }
            let primary = if basis == MergeBasis::ZZ {
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
            let reported = raw[..code.distance()].iter().fold(false, |v, &b| v ^ b);
            let mut actual = 0;
            for (i, (data, reference)) in model
                .logical_data_observables()
                .iter()
                .zip(&refs)
                .enumerate()
            {
                let mut terms = data.terms().to_vec();
                terms.extend(reference.iter().copied());
                let sign = state
                    .expectation(&PauliProduct::new(terms).unwrap())
                    .unwrap();
                assert_ne!(sign, 0);
                let wrong = (sign < 0) ^ (i == 3 && reported);
                actual |= u64::from(wrong) << i;
            }
            assert_eq!(actual, graph.logical_flips(&faults).unwrap());
            let mut actual_corrected = 0;
            for (i, (data, reference)) in model
                .logical_data_observables()
                .iter()
                .zip(&refs)
                .enumerate()
            {
                let mut terms = data.terms().to_vec();
                terms.extend(reference.iter().copied());
                let sign = physically_recovered
                    .expectation(&PauliProduct::new(terms).unwrap())
                    .unwrap();
                assert_ne!(sign, 0);
                let wrong = (sign < 0) ^ (i == 3 && proposal.parity.unwrap());
                actual_corrected |= u64::from(wrong) << i;
            }
            let modeled_corrected = graph.logical_flips(&faults).unwrap() ^ recovered.logical;
            assert_eq!(actual_corrected, modeled_corrected);
        }
    }
}
#[test]
fn surgery_independent_fault_sampling_never_postselects_decoder_limits() {
    let model = SurgeryDetectorModel::new(
        PlanarSurgery::new(3, MergeBasis::ZZ, Default::default()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let graph = model.graph();
    let out = graph
        .run_independent_faults(
            0.01,
            0,
            100,
            129,
            DecoderBudget {
                max_edge_visits: 0,
                max_growth_rounds: 0,
            },
        )
        .unwrap();
    assert_eq!(out.counts.attempts, 100);
    assert_eq!(out.counts.unsuccessful().unwrap(), 100);
    assert_eq!(out.counts.decoder_limits, 100);
    assert!(
        graph
            .run_independent_faults(f64::NAN, 0, 10, 0, Default::default())
            .is_err()
    );
    assert!(
        graph
            .run_independent_faults(0.01, u64::MAX, 1, 0, Default::default())
            .is_err()
    );
}
