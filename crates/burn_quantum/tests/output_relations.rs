//! Direct branch oracle for the signed nullspace flows returned with output generators.
use burn_quantum::photonic::{qec::MeasurementRelations, stabilizer::*};

fn p(terms: &[(usize, Pauli)]) -> PauliProduct {
    PauliProduct::new(terms.to_vec()).unwrap()
}

fn branches(
    state: Tableau,
    measurements: &[PauliProduct],
    records: Vec<bool>,
    result: &mut Vec<(Tableau, Vec<bool>)>,
) {
    if records.len() == measurements.len() {
        result.push((state, records));
        return;
    }
    let observable = &measurements[records.len()];
    let random = state.expectation(observable).unwrap() == 0;
    for selected in [false, true].into_iter().take(if random { 2 } else { 1 }) {
        let mut child = state.clone();
        let mut record = records.clone();
        record.push(child.measure(observable, selected).unwrap().negative);
        branches(child, measurements, record, result);
    }
}

#[test]
fn reused_nullspace_flows_preserve_all_branch_signs_and_complex_output_axes() {
    let mut cases = vec![];
    // A negative Y stabilizer must retain its sign even when no record is needed.
    let mut negative_y = Tableau::zero(1, Default::default()).unwrap();
    negative_y.h(0).unwrap();
    negative_y.s(0).unwrap();
    negative_y.z(0).unwrap();
    cases.push((negative_y, vec![], vec![0]));

    let mut graph = Tableau::zero(5, Default::default()).unwrap();
    for q in 0..5 {
        graph.h(q).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (2, 4), (3, 4)] {
        graph.cz(a, b).unwrap();
    }
    graph.s(3).unwrap();
    graph.s(4).unwrap();
    graph.z(1).unwrap();
    let measured = vec![
        p(&[(0, Pauli::X)]),
        p(&[(1, Pauli::X), (2, Pauli::X)]),
        p(&[(1, Pauli::Z), (2, Pauli::Z)]),
        // YY = -XX*ZZ gives a signed dependency, without adding a random bit.
        p(&[(1, Pauli::Y), (2, Pauli::Y)]),
    ];
    cases.push((graph.clone(), measured.clone(), vec![4, 3]));
    let repeated = (0..70)
        .map(|i| measured[i % measured.len()].clone())
        .collect();
    cases.push((graph, repeated, vec![3, 4]));

    let mut saw_y = false;
    let mut saw_negative_flow = false;
    let mut saw_negative_check = false;
    for (initial, measurements, outputs) in cases {
        let relations = MeasurementRelations::derive(&initial, &measurements, 1 << 20).unwrap();
        let generators = relations.output_relations(&outputs, 1 << 20).unwrap();
        assert_eq!(generators.len(), outputs.len());
        let independent = generators
            .iter()
            .map(|(operator, _)| relations.correlation(operator).unwrap().unwrap())
            .collect::<Vec<_>>();
        let mut outcomes = vec![];
        branches(initial, &measurements, vec![], &mut outcomes);
        assert!(!outcomes.is_empty());
        for (state, records) in outcomes {
            for ((operator, direct), solved) in generators.iter().zip(&independent) {
                let expected = state.expectation(operator).unwrap();
                assert_ne!(expected, 0);
                for flow in [direct, solved] {
                    let bit = flow
                        .terms
                        .iter()
                        .fold(flow.negative, |v, &r| v ^ records[r]);
                    assert_eq!(expected, if bit { -1 } else { 1 });
                }
                saw_y |= operator.terms().iter().any(|&(_, axis)| axis == Pauli::Y);
                saw_negative_flow |= direct.negative;
            }
            for check in relations.checks() {
                assert!(
                    !check
                        .terms
                        .iter()
                        .fold(check.negative, |v, &r| v ^ records[r])
                );
                saw_negative_check |= check.negative;
            }
        }
    }
    assert!(saw_y && saw_negative_flow && saw_negative_check);
}
