//! Exact branch enumeration tests for bit-sliced reference sampling and check inference.
use burn_quantum::photonic::{Parity, qec::*, stabilizer::*};
use std::collections::BTreeMap;
fn p(terms: &[(usize, Pauli)]) -> PauliProduct {
    PauliProduct::new(terms.to_vec()).unwrap()
}
fn branch_distribution(
    state: Tableau,
    obs: &[PauliProduct],
    i: usize,
    record: u64,
    weight: f64,
    out: &mut BTreeMap<u64, f64>,
) {
    if i == obs.len() {
        *out.entry(record).or_default() += weight;
        return;
    }
    let random = state.expectation(&obs[i]).unwrap() == 0;
    for bit in 0..if random { 2 } else { 1 } {
        let mut next = state.clone();
        let result = next.measure(&obs[i], bit == 1).unwrap();
        branch_distribution(
            next,
            obs,
            i + 1,
            record | (u64::from(result.negative) << i),
            weight / if random { 2. } else { 1. },
            out,
        );
    }
}
#[test]
fn noncommuting_reference_frames_match_exact_branch_distribution_and_pauli_mixture() {
    let mut initial = Tableau::zero(3, TableauLimits::default()).unwrap();
    initial.h(0).unwrap();
    initial.cnot(0, 1).unwrap();
    initial.s(1).unwrap();
    initial.h(2).unwrap();
    let obs = vec![
        p(&[(0, Pauli::Z), (2, Pauli::Z)]),
        p(&[(0, Pauli::X), (1, Pauli::Y)]),
        p(&[(2, Pauli::Y)]),
        p(&[(0, Pauli::Z), (1, Pauli::Z)]),
        p(&[(0, Pauli::X), (1, Pauli::X)]),
        p(&[(0, Pauli::Z), (1, Pauli::Z)]),
    ];
    let sampler = ReferenceSampler::compile(&initial, &obs, 100).unwrap();
    for (noise, probs) in [
        (PauliNoise::NONE, [1., 0., 0., 0.]),
        (PauliNoise::new(1., 0., 0.).unwrap(), [0., 1., 0., 0.]),
        (
            PauliNoise::new(0.17, 0.13, 0.09).unwrap(),
            [0.61, 0.17, 0.13, 0.09],
        ),
    ] {
        let mut expected = BTreeMap::new();
        for errors in 0..64usize {
            let mut state = initial.clone();
            let mut weight = 1.;
            for q in 0..3 {
                let fault = (errors >> (2 * q)) & 3;
                weight *= probs[fault];
                if fault == 1 || fault == 2 {
                    state.x(q).unwrap();
                }
                if fault == 2 || fault == 3 {
                    state.z(q).unwrap();
                }
            }
            if weight > 0. {
                branch_distribution(state, &obs, 0, 0, weight, &mut expected);
            }
        }
        let samples = sampler.sample(0, 16384, 51, noise, 1 << 24).unwrap();
        let mut measured = BTreeMap::new();
        for s in 0..samples.shots() {
            let value =
                (0..obs.len()).fold(0, |v, m| v | (u64::from(samples.bit(m, s).unwrap()) << m));
            *measured.entry(value).or_insert(0usize) += 1;
        }
        for (&record, &probability) in &expected {
            assert!(
                (measured.get(&record).copied().unwrap_or(0) as f64 / samples.shots() as f64
                    - probability)
                    .abs()
                    < 0.016,
                "record {record}: expected {probability}, observed {:?}",
                measured.get(&record)
            );
        }
        assert!(measured.keys().all(|key| expected.contains_key(key)));
    }
}
#[test]
fn reference_samples_replay_across_shards_with_checked_padding_and_budgets() {
    let state = Tableau::zero(2, TableauLimits::default()).unwrap();
    let obs = vec![
        p(&[(0, Pauli::X)]),
        p(&[(0, Pauli::Z)]),
        p(&[(1, Pauli::Y)]),
    ];
    let sampler = ReferenceSampler::compile(&state, &obs, 6).unwrap();
    let noise = PauliNoise::new(0.2, 0.3, 0.1).unwrap();
    let all = sampler.sample(17, 140, 99, noise, 4096).unwrap();
    for (start, len) in [(17, 37), (54, 103)] {
        let shard = sampler.sample(start, len, 99, noise, 4096).unwrap();
        for m in 0..3 {
            for s in 0..len {
                assert_eq!(
                    shard.bit(m, s).unwrap(),
                    all.bit(m, start as usize - 17 + s).unwrap()
                );
            }
            assert_eq!(shard.rows()[m].last().unwrap() >> (len % 64), 0);
        }
    }
    assert!(sampler.sample(u64::MAX, 1, 0, noise, 4096).is_err());
    assert!(sampler.sample(0, 1, 0, noise, 1).is_err());
    assert!(ReferenceSampler::compile(&state, &obs, 5).is_err());
    assert!(PauliNoise::new(0.5, 0.6, 0.).is_err());
    assert!(PauliNoise::new(f64::NAN, 0., 0.).is_err());
    assert!(all.bit(3, 0).is_err());
}
#[test]
fn signed_fusion_relations_infer_swapped_bell_outputs_and_handle_erasures() {
    let mut state = Tableau::zero(4, TableauLimits::default()).unwrap();
    for (a, b) in [(0, 1), (2, 3)] {
        state.h(a).unwrap();
        state.cnot(a, b).unwrap();
    }
    let obs = vec![
        p(&[(1, Pauli::X), (2, Pauli::X)]),
        p(&[(1, Pauli::Z), (2, Pauli::Z)]),
    ];
    let relations = MeasurementRelations::derive(&state, &obs, 100).unwrap();
    assert!(relations.checks().is_empty());
    let xx = p(&[(0, Pauli::X), (3, Pauli::X)]);
    let yy = p(&[(0, Pauli::Y), (3, Pauli::Y)]);
    let zz = p(&[(0, Pauli::Z), (3, Pauli::Z)]);
    for a in [false, true] {
        for b in [false, true] {
            assert_eq!(
                relations.infer(&[Some(a), Some(b)], &xx).unwrap(),
                ParityInference::Known(if a { Parity::Odd } else { Parity::Even })
            );
            assert_eq!(
                relations.infer(&[Some(a), Some(b)], &yy).unwrap(),
                ParityInference::Known(if !(a ^ b) { Parity::Odd } else { Parity::Even })
            );
            assert_eq!(
                relations.infer(&[None, Some(b)], &xx).unwrap(),
                ParityInference::Erased
            );
            assert_eq!(
                relations.infer(&[None, Some(b)], &zz).unwrap(),
                ParityInference::Known(if b { Parity::Odd } else { Parity::Even })
            );
        }
    }
    assert_eq!(
        relations
            .infer(&[None, None], &p(&[(0, Pauli::Z)]))
            .unwrap(),
        ParityInference::Unconstrained
    );
    let bell_obs = vec![
        p(&[(0, Pauli::X), (1, Pauli::X)]),
        p(&[(0, Pauli::Y), (1, Pauli::Y)]),
        p(&[(0, Pauli::Z), (1, Pauli::Z)]),
    ];
    let known = MeasurementRelations::derive(&state, &bell_obs, 100).unwrap();
    assert_eq!(known.checks().len(), 3);
    assert_eq!(
        known.infer(&[None, None, None], &bell_obs[1]).unwrap(),
        ParityInference::Known(Parity::Odd)
    );
    assert_eq!(
        known
            .infer(&[Some(true), None, None], &bell_obs[1])
            .unwrap(),
        ParityInference::Inconsistent
    );
    assert!(
        MeasurementRelations::derive(&state, &[p(&[(0, Pauli::X)]), p(&[(0, Pauli::Z)])], 100)
            .is_err()
    );
}
#[test]
fn continuation_preserves_all_frames_across_json_restart_and_bounds_before_mutation() {
    let mut initial = Tableau::zero(8, TableauLimits::default()).unwrap();
    initial.h(0).unwrap();
    for q in 1..8 {
        initial.cnot(0, q).unwrap();
    }
    let obs: Vec<_> = (0..24)
        .map(|i| p(&[(i % 8, [Pauli::X, Pauli::Y, Pauli::Z][i / 8])]))
        .collect();
    let sampler = ReferenceSampler::compile(&initial, &obs, 1000).unwrap();
    let noise = PauliNoise::new(0.1, 0.1, 0.1).unwrap();
    let all = sampler.sample(7, 129, 11, noise, 100000).unwrap();
    let mut session = sampler.start(7, 129, 11, noise, 100000).unwrap();
    let mut windows = vec![];
    for chunk in [3, 9, 1, 11] {
        let before = serde_json::to_string(&session.checkpoint()).unwrap();
        assert!(session.advance(chunk, 1).is_err());
        assert_eq!(
            serde_json::to_string(&session.checkpoint()).unwrap(),
            before
        );
        let batch = session.advance(chunk, 100000).unwrap();
        windows.extend(batch.rows().iter().cloned());
        let json = serde_json::to_string(&session.checkpoint()).unwrap();
        session = sampler
            .resume(serde_json::from_str(&json).unwrap(), 100000)
            .unwrap();
    }
    assert_eq!(windows, all.rows());
    assert_eq!(session.position(), 24);
    assert_eq!(session.frame_bytes(), 8 * 2 * 3 * 8);
    assert_eq!(session.advance(4, 100000).unwrap().measurements(), 0);
    let checkpoint = session.checkpoint();
    let other = ReferenceSampler::compile(&initial, &obs[..23], 1000).unwrap();
    assert!(other.resume(checkpoint.clone(), 100000).is_err());
    let mut json = serde_json::to_value(checkpoint).unwrap();
    json["x"][0][2] = serde_json::json!(2);
    assert!(
        sampler
            .resume(serde_json::from_value(json).unwrap(), 100000)
            .is_err()
    );
}
