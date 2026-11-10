//! Open-boundary code, physical error/erasure correction and logical channel tests.
use burn_quantum::Result;
use burn_quantum::photonic::qec::{PlanarCode, PlanarDecodeBudget, PlanarRecovery};
use burn_quantum::photonic::stabilizer::{Pauli, PauliProduct, Tableau, TableauLimits};
fn product(parts: &[PauliProduct]) -> Result<PauliProduct> {
    PauliProduct::new(
        parts
            .iter()
            .flat_map(|p| p.terms().iter().copied())
            .collect(),
    )
}
fn singleton(q: usize, p: Pauli) -> PauliProduct {
    PauliProduct::new(vec![(q, p)]).unwrap()
}
fn bell(code: &PlanarCode, offset: usize, reference: usize, state: &mut Tableau) -> Result<()> {
    state.h(reference)?;
    state.cnot(reference, offset + code.input_site())?;
    code.encode(state, offset)?;
    Ok(())
}
fn assert_code(code: &PlanarCode, state: &Tableau, offset: usize) -> Result<()> {
    for p in code
        .x_checks(offset)?
        .into_iter()
        .chain(code.z_checks(offset)?)
    {
        assert_eq!(state.expectation(&p)?, 1);
    }
    Ok(())
}
fn assert_bell(code: &PlanarCode, state: &Tableau, offset: usize, reference: usize) -> Result<()> {
    assert_code(code, state, offset)?;
    assert_eq!(
        state.expectation(&product(&[
            code.logical_x(offset)?,
            singleton(reference, Pauli::X)
        ])?)?,
        1
    );
    assert_eq!(
        state.expectation(&product(&[
            code.logical_z(offset)?,
            singleton(reference, Pauli::Z)
        ])?)?,
        1
    );
    Ok(())
}
#[test]
fn open_boundaries_encode_unknown_states_and_have_distance_three() -> Result<()> {
    for d in [3, 5, 7] {
        let code = PlanarCode::new(d, 100)?;
        let n = code.qubits();
        assert_eq!(n, d * d + (d - 1) * (d - 1));
        let mut state = Tableau::zero(n + 1, TableauLimits::default())?;
        bell(&code, 0, n, &mut state)?;
        assert_bell(&code, &state, 0, n)?;
        let x = code.x_checks(0)?;
        let z = code.z_checks(0)?;
        assert_eq!(x.len() + z.len(), n - 1);
        for a in &x {
            assert!((3..=4).contains(&a.terms().len()));
            for b in &z {
                assert!(!a.anticommutes(b));
            }
        }
        assert!(code.logical_x(0)?.anticommutes(&code.logical_z(0)?));
        assert_eq!(code.logical_x(0)?.terms().len(), d);
        assert_eq!(code.logical_z(0)?.terms().len(), d);
    }
    let code = PlanarCode::new(3, 13)?;
    for parity in [code.x_error_code()?, code.z_error_code()?] {
        for a in 0..13 {
            for b in a..13 {
                let mut e = vec![false; 13];
                e[a] = true;
                e[b] = true;
                if parity.syndrome(&e)?.iter().all(|&s| !s) {
                    assert!(!parity.logical_flips(&e)?[0]);
                }
            }
        }
    }
    assert!(PlanarCode::new(2, 100).is_err());
    assert!(PlanarCode::new(5, 40).is_err());
    Ok(())
}
#[test]
fn all_single_pauli_faults_and_every_correctable_two_site_erasure_preserve_reference() -> Result<()>
{
    let code = PlanarCode::new(3, 13)?;
    let n = code.qubits();
    let mut initial = Tableau::zero(n + 2, TableauLimits::default())?;
    bell(&code, 0, n, &mut initial)?;
    let budget = PlanarDecodeBudget {
        max_weight: 1,
        max_candidates: 100,
    };
    for q in 0..n {
        for p in [Pauli::X, Pauli::Y, Pauli::Z] {
            let mut state = initial.clone();
            if p != Pauli::Z {
                state.x(q)?;
            }
            if p != Pauli::X {
                state.z(q)?;
            }
            let (x, z, cost) = code.measure_syndrome(&mut state, 0, n + 1, || false)?;
            assert_eq!(cost.measurements, n - 1);
            assert_eq!(cost.preparations, n - 1);
            assert_eq!(cost.cnots, 40);
            let recovery = code.decode(&x, &z, &vec![false; n], budget)?;
            assert!(code.correct(&mut state, 0, &recovery)?);
            assert_bell(&code, &state, 0, n)?;
        }
    }
    for a in 0..n {
        for b in a..n {
            for hidden in [false, true] {
                let mut state = initial.clone();
                let mut erased = vec![false; n];
                erased[a] = true;
                erased[b] = true;
                let cost = code.replace_erasures(&mut state, 0, &erased, || hidden)?;
                assert_eq!(cost.preparations, 1 + usize::from(a != b));
                let (x, z, _) = code.measure_syndrome(&mut state, 0, n + 1, || hidden)?;
                let recovery = code.decode(
                    &x,
                    &z,
                    &erased,
                    PlanarDecodeBudget {
                        max_weight: 0,
                        max_candidates: 1,
                    },
                )?;
                assert!(code.correct(&mut state, 0, &recovery)?);
                assert_bell(&code, &state, 0, n)?;
            }
        }
    }
    let mut erased = vec![false; n];
    for &(q, _) in code.logical_z(0)?.terms() {
        erased[q] = true;
    }
    assert_eq!(
        code.decode(&[false; 6], &[false; 6], &erased, budget)?,
        PlanarRecovery::Ambiguous
    );
    assert_eq!(
        code.decode(
            &[false; 6],
            &[false; 6],
            &vec![false; n],
            PlanarDecodeBudget {
                max_weight: 1,
                max_candidates: 0
            }
        )?,
        PlanarRecovery::BudgetExceeded
    );
    Ok(())
}
#[test]
fn logical_h_cnot_and_both_phase_teleportation_branches_have_exact_choi_correlations() -> Result<()>
{
    for d in [3, 5] {
        let code = PlanarCode::new(d, 100)?;
        let n = code.qubits();
        let mut state = Tableau::zero(n + 1, TableauLimits::default())?;
        bell(&code, 0, n, &mut state)?;
        let cost = code.logical_h(&mut state, 0)?;
        assert_eq!(cost.hadamards, n);
        assert_eq!(cost.cnots, 3 * cost.swaps);
        assert_code(&code, &state, 0)?;
        assert_eq!(
            state.expectation(&product(&[code.logical_z(0)?, singleton(n, Pauli::X)])?)?,
            1
        );
        assert_eq!(
            state.expectation(&product(&[code.logical_x(0)?, singleton(n, Pauli::Z)])?)?,
            1
        );
        code.logical_h(&mut state, 0)?;
        assert_bell(&code, &state, 0, n)?;
        let mut two = Tableau::zero(2 * n + 2, TableauLimits::default())?;
        bell(&code, 0, 2 * n, &mut two)?;
        bell(&code, n, 2 * n + 1, &mut two)?;
        assert_eq!(code.logical_cnot(&mut two, 0, n)?.cnots, n);
        assert_code(&code, &two, 0)?;
        assert_code(&code, &two, n)?;
        for terms in [
            vec![
                code.logical_x(0)?,
                code.logical_x(n)?,
                singleton(2 * n, Pauli::X),
            ],
            vec![code.logical_z(0)?, singleton(2 * n, Pauli::Z)],
            vec![code.logical_x(n)?, singleton(2 * n + 1, Pauli::X)],
            vec![
                code.logical_z(0)?,
                code.logical_z(n)?,
                singleton(2 * n + 1, Pauli::Z),
            ],
        ] {
            assert_eq!(two.expectation(&product(&terms)?)?, 1);
        }
        let mut branches = [0; 2];
        for pattern in 0..32u64 {
            let mut state = Tableau::zero(2 * n + 1, TableauLimits::default())?;
            bell(&code, 0, 2 * n, &mut state)?;
            let mut cursor = 0;
            let (m, cost) = code.logical_s(&mut state, 0, n, || {
                let b = (pattern >> (cursor % 5)) & 1 == 1;
                cursor += 1;
                b
            })?;
            branches[usize::from(m)] += 1;
            assert_eq!(cost.preparations, n);
            assert_eq!(cost.measurements, n);
            assert_code(&code, &state, 0)?;
            let mut yterms = std::collections::BTreeMap::new();
            for &(q, _) in code.logical_x(0)?.terms() {
                yterms.insert(q, Pauli::X);
            }
            for &(q, _) in code.logical_z(0)?.terms() {
                yterms
                    .entry(q)
                    .and_modify(|p| *p = Pauli::Y)
                    .or_insert(Pauli::Z);
            }
            let logical_y = PauliProduct::new(yterms.into_iter().collect())?;
            assert_eq!(
                state.expectation(&product(&[logical_y, singleton(2 * n, Pauli::X)])?)?,
                1
            );
            assert_eq!(
                state.expectation(&product(&[code.logical_z(0)?, singleton(2 * n, Pauli::Z)])?)?,
                1
            );
        }
        assert!(branches.iter().all(|&n| n > 0));
    }
    Ok(())
}

#[test]
fn noisy_multiround_memory_accounts_for_loss_costs_and_replays_shards() -> Result<()> {
    use burn_quantum::photonic::{
        RailTransmission,
        qec::{PlanarMemoryNoise, run_planar_memory},
        stabilizer::PauliNoise,
    };
    let code = PlanarCode::new(3, 13)?;
    let budget = PlanarDecodeBudget {
        max_weight: 1,
        max_candidates: 100,
    };
    let ideal = run_planar_memory(
        &code,
        3,
        PlanarMemoryNoise::new(0., PauliNoise::NONE)?,
        0,
        8,
        51,
        budget,
    )?;
    assert_eq!(ideal.outcomes.corrected, 8);
    assert_eq!(ideal.rounds, 24);
    assert_eq!(ideal.cost.measurements, 24 * 12);
    assert_eq!(ideal.cost.preparations, 8 * 13 + 24 * 12);
    assert_eq!(ideal.cost.cnots, 8 * code.encoding_cnots().len() + 24 * 40);
    let lost = run_planar_memory(
        &code,
        3,
        PlanarMemoryNoise::new(1., PauliNoise::NONE)?,
        0,
        8,
        51,
        budget,
    )?;
    assert_eq!(lost.outcomes.erased_logicals, 8);
    assert_eq!(lost.rounds, 8);
    assert_eq!(lost.replacements, 8 * 13);
    let noise = PlanarMemoryNoise::storage(
        RailTransmission::new(0.99, 0.99)?,
        2,
        PauliNoise::new(0.015, 0.015, 0.015)?,
    )?;
    let all = run_planar_memory(&code, 3, noise, 70, 64, 51, budget)?;
    let a = run_planar_memory(&code, 3, noise, 70, 17, 51, budget)?;
    let b = run_planar_memory(&code, 3, noise, 87, 47, 51, budget)?;
    assert_eq!(
        all.outcomes.corrected,
        a.outcomes.corrected + b.outcomes.corrected
    );
    assert_eq!(
        all.outcomes.logical_errors,
        a.outcomes.logical_errors + b.outcomes.logical_errors
    );
    assert_eq!(
        all.outcomes.erased_logicals,
        a.outcomes.erased_logicals + b.outcomes.erased_logicals
    );
    assert_eq!(
        all.outcomes.decoder_budget,
        a.outcomes.decoder_budget + b.outcomes.decoder_budget
    );
    assert_eq!(all.cost.cnots, a.cost.cnots + b.cost.cnots);
    assert_eq!(all.replacements, a.replacements + b.replacements);
    assert_eq!(
        all.outcomes.attempts,
        all.outcomes.corrected
            + all.outcomes.logical_errors
            + all.outcomes.erased_logicals
            + all.outcomes.decoder_budget
    );
    assert!(
        PlanarMemoryNoise::storage(RailTransmission::new(0.9, 0.8)?, 1, PauliNoise::NONE).is_err()
    );
    Ok(())
}

#[test]
fn encoded_phase_gate_corrects_every_single_ancilla_pauli_fault_and_erased_readout() -> Result<()> {
    use burn_quantum::photonic::qec::PlanarReadout;
    let code = PlanarCode::new(3, 13)?;
    let n = code.qubits();
    let budget = PlanarDecodeBudget {
        max_weight: 1,
        max_candidates: 100,
    };
    for q in 0..n {
        for p in [Pauli::X, Pauli::Y, Pauli::Z] {
            let mut state = Tableau::zero(2 * n + 2, TableauLimits::default())?;
            bell(&code, 0, 2 * n, &mut state)?;
            state.h(n + code.input_site())?;
            state.s(n + code.input_site())?;
            code.encode(&mut state, n)?;
            if p != Pauli::Z {
                state.x(n + q)?;
            }
            if p != Pauli::X {
                state.z(n + q)?;
            }
            code.logical_cnot(&mut state, 0, n)?;
            let (PlanarReadout::Known(negative), cost) =
                code.readout_z(&mut state, n, &vec![false; n], budget, || q % 2 == 1)?
            else {
                panic!("single ancillary fault rejected")
            };
            assert_eq!(cost.measurements, n);
            if negative {
                for &(q, _) in code.logical_z(0)?.terms() {
                    state.z(q)?;
                }
            }
            let (x, z, _) = code.measure_syndrome(&mut state, 0, 2 * n + 1, || false)?;
            let recovery = code.decode(&x, &z, &vec![false; n], budget)?;
            assert!(code.correct(&mut state, 0, &recovery)?);
            // Inverse logical S uses the same noiseless primitive three times.
            // Instead inspect Y_L X_ref and Z_L Z_ref directly, with one overlap.
            let mut terms = std::collections::BTreeMap::new();
            for &(q, _) in code.logical_x(0)?.terms() {
                terms.insert(q, Pauli::X);
            }
            for &(q, _) in code.logical_z(0)?.terms() {
                terms
                    .entry(q)
                    .and_modify(|p| *p = Pauli::Y)
                    .or_insert(Pauli::Z);
            }
            terms.insert(2 * n, Pauli::X);
            assert_eq!(
                state.expectation(&PauliProduct::new(terms.into_iter().collect())?)?,
                1
            );
            assert_eq!(
                state.expectation(&product(&[code.logical_z(0)?, singleton(2 * n, Pauli::Z)])?)?,
                1
            );
        }
    }
    for a in 0..n {
        for b in a..n {
            let mut state = Tableau::zero(n + 1, TableauLimits::default())?;
            bell(&code, 0, n, &mut state)?;
            let mut erased = vec![false; n];
            erased[a] = true;
            erased[b] = true;
            let (PlanarReadout::Known(bit), cost) = code.readout_z(
                &mut state,
                0,
                &erased,
                PlanarDecodeBudget {
                    max_weight: 0,
                    max_candidates: 1,
                },
                || true,
            )?
            else {
                panic!("two-erasure readout rejected")
            };
            assert_eq!(cost.measurements, n - 1 - usize::from(a != b));
            assert_eq!(
                state.expectation(&singleton(n, Pauli::Z))?,
                if bit { -1 } else { 1 }
            );
        }
    }
    Ok(())
}

#[test]
fn invalid_destructive_readout_budget_does_not_consume_the_state() -> Result<()> {
    use burn_quantum::photonic::qec::PlanarDecodeBudget;
    let code = PlanarCode::new(3, 13)?;
    let mut state = Tableau::zero(14, TableauLimits::default())?;
    bell(&code, 0, 13, &mut state)?;
    assert!(
        code.readout_z(
            &mut state,
            0,
            &[false; 13],
            PlanarDecodeBudget {
                max_weight: 14,
                max_candidates: 100
            },
            || false
        )
        .is_err()
    );
    assert_bell(&code, &state, 0, 13)?;
    Ok(())
}
