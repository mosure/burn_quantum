//! Search conformance using independent references.
use super::*;

#[test]
fn unknown_count_search_returns_only_verified_reproducible_witnesses() {
    for clauses in [
        vec![vec![1], vec![-2], vec![3]],
        vec![vec![1, -2], vec![2, 3]],
    ] {
        let sat = solver(3, clauses);
        for seed in [0, 7, 23] {
            let config = SearchConfig {
                seed,
                max_trials: 64,
                max_oracle_calls: 256,
            };
            let result = sat.search(config, &device()).unwrap();
            let SatOutcome::Satisfiable { ref assignment } = result.outcome else {
                panic!("{result:?}");
            };
            assert!(sat.formula().evaluate(assignment).unwrap());
            assert!(result.trials > 0 && result.trials <= config.max_trials);
            assert!(result.oracle_calls <= config.max_oracle_calls);
            assert_eq!(result, sat.search(config, &device()).unwrap());
        }
    }
}

#[test]
fn budgets_never_turn_failed_search_into_an_unsat_claim() {
    let unsat = solver(1, vec![vec![1], vec![-1]]);
    for config in [
        SearchConfig {
            max_trials: 0,
            max_oracle_calls: 64,
            seed: 7,
        },
        SearchConfig {
            max_trials: 8,
            max_oracle_calls: 0,
            seed: 7,
        },
        SearchConfig {
            max_trials: 8,
            max_oracle_calls: 3,
            seed: 7,
        },
    ] {
        let report = unsat.search(config, &device()).unwrap();
        assert_eq!(report.outcome, SatOutcome::Inconclusive);
        assert!(
            report.oracle_calls <= config.max_oracle_calls && report.trials <= config.max_trials
        );
    }
    let config = SearchConfig {
        max_trials: 0,
        ..Default::default()
    };
    assert_eq!(
        solver(2, vec![vec![1]])
            .search(config, &device())
            .unwrap()
            .outcome,
        SatOutcome::Inconclusive
    );
    assert_eq!(
        solver(0, vec![vec![]])
            .search(config, &device())
            .unwrap()
            .outcome,
        SatOutcome::Unsatisfiable
    );
    assert_eq!(
        solver(0, vec![]).search(config, &device()).unwrap().outcome,
        SatOutcome::Satisfiable { assignment: vec![] }
    );
    assert_eq!(
        solver(2, vec![vec![1, -1]])
            .search(config, &device())
            .unwrap()
            .outcome,
        SatOutcome::Satisfiable {
            assignment: vec![false, false]
        }
    );
}
