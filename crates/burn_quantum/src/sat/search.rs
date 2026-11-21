//! Bounded unknown-solution-count search and verified logical outcomes.
use super::GroverSat;
pub use crate::algorithms::amplification::SearchConfig;
use crate::algorithms::amplification::exponential_search;
use crate::{Result, Samples};
use burn::tensor::Device;
use serde::{Deserialize, Serialize};

/// A verified witness, a trivial contradiction, or an exhausted search budget.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SatOutcome {
    /// Every clause was checked against this assignment.
    Satisfiable {
        /// Values for variables 1 through n; variable 1 is the most significant qubit.
        assignment: Vec<bool>,
    },
    /// Proven by an explicit empty clause. General failed searches never use this status.
    Unsatisfiable,
    /// No witness found within the budget; this is not an UNSAT proof or confidence bound.
    Inconclusive,
}

/// Search outcome and executed work; preprocessing and classical verification are separate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchReport {
    /// Verified logical outcome.
    #[serde(flatten)]
    pub outcome: SatOutcome,
    /// Number of sampled candidates (and classical candidate checks).
    pub trials: usize,
    /// Number of complete compute/phase/uncompute oracle applications.
    pub oracle_calls: usize,
    /// Local random stream seed.
    pub seed: u64,
}

impl GroverSat {
    /// Bounded BBHT-style search: choose j uniformly from integers below m,
    /// measure after j iterations, and increase m by 6/5 up to sqrt(2^n).
    /// Does not need the number of satisfying assignments. No classical exhaustive
    /// fallback is used. Exceeding either budget yields `Inconclusive`.
    pub fn search(&self, config: SearchConfig, device: &Device) -> Result<SearchReport> {
        let mut report = SearchReport {
            outcome: SatOutcome::Inconclusive,
            trials: 0,
            oracle_calls: 0,
            seed: config.seed,
        };
        if self.formula().clauses().iter().any(Vec::is_empty) {
            report.outcome = SatOutcome::Unsatisfiable;
            return Ok(report);
        }
        if self.formula().clauses().is_empty() {
            report.outcome = SatOutcome::Satisfiable {
                assignment: vec![false; self.formula().num_variables()],
            };
            return Ok(report);
        }
        if config.max_trials == 0 {
            return Ok(report);
        }
        let uniform = self.initial_state(device)?;
        let variables = self.formula().num_variables();
        let search = exponential_search(
            Some(1. / (1usize << variables) as f64),
            config,
            |rounds, seed| {
                let state = self.iterate(uniform.clone(), rounds)?;
                let samples = Samples::from_state(&state, 1, seed, 1)?;
                let digits = state.register().decode(samples.outcomes[0][0])?;
                Ok(digits[..variables]
                    .iter()
                    .map(|&x| x == 1)
                    .collect::<Vec<bool>>())
            },
            |assignment| self.formula().evaluate(assignment),
        )?;
        report.trials = search.trials;
        report.oracle_calls = search.oracle_calls;
        if let Some(assignment) = search.witness {
            report.outcome = SatOutcome::Satisfiable { assignment };
        }
        Ok(report)
    }
}
