//! Bounded unknown-solution-count search and verified logical outcomes.
use super::GroverSat;
use crate::{Result, Samples};
use burn::tensor::Device;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Reproducible limits for randomized search with an unknown solution count.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Maximum preparations/measurements. Zero permits only trivial logical results.
    pub max_trials: usize,
    /// Total Grover oracle applications across all trials; zero permits uniform sampling.
    pub max_oracle_calls: usize,
    /// Seed for iteration selection and terminal measurement, independent of Burn's RNG.
    pub seed: u64,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_trials: 64,
            max_oracle_calls: 4096,
            seed: 0,
        }
    }
}

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
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
        let mut window = 1.0f64;
        let cap = (2.0f64).powf(self.formula().num_variables() as f64 / 2.0);
        for _ in 0..config.max_trials {
            let iterations = rng.random_range(0..window.ceil() as usize);
            if iterations > config.max_oracle_calls - report.oracle_calls {
                break;
            }
            let state = self.iterate(uniform.clone(), iterations)?;
            report.oracle_calls += iterations;
            let samples = Samples::from_state(&state, 1, rng.random(), 1)?;
            report.trials += 1;
            let digits = state.register().decode(samples.outcomes[0][0])?;
            let assignment: Vec<bool> = digits[..self.formula().num_variables()]
                .iter()
                .map(|&x| x == 1)
                .collect();
            if self.formula().evaluate(&assignment)? {
                report.outcome = SatOutcome::Satisfiable { assignment };
                return Ok(report);
            }
            window = (window * 1.2).min(cap);
        }
        Ok(report)
    }
}
