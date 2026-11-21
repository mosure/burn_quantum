use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Reproducible budgets for randomized search with an unknown success probability.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Maximum independent preparations, measurements and candidate checks.
    pub max_trials: usize,
    /// Maximum complete good-space phase applications across all trials.
    pub max_oracle_calls: usize,
    /// Seed for both round selection and per-trial measurement seeds.
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

/// A classically verified candidate, or an inconclusive bounded search, with actual work.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExponentialSearchResult<T> {
    /// First candidate accepted by the supplied verifier.
    pub witness: Option<T>,
    /// Number of sampled and verified candidates.
    pub trials: usize,
    /// Total executed amplification rounds, excluding classical verification.
    pub oracle_calls: usize,
}

/// Boyer–Brassard–Høyer–Tapp randomized exponential search. Start at window 1,
/// grow by 6/5 after each failure. Each trial
/// draws j uniformly from the integers 0 <= j < window. The sampler must prepare
/// afresh, apply j standard rounds and return one measured candidate using its seed.
///
/// For N uniform candidates, `Some(1.0 / N)` caps the window at sqrt(N).
/// More generally, a promised lower bound b on *nonzero* good mass caps it at
/// 1/sqrt(b). Pass `None` for an arbitrary preparation with unknown overlap.
/// In that case only the caller's budgets bound the search, since good mass can
/// be arbitrarily small even on a single qubit. Exhaustion returns `witness: None`.
pub fn exponential_search<T>(
    minimum_nonzero_mass: Option<f64>,
    config: SearchConfig,
    mut sample: impl FnMut(usize, u64) -> Result<T>,
    mut verify: impl FnMut(&T) -> Result<bool>,
) -> Result<ExponentialSearchResult<T>> {
    let cap = match minimum_nonzero_mass {
        Some(p) if 0. < p && p <= 1. => 1. / p.sqrt(),
        Some(_) => {
            return Err(Error::InvalidValue(
                "nonzero mass bound must be in (0, 1]".into(),
            ));
        }
        None => config.max_oracle_calls.saturating_add(1) as f64,
    };
    let mut result = ExponentialSearchResult {
        witness: None,
        trials: 0,
        oracle_calls: 0,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut window = 1_f64;
    for _ in 0..config.max_trials {
        let rounds = rng.random_range(0..window.ceil() as usize);
        if rounds > config.max_oracle_calls - result.oracle_calls {
            break;
        }
        let candidate = sample(rounds, rng.random())?;
        result.trials += 1;
        result.oracle_calls += rounds;
        if verify(&candidate)? {
            result.witness = Some(candidate);
            break;
        }
        window = (window * 1.2).min(cap);
    }
    Ok(result)
}
