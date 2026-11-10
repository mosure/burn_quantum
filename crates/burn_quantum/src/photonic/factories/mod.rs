//! Bounded heralded source attempts and capacity-constrained classical scheduling.
//! Models availability/yield; quantum resource quality is supplied separately by the caller.
mod bank;
mod schedule;
use super::runtime::shot_seed;
use crate::{Error, Result};
pub use bank::{CorrelatedFactoryBank, FactoryBankReport, FactoryMode};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
pub use schedule::{CapacityCalendar, Reservation};
use serde::{Deserialize, Serialize};

/// Independent heralded resource factory with a fixed cycle time and attempt budget.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "FactoryData", into = "FactoryData")]
pub struct ResourceFactory(FactoryData);
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FactoryData {
    success: f64,
    photons_per_attempt: usize,
    cycle_ticks: u64,
    max_attempts: usize,
}
/// One bounded factory invocation. Failure is not a delayed successful resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactoryReport {
    /// Number of physical generation attempts made.
    pub attempts: usize,
    /// Total photons consumed by the source attempts.
    pub photons_consumed: usize,
    /// Time at which generation stops, successful or exhausted.
    pub finish_tick: u64,
    /// Resource availability, or None on exhausted generation budget.
    pub available_at: Option<u64>,
}
impl ResourceFactory {
    /// Construct a finite, independent-attempt source model.
    pub fn new(
        success: f64,
        photons_per_attempt: usize,
        cycle_ticks: u64,
        max_attempts: usize,
    ) -> Result<Self> {
        Self::try_from(FactoryData {
            success,
            photons_per_attempt,
            cycle_ticks,
            max_attempts,
        })
    }
    /// Probability of obtaining at least one resource before exhausting the attempt budget.
    pub fn yield_probability(&self) -> f64 {
        -((self.0.max_attempts as f64) * (-self.0.success).ln_1p()).exp_m1()
    }
    /// Expected physical attempts, including the final exhausted attempt.
    pub fn expected_attempts(&self) -> f64 {
        if self.0.success == 0. {
            self.0.max_attempts as f64
        } else {
            self.yield_probability() / self.0.success
        }
    }
    /// Maximum elapsed duration of one invocation, including all exhausted attempts.
    /// Construction and deserialization validate that this product fits in `u64`.
    pub fn worst_case_ticks(&self) -> u64 {
        self.0.cycle_ticks * self.0.max_attempts as u64
    }
    /// Supplied photons consumed by each generation attempt, including failed attempts.
    pub fn photons_per_attempt(&self) -> usize {
        self.0.photons_per_attempt
    }
    /// Mean supplied photons consumed before success or budget exhaustion.
    pub fn expected_photons(&self) -> f64 {
        self.expected_attempts() * self.0.photons_per_attempt as f64
    }
    /// Maximum photon consumption when all generation attempts are exhausted.
    pub fn max_photons(&self) -> usize {
        self.0.photons_per_attempt * self.0.max_attempts
    }
    /// Sample an invocation with global shot and factory identities; enforce its deadline.
    pub fn generate(
        &self,
        seed: u64,
        shot: u64,
        factory: u64,
        start: u64,
        deadline: u64,
    ) -> Result<FactoryReport> {
        let budget_end = start
            .checked_add(self.worst_case_ticks())
            .ok_or(Error::Overflow)?;
        if budget_end > deadline {
            return Err(Error::InvalidValue(
                "factory worst-case completion exceeds deadline".into(),
            ));
        }
        // FACTORY1 separates source randomness from simulator/component streams.
        let domain = shot_seed(seed, 0, 0x4641_4354_4f52_5931);
        let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(domain, shot, factory));
        for attempt in 1..=self.0.max_attempts {
            if rng.random::<f64>() < self.0.success {
                let tick = start + self.0.cycle_ticks * attempt as u64;
                return Ok(FactoryReport {
                    attempts: attempt,
                    photons_consumed: attempt * self.0.photons_per_attempt,
                    finish_tick: tick,
                    available_at: Some(tick),
                });
            }
        }
        Ok(FactoryReport {
            attempts: self.0.max_attempts,
            photons_consumed: self.0.max_attempts * self.0.photons_per_attempt,
            finish_tick: budget_end,
            available_at: None,
        })
    }
}
impl TryFrom<FactoryData> for ResourceFactory {
    type Error = Error;
    fn try_from(v: FactoryData) -> Result<Self> {
        if !v.success.is_finite()
            || !(0.0..=1.0).contains(&v.success)
            || v.photons_per_attempt == 0
            || v.cycle_ticks == 0
            || v.max_attempts == 0
            || v.max_attempts > 1_048_576
        {
            return Err(Error::InvalidValue(
                "invalid factory probability, duration or budget".into(),
            ));
        }
        v.photons_per_attempt
            .checked_mul(v.max_attempts)
            .ok_or(Error::Overflow)?;
        v.cycle_ticks
            .checked_mul(v.max_attempts as u64)
            .ok_or(Error::Overflow)?;
        Ok(Self(v))
    }
}
impl From<ResourceFactory> for FactoryData {
    fn from(v: ResourceFactory) -> Self {
        v.0
    }
}
