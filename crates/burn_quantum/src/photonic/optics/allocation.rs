//! Bounded adaptive minibatch shot allocation and complete sampling progress.
use super::ShotEstimate;
use crate::{Error, Result};
use serde::Serialize;
/// One item's completed unconditional readout statistics and objective coefficient.
#[derive(Clone, Copy, Debug)]
pub struct ShotAllocationItem {
    /// Completed physical attempts, including no clicks.
    pub completed: usize,
    /// Estimated per-shot variance, not variance of the current mean.
    pub variance: f64,
    /// Coefficient of this item's mean in the declared scalar objective.
    pub coefficient: f64,
}
/// Box-constrained Neyman allocation for the next finite minibatch measurement round.
#[derive(Clone, Copy, Debug)]
pub struct ShotAllocationPolicy {
    /// Minimum planned total for each item, including its existing completed shots.
    pub minimum_per_item: usize,
    /// Maximum total shots per item.
    pub maximum_per_item: usize,
    /// Maximum new attempts this round, across all items.
    pub round_budget: usize,
    /// Maximum cumulative attempts, including earlier completed measurements.
    pub total_budget: usize,
    /// Positive variance floor prevents zero pilot variance from permanently starving an item.
    pub variance_floor: f64,
}
/// A reservation plan: planned totals are not falsely reported as completed experiments.
#[derive(Clone, Debug, Serialize)]
pub struct ShotAllocation {
    /// Already completed attempts at allocation time, per item.
    pub completed: Vec<usize>,
    /// New attempts to execute, per item.
    pub additional: Vec<usize>,
    /// Planned totals after the reserved experiments finish.
    pub planned_totals: Vec<usize>,
    /// Sum of new reservations.
    pub reserved_attempts: usize,
    /// Plug-in standard error predicted for the scalar objective after execution.
    /// Pilot estimates/adaptive stopping mean this is not a rigorous confidence interval.
    pub estimated_standard_error: f64,
}
impl ShotAllocationPolicy {
    /// Allocate using completed counts and pilot variances; reject infeasible minima atomically.
    /// Solves the continuous allocation by bisection, then rounds within the same hard budgets.
    pub fn allocate(&self, items: &[ShotAllocationItem]) -> Result<ShotAllocation> {
        if items.is_empty()
            || items.len() > 65_536
            || self.minimum_per_item < 2
            || self.maximum_per_item < self.minimum_per_item
            || self.round_budget == 0
            || self.total_budget as u64 > 1u64 << 40
            || !self.variance_floor.is_finite()
            || self.variance_floor <= 0.
            || items.iter().any(|x| {
                x.completed > self.maximum_per_item
                    || !x.variance.is_finite()
                    || x.variance < 0.
                    || !x.coefficient.is_finite()
            })
        {
            return Err(Error::InvalidValue(
                "invalid minibatch shot allocation".into(),
            ));
        }
        let completed = items
            .iter()
            .try_fold(0usize, |sum, item| sum.checked_add(item.completed))
            .ok_or(Error::Overflow)?;
        if completed > self.total_budget {
            return Err(Error::ResourceLimit {
                resource: "completed optical shots",
                required: completed,
                limit: self.total_budget,
            });
        }
        let base: Vec<_> = items
            .iter()
            .map(|x| x.completed.max(self.minimum_per_item))
            .collect();
        let minimum = base
            .iter()
            .try_fold(0usize, |s, n| s.checked_add(*n))
            .ok_or(Error::Overflow)?;
        let capacity = items
            .len()
            .checked_mul(self.maximum_per_item)
            .ok_or(Error::Overflow)?;
        let available = self
            .round_budget
            .min(self.total_budget - completed)
            .min(capacity - completed);
        let mut target = completed + available;
        if minimum > target {
            return Err(Error::ResourceLimit {
                resource: "minimum minibatch shots",
                required: minimum - completed,
                limit: available,
            });
        }
        let mut weight: Vec<_> = items
            .iter()
            .map(|x| x.coefficient.abs() * x.variance.max(self.variance_floor).sqrt())
            .collect();
        if weight.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidValue(
                "shot allocation weight overflow".into(),
            ));
        }
        let maximum = weight.iter().copied().fold(0f64, f64::max);
        if maximum == 0. {
            weight.fill(1.);
        } else {
            for w in &mut weight {
                *w /= maximum;
            }
            // Zero-coefficient items receive their declared minimum but do not
            // consume surplus after every relevant item reaches its cap.
            let useful_capacity = weight
                .iter()
                .zip(&base)
                .map(|(&w, &n)| if w == 0. { n } else { self.maximum_per_item })
                .sum::<usize>();
            target = target.min(useful_capacity);
        }
        let mut lo = 0.;
        let mut hi = 1.;
        for _ in 0..100 {
            let lambda = (lo + hi) / 2.;
            let sum = weight
                .iter()
                .zip(&base)
                .map(|(w, &n)| (w / lambda).max(n as f64).min(self.maximum_per_item as f64))
                .sum::<f64>();
            if sum > target as f64 {
                lo = lambda;
            } else {
                hi = lambda;
            }
        }
        let mut totals: Vec<_> = weight
            .iter()
            .zip(&base)
            .map(|(w, &n)| ((w / hi).floor() as usize).clamp(n, self.maximum_per_item))
            .collect();
        let sum = totals.iter().sum::<usize>();
        // Bisection floors should undershoot by at most one per unconstrained item.
        if sum > target || target - sum > items.len() {
            return Err(Error::InvalidValue(
                "shot allocation rounding unresolved".into(),
            ));
        }
        for _ in sum..target {
            let best = (0..items.len())
                .filter(|&i| totals[i] < self.maximum_per_item)
                .max_by(|&a, &b| {
                    let gain =
                        |i: usize| weight[i].powi(2) / (totals[i] as f64 * (totals[i] + 1) as f64);
                    gain(a).total_cmp(&gain(b)).then_with(|| b.cmp(&a))
                })
                .ok_or_else(|| Error::InvalidValue("shot allocation capacity exhausted".into()))?;
            totals[best] += 1;
        }
        let variance = items
            .iter()
            .zip(&totals)
            .map(|(item, &n)| item.coefficient.powi(2) * item.variance / n as f64)
            .sum::<f64>();
        if !variance.is_finite() {
            return Err(Error::InvalidValue(
                "allocation uncertainty overflow".into(),
            ));
        }
        let additional = totals
            .iter()
            .zip(items)
            .map(|(&n, item)| n - item.completed)
            .collect();
        Ok(ShotAllocation {
            completed: items.iter().map(|x| x.completed).collect(),
            additional,
            planned_totals: totals,
            reserved_attempts: target - completed,
            estimated_standard_error: variance.sqrt(),
        })
    }
}
/// Mergeable completed-shot statistics; reservations never enter this state.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ShotAccumulator {
    shots: usize,
    mean: f64,
    m2: f64,
}
impl ShotAccumulator {
    /// Add one independently executed shot estimate, preserving total attempts and uncertainty.
    /// Errors leave the accumulator unchanged. Adaptive samples can introduce stopping bias.
    pub fn merge(&mut self, sample: ShotEstimate) -> Result<()> {
        if sample.attempts < 2
            || !sample.mean.is_finite()
            || !sample.variance.is_finite()
            || sample.variance < 0.
        {
            return Err(Error::InvalidValue(
                "invalid completed shot estimate".into(),
            ));
        }
        let total = self
            .shots
            .checked_add(sample.attempts)
            .ok_or(Error::Overflow)?;
        let delta = sample.mean - self.mean;
        let mean = if self.shots == 0 {
            sample.mean
        } else {
            self.mean + delta * sample.attempts as f64 / total as f64
        };
        let m2 = if self.shots == 0 {
            sample.variance * (sample.attempts - 1) as f64
        } else {
            self.m2
                + sample.variance * (sample.attempts - 1) as f64
                + delta.powi(2) * self.shots as f64 * sample.attempts as f64 / total as f64
        };
        if !mean.is_finite() || !m2.is_finite() {
            return Err(Error::InvalidValue(
                "completed shot statistics overflow".into(),
            ));
        }
        *self = Self {
            shots: total,
            mean,
            m2,
        };
        Ok(())
    }
    /// Completed observations only; None before any measured batch has been merged.
    pub fn estimate(&self) -> Option<ShotEstimate> {
        (self.shots >= 2).then(|| {
            let variance = self.m2 / (self.shots - 1) as f64;
            ShotEstimate {
                mean: self.mean,
                variance,
                standard_error: (variance / self.shots as f64).sqrt(),
                attempts: self.shots,
            }
        })
    }
}
