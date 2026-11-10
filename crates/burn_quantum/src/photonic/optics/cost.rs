//! Explicit calibrated resource assumptions for local optical experiment comparisons.
use crate::{Error, Result};
use serde::Serialize;
/// Caller-declared device/resource model; these are assumptions, not measured hardware telemetry.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct OpticalCostModel {
    /// Mean photons injected per attempted preparation, including supplied ancillary photons.
    pub mean_photons_per_attempt: f64,
    /// Preparation/measurement period per independent lane in seconds.
    pub seconds_per_attempt: f64,
    /// Energy per attempted preparation/measurement, in joules.
    pub joules_per_attempt: f64,
    /// Number of independent concurrently available lanes.
    pub parallel_lanes: usize,
}
/// Work accounting under the explicitly supplied model, retaining failed/no-click attempts.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct OpticalResourceCost {
    /// Total attempted preparations across every probe and state.
    pub attempts: usize,
    /// Expected number of injected photons, before loss.
    pub expected_photons: f64,
    /// Serial duration in seconds.
    pub serial_seconds: f64,
    /// Ideal-lane duration in seconds; excludes unspecified routing/feedback contention.
    pub ideal_parallel_seconds: f64,
    /// Modeled total energy in joules.
    pub modeled_joules: f64,
}
impl OpticalCostModel {
    /// Validate assumptions and compute finite resource totals without hidden conditioning.
    pub fn cost(&self, attempts: usize) -> Result<OpticalResourceCost> {
        if self.parallel_lanes == 0
            || [
                self.mean_photons_per_attempt,
                self.seconds_per_attempt,
                self.joules_per_attempt,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err(Error::InvalidValue("invalid optical resource model".into()));
        }
        let cost = OpticalResourceCost {
            attempts,
            expected_photons: attempts as f64 * self.mean_photons_per_attempt,
            serial_seconds: attempts as f64 * self.seconds_per_attempt,
            ideal_parallel_seconds: attempts.div_ceil(self.parallel_lanes) as f64
                * self.seconds_per_attempt,
            modeled_joules: attempts as f64 * self.joules_per_attempt,
        };
        if [
            cost.expected_photons,
            cost.serial_seconds,
            cost.ideal_parallel_seconds,
            cost.modeled_joules,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err(Error::InvalidValue(
                "optical resource accounting overflow".into(),
            ));
        }
        Ok(cost)
    }
}
