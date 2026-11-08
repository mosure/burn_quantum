//! Burn configuration, schedule parameters, and preflight validation.
use super::validation::{invalid, matrix_size, nonnegative, positive};
use crate::{Error, Result as QuantumResult};
use burn::config::Config;

/// Native Burn configuration for fidelity-based quantum natural SPSA.
/// The k-th attempt (k starts at zero) uses a/(k+1+A)^alpha and c/(k+1)^gamma.
#[derive(Config, Debug)]
pub struct QnSpsaConfig {
    /// Learning-rate numerator a.
    #[config(default = 0.05)]
    pub learning_rate: f64,
    /// Perturbation numerator c. Small values amplify fidelity noise as 1/c².
    #[config(default = 0.1)]
    pub perturbation: f64,
    /// Learning-rate decay alpha; zero selects a constant schedule.
    #[config(default = 0.602)]
    pub learning_rate_decay: f64,
    /// Perturbation decay gamma; zero selects a constant schedule.
    #[config(default = 0.101)]
    pub perturbation_decay: f64,
    /// Nonnegative learning-rate stability offset A.
    #[config(default = 0.0)]
    pub stability_constant: f64,
    /// Positive beta in |averaged metric| + beta I.
    #[config(default = 0.01)]
    pub regularization: f64,
    /// The initial metric is this nonnegative multiple of identity, with weight one.
    #[config(default = 1.0)]
    pub initial_metric_scale: f64,
    /// Independent pairs of Rademacher directions per attempt. Each costs 2 loss + 4 fidelity calls.
    #[config(default = 1)]
    pub resamplings: usize,
    /// First this many attempts use the ordinary SPSA direction while still learning the metric.
    #[config(default = 0)]
    pub metric_delay: u64,
    /// If set, accept only when candidate loss <= current loss + tolerance.
    /// This adds two loss calls per attempt; no cached loss can become stale.
    #[config(default = "None")]
    pub blocking_tolerance: Option<f64>,
    /// Local ChaCha8 seed; attempt index selects an independent reproducible stream.
    #[config(default = 0)]
    pub seed: u64,
    /// Maximum entries in each dense host metric; default allows 256 parameters.
    /// Eigensolver scratch storage is additional; this is not a total memory cap.
    #[config(default = 65_536)]
    pub max_metric_elements: usize,
}
impl QnSpsaConfig {
    pub(super) fn validate(&self, parameters: usize) -> QuantumResult<usize> {
        if !positive(self.learning_rate)
            || !positive(self.perturbation)
            || !positive(self.regularization)
            || !nonnegative(self.initial_metric_scale)
            || !nonnegative(self.learning_rate_decay)
            || !nonnegative(self.perturbation_decay)
            || !nonnegative(self.stability_constant)
            || self.blocking_tolerance.is_some_and(|x| !nonnegative(x))
            || self.resamplings == 0
        {
            return Err(invalid(
                "invalid rate, perturbation, regularization, schedule, or resampling count",
            ));
        }
        self.resamplings
            .checked_mul(4)
            .and_then(|n| n.checked_add(2))
            .ok_or(Error::Overflow)?;
        matrix_size(parameters, self.max_metric_elements)
    }
}
