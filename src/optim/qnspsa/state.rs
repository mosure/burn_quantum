//! Portable optimizer checkpoints and step diagnostics; model weights are separate.
use super::QnSpsaConfig;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Portable checkpoint; restore through [`QnSpsa::from_state`](super::QnSpsa::from_state) to validate it.
/// Save the model's parameter record separately, along with evaluator RNG state if noisy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QnSpsaState {
    /// Checkpoint schema, currently 1.
    pub version: u32,
    /// Full schedule, seed and numerical configuration.
    pub config: QnSpsaConfig,
    /// Parameter-vector length.
    pub parameters: usize,
    /// Completed attempts, including blocked updates; determines the next RNG stream.
    pub iteration: u64,
    /// Raw symmetric running metric, before taking its matrix absolute value.
    pub metric: Vec<f64>,
}

/// Diagnostics for one completed attempt. Counts are evaluator calls, not shots/circuits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QnSpsaStepReport {
    /// One-based completed attempt index.
    pub iteration: u64,
    /// Actual scheduled learning rate.
    pub learning_rate: f64,
    /// Actual scheduled perturbation.
    pub perturbation: f64,
    /// Whether the proposed parameter update was accepted.
    pub accepted: bool,
    /// Mean perturbed loss, averaged over resamplings.
    pub loss_estimate: f64,
    /// Exact/sample loss at the old point, present only with blocking.
    pub loss_before: Option<f64>,
    /// Exact/sample loss at the proposed point, present only with blocking.
    pub loss_proposed: Option<f64>,
    /// Euclidean norm of the proposed displacement, even if blocked.
    pub proposed_step_norm: f64,
    /// Two per resampling, plus two if blocking is enabled.
    pub loss_evaluations: usize,
    /// Four per resampling.
    pub fidelity_evaluations: usize,
}

/// Updated tensor and diagnostics. The tensor keeps its input device and dtype.
#[derive(Debug)]
pub struct QnSpsaStep {
    /// Accepted parameters, or the original parameters after a blocked proposal.
    pub parameters: Tensor<1>,
    /// Reproducible work accounting and acceptance outcome.
    pub report: QnSpsaStepReport,
}
