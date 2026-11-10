//! Renderer-independent, owned scientific records. No tensors or UI state are retained.
mod fabric;
mod validation;
use burn_quantum::{Circuit, Error, Result};
pub use fabric::*;
use serde::{Deserialize, Serialize};

/// A complex scalar in the circuit's computational basis convention.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ComplexValue {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}
/// One subsystem's reduced density matrix. Built-in simulations compute an exact partial trace.
/// Imported mixed-state summaries are checked locally; they do not specify a complete global state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubsystemState {
    /// Register wire index, wire zero most significant.
    pub wire: usize,
    /// Local Hilbert-space dimension.
    pub dimension: usize,
    /// Diagonal of the reduced density matrix.
    pub probabilities: Vec<f64>,
    /// Row-major reduced density matrix.
    pub density: Vec<ComplexValue>,
    /// `[X,Y,Z]` expectations for dimension two; qudits have no qubit Bloch vector.
    pub bloch: Option<[f64; 3]>,
    /// Tr(rho squared); one for a pure local state.
    pub purity: f64,
}
/// A cached state after `step` operations or optimizer updates, according to trace kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelineFrame {
    /// Number of operations or optimizer updates already completed.
    pub step: usize,
    /// Human-readable event label.
    pub label: String,
    /// Full computational-basis distribution in big-endian mixed-radix order.
    pub probabilities: Vec<f64>,
    /// Pure-state amplitudes when supplied. None means global coherences are not retained;
    /// it does not by itself establish that the source state was mixed.
    pub amplitudes: Option<Vec<ComplexValue>>,
    /// One partial trace per register wire.
    pub subsystems: Vec<SubsystemState>,
    /// Total probability mass before any display formatting; never silently normalized.
    pub norm: f64,
}
/// Result of one explicitly requested simulation, reused by all views and playback.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentTrace {
    /// Meaning of the timeline's step axis.
    pub kind: TraceKind,
    /// Ordered cached frames, interpreted according to `kind`.
    pub frames: Vec<TimelineFrame>,
    /// Actual Burn device description, independent of the graphics renderer.
    pub backend: String,
}
/// Scientific meaning of each cached frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceKind {
    /// Input followed by each circuit operation.
    Circuit,
    /// Initial state followed by actual optimizer iterates.
    Optimization,
}
/// One measured optimizer iterate. Evaluation counts are cumulative.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationPoint {
    /// Completed update count; zero is the initial parameter vector.
    pub iteration: usize,
    /// Actual parameters at this iterate.
    pub parameters: Vec<f64>,
    /// Objective evaluated at these parameters, not the perturbed-loss average.
    pub loss: f64,
    /// Row-major running fidelity metric before regularization.
    pub metric: Vec<f64>,
    /// Reduced-qubit Bloch vectors at this iterate.
    pub bloch: Vec<Option<[f64; 3]>>,
    /// Computational-basis probabilities at this iterate.
    pub probabilities: Vec<f64>,
    /// Number of objective evaluations including center and probe evaluations.
    pub loss_evaluations: usize,
    /// Number of squared-overlap evaluations.
    pub fidelity_evaluations: usize,
}
/// Portable experiment plus computed evidence. Inputs/outputs are the first/last frames.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualizationDocument {
    /// Current format is version one.
    pub schema_version: u32,
    /// Experiment title.
    pub title: String,
    /// Scientific scope and assumptions, shown with the experiment.
    pub description: String,
    /// Original checked circuit, including symbolic angles.
    pub circuit: Circuit,
    /// Parameter bindings used for the displayed circuit timeline.
    pub parameters: Vec<f64>,
    /// Feature bindings for the single displayed input example.
    pub inputs: Vec<f64>,
    /// Gate labels in the same order as circuit operations.
    pub operation_labels: Vec<String>,
    /// Cached circuit trajectory.
    pub trace: ExperimentTrace,
    /// A separately identified actual compiled physical schedule, when present.
    pub fabric: Option<FabricSnapshot>,
    /// Actual optimizer iterates, empty for nontraining experiments.
    pub optimization: Vec<OptimizationPoint>,
}

/// One probability bucket; adjacent basis states are aggregated without losing mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbabilityBin {
    /// Inclusive first flat basis index.
    pub start: usize,
    /// Exclusive final flat basis index.
    pub end: usize,
    /// Sum of probabilities on this interval.
    pub probability: f64,
}
/// Aggregate all entries, rather than dropping low-probability states from a plot.
pub fn aggregate_probabilities(values: &[f64], max_bins: usize) -> Result<Vec<ProbabilityBin>> {
    if max_bins == 0
        || values.is_empty()
        || values.iter().any(|v| !v.is_finite() || *v < 0.)
        || !values.iter().sum::<f64>().is_finite()
    {
        return Err(Error::InvalidValue(
            "invalid probability aggregation".into(),
        ));
    }
    let width = values.len().div_ceil(max_bins);
    Ok(values
        .chunks(width)
        .enumerate()
        .map(|(i, chunk)| ProbabilityBin {
            start: i * width,
            end: i * width + chunk.len(),
            probability: chunk.iter().sum(),
        })
        .collect())
}
