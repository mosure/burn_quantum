//! Small reporting and deterministic-run utilities for the executable examples.
//!
//! Quantum operations live in `burn_quantum`; datasets, training loops, and each
//! algorithm's acceptance criteria live in that algorithm's single binary file.
use burn::tensor::{Device, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::collections::BTreeMap;

#[path = "../../../support/device.rs"]
mod runtime;

/// Explicit device selection shared by all examples, controlled by BURN_QUANTUM_DEVICE.
pub fn execution_device() -> ExampleResult<Device> {
    Ok(runtime::execution()?.device)
}

/// Example runner result, including tensor validation and reporting failures.
pub type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Machine-readable outcomes for a bounded deterministic run.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Cargo binary name.
    pub example: &'static str,
    /// Seed for initialization/sampling; datasets are deterministic grids.
    pub seed: u64,
    /// Number of optimizer updates (zero for a closed-form kernel fit).
    pub steps: usize,
    /// Requested execution backend; hardware requests cannot fall back to CPU.
    pub backend: &'static str,
    /// Actual adapter name and, for WGPU, adapter type and graphics API.
    pub adapter: String,
    /// Example tensor precision.
    pub dtype: &'static str,
    /// Whether the executable was compiled with GPU kernel fusion.
    pub fusion: bool,
    /// Finite numerical results, with descriptive metric names.
    pub metrics: BTreeMap<&'static str, f64>,
}
impl Report {
    /// Construct a report, rejecting nonfinite metrics.
    pub fn new(
        example: &'static str,
        seed: u64,
        steps: usize,
        metrics: &[(&'static str, f64)],
    ) -> ExampleResult<Self> {
        ensure(
            metrics.iter().all(|(_, v)| v.is_finite()),
            "nonfinite example metric",
        )?;
        let execution = runtime::execution()?;
        Ok(Self {
            example,
            seed,
            steps,
            backend: execution.backend,
            adapter: execution.adapter,
            dtype: "f32",
            fusion: cfg!(feature = "fusion"),
            metrics: metrics.iter().copied().collect(),
        })
    }
    /// Print a JSON report to standard output.
    pub fn print(&self) -> ExampleResult<()> {
        println!("{}", serde_json::to_string_pretty(self)?);
        Ok(())
    }
}

/// Fail an executable or its test when its stated acceptance condition is unmet.
pub fn ensure(condition: bool, message: &str) -> ExampleResult<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

/// Explicit local initialization, independent of Burn's process-global RNG.
pub fn initial_angles(count: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count).map(|_| rng.random_range(-0.8..0.8)).collect()
}

/// Explicit synchronous tensor readback for evaluation/reporting.
pub fn values<const D: usize>(tensor: Tensor<D>) -> ExampleResult<Vec<f64>> {
    tensor
        .into_data()
        .convert::<f64>()
        .try_to_vec::<f64>()
        .map_err(|e| format!("tensor readback: {e:?}").into())
}

/// Classification accuracy for `[batch, classes]` predictions and one-hot targets.
/// This is an evaluation readback, never part of the training graph.
pub fn accuracy(probabilities: Tensor<2>, targets: Tensor<2>) -> ExampleResult<f64> {
    let [batch, classes] = probabilities.dims();
    ensure(
        batch > 0 && classes > 1 && targets.dims() == [batch, classes],
        "invalid accuracy shapes",
    )?;
    let predictions = values(probabilities)?;
    let labels = values(targets)?;
    ensure(
        predictions.iter().chain(&labels).all(|x| x.is_finite()),
        "nonfinite classification values",
    )?;
    let argmax = |row: &[f64]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let correct = predictions
        .chunks_exact(classes)
        .zip(labels.chunks_exact(classes))
        .filter(|(p, y)| argmax(p) == argmax(y))
        .count();
    Ok(correct as f64 / batch as f64)
}
