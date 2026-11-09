//! Independent analytic geometry, estimator accounting, and selected-device checks.
mod common;
#[path = "../support/device.rs"]
mod runtime;

use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{gate::Axis, optim::*, *};
use common::{close, complex_values, values};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cell::Cell;

fn selected() -> Device {
    runtime::execution()
        .expect("requested backend unavailable")
        .device
}
fn tensor(x: &[f64], device: &Device, dtype: DType) -> Tensor<1> {
    Tensor::from_data(TensorData::new(x.to_vec(), [x.len()]), (device, dtype))
}
fn config() -> QnSpsaConfig {
    QnSpsaConfig::new()
        .with_learning_rate(0.1)
        .with_perturbation(0.05)
        .with_learning_rate_decay(0.0)
        .with_perturbation_decay(0.0)
        .with_seed(13)
}
fn scalar_loss(p: Tensor<1>) -> Result<f64> {
    Ok(values(p)[0].cos())
}
fn scalar_fidelity(a: Tensor<1>, b: Tensor<1>) -> Result<f64> {
    Ok(((values(a)[0] - values(b)[0]) / 2.0).cos().powi(2))
}

fn directions(n: usize) -> Vec<Vec<f64>> {
    (0..1usize << n)
        .map(|bits| {
            (0..n)
                .map(|i| if bits & (1 << i) == 0 { -1. } else { 1. })
                .collect()
        })
        .collect()
}

#[path = "qnspsa/estimator.rs"]
mod estimator;
#[path = "qnspsa/geometry.rs"]
mod geometry;
#[path = "qnspsa/optimizer.rs"]
mod optimizer;
