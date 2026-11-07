//! Quantum fidelity-kernel classification with a product RY feature map.
//! Fit kernel ridge regression: (K + lambda I) alpha = y. The product feature
//! map induces nonlinear feature interactions sufficient for continuous parity.
//! The tiny classical Cholesky solve is intentionally visible in this file.
use burn::tensor::{Device, Tensor, TensorData};
use burn_quantum::{
    Circuit, DenseSimulator, Register,
    gate::Axis,
    qml::{angle_embedding, fidelity_kernel},
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

const RIDGE: f64 = 1e-3;

fn dataset(magnitudes: &[f64], device: &Device) -> (Tensor<2>, Vec<f64>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for &a in magnitudes {
        for &b in magnitudes {
            for sx in [-1., 1.] {
                for sy in [-1., 1.] {
                    features.extend([sx * a, sy * b]);
                    labels.push(sx * sy);
                }
            }
        }
    }
    (
        Tensor::from_data(TensorData::new(features, [labels.len(), 2]), device),
        labels,
    )
}

fn solve(gram: &[f64], labels: &[f64]) -> ExampleResult<Vec<f64>> {
    let n = labels.len();
    let mut lower = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let entry = gram[i * n + j] + if i == j { RIDGE } else { 0. };
            let residual = entry
                - (0..j)
                    .map(|k| lower[i * n + k] * lower[j * n + k])
                    .sum::<f64>();
            lower[i * n + j] = if i == j {
                ensure(
                    residual.is_finite() && residual > 0.,
                    "regularized kernel is not positive definite",
                )?;
                residual.sqrt()
            } else {
                residual / lower[j * n + j]
            };
        }
    }
    let mut z = vec![0.; n];
    for i in 0..n {
        z[i] =
            (labels[i] - (0..i).map(|j| lower[i * n + j] * z[j]).sum::<f64>()) / lower[i * n + i];
    }
    let mut alpha = vec![0.; n];
    for i in (0..n).rev() {
        alpha[i] = (z[i] - (i + 1..n).map(|j| lower[j * n + i] * alpha[j]).sum::<f64>())
            / lower[i * n + i];
    }
    let residual = (0..n)
        .map(|i| {
            ((0..n).map(|j| gram[i * n + j] * alpha[j]).sum::<f64>() + RIDGE * alpha[i] - labels[i])
                .abs()
        })
        .fold(0., f64::max);
    ensure(residual < 1e-8, "kernel ridge solve residual is too large")?;
    Ok(alpha)
}

fn classify(kernel: &[f64], alpha: &[f64], labels: &[f64]) -> f64 {
    let n = alpha.len();
    let correct = kernel
        .chunks_exact(n)
        .zip(labels)
        .filter(|(row, y)| {
            let prediction: f64 = row.iter().zip(alpha).map(|(k, a)| k * a).sum();
            prediction.signum() == y.signum()
        })
        .count();
    correct as f64 / labels.len() as f64
}

fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let engine = DenseSimulator::default();
    let (train_x, train_y) = dataset(&[0.35, 0.8], &device);
    let (test_x, test_y) = dataset(&[0.2, 0.55, 1.0], &device);
    let mut feature_map = Circuit::new(Register::qubits(2)?, 0, 2);
    angle_embedding(&mut feature_map, &[0, 1], Axis::Y, 0)?;
    let train = feature_map.run(&engine, None, Some(train_x), train_y.len(), &device)?;
    let test = feature_map.run(&engine, None, Some(test_x), test_y.len(), &device)?;
    train.validate(1e-5)?;
    test.validate(1e-5)?;
    let gram = values(fidelity_kernel(&train, &train, engine.limits)?)?;
    let cross = values(fidelity_kernel(&test, &train, engine.limits)?)?;
    let n = train_y.len();
    let mut symmetry_error: f64 = 0.;
    let mut diagonal_error: f64 = 0.;
    for i in 0..n {
        diagonal_error = diagonal_error.max((gram[i * n + i] - 1.).abs());
        for j in 0..n {
            symmetry_error = symmetry_error.max((gram[i * n + j] - gram[j * n + i]).abs());
        }
    }
    ensure(
        symmetry_error < 1e-6 && diagonal_error < 1e-5,
        "invalid fidelity Gram matrix",
    )?;
    let alpha = solve(&gram, &train_y)?;
    let train_accuracy = classify(&gram, &alpha, &train_y);
    let test_accuracy = classify(&cross, &alpha, &test_y);
    ensure(
        train_accuracy == 1. && test_accuracy >= 0.95,
        "quantum kernel failed held-out parity classification",
    )?;
    Report::new(
        "quantum_kernel",
        0,
        0,
        &[
            ("train_accuracy", train_accuracy),
            ("test_accuracy", test_accuracy),
            ("majority_baseline_accuracy", 0.5),
            ("ridge", RIDGE),
            ("gram_symmetry_error", symmetry_error),
            ("gram_diagonal_error", diagonal_error),
        ],
    )
}

fn main() -> ExampleResult<()> {
    run()?.print()
}

#[cfg(test)]
mod tests {
    #[test]
    fn fidelity_kernel_classifies_disjoint_data() {
        super::run().unwrap();
    }
}
