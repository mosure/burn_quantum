//! Angle-encoded variational binary classifier for a continuous parity problem.
//! Synthetic train/test grids are disjoint; labels are the sign of x0*x1.
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor, TensorData};
use burn_quantum::{
    Circuit, Observable, QuantumLayer, QuantumLayerConfig, Register,
    gate::Axis,
    qml::{angle_embedding, distribution_cross_entropy, pauli_z_probabilities, ry_cnot_layer},
};
use burn_quantum_example::{ExampleResult, Report, accuracy, ensure, execution_device};

const SEED: u64 = 0; // Explicit fixed angles; no random data or initialization.
const STEPS: usize = 120;

fn dataset(magnitudes: &[f64], device: &Device) -> (Tensor<2>, Tensor<2>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for &a in magnitudes {
        for &b in magnitudes {
            for sx in [-1., 1.] {
                for sy in [-1., 1.] {
                    let x = sx * a;
                    let y = sy * b;
                    // A fixed physical encoding range, fitted to neither train nor test data.
                    features.extend([
                        (x + 1.) * std::f64::consts::FRAC_PI_2,
                        (y + 1.) * std::f64::consts::FRAC_PI_2,
                    ]);
                    labels.extend(if x * y < 0. { [0., 1.] } else { [1., 0.] });
                }
            }
        }
    }
    let batch = features.len() / 2;
    (
        Tensor::from_data(TensorData::new(features, [batch, 2]), device),
        Tensor::from_data(TensorData::new(labels, [batch, 2]), device),
    )
}

fn predict(model: &QuantumLayer, input: Tensor<2>) -> burn_quantum::Result<Tensor<2>> {
    let batch = input.dims()[0];
    Ok(pauli_z_probabilities(
        model.forward(Some(input), batch)?.reshape([batch]),
    ))
}

fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let (train_x, train_y) = dataset(&[0.35, 0.8], &device);
    let (test_x, test_y) = dataset(&[0.2, 0.55, 0.95], &device);
    let mut circuit = Circuit::new(Register::qubits(2)?, 2, 2);
    angle_embedding(&mut circuit, &[0, 1], Axis::Y, 0)?;
    ry_cnot_layer(&mut circuit, &[0, 1], 0)?;
    let mut model =
        QuantumLayerConfig::new(circuit, vec![Observable::z(1)], vec![0.8, -0.6]).init(&device)?;
    let initial =
        distribution_cross_entropy(predict(&model, train_x.clone())?, train_y.clone(), 1e-7)?
            .into_scalar::<f64>();
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..STEPS {
        let loss =
            distribution_cross_entropy(predict(&model, train_x.clone())?, train_y.clone(), 1e-7)?;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.04, model, gradients);
    }
    let train_prob = predict(&model, train_x)?;
    let test_prob = predict(&model, test_x)?;
    let final_loss =
        distribution_cross_entropy(train_prob.clone(), train_y.clone(), 1e-7)?.into_scalar::<f64>();
    let train_accuracy = accuracy(train_prob, train_y)?;
    let test_accuracy = accuracy(test_prob, test_y)?;
    ensure(
        final_loss < initial * 0.8 && test_accuracy >= 0.95,
        "variational classifier failed to learn held-out parity",
    )?;
    Report::new(
        "variational_classifier",
        SEED,
        STEPS,
        &[
            ("initial_cross_entropy", initial),
            ("cross_entropy", final_loss),
            ("train_accuracy", train_accuracy),
            ("test_accuracy", test_accuracy),
            ("majority_baseline_accuracy", 0.5),
        ],
    )
}

fn main() -> ExampleResult<()> {
    run()?.print()
}

#[cfg(test)]
mod tests {
    #[test]
    fn classifies_disjoint_parity_grid() {
        super::run().unwrap();
    }
}
