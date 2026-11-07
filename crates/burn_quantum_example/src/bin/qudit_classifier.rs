//! Three-class variational classifier using a single qutrit's Born probabilities.
//! The synthetic clusters are noisy angle encodings near the three basis states.
//! Trainable real rotations mix the three pairs of qutrit levels.
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor, TensorData};
use burn_quantum::{
    Angle, Circuit, QuantumLayerConfig, Register, Rotation, gate::Axis,
    qml::distribution_cross_entropy,
};
use burn_quantum_example::{ExampleResult, Report, accuracy, ensure, execution_device};

const STEPS: usize = 120;

fn dataset(jitters: &[f64], device: &Device) -> (Tensor<2>, Tensor<2>) {
    let mut input = Vec::new();
    let mut target = Vec::new();
    let pi = std::f64::consts::PI;
    for (class, center) in [[0., 0.], [pi, 0.], [pi, pi]].into_iter().enumerate() {
        for &a in jitters {
            for &b in jitters {
                input.extend([center[0] + a, center[1] + b]);
                target.extend((0..3).map(|j| if j == class { 1. } else { 0. }));
            }
        }
    }
    let batch = input.len() / 2;
    (
        Tensor::from_data(TensorData::new(input, [batch, 2]), device),
        Tensor::from_data(TensorData::new(target, [batch, 3]), device),
    )
}

fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let mut circuit = Circuit::new(Register::new(vec![3])?, 3, 2);
    circuit.rotate(
        0,
        Rotation {
            axis: Axis::Y,
            levels: [0, 1],
            angle: Angle::input(0),
        },
    )?;
    circuit.rotate(
        0,
        Rotation {
            axis: Axis::Y,
            levels: [1, 2],
            angle: Angle::input(1),
        },
    )?;
    for (index, levels) in [[0, 1], [1, 2], [0, 2]].into_iter().enumerate() {
        circuit.rotate(
            0,
            Rotation {
                axis: Axis::Y,
                levels,
                angle: Angle::parameter(index),
            },
        )?;
    }
    let mut model = QuantumLayerConfig::new(circuit, vec![], vec![0.8, -0.6, 0.5]).init(&device)?;
    let (train_x, train_y) = dataset(&[-0.2, 0., 0.2], &device);
    let (test_x, test_y) = dataset(&[-0.3, -0.1, 0.1, 0.3], &device);
    let batch = train_x.dims()[0];
    let test_batch = test_x.dims()[0];
    let initial = distribution_cross_entropy(
        model.state(Some(train_x.clone()), batch)?.probabilities(),
        train_y.clone(),
        1e-7,
    )?
    .into_scalar::<f64>();
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..STEPS {
        let probabilities = model.state(Some(train_x.clone()), batch)?.probabilities();
        let loss = distribution_cross_entropy(probabilities, train_y.clone(), 1e-7)?;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.04, model, gradients);
    }
    let train = model.state(Some(train_x), batch)?;
    let test = model.state(Some(test_x), test_batch)?;
    train.validate(1e-5)?;
    test.validate(1e-5)?;
    let final_loss = distribution_cross_entropy(train.probabilities(), train_y.clone(), 1e-7)?
        .into_scalar::<f64>();
    let train_accuracy = accuracy(train.probabilities(), train_y)?;
    let test_accuracy = accuracy(test.probabilities(), test_y)?;
    ensure(
        final_loss < initial / 5. && test_accuracy >= 0.95,
        "qutrit classifier failed held-out classification",
    )?;
    Report::new(
        "qudit_classifier",
        0,
        STEPS,
        &[
            ("initial_cross_entropy", initial),
            ("cross_entropy", final_loss),
            ("train_accuracy", train_accuracy),
            ("test_accuracy", test_accuracy),
            ("majority_baseline_accuracy", 1. / 3.),
        ],
    )
}

fn main() -> ExampleResult<()> {
    run()?.print()
}

#[cfg(test)]
mod tests {
    #[test]
    fn learns_three_class_qutrit_readout() {
        super::run().unwrap();
    }
}
