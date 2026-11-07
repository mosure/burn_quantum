//! End-to-end classical encoder -> variational circuit -> classical head regression.
//! Training and evaluation use disjoint deterministic input grids.
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor, TensorData};
use burn_quantum::{
    Circuit, Observable, QuantumLayer, QuantumLayerConfig, Register,
    gate::Axis,
    qml::{angle_embedding, ry_cnot_layer},
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

const STEPS: usize = 120;

#[derive(Module, Debug)]
struct Hybrid {
    encoder: Linear,
    quantum: QuantumLayer,
    head: Linear,
}
impl Hybrid {
    fn new(device: &Device) -> burn_quantum::Result<Self> {
        let mut circuit = Circuit::new(Register::qubits(2)?, 2, 2);
        angle_embedding(&mut circuit, &[0, 1], Axis::Y, 0)?;
        ry_cnot_layer(&mut circuit, &[0, 1], 0)?;
        let quantum = QuantumLayerConfig::new(
            circuit,
            vec![Observable::z(0), Observable::z(1)],
            vec![0.2, -0.1],
        )
        .init(device)?;
        // All trainable values are explicit; there is no stochastic initialization.
        let mut encoder = LinearConfig::new(1, 2).with_bias(false).init(device);
        encoder.weight = Param::from_tensor(Tensor::from_floats([[0.6, 1.1]], device));
        let mut head = LinearConfig::new(2, 1).with_bias(false).init(device);
        head.weight = Param::from_tensor(Tensor::from_floats([[0.3], [0.2]], device));
        Ok(Self {
            encoder,
            quantum,
            head,
        })
    }
    fn forward(&self, input: Tensor<2>) -> burn_quantum::Result<Tensor<2>> {
        let batch = input.dims()[0];
        Ok(self.head.forward(
            self.quantum
                .forward(Some(self.encoder.forward(input)), batch)?,
        ))
    }
}

fn dataset(count: usize, midpoints: bool, device: &Device) -> (Tensor<2>, Tensor<2>) {
    let xs: Vec<f64> = (0..count)
        .map(|i| {
            if midpoints {
                -1.5 + 3. * (i as f64 + 0.5) / count as f64
            } else {
                -1.5 + 3. * i as f64 / (count - 1) as f64
            }
        })
        .collect();
    let ys = xs.iter().map(|x| (1.2 * x + 0.3).cos()).collect();
    (
        Tensor::from_data(TensorData::new(xs, [count, 1]), device),
        Tensor::from_data(TensorData::new(ys, [count, 1]), device),
    )
}

fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let mut model = Hybrid::new(&device)?;
    let (train_x, train_y) = dataset(24, false, &device);
    let (test_x, test_y) = dataset(31, true, &device);
    let initial = (model.forward(train_x.clone())? - train_y.clone())
        .square()
        .mean()
        .into_scalar::<f64>();
    let before_encoder = values(model.encoder.weight.val())?;
    let before_quantum = values(model.quantum.parameters.val())?;
    let before_head = values(model.head.weight.val())?;
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..STEPS {
        let loss = (model.forward(train_x.clone())? - train_y.clone())
            .square()
            .mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.03, model, gradients);
    }
    let train_mse = (model.forward(train_x)? - train_y)
        .square()
        .mean()
        .into_scalar::<f64>();
    let test_mse = (model.forward(test_x)? - test_y)
        .square()
        .mean()
        .into_scalar::<f64>();
    let changed = |before: Vec<f64>, after: Vec<f64>| {
        before
            .iter()
            .zip(after)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
    };
    let encoder_change = changed(before_encoder, values(model.encoder.weight.val())?);
    let quantum_change = changed(before_quantum, values(model.quantum.parameters.val())?);
    let head_change = changed(before_head, values(model.head.weight.val())?);
    ensure(
        train_mse < initial / 20. && test_mse < 0.002,
        "hybrid regression failed held-out evaluation",
    )?;
    ensure(
        encoder_change > 1e-3 && quantum_change > 1e-3 && head_change > 1e-3,
        "hybrid training failed to update every stage",
    )?;
    Report::new(
        "hybrid_regression",
        0,
        STEPS,
        &[
            ("initial_train_mse", initial),
            ("train_mse", train_mse),
            ("test_mse", test_mse),
            ("encoder_parameter_change", encoder_change),
            ("quantum_parameter_change", quantum_change),
            ("head_parameter_change", head_change),
        ],
    )
}

fn main() -> ExampleResult<()> {
    run()?.print()
}

#[cfg(test)]
mod tests {
    #[test]
    fn learns_through_all_three_stages() {
        super::run().unwrap();
    }
}
