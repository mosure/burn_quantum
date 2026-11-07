//! Classical encoder -> entangled quantum layer -> classical head, trained end to end.
#[path = "../support/device.rs"]
mod runtime;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor, TensorData};
use burn_quantum::{
    Angle, Circuit, Observable, QuantumLayer, QuantumLayerConfig, Register, Rotation, Unitary,
    gate::Axis,
};

#[derive(Module, Debug)]
struct Hybrid {
    encoder: Linear,
    quantum: QuantumLayer,
    head: Linear,
}
impl Hybrid {
    fn new(device: &Device) -> burn_quantum::Result<Self> {
        let mut circuit = Circuit::new(Register::qubits(2)?, 2, 2);
        for wire in 0..2 {
            circuit.rotate(wire, Rotation::qubit(Axis::Y, Angle::input(wire)))?;
            circuit.rotate(wire, Rotation::qubit(Axis::Y, Angle::parameter(wire)))?;
        }
        circuit.unitary(vec![0, 1], Unitary::cnot())?;
        let quantum = QuantumLayerConfig::new(
            circuit,
            vec![Observable::z(0), Observable::z(1)],
            vec![0.2, -0.1],
        )
        .init(device)?;
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
        let angles = self.encoder.forward(input);
        Ok(self
            .head
            .forward(self.quantum.forward(Some(angles), batch)?))
    }
}

fn main() -> burn_quantum::Result<()> {
    let device = runtime::execution()
        .map_err(burn_quantum::Error::Unsupported)?
        .device
        .autodiff();
    let mut model = Hybrid::new(&device)?;
    let xs: Vec<f32> = (0..24).map(|i| -1.5 + 3.0 * i as f32 / 23.0).collect();
    let ys: Vec<f32> = xs.iter().map(|x| (1.2 * x + 0.3).cos()).collect();
    let input = Tensor::from_data(TensorData::new(xs, [24, 1]), &device);
    let target = Tensor::from_data(TensorData::new(ys, [24, 1]), &device);
    let mut optimizer = AdamConfig::new().init();
    let initial = (model.forward(input.clone())? - target.clone())
        .square()
        .mean()
        .into_scalar::<f32>();
    for step in 0..120 {
        let loss = (model.forward(input.clone())? - target.clone())
            .square()
            .mean();
        if step % 40 == 0 {
            println!("step {step}, MSE={:.7}", loss.clone().into_scalar::<f32>());
        }
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.03, model, gradients);
    }
    let final_loss = (model.forward(input)? - target)
        .square()
        .mean()
        .into_scalar::<f32>();
    println!("Hybrid regression MSE: {initial:.7} -> {final_loss:.7}");
    assert!(
        final_loss < initial / 20. && final_loss < 0.01,
        "hybrid training did not converge"
    );
    Ok(())
}
