//! Gradient oracles, shared bindings, input encoding, and optimizer integration.
mod common;
use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{gate::Axis, gradient::parameter_shift, *};
use common::*;

fn parameter(x: f64, device: &Device) -> Tensor<1> {
    Tensor::from_data(TensorData::new(vec![x], [1]), (device, DType::F64))
}
fn evaluate(circuit: &Circuit, x: f64, observable: &Observable) -> f64 {
    let engine = DenseSimulator::default();
    let device = Device::flex();
    let state = circuit
        .run(&engine, Some(parameter(x, &device)), None, 1, &device)
        .unwrap();
    values(observable.expectation(&engine, &state).unwrap())[0]
}

#[test]
fn rotations_autodiff_parameter_shift_finite_difference() {
    for dimension in [2, 3, 4] {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let mut circuit = Circuit::new(Register::new(vec![dimension]).unwrap(), 1, 0);
            circuit
                .unitary(vec![0], Unitary::fourier(dimension).unwrap())
                .unwrap();
            circuit
                .rotate(
                    0,
                    Rotation {
                        axis,
                        levels: [0, dimension - 1],
                        angle: Angle::parameter(0),
                    },
                )
                .unwrap();
            // Coherence with the spectator level makes the half-frequency term observable.
            let mut entries = vec![0.; dimension * dimension];
            entries[0] = 0.7;
            entries[dimension * dimension - 1] = -0.3;
            for j in 1..dimension {
                entries[j] = 0.4;
                entries[j * dimension] = 0.4;
            }
            let observable =
                Observable::new(vec![0], Matrix::real(dimension, entries).unwrap()).unwrap();
            let engine = DenseSimulator::default();
            let device = Device::flex().autodiff();
            let theta = parameter(0.73, &device).require_grad();
            let state = circuit
                .run(&engine, Some(theta.clone()), None, 1, &device)
                .unwrap();
            let gradients = observable
                .expectation(&engine, &state)
                .unwrap()
                .sum()
                .backward();
            let autodiff = values(theta.grad(&gradients).unwrap())[0];
            let shift = values(
                parameter_shift(
                    &engine,
                    &circuit,
                    parameter(0.73, &Device::flex()),
                    None,
                    1,
                    &Device::flex(),
                    &observable,
                )
                .unwrap(),
            )[0];
            let fd = (evaluate(&circuit, 0.730001, &observable)
                - evaluate(&circuit, 0.729999, &observable))
                / 0.000002;
            close(&[autodiff, shift], &[fd, fd], 1e-7);
        }
    }
}

#[test]
fn shared_parameters_accumulate_occurrence_chain_rule() {
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    circuit
        .rotate(
            0,
            Rotation::qubit(
                Axis::Y,
                Angle::Parameter {
                    index: 0,
                    scale: 2.,
                    offset: 0.1,
                },
            ),
        )
        .unwrap();
    circuit
        .rotate(
            0,
            Rotation::qubit(
                Axis::Y,
                Angle::Parameter {
                    index: 0,
                    scale: -0.5,
                    offset: 0.2,
                },
            ),
        )
        .unwrap();
    let engine = DenseSimulator::default();
    let device = Device::flex().autodiff();
    let theta = parameter(0.4, &device).require_grad();
    let state = circuit
        .run(&engine, Some(theta.clone()), None, 1, &device)
        .unwrap();
    let gradients = Observable::z(0)
        .expectation(&engine, &state)
        .unwrap()
        .backward();
    let autodiff = values(theta.grad(&gradients).unwrap())[0];
    let shift = values(
        parameter_shift(
            &engine,
            &circuit,
            theta.detach(),
            None,
            1,
            &device,
            &Observable::z(0),
        )
        .unwrap(),
    )[0];
    close(&[autodiff, shift], &[-1.5 * 0.9f64.sin(); 2], 1e-10);
}

#[test]
fn batched_input_and_parameter_gradients() {
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 1);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::input(0)))
        .unwrap();
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let device = Device::flex().autodiff();
    let engine = DenseSimulator::default();
    let theta = parameter(0.2, &device).require_grad();
    let inputs = Tensor::<2>::from_data(
        TensorData::new(vec![0.1f64, 0.4, 0.7], [3, 1]),
        (&device, DType::F64),
    )
    .require_grad();
    let state = circuit
        .run(
            &engine,
            Some(theta.clone()),
            Some(inputs.clone()),
            3,
            &device,
        )
        .unwrap();
    let output = Observable::z(0).expectation(&engine, &state).unwrap();
    close(
        &values(output.clone()),
        &[0.3f64.cos(), 0.6f64.cos(), 0.9f64.cos()],
        1e-10,
    );
    let gradients = output.sum().backward();
    let expected: Vec<_> = [0.3f64, 0.6, 0.9].into_iter().map(|v| -v.sin()).collect();
    close(
        &values(theta.grad(&gradients).unwrap()),
        &[expected.iter().sum()],
        1e-10,
    );
    close(&values(inputs.grad(&gradients).unwrap()), &expected, 1e-10);
}

#[test]
fn native_burn_module_optimizer_learns_and_moves_to_inference() {
    let device = Device::flex().autodiff();
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let mut layer = QuantumLayer::new(
        circuit,
        vec![Observable::z(0)],
        vec![0.3],
        DenseSimulator::default(),
        &device,
    )
    .unwrap();
    assert_eq!(layer.num_params(), 1);
    let initial = values((layer.forward(None, 1).unwrap() + 1.).square())[0];
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..90 {
        let loss = (layer.forward(None, 1).unwrap() + 1.).square().mean();
        let grads = GradientsParams::from_grads(loss.backward(), &layer);
        layer = optimizer.step(0.08, layer, grads);
    }
    let final_loss = values((layer.forward(None, 1).unwrap() + 1.).square())[0];
    assert!(
        final_loss < 0.001 && final_loss < initial / 1000.,
        "{initial} -> {final_loss}"
    );
    let inference = layer.clone().to_device(&Device::flex());
    close(
        &values(inference.forward(None, 1).unwrap()),
        &values(layer.forward(None, 1).unwrap()),
        1e-6,
    );
}

#[test]
fn gradients_through_pure_to_density_and_noise() {
    let device = Device::flex().autodiff();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let theta = parameter(0.6, &device).require_grad();
    let state = circuit
        .run(&engine, Some(theta.clone()), None, 1, &device)
        .unwrap();
    let density = DensityMatrix::from_pure(&state, engine.limits)
        .unwrap()
        .channel(
            &engine,
            &[0],
            &KrausChannel::amplitude_damping(0.3).unwrap(),
        )
        .unwrap();
    let loss = density.expectation(&engine, &Observable::z(0)).unwrap();
    let grads = loss.backward();
    close(
        &values(theta.grad(&grads).unwrap()),
        &[-0.7 * 0.6f64.sin()],
        1e-10,
    );
}

#[test]
fn burn_config_and_burnpack_weight_record_roundtrip() {
    use burn::store::ModuleRecord;
    let device = Device::flex().autodiff();
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let config = QuantumLayerConfig::new(circuit, vec![Observable::z(0)], vec![0.8]);
    let original = config.init(&device).unwrap();
    let config_json = serde_json::to_string(&config).unwrap();
    let mut config: QuantumLayerConfig = serde_json::from_str(&config_json).unwrap();
    let bytes = original.clone().into_record().into_bytes().unwrap();
    config.initial = vec![0.1];
    let fresh = config.init(&device).unwrap();
    let restored = fresh
        .try_load_record(ModuleRecord::from_bytes(bytes).unwrap())
        .unwrap();
    close(
        &values(original.forward(None, 1).unwrap()),
        &values(restored.forward(None, 1).unwrap()),
        1e-7,
    );
}

#[test]
fn burn_balanced_checkpointing_preserves_gradients() {
    let device = Device::flex().autodiff().gradient_checkpointing();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(Register::qubits(2).unwrap(), 1, 0);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    circuit.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    let theta = parameter(0.7, &device).require_grad();
    let state = circuit
        .run(&engine, Some(theta.clone()), None, 1, &device)
        .unwrap();
    let gradients = Observable::z(1)
        .expectation(&engine, &state)
        .unwrap()
        .backward();
    close(
        &values(theta.grad(&gradients).unwrap()),
        &[-0.7f64.sin()],
        1e-10,
    );
}
