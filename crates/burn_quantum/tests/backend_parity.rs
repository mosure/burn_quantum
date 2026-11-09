//! Portable F32 checks against Flex and independent analytic/scalar references.
//! Set BURN_QUANTUM_DEVICE=wgpu or cuda and enable that feature for hardware runs.
mod common;
#[path = "../support/device.rs"]
mod runtime;

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::store::ModuleRecord;
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{gate::Axis, gradient::parameter_shift, qml::*, *};
use common::{apply_reference, close, complex_close, complex_values, normalized, values};
use num_complex::Complex64;

const VALUE_TOL: f64 = 2e-5;
const GRAD_TOL: f64 = 5e-5;

fn devices() -> [Device; 2] {
    [
        Device::flex(),
        runtime::execution()
            .expect("requested backend unavailable")
            .device,
    ]
}

fn state32(register: Register, data: &[Complex64], batch: usize, device: &Device) -> StateVector {
    let shape = [batch, register.size()];
    StateVector::from_amplitudes(
        register,
        ComplexTensor::new(
            Tensor::from_data(
                TensorData::new(data.iter().map(|v| v.re as f32).collect(), shape),
                device,
            ),
            Tensor::from_data(
                TensorData::new(data.iter().map(|v| v.im as f32).collect(), shape),
                device,
            ),
        )
        .unwrap(),
        ResourceLimits::default(),
    )
    .unwrap()
}

#[test]
fn density_outer_products_match_scalar_complex_reference() {
    let data: Vec<_> = normalized(4, 3)
        .into_iter()
        .chain(normalized(4, 19))
        .collect();
    let expected: Vec<_> = data
        .chunks_exact(4)
        .flat_map(|row| {
            row.iter()
                .flat_map(move |a| row.iter().map(move |b| a * b.conj()))
        })
        .collect();
    for device in devices() {
        let state = state32(Register::qubits(2).unwrap(), &data, 2, &device);
        let density = DensityMatrix::from_pure(&state, ResourceLimits::default()).unwrap();
        let actual: Vec<_> = values(density.entries().real())
            .into_iter()
            .zip(values(density.entries().imag()))
            .map(|(re, im)| Complex64::new(re, im))
            .collect();
        complex_close(&actual, &expected, VALUE_TOL);
        density.validate(VALUE_TOL).unwrap();
    }
}

#[test]
fn entanglement_readout_density_and_sampling_on_selected_backend() {
    let engine = DenseSimulator::default();
    for d in [2, 3, 5] {
        let mut circuit = Circuit::new(Register::new(vec![d, d]).unwrap(), 0, 0);
        circuit
            .unitary(vec![0], Unitary::fourier(d).unwrap())
            .unwrap();
        circuit
            .unitary(vec![0, 1], Unitary::sum(d).unwrap())
            .unwrap();
        let expected: Vec<_> = (0..2 * d * d)
            .map(|i| {
                if (i % (d * d)) / d == i % d {
                    1. / d as f64
                } else {
                    0.
                }
            })
            .collect();
        for device in devices() {
            let state = circuit.run(&engine, None, None, 2, &device).unwrap();
            state.validate(VALUE_TOL).unwrap();
            close(&values(state.probabilities()), &expected, VALUE_TOL);
            let full = DensityMatrix::from_pure(&state, engine.limits).unwrap();
            for wire in 0..2 {
                let reduced = state.reduced_density(&[wire], engine.limits).unwrap();
                let traced = full.reduced(&[wire], engine.limits).unwrap();
                reduced.validate(VALUE_TOL).unwrap();
                close(&values(reduced.purity()), &[1. / d as f64; 2], VALUE_TOL);
                close(
                    &values(reduced.entries().real()),
                    &values(traced.entries().real()),
                    VALUE_TOL,
                );
                close(
                    &values(reduced.entries().imag()),
                    &values(traced.entries().imag()),
                    VALUE_TOL,
                );
            }
            let samples = Samples::from_state(&state, 256, 91, 512).unwrap();
            assert_eq!(samples, Samples::from_state(&state, 256, 91, 512).unwrap());
            assert!(
                samples
                    .outcomes
                    .iter()
                    .flatten()
                    .all(|&index| index / d == index % d)
            );
        }
    }
}

#[test]
fn mixed_radix_complex_contractions_match_independent_oracle() {
    let engine = DenseSimulator::default();
    for dimensions in [vec![2, 3, 2], vec![3, 2, 4]] {
        let register = Register::new(dimensions.clone()).unwrap();
        let initial: Vec<_> = normalized(register.size(), 3)
            .into_iter()
            .chain(normalized(register.size(), 17))
            .collect();
        for wires in [vec![0], vec![2], vec![2, 0], vec![1, 0]] {
            let gate = Unitary::fourier(register.local_size(&wires).unwrap()).unwrap();
            let expected = apply_reference(&initial, &dimensions, &wires, gate.matrix());
            for device in devices() {
                let input = state32(register.clone(), &initial, 2, &device);
                let output = engine.apply(&input, &wires, &gate).unwrap();
                complex_close(&complex_values(&output), &expected, VALUE_TOL);
                let restored = engine.apply(&output, &wires, &gate.adjoint()).unwrap();
                complex_close(&complex_values(&restored), &initial, VALUE_TOL);
                let density = DensityMatrix::from_pure(&output, engine.limits).unwrap();
                let reduced = output.reduced_density(&[2, 0], engine.limits).unwrap();
                let traced = density.reduced(&[2, 0], engine.limits).unwrap();
                close(
                    &values(reduced.entries().real()),
                    &values(traced.entries().real()),
                    VALUE_TOL,
                );
                close(
                    &values(reduced.entries().imag()),
                    &values(traced.entries().imag()),
                    VALUE_TOL,
                );
            }
        }
    }
}

fn reference_value(circuit: &Circuit, observable: &Observable, theta: f64) -> f64 {
    let device = Device::flex();
    let engine = DenseSimulator::default();
    let parameter = Tensor::from_data([theta], (&device, DType::F64));
    let state = circuit
        .run(&engine, Some(parameter), None, 1, &device)
        .unwrap();
    observable
        .expectation(&engine, &state)
        .unwrap()
        .into_scalar::<f64>()
}

#[test]
fn qubit_qudit_shared_rotation_gradients_match_f64_finite_difference() {
    let engine = DenseSimulator::default();
    for dimension in [2, 3, 4] {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let mut circuit = Circuit::new(Register::new(vec![dimension]).unwrap(), 1, 0);
            circuit
                .unitary(vec![0], Unitary::fourier(dimension).unwrap())
                .unwrap();
            for scale in [1., -0.3] {
                circuit
                    .rotate(
                        0,
                        Rotation {
                            axis,
                            levels: [0, dimension - 1],
                            angle: Angle::Parameter {
                                index: 0,
                                scale,
                                offset: 0.1,
                            },
                        },
                    )
                    .unwrap();
            }
            let mut entries = vec![0.; dimension * dimension];
            entries[0] = 0.7;
            entries[dimension * dimension - 1] = -0.3;
            for j in 1..dimension {
                entries[j] = 0.4;
                entries[j * dimension] = 0.4;
            }
            let observable =
                Observable::new(vec![0], Matrix::real(dimension, entries).unwrap()).unwrap();
            let fd = (reference_value(&circuit, &observable, 0.730001)
                - reference_value(&circuit, &observable, 0.729999))
                / 0.000002;
            for device in devices() {
                let parameter = Tensor::<1>::from_floats([0.73], &device);
                let shift =
                    parameter_shift(&engine, &circuit, parameter, None, 1, &device, &observable)
                        .unwrap();
                close(&values(shift), &[fd], GRAD_TOL);
                let device = device.autodiff();
                let theta = Tensor::<1>::from_floats([0.73], &device).require_grad();
                let state = circuit
                    .run(&engine, Some(theta.clone()), None, 1, &device)
                    .unwrap();
                let gradients = observable
                    .expectation(&engine, &state)
                    .unwrap()
                    .sum()
                    .backward();
                close(&values(theta.grad(&gradients).unwrap()), &[fd], GRAD_TOL);
            }
        }
    }
}

#[test]
fn batched_input_gradients_and_checkpointing_match_analytic_reference() {
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(Register::qubits(2).unwrap(), 1, 1);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::input(0)))
        .unwrap();
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    circuit.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    let expected: Vec<_> = [0.3f64, 0.6, 0.9].into_iter().map(|v| -v.sin()).collect();
    for base in devices() {
        for checkpoint in [false, true] {
            let device = if checkpoint {
                base.clone().autodiff().gradient_checkpointing()
            } else {
                base.clone().autodiff()
            };
            let theta = Tensor::<1>::from_floats([0.2], &device).require_grad();
            let inputs = Tensor::<2>::from_floats([[0.1], [0.4], [0.7]], &device).require_grad();
            let state = circuit
                .run(
                    &engine,
                    Some(theta.clone()),
                    Some(inputs.clone()),
                    3,
                    &device,
                )
                .unwrap();
            let output = Observable::z(1).expectation(&engine, &state).unwrap();
            close(
                &values(output.clone()),
                &[0.3f64.cos(), 0.6f64.cos(), 0.9f64.cos()],
                VALUE_TOL,
            );
            let gradients = output.sum().backward();
            close(
                &values(theta.grad(&gradients).unwrap()),
                &[expected.iter().sum()],
                GRAD_TOL,
            );
            close(
                &values(inputs.grad(&gradients).unwrap()),
                &expected,
                GRAD_TOL,
            );
        }
    }
}

#[test]
fn channels_and_density_gradients_match_analytic_reference() {
    let engine = DenseSimulator::default();
    for base in devices() {
        let one = StateVector::basis(
            Register::qubits(1).unwrap(),
            &[1],
            1,
            &base,
            DType::F32,
            engine.limits,
        )
        .unwrap();
        let rho = DensityMatrix::from_pure(&one, engine.limits).unwrap();
        for p in [0., 0.25, 1.] {
            let damped = rho
                .channel(&engine, &[0], &KrausChannel::amplitude_damping(p).unwrap())
                .unwrap();
            damped.validate(VALUE_TOL).unwrap();
            close(&values(damped.probabilities()), &[p, 1. - p], VALUE_TOL);
            let mixed = rho
                .channel(&engine, &[0], &KrausChannel::depolarizing(p).unwrap())
                .unwrap();
            close(
                &values(mixed.probabilities()),
                &[p / 2., 1. - p / 2.],
                VALUE_TOL,
            );
        }
        let device = base.autodiff();
        let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
        circuit
            .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
            .unwrap();
        let theta = Tensor::<1>::from_floats([0.6], &device).require_grad();
        let state = circuit
            .run(&engine, Some(theta.clone()), None, 1, &device)
            .unwrap();
        let damped = DensityMatrix::from_pure(&state, engine.limits)
            .unwrap()
            .channel(
                &engine,
                &[0],
                &KrausChannel::amplitude_damping(0.3).unwrap(),
            )
            .unwrap();
        let gradients = damped
            .expectation(&engine, &Observable::z(0))
            .unwrap()
            .sum()
            .backward();
        close(
            &values(theta.grad(&gradients).unwrap()),
            &[-0.7 * 0.6f64.sin()],
            GRAD_TOL,
        );
    }
}

fn kernel_loss_snapshot(device: Device) -> Vec<f64> {
    let device = device.autodiff();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 2);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::input(0)))
        .unwrap();
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    circuit
        .rotate(0, Rotation::qubit(Axis::Z, Angle::input(1)))
        .unwrap();
    let theta = Tensor::<1>::from_floats([0.3], &device).require_grad();
    let inputs =
        Tensor::<2>::from_floats([[0.4, 0.2], [1., -0.7], [0.8, 0.9]], &device).require_grad();
    let state = circuit
        .run(
            &engine,
            Some(theta.clone()),
            Some(inputs.clone()),
            3,
            &device,
        )
        .unwrap();
    let kernel = fidelity_kernel(&state, &state, engine.limits).unwrap();
    let targets = Tensor::from_floats([[1., 0.], [0., 1.], [0.5, 0.5]], &device);
    let probabilities =
        pauli_z_probabilities(Observable::z(0).expectation(&engine, &state).unwrap());
    close(
        &values(probabilities.clone()),
        &values(state.probabilities()),
        VALUE_TOL,
    );
    let loss =
        distribution_cross_entropy(probabilities, targets, 1e-7).unwrap() + kernel.clone().mean();
    let mut snapshot = values(kernel);
    snapshot.extend(values(loss.clone()));
    let gradients = loss.backward();
    snapshot.extend(values(theta.grad(&gradients).unwrap()));
    snapshot.extend(values(inputs.grad(&gradients).unwrap()));
    snapshot
}

#[test]
fn fidelity_kernel_loss_and_parameter_input_gradients_match_cpu() {
    let [cpu, selected] = devices();
    close(
        &kernel_loss_snapshot(selected),
        &kernel_loss_snapshot(cpu),
        GRAD_TOL,
    );
}

fn trained_layer(device: Device) -> (QuantumLayer, Vec<f64>) {
    let device = device.autodiff();
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    ry_cnot_layer(&mut circuit, &[0], 0).unwrap();
    let config = QuantumLayerConfig::new(circuit, vec![Observable::z(0)], vec![0.3]);
    let mut layer = config.init(&device).unwrap();
    let mut optimizer = AdamConfig::new().init();
    let mut curve = Vec::new();
    for _ in 0..20 {
        let loss = (layer.forward(None, 1).unwrap() + 1.).square().mean();
        curve.extend(values(loss.clone()));
        let gradients = GradientsParams::from_grads(loss.backward(), &layer);
        layer = optimizer.step(0.08, layer, gradients);
    }
    assert!(curve[19] < curve[0] / 3.);
    let bytes = layer.clone().into_record().into_bytes().unwrap();
    let restored = config
        .init(&device)
        .unwrap()
        .try_load_record(ModuleRecord::from_bytes(bytes).unwrap())
        .unwrap();
    close(
        &values(layer.parameters.val()),
        &values(restored.parameters.val()),
        1e-7,
    );
    (layer, curve)
}

#[test]
fn optimizer_records_and_gpu_to_cpu_movement_preserve_results() {
    let [cpu, selected] = devices();
    let (reference, cpu_curve) = trained_layer(cpu.clone());
    let (layer, selected_curve) = trained_layer(selected);
    close(&selected_curve, &cpu_curve, 2e-4);
    close(
        &values(layer.parameters.val()),
        &values(reference.parameters.val()),
        GRAD_TOL,
    );
    let moved = layer.clone().to_device(&cpu);
    close(
        &values(moved.forward(None, 1).unwrap()),
        &values(layer.forward(None, 1).unwrap()),
        VALUE_TOL,
    );
}

#[test]
fn cross_device_inputs_are_rejected_and_explicit_transfers_work() {
    let [cpu, selected] = devices();
    let re = Tensor::<2>::from_floats([[1., 0.]], &cpu);
    let im = Tensor::<2>::zeros([1, 2], &selected);
    if cpu != selected {
        assert!(matches!(
            ComplexTensor::new(re.clone(), im),
            Err(Error::TensorContext)
        ));
    }
    let amplitudes = ComplexTensor::new(re, Tensor::zeros([1, 2], &cpu))
        .unwrap()
        .to_device(&selected);
    assert_eq!(amplitudes.device(), selected);
    let state = StateVector::from_amplitudes(
        Register::qubits(1).unwrap(),
        amplitudes,
        ResourceLimits::default(),
    )
    .unwrap();
    state.validate(VALUE_TOL).unwrap();
    close(&values(state.probabilities()), &[1., 0.], VALUE_TOL);
}
