//! Wide full-F32 complex contraction and AD regression for buffered GPU execution.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{Tensor, TensorData};
use burn_quantum::{ComplexTensor, DenseSimulator, Register, StateVector};
use common::{close, values};
use num_complex::Complex64;
use std::time::Instant;
#[test]
fn wide_batched_complex_contraction_and_gradients_match_scalar_oracle() {
    let device = runtime::execution().unwrap().device.autodiff();
    const B: usize = 2;
    const N: usize = 330;
    let inputs: Vec<_> = (0..B * N)
        .map(|i| Complex64::new((i as f64 * 0.13).sin() / 20., (i as f64 * 0.19).cos() / 20.))
        .collect();
    let operators: Vec<_> = (0..B * N * N)
        .map(|i| {
            Complex64::new(
                (i as f64 * 0.023).cos() / 40.,
                (i as f64 * 0.037).sin() / 40.,
            )
        })
        .collect();
    let weight: Vec<_> = (0..B * N)
        .map(|i| Complex64::new((i as f64 * 0.11).cos() / 10., (i as f64 * 0.07).sin() / 10.))
        .collect();
    let re = Tensor::from_data(
        TensorData::new(inputs.iter().map(|v| v.re as f32).collect(), [B, N]),
        &device,
    )
    .require_grad();
    let im = Tensor::from_data(
        TensorData::new(inputs.iter().map(|v| v.im as f32).collect(), [B, N]),
        &device,
    )
    .require_grad();
    let state = StateVector::from_amplitudes(
        Register::new(vec![N]).unwrap(),
        ComplexTensor::new(re.clone(), im.clone()).unwrap(),
        Default::default(),
    )
    .unwrap();
    let op_re = Tensor::from_data(
        TensorData::new(operators.iter().map(|v| v.re as f32).collect(), [B, N, N]),
        &device,
    )
    .require_grad();
    let op_im = Tensor::from_data(
        TensorData::new(operators.iter().map(|v| v.im as f32).collect(), [B, N, N]),
        &device,
    )
    .require_grad();
    let op = ComplexTensor::new(op_re.clone(), op_im.clone()).unwrap();
    let mut oracle = vec![Complex64::new(0., 0.); B * N];
    let mut gradient = vec![Complex64::new(0., 0.); B * N];
    let mut operator_gradient = vec![Complex64::new(0., 0.); B * N * N];
    for b in 0..B {
        for row in 0..N {
            for col in 0..N {
                let u = operators[b * N * N + row * N + col];
                oracle[b * N + row] += u * inputs[b * N + col];
                gradient[b * N + col] += u.conj() * weight[b * N + row];
                operator_gradient[b * N * N + row * N + col] =
                    weight[b * N + row] * inputs[b * N + col].conj();
            }
        }
    }
    let started = Instant::now();
    eprintln!("wide contraction starting apply");
    let output = DenseSimulator::default()
        .apply_tensor(&state, &[0], &op)
        .unwrap();
    eprintln!("wide contraction apply built; reading outputs");
    close(
        &values(output.amplitudes().real()),
        &oracle.iter().map(|z| z.re).collect::<Vec<_>>(),
        3e-6,
    );
    close(
        &values(output.amplitudes().imag()),
        &oracle.iter().map(|z| z.im).collect::<Vec<_>>(),
        3e-6,
    );
    let forward = started.elapsed();
    eprintln!("wide contraction forward completed in {forward:?}; starting backward");
    let wr = Tensor::from_data(
        TensorData::new(weight.iter().map(|v| v.re as f32).collect(), [B, N]),
        &device,
    );
    let wi = Tensor::from_data(
        TensorData::new(weight.iter().map(|v| v.im as f32).collect(), [B, N]),
        &device,
    );
    let grads = (output.amplitudes().real() * wr + output.amplitudes().imag() * wi)
        .sum()
        .backward();
    close(
        &values(re.grad(&grads).unwrap()),
        &gradient.iter().map(|z| z.re).collect::<Vec<_>>(),
        3e-6,
    );
    close(
        &values(im.grad(&grads).unwrap()),
        &gradient.iter().map(|z| z.im).collect::<Vec<_>>(),
        3e-6,
    );
    close(
        &values(op_re.grad(&grads).unwrap()),
        &operator_gradient.iter().map(|z| z.re).collect::<Vec<_>>(),
        3e-6,
    );
    close(
        &values(op_im.grad(&grads).unwrap()),
        &operator_gradient.iter().map(|z| z.im).collect::<Vec<_>>(),
        3e-6,
    );
    eprintln!(
        "wide contraction forward={forward:?}, forward+backward={:?}",
        started.elapsed()
    );
}
