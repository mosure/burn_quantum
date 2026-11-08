//! Independent Boolean and scalar-complex references, on the selected CPU/GPU.
mod common;
#[path = "../support/device.rs"]
mod runtime;

use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{sat::*, *};
use common::{close, complex_close, complex_values, normalized, values};
use num_complex::Complex64;

fn device() -> Device {
    runtime::execution()
        .expect("requested backend unavailable")
        .device
}
fn state(register: Register, data: &[Complex64], batch: usize) -> StateVector {
    let shape = [batch, register.size()];
    StateVector::from_amplitudes(
        register,
        ComplexTensor::new(
            Tensor::from_data(
                TensorData::new(data.iter().map(|z| z.re as f32).collect(), shape),
                &device(),
            ),
            Tensor::from_data(
                TensorData::new(data.iter().map(|z| z.im as f32).collect(), shape),
                &device(),
            ),
        )
        .unwrap(),
        ResourceLimits::default(),
    )
    .unwrap()
}
fn truth(clauses: &[Vec<i32>], n: usize, x: usize) -> bool {
    clauses.iter().all(|clause| {
        clause.iter().any(|&literal| {
            let bit = (x >> (n - literal.unsigned_abs() as usize)) & 1;
            (bit == 1) == (literal > 0)
        })
    })
}
fn solver(n: usize, clauses: Vec<Vec<i32>>) -> GroverSat {
    GroverSat::new(Cnf::new(n, clauses).unwrap(), ResourceLimits::default()).unwrap()
}

#[path = "sat/cnf.rs"]
mod cnf;
#[path = "sat/contracts.rs"]
mod contracts;
#[path = "sat/oracle.rs"]
mod oracle;
#[path = "sat/search.rs"]
mod search;
