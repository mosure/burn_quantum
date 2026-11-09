#![allow(dead_code)]
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{ComplexTensor, Matrix, Register, ResourceLimits, StateVector};
use num_complex::Complex64;

pub fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f64> {
    tensor
        .into_data()
        .convert::<f64>()
        .try_to_vec::<f64>()
        .unwrap()
}
pub fn close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= tol,
            "entry {i}: {a} != {e}, tolerance {tol}"
        );
    }
}
pub fn complex_values(state: &StateVector) -> Vec<Complex64> {
    values(state.amplitudes().real())
        .into_iter()
        .zip(values(state.amplitudes().imag()))
        .map(|(re, im)| Complex64::new(re, im))
        .collect()
}
pub fn complex_close(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((*a - *e).norm() <= tol, "entry {i}: {a} != {e}");
    }
}
pub fn state(
    register: Register,
    amplitudes: &[Complex64],
    batch: usize,
    device: &Device,
) -> StateVector {
    let shape = [batch, register.size()];
    let re = Tensor::from_data(
        TensorData::new(amplitudes.iter().map(|x| x.re).collect(), shape),
        (device, DType::F64),
    );
    let im = Tensor::from_data(
        TensorData::new(amplitudes.iter().map(|x| x.im).collect(), shape),
        (device, DType::F64),
    );
    StateVector::from_amplitudes(
        register,
        ComplexTensor::new(re, im).unwrap(),
        ResourceLimits::default(),
    )
    .unwrap()
}
pub fn normalized(n: usize, offset: usize) -> Vec<Complex64> {
    let mut values: Vec<_> = (0..n)
        .map(|i| {
            Complex64::new(
                ((i + offset) as f64 * 1.23).sin(),
                ((i + offset) as f64 * 0.71).cos(),
            )
        })
        .collect();
    let norm = values.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    for value in &mut values {
        *value /= norm;
    }
    values
}
// Deliberately scalar oracle: enumerate every input/output local basis pair.
// It does not use the simulator's permutation or tensor contraction code.
pub fn apply_reference(
    amplitudes: &[Complex64],
    dimensions: &[usize],
    wires: &[usize],
    matrix: &Matrix,
) -> Vec<Complex64> {
    let n: usize = dimensions.iter().product();
    let k = matrix.dimension();
    let mut output = vec![Complex64::new(0., 0.); amplitudes.len()];
    for batch in 0..amplitudes.len() / n {
        for old in 0..n {
            let mut digits = vec![0; dimensions.len()];
            let mut index = old;
            for w in (0..dimensions.len()).rev() {
                digits[w] = index % dimensions[w];
                index /= dimensions[w];
            }
            let mut local = 0;
            for &w in wires {
                local = local * dimensions[w] + digits[w];
            }
            for target in 0..k {
                let mut result = digits.clone();
                let mut remaining = target;
                for &w in wires.iter().rev() {
                    result[w] = remaining % dimensions[w];
                    remaining /= dimensions[w];
                }
                let mut new = 0;
                for (digit, dimension) in result.iter().zip(dimensions) {
                    new = new * dimension + digit;
                }
                output[batch * n + new] +=
                    matrix.entries()[target * k + local] * amplitudes[batch * n + old];
            }
        }
    }
    output
}
