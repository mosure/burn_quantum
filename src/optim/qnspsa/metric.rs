//! Host F64 spectral regularization and solves; no tensor backend assumptions.
use super::validation::{finite, invalid, positive};
use crate::Result as QuantumResult;
use nalgebra::{DMatrix, DVector, linalg::SymmetricEigen};

pub(super) fn regularized(metric: &[f64], n: usize, beta: f64) -> QuantumResult<Vec<f64>> {
    let (vectors, spectrum) = regularize(metric, n, beta)?;
    let matrix = &vectors * DMatrix::from_diagonal(&spectrum) * vectors.transpose();
    let output: Vec<_> = (0..n)
        .flat_map(|i| {
            let matrix = &matrix;
            (0..n).map(move |j| matrix[(i, j)])
        })
        .collect();
    finite(&output)?;
    Ok(output)
}

pub(super) fn solve(
    metric: &[f64],
    n: usize,
    beta: f64,
    gradient: Vec<f64>,
) -> QuantumResult<Vec<f64>> {
    let (vectors, spectrum) = regularize(metric, n, beta)?;
    let projected = vectors.transpose() * DVector::from_vec(gradient);
    let solved = &vectors * projected.component_div(&spectrum);
    Ok(solved.as_slice().to_vec())
}

// Spectral absolute value, not elementwise abs and not the objective's Hessian.
fn regularize(metric: &[f64], n: usize, beta: f64) -> QuantumResult<(DMatrix<f64>, DVector<f64>)> {
    finite(metric)?;
    let matrix = DMatrix::from_row_slice(n, n, metric);
    let eigen = SymmetricEigen::try_new(matrix, 16.0 * f64::EPSILON, 10_000)
        .ok_or_else(|| invalid("metric eigendecomposition did not converge"))?;
    finite(eigen.eigenvalues.as_slice())?;
    finite(eigen.eigenvectors.as_slice())?;
    let spectrum = eigen.eigenvalues.map(|x| x.abs() + beta);
    if spectrum.iter().any(|&x| !positive(x)) {
        return Err(invalid(
            "regularized metric is not finite positive definite",
        ));
    }
    Ok((eigen.eigenvectors, spectrum))
}
