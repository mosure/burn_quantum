//! Full-precision contractions for quantum amplitudes.
//!
//! CubeCL 0.11's accelerated matmul may convert F32 operands to TF32. That can
//! violate normalization even for a short unitary circuit. Keep native Flex
//! GEMM, but express other devices' dot products as ordinary multiply/reduce
//! operations: no lower-precision tensor-core matmul, no readback, native AD.

use burn::tensor::Tensor;

/// Batched matrix multiplication with explicit same-dtype products/reductions.
/// Callers validate shapes, contexts, and allocation budgets before entering.
pub(crate) fn matmul(lhs: Tensor<3>, rhs: Tensor<3>) -> Tensor<3> {
    #[cfg(feature = "flex")]
    if lhs.device() == burn::tensor::Device::flex() {
        return lhs.matmul(rhs);
    }

    // Evaluate one output column at a time. Each product is the size of the
    // broadcast LHS, avoiding a [batch, rows, columns, reduction] allocation.
    // Both that product and the completed output fit the caller's state budget.
    let columns = rhs.dims()[2];
    let rhs = rhs.transpose();
    let outputs = (0..columns)
        .map(|column| {
            let vector = rhs.clone().slice_dim(1, column..column + 1);
            (lhs.clone() * vector).sum_dim(2)
        })
        .collect();
    Tensor::cat(outputs, 2)
}

pub(crate) fn matmul2(lhs: Tensor<2>, rhs: Tensor<2>) -> Tensor<2> {
    let [m, k] = lhs.dims();
    let [_, n] = rhs.dims();
    matmul(lhs.reshape([1, m, k]), rhs.reshape([1, k, n])).reshape([m, n])
}
