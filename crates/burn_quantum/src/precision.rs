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
    let mut outputs = Vec::with_capacity(columns);
    // A wide fan-out of reduction branches is expensive for Burn's fusion planner.
    // Flush bounded groups without reading tensors, changing device or detaching AD.
    // Device::flush only dispatches buffered work; eager devices treat it as a no-op.
    const COLUMN_BLOCK: usize = 16;
    let bounded_fusion = cfg!(feature = "fusion") && columns > COLUMN_BLOCK;
    for column in 0..columns {
        let vector = rhs.clone().slice_dim(1, column..column + 1);
        outputs.push((lhs.clone() * vector).sum_dim(2));
        if bounded_fusion && ((column + 1) % COLUMN_BLOCK == 0 || column + 1 == columns) {
            lhs.device()
                .flush()
                .expect("dispatching full-precision contraction failed");
        }
    }
    if !bounded_fusion {
        return Tensor::cat(outputs, 2);
    }
    // A single hundreds-input concat remains expensive even when its inputs have
    // been dispatched. Bound that fan-in too; each intermediate fits the final
    // output allocation, and ordinary Tensor::cat preserves native reverse AD.
    const CONCAT_FAN_IN: usize = 16;
    while outputs.len() > 1 {
        let mut next = Vec::with_capacity(outputs.len().div_ceil(CONCAT_FAN_IN));
        let mut inputs = outputs.into_iter();
        loop {
            let group: Vec<_> = inputs.by_ref().take(CONCAT_FAN_IN).collect();
            if group.is_empty() {
                break;
            }
            next.push(Tensor::cat(group, 2));
            lhs.device()
                .flush()
                .expect("dispatching contraction concatenation failed");
        }
        outputs = next;
    }
    outputs.pop().expect("nonempty matrix output")
}

pub(crate) fn matmul2(lhs: Tensor<2>, rhs: Tensor<2>) -> Tensor<2> {
    let [m, k] = lhs.dims();
    let [_, n] = rhs.dims();
    matmul(lhs.reshape([1, m, k]), rhs.reshape([1, k, n])).reshape([m, n])
}
