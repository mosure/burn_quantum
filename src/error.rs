//! Recoverable validation errors at the quantum API boundary.

/// Errors detected before dispatching invalid quantum operations to Burn.
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum Error {
    /// A local dimension was less than two, or a register was empty.
    #[error("invalid register dimensions: {0}")]
    InvalidDimensions(String),
    /// A size product exceeded the host's address space.
    #[error("dimension or allocation size overflow")]
    Overflow,
    /// A configured allocation budget would be exceeded.
    #[error("{resource} requires {required} elements, limit is {limit}")]
    ResourceLimit {
        /// Resource being allocated.
        resource: &'static str,
        /// Required element count.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Tensor, matrix, batch, or parameter shape mismatch.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// Target wires were invalid or repeated.
    #[error("invalid wires: {0}")]
    Wires(String),
    /// A basis digit or index was out of range.
    #[error("basis index out of range")]
    Basis,
    /// A scalar, matrix, or state violated a mathematical invariant.
    #[error("invalid quantum value: {0}")]
    InvalidValue(String),
    /// Inputs were on different hardware devices or had different dtypes.
    #[error("incompatible tensor devices, dtypes, or autodiff contexts")]
    TensorContext,
    /// Operation requires a capability the engine does not implement.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    /// Tensor readback could not be decoded.
    #[error("tensor readback failed: {0}")]
    Readback(String),
}

/// Quantum API result.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn checked_product(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values
        .into_iter()
        .try_fold(1usize, |n, d| n.checked_mul(d).ok_or(Error::Overflow))
}

pub(crate) fn tolerance(value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(Error::InvalidValue(
            "tolerance must be finite and in (0, 1)".into(),
        ))
    }
}
