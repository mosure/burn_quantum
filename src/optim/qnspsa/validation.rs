//! Shared numerical and allocation invariants for configuration and measured samples.
use crate::{Error, Result as QuantumResult};

pub(super) fn invalid(message: &str) -> Error {
    Error::InvalidValue(format!("QN-SPSA: {message}"))
}
pub(super) fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}
pub(super) fn nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}
pub(super) fn finite(values: &[f64]) -> QuantumResult<()> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(invalid(
            "nonfinite parameters, samples, or numerical result",
        ))
    }
}
pub(super) fn matrix_size(parameters: usize, limit: usize) -> QuantumResult<usize> {
    if parameters == 0 {
        return Err(Error::Shape(
            "QN-SPSA needs a nonempty parameter vector".into(),
        ));
    }
    let size = parameters.checked_mul(parameters).ok_or(Error::Overflow)?;
    if size > limit {
        return Err(Error::ResourceLimit {
            resource: "QN-SPSA metric",
            required: size,
            limit,
        });
    }
    Ok(size)
}
