//! Device-independent instructions and parameter expressions.
use crate::gate::Axis;
use crate::{Error, Result, Unitary};
use serde::{Deserialize, Serialize};

/// An affine angle expression. References may be shared across gate occurrences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Angle {
    /// Constant angle in radians.
    Constant(f64),
    /// `scale * parameters[index] + offset`.
    Parameter {
        /// Parameter-vector index.
        index: usize,
        /// Chain-rule scale.
        scale: f64,
        /// Constant phase offset.
        offset: f64,
    },
    /// `scale * inputs[batch, index] + offset`.
    Input {
        /// Input-feature index.
        index: usize,
        /// Feature scale.
        scale: f64,
        /// Constant phase offset.
        offset: f64,
    },
}
impl Angle {
    /// Reference a trainable parameter without affine scaling.
    pub fn parameter(index: usize) -> Self {
        Self::Parameter {
            index,
            scale: 1.,
            offset: 0.,
        }
    }
    /// Reference a per-example input feature without affine scaling.
    pub fn input(index: usize) -> Self {
        Self::Input {
            index,
            scale: 1.,
            offset: 0.,
        }
    }
    pub(super) fn validate(&self, parameters: usize, inputs: usize) -> Result<()> {
        let valid = match self {
            Self::Constant(x) => x.is_finite(),
            Self::Parameter {
                index,
                scale,
                offset,
            } => *index < parameters && scale.is_finite() && offset.is_finite(),
            Self::Input {
                index,
                scale,
                offset,
            } => *index < inputs && scale.is_finite() && offset.is_finite(),
        };
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidValue(
                "invalid angle expression or binding index".into(),
            ))
        }
    }
}

/// Analytic two-level SU(2) rotation, including qubit Pauli rotations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rotation {
    /// Rotation generator axis.
    pub axis: Axis,
    /// Ordered levels defining the embedded Pauli basis.
    pub levels: [usize; 2],
    /// Angle expression.
    pub angle: Angle,
}
impl Rotation {
    /// A qubit rotation in the |0>, |1> subspace.
    pub fn qubit(axis: Axis, angle: Angle) -> Self {
        Self {
            axis,
            levels: [0, 1],
            angle,
        }
    }
}

/// Executable instructions. Structured controlled operations require circuit schema v2.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    /// Constant unitary on ordered target wires.
    Unitary {
        /// Local basis follows this target order.
        wires: Vec<usize>,
        /// Checked local operator.
        gate: Unitary,
    },
    /// Differentiable rotation on one finite-dimensional subsystem.
    Rotate {
        /// Target wire.
        wire: usize,
        /// Generator, levels, and binding.
        rotation: Rotation,
    },
    /// X on a qubit target, conditioned on arbitrary subsystem levels.
    ControlledX {
        /// Distinct `(wire, level)` controls; empty means unconditional X.
        controls: Vec<(usize, usize)>,
        /// Qubit wire, distinct from every control.
        target: usize,
    },
    /// Minus sign on the subspace selected by all controls.
    PhaseFlip {
        /// Distinct `(wire, level)` controls; empty means a global minus sign.
        controls: Vec<(usize, usize)>,
    },
}
