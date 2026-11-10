//! Complete destructive quantum instrument for an unboosted dual-rail Bell analyzer.
use super::DualRail;
use crate::{ComplexTensor, DenseSimulator, DensityMatrix, Error, Matrix, Result};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// A Pauli-product eigenvalue, represented without an ambiguous bit convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Parity {
    /// Eigenvalue +1.
    Even,
    /// Eigenvalue -1.
    Odd,
}
/// Available fusion parities. Missing information is an erasure, not a random bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusionParities {
    /// X tensor X parity, when resolved.
    pub xx: Option<Parity>,
    /// Z tensor Z parity, when resolved.
    pub zz: Option<Parity>,
}
/// Outcomes for ideal PNR detection after the specified standard Bell analyzer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionOutcome {
    /// Resolved Psi+ or Psi-; ZZ is always odd in this analyzer convention.
    Success {
        /// XX parity (even for Psi+, odd for Psi-).
        xx: Parity,
    },
    /// Two identical logical Z results. XX is erased but ZZ is known even.
    Failure {
        /// False means 00, true means 11. These are destructive Z measurements.
        bit: bool,
    },
    /// At least one input is vacuum. This coarse record discards all parity information.
    Loss,
}
impl FusionOutcome {
    /// Return only the parity information actually revealed by this outcome.
    pub fn parities(self) -> FusionParities {
        match self {
            Self::Success { xx } => FusionParities {
                xx: Some(xx),
                zz: Some(Parity::Odd),
            },
            Self::Failure { .. } => FusionParities {
                xx: None,
                zz: Some(Parity::Even),
            },
            Self::Loss => FusionParities { xx: None, zz: None },
        }
    }
}

/// An exact branch with its unconditional weight and subnormalized surviving state.
/// Zero-weight branches are retained; no probability is clipped or postselected.
#[derive(Clone, Debug)]
pub struct FusionBranch {
    pub(crate) outcomes: Vec<FusionOutcome>,
    pub(crate) probability: Tensor<1>,
    pub(crate) state: Option<DensityMatrix>,
}
impl FusionBranch {
    /// Ordered fusion observations in this branch.
    pub fn outcomes(&self) -> &[FusionOutcome] {
        &self.outcomes
    }
    /// Absolute branch probability per batch entry, retaining the autodiff graph.
    pub fn probability(&self) -> Tensor<1> {
        self.probability.clone()
    }
    /// Subnormalized state whose trace equals the branch probability.
    /// None means all photons have been destructively measured.
    pub fn unnormalized_state(&self) -> Option<&DensityMatrix> {
        self.state.as_ref()
    }
    /// Explicit conditional normalization, rejecting impossible/too-small branches.
    ///
    /// This synchronizes for validation. All batch weights must exceed the given
    /// finite positive threshold. The returned density retains quotient gradients.
    pub fn conditional_state(&self, min_probability: f64) -> Result<Option<DensityMatrix>> {
        crate::error::tolerance(min_probability)?;
        if crate::complex::read(self.probability())?
            .iter()
            .any(|p| !p.is_finite() || *p <= min_probability)
        {
            return Err(Error::InvalidValue(
                "cannot condition on an impossible or too-small fusion branch".into(),
            ));
        }
        self.state
            .as_ref()
            .map(|state| {
                let b = state.batch_size();
                let probability = self.probability().reshape([b, 1, 1]);
                let elements = crate::error::checked_product(state.entries().dims())?;
                let limits = crate::ResourceLimits {
                    max_state_elements: elements,
                    ..Default::default()
                };
                DensityMatrix::from_entries(
                    state.register().clone(),
                    ComplexTensor::new(
                        state.entries().real() / probability.clone(),
                        state.entries().imag() / probability,
                    )?,
                    limits,
                )
            })
            .transpose()
    }
}

/// Ideal, unboosted type-II-style Bell analyzer resolving Psi+ and Psi-.
///
/// Success is 50% on independent maximally mixed code inputs, not a state-independent
/// Bernoulli event. Each branch projects then traces out both consumed sites.
/// The five projectors sum to identity, including the five-dimensional vacuum
/// sector. No multiphoton inputs, distinguishability errors or dark counts are modeled.
pub struct StandardFusion;
impl StandardFusion {
    /// Enumerate all five outcomes, preserving batch weights and surviving coherences.
    pub fn branches(
        engine: &DenseSimulator,
        state: &DensityMatrix,
        sites: [usize; 2],
    ) -> Result<Vec<FusionBranch>> {
        DualRail::validate(state.register())?;
        state.register().local_size(&sites)?;
        let keep: Vec<_> = (0..state.register().wires())
            .filter(|w| !sites.contains(w))
            .collect();
        let mut branches = Vec::with_capacity(5);
        for outcome in [
            FusionOutcome::Success { xx: Parity::Even },
            FusionOutcome::Success { xx: Parity::Odd },
            FusionOutcome::Failure { bit: false },
            FusionOutcome::Failure { bit: true },
            FusionOutcome::Loss,
        ] {
            let projected = state.left_right(engine, &sites, &projector(outcome)?)?;
            let probability = projected.trace();
            let reduced = if keep.is_empty() {
                None
            } else {
                Some(projected.reduced(&keep, engine.limits)?)
            };
            branches.push(FusionBranch {
                outcomes: vec![outcome],
                probability,
                state: reduced,
            });
        }
        Ok(branches)
    }
}
fn projector(outcome: FusionOutcome) -> Result<Matrix> {
    let mut data = vec![0.; 81];
    match outcome {
        FusionOutcome::Success { xx } => {
            // |01> and |10> have mixed-radix indices 1 and 3.
            data[10] = 0.5;
            data[3 * 9 + 3] = 0.5;
            let sign = if xx == Parity::Even { 1. } else { -1. };
            data[12] = sign * 0.5;
            data[3 * 9 + 1] = sign * 0.5;
        }
        FusionOutcome::Failure { bit } => {
            let index = if bit { 4 } else { 0 };
            data[index * 9 + index] = 1.;
        }
        FusionOutcome::Loss => {
            for a in 0..3 {
                for b in 0..3 {
                    if a == 2 || b == 2 {
                        let index = a * 3 + b;
                        data[index * 9 + index] = 1.;
                    }
                }
            }
        }
    }
    Matrix::real(9, data)
}
