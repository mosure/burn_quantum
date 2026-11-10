//! Explicit herald probabilities and branch-weighted objectives for exact photonic models.
use super::{FusionBranch, FusionOutcome};
use crate::{DenseSimulator, Error, Observable, Result};
use burn::tensor::Tensor;

/// A heralded observable retaining both success cost and its unnormalized numerator.
#[derive(Clone, Debug)]
pub struct HeraldedReadout {
    probability: Tensor<1>,
    numerator: Tensor<1>,
}
impl HeraldedReadout {
    /// Probability per attempt, including all failed attempts in the denominator.
    pub fn probability(&self) -> Tensor<1> {
        self.probability.clone()
    }
    /// E[accept * observable], with no implicit postselection.
    pub fn unconditional(&self) -> Tensor<1> {
        self.numerator.clone()
    }
    /// E[observable | accept], with gradients through the success denominator.
    /// Synchronizes for explicit zero/near-zero success validation.
    pub fn conditional(&self, minimum_probability: f64) -> Result<Tensor<1>> {
        crate::error::tolerance(minimum_probability)?;
        if crate::complex::read(self.probability())?
            .iter()
            .any(|p| !p.is_finite() || *p <= minimum_probability)
        {
            return Err(Error::InvalidValue(
                "herald probability below conditioning threshold".into(),
            ));
        }
        Ok(self.unconditional() / self.probability())
    }
}
/// Accumulate exact subnormalized branch readouts selected only by classical records.
///
/// The observable uses the surviving register's wire indices. Selection is a static
/// predicate on records, not tensor values. Gradients include the branch weights.
/// An accepted terminal branch has no quantum register and is an error here.
pub fn heralded_readout(
    branches: &[FusionBranch],
    engine: &DenseSimulator,
    observable: &Observable,
    accept: impl Fn(&[FusionOutcome]) -> bool,
) -> Result<HeraldedReadout> {
    let first = branches
        .first()
        .ok_or_else(|| Error::InvalidValue("empty instrument".into()))?;
    let mut probability = first.probability() * 0.;
    let mut numerator = probability.clone();
    for branch in branches {
        let p = branch.probability();
        if p.dims() != probability.dims() {
            return Err(Error::Shape("inconsistent branch batches".into()));
        }
        crate::complex::compatible(&p, &probability)?;
        if accept(branch.outcomes()) {
            let rho = branch.unnormalized_state().ok_or_else(|| {
                Error::InvalidValue("accepted branch has no quantum output".into())
            })?;
            numerator = numerator + rho.expectation(engine, observable)?;
            probability = probability + p;
        }
    }
    Ok(HeraldedReadout {
        probability,
        numerator,
    })
}
