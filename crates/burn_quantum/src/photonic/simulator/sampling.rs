use super::FusionRun;
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// One unconditional attempt; failed/lost fusions and vacuum outputs remain observable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionShot {
    /// Index into FusionRun::branches, identifying the complete fusion record.
    pub branch: usize,
    /// Computational output digits in FusionRun::outputs order; two means vacuum.
    pub output_digits: Vec<usize>,
}
impl FusionRun {
    /// Seeded host sampling of the joint branch/output distribution, for every batch.
    ///
    /// This is an explicit synchronous nondifferentiable readback. It bounds the
    /// total number of returned attempts; no retry-until-success postselection.
    pub fn sample(
        &self,
        shots_per_batch: usize,
        seed: u64,
        max_total_shots: usize,
    ) -> Result<Vec<Vec<FusionShot>>> {
        let required = self
            .batch
            .checked_mul(shots_per_batch)
            .ok_or(Error::Overflow)?;
        if required > max_total_shots {
            return Err(Error::ResourceLimit {
                resource: "photonic shots",
                required,
                limit: max_total_shots,
            });
        }
        let mut distributions = Vec::with_capacity(self.branches.len());
        let mut dimensions = Vec::with_capacity(self.branches.len());
        for branch in &self.branches {
            if let Some(state) = branch.unnormalized_state() {
                distributions.push(crate::complex::read(state.probabilities())?);
                dimensions.push(state.register().size());
            } else {
                distributions.push(crate::complex::read(branch.probability())?);
                dimensions.push(1);
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut batches = Vec::with_capacity(self.batch);
        for batch in 0..self.batch {
            let mut cdf = vec![];
            let mut total = 0.;
            for (branch, values) in distributions.iter().enumerate() {
                let n = dimensions[branch];
                for (index, &p) in values[batch * n..(batch + 1) * n].iter().enumerate() {
                    if !p.is_finite() || p < -1e-6 {
                        return Err(Error::InvalidValue(
                            "invalid joint photonic probability".into(),
                        ));
                    }
                    total += p.max(0.);
                    cdf.push((total, branch, index));
                }
            }
            if (total - 1.).abs() > 1e-5 {
                return Err(Error::InvalidValue(
                    "joint photonic probabilities are not normalized".into(),
                ));
            }
            let mut shots = Vec::with_capacity(shots_per_batch);
            for _ in 0..shots_per_batch {
                let draw = rng.random::<f64>() * total;
                let index = cdf
                    .partition_point(|&(p, _, _)| p <= draw)
                    .min(cdf.len() - 1);
                let (_, branch, basis) = cdf[index];
                let output_digits = self.branches[branch]
                    .state
                    .as_ref()
                    .map(|s| s.register().decode(basis))
                    .transpose()?
                    .unwrap_or_default();
                shots.push(FusionShot {
                    branch,
                    output_digits,
                });
            }
            batches.push(shots);
        }
        Ok(batches)
    }
}
