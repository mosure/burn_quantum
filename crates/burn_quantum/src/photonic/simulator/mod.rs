//! Exact subnormalized density branches for small encoded photonic networks.
mod sampling;
use super::{
    CompiledFusionNetwork, DualRail, FusionBranch, PhotonId, RailTransmission, StandardFusion,
};
use crate::{Circuit, DenseSimulator, DensityMatrix, Error, ResourceLimits, Result, StateVector};
use burn::tensor::{DType, Device, Tensor};
pub use sampling::FusionShot;

/// Per-state and branch-count preflights; not a total allocator/autodiff memory cap.
#[derive(Clone, Copy, Debug)]
pub struct SimulationLimits {
    /// Existing dense tensor/operator/index limits.
    pub dense: ResourceLimits,
    /// Maximum exact branches, including zero-weight outcomes (five per fusion).
    pub max_branches: usize,
}
impl Default for SimulationLimits {
    fn default() -> Self {
        Self {
            dense: ResourceLimits::default(),
            max_branches: 125,
        }
    }
}
/// Dense reference executor for the documented encoded single-photon model.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseFusionSimulator {
    /// Allocation and exact-enumeration budgets.
    pub limits: SimulationLimits,
}
/// Exact result with every branch weight and stable output identity retained.
#[derive(Clone, Debug)]
pub struct FusionRun {
    pub(crate) branches: Vec<FusionBranch>,
    outputs: Vec<PhotonId>,
    pub(crate) batch: usize,
}
impl FusionRun {
    /// Complete unpostselected instrument outcomes, including impossible branches.
    pub fn branches(&self) -> &[FusionBranch] {
        &self.branches
    }
    /// Physical site IDs corresponding to the remaining state/register order.
    pub fn outputs(&self) -> &[PhotonId] {
        &self.outputs
    }
    /// Sum of absolute branch probabilities; normalized physical inputs give one.
    pub fn total_probability(&self) -> Tensor<1> {
        self.branches
            .iter()
            .skip(1)
            .fold(self.branches[0].probability(), |sum, b| {
                sum + b.probability()
            })
    }
}
impl DenseFusionSimulator {
    /// Prepare all ideal resources, apply compiled local losses and enumerate fusions.
    ///
    /// Independent loss channels can be commuted before measurements on other
    /// photons because each photon is transported once and consumed at most once.
    /// There is no sampling, renormalization or host value readback in this method.
    pub fn execute(
        &self,
        plan: &CompiledFusionNetwork,
        batch: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<FusionRun> {
        let mut branches = 1usize;
        for _ in plan.fusions() {
            branches = branches.checked_mul(5).ok_or(Error::Overflow)?;
            if branches > self.limits.max_branches {
                return Err(Error::ResourceLimit {
                    resource: "fusion branches",
                    required: branches,
                    limit: self.limits.max_branches,
                });
            }
        }
        if self.limits.max_branches == 0 {
            return Err(Error::ResourceLimit {
                resource: "fusion branches",
                required: 1,
                limit: 0,
            });
        }
        let register = DualRail::register(plan.program().num_photons())?;
        let n = register.size();
        self.limits
            .dense
            .check_state(batch, n.checked_mul(n).ok_or(Error::Overflow)?)?;
        let engine = DenseSimulator::new(self.limits.dense);
        let mut circuit = Circuit::new(register.clone(), 0, 0);
        let mut offset = 0;
        for instance in &plan.program.resources {
            instance.state.append_to(&mut circuit, offset)?;
            offset += instance.state.qubits();
        }
        let initial = StateVector::zero(register, batch, device, dtype, self.limits.dense)?;
        let prepared = circuit.run_from(&engine, initial, None, None)?;
        let mut density = DensityMatrix::from_pure(&prepared, self.limits.dense)?;
        for (site, &transmission) in plan.transmissions.iter().enumerate() {
            if transmission != RailTransmission::IDEAL {
                density = density.channel(&engine, &[site], &transmission.channel()?)?;
            }
        }
        let mut current = vec![FusionBranch {
            outcomes: vec![],
            probability: density.trace(),
            state: Some(density),
        }];
        let mut active: Vec<_> = (0..plan.program().num_photons()).map(PhotonId).collect();
        for fusion in &plan.fusions {
            let sites = fusion.photons.map(|p| {
                active
                    .iter()
                    .position(|x| *x == p)
                    .expect("compiled photon consumed once")
            });
            let mut next = Vec::with_capacity(current.len() * 5);
            for branch in &current {
                for mut child in StandardFusion::branches(
                    &engine,
                    branch
                        .state
                        .as_ref()
                        .expect("compiled fusion has live inputs"),
                    sites,
                )? {
                    let mut outcomes = branch.outcomes.clone();
                    outcomes.extend(child.outcomes);
                    child.outcomes = outcomes;
                    next.push(child);
                }
            }
            active.retain(|p| !fusion.photons.contains(p));
            current = next;
        }
        Ok(FusionRun {
            branches: current,
            outputs: active,
            batch,
        })
    }
}
