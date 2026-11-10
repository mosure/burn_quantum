use super::*;
use crate::photonic::qec::{PlanarSurgery, SurgeryAttempt};
use crate::photonic::runtime::shot_seed;
use crate::photonic::stabilizer::Tableau;
use crate::{Error, Result};
/// Preflight limits for an explicitly consumed six-ring surgery circuit.
#[derive(Clone, Copy, Debug)]
pub struct SixRingSurgeryLimits {
    /// Maximum source invocations if every check completes.
    pub max_rings: usize,
    /// Maximum retained check-attempt transcripts.
    pub max_checks: usize,
}
impl Default for SixRingSurgeryLimits {
    fn default() -> Self {
        Self {
            max_rings: 1 << 20,
            max_checks: 1 << 18,
        }
    }
}
/// Growing-distance local seam surgery lowered to actual six-ring gate teleportation.
///
/// Spatial code distance belongs to the supplied planar geometry. The unencoded
/// standard analyzers make this particular physical realization exponentially
/// unlikely to finish. It does not inherit the six-ring bulk erasure threshold.
#[derive(Clone, Debug)]
pub struct SixRingSurgery {
    surgery: PlanarSurgery,
    rings: usize,
}
/// A complete physical transcript, including rejected checks and controller errors.
#[derive(Clone, Debug)]
pub struct OpticalSurgeryRun {
    /// The ideal-code controller can reject a noisy syndrome. Its error and all
    /// physical records are retained rather than converted into successful data.
    pub logical: Result<SurgeryAttempt>,
    /// Actual optical instruments, including the last rejected one.
    pub checks: Vec<OpticalCheckAttempt>,
    /// Summed actual operations, not the maximum successful schedule.
    pub cost: OpticalSurgeryCost,
    /// Ideal forced-success importance weight in log space; None for physical attempts.
    pub log_herald_weight: Option<f64>,
}
impl SixRingSurgery {
    /// Validate every bounded-support check and complete worst-case resource budget.
    pub fn compile(surgery: &PlanarSurgery, limits: SixRingSurgeryLimits) -> Result<Self> {
        let per_round = surgery
            .merged_checks()
            .iter()
            .chain(surgery.split_checks())
            .try_fold(0usize, |sum, p| {
                let w = p.terms().len();
                if !(1..=6).contains(&w) {
                    return Err(Error::Unsupported(
                        "six-ring lowering requires weight 1..=6 checks".into(),
                    ));
                }
                sum.checked_add(w).ok_or(Error::Overflow)
            })?;
        let rings = per_round
            .checked_mul(surgery.code().distance())
            .ok_or(Error::Overflow)?;
        for (resource, required, limit) in [
            ("six-ring surgery resources", rings, limits.max_rings),
            (
                "six-ring surgery check records",
                surgery.measurements(),
                limits.max_checks,
            ),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        // Every possible counter increment is bounded by this checked number.
        rings.checked_mul(64).ok_or(Error::Overflow)?;
        Ok(Self {
            surgery: surgery.clone(),
            rings,
        })
    }
    /// Number of ring resources on a fully successful physical path.
    pub fn scheduled_rings(&self) -> usize {
        self.rings
    }
    /// Number of fresh syndrome photons on a fully successful physical path.
    pub fn scheduled_checks(&self) -> usize {
        self.surgery.measurements()
    }
    /// Ideal standard-fusion completion probability in logarithmic form.
    /// Assumes unit source availability and ideal detection/storage.
    pub fn ideal_log_herald_probability(&self) -> f64 {
        -(self.rings as f64) * 4f64.ln()
    }
    /// Execute physical checks through the surgery controller, preserving partial outcomes.
    /// Scratch must be outside both complete data patches, not just the current check.
    pub fn execute(
        &self,
        state: &mut Tableau,
        offsets: [usize; 2],
        workspace: [usize; 7],
        instrument: SixRingCheckInstrument,
        sampling: SixRingSampling,
        seed: u64,
    ) -> Result<OpticalSurgeryRun> {
        instrument.validate_sampling(sampling)?;
        let n = self.surgery.code().qubits();
        let ends = offsets.map(|o| o.checked_add(n).ok_or(Error::Overflow));
        let [a, b] = ends;
        let ends = [a?, b?];
        if offsets[0].abs_diff(offsets[1]) < n
            || ends.iter().any(|&end| end > state.qubits())
            || workspace.iter().enumerate().any(|(i, &q)| {
                q >= state.qubits()
                    || workspace[..i].contains(&q)
                    || (0..2).any(|j| (offsets[j]..ends[j]).contains(&q))
            })
        {
            return Err(Error::Wires(
                "invalid optical surgery data/scratch layout".into(),
            ));
        }
        let mut checks = Vec::with_capacity(self.scheduled_checks());
        let mut cost = OpticalSurgeryCost::default();
        let logical = self.surgery.execute_with(state, offsets, |state, request| {
            let record = instrument.measure(
                state,
                &request.observable,
                workspace,
                sampling,
                shot_seed(seed, checks.len() as u64, 0x5355_5247_4552_0001),
            )?;
            let measurement = record.measurement;
            cost.add(record.cost);
            checks.push(record);
            Ok(measurement)
        });
        let log_herald_weight = (sampling == SixRingSampling::ConditionedReference)
            .then_some(-(cost.rings_emitted as f64) * 4f64.ln());
        Ok(OpticalSurgeryRun {
            logical,
            checks,
            cost,
            log_herald_weight,
        })
    }
}
