use super::{Pauli, PauliNoise, PauliProduct, Tableau, TableauLimits};
use crate::photonic::runtime::shot_seed;
use crate::photonic::{CompiledFusionNetwork, FusionOutcome, Parity};
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Limits for independent host stabilizer trajectories.
#[derive(Clone, Copy, Debug)]
pub struct StabilizerLimits {
    /// Bounds per tableau; one prepared and one working tableau coexist during sampling.
    pub tableau: TableauLimits,
    /// Maximum attempts in a single call.
    pub max_shots: usize,
    /// Maximum returned scalar records, including outputs, fusion outcomes and readouts.
    pub max_result_elements: usize,
}
impl Default for StabilizerLimits {
    fn default() -> Self {
        Self {
            tableau: TableauLimits::default(),
            max_shots: 1_000_000,
            max_result_elements: 1 << 24,
        }
    }
}
/// One unpostselected attempt, with stable global shot identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StabilizerShot {
    /// Global shot ID, used for partition-independent randomness.
    pub shot_id: u64,
    /// Complete fusion observations in compiled operation order.
    pub outcomes: Vec<FusionOutcome>,
    /// Digits in the compiled output order; 2 is vacuum.
    pub output_digits: Vec<usize>,
    /// Conditional-trajectory Pauli expectations before terminal Z measurement.
    /// Values are -1/0/+1, or zero when any requested factor is on a lost photon.
    /// Ensemble averages estimate unconditional encoded Pauli expectations.
    pub pauli_expectations: Vec<i8>,
}
/// Polynomial-memory reference Clifford engine for the compiled static fabric model.
#[derive(Clone, Copy, Debug, Default)]
pub struct StabilizerSimulator {
    /// Allocation and result limits.
    pub limits: StabilizerLimits,
}
impl StabilizerSimulator {
    /// Prepare ideal graph resources in a tableau without any dense amplitudes.
    /// Reject asymmetric attenuation before allocation.
    pub fn prepare(&self, plan: &CompiledFusionNetwork) -> Result<Tableau> {
        if plan.transmissions().iter().any(|eta| {
            let [a, b] = eta.values();
            a != b
        }) {
            return Err(Error::Unsupported(
                "stabilizer engine requires equal rail transmissions".into(),
            ));
        }
        let mut tableau = Tableau::zero(plan.program().num_photons(), self.limits.tableau)?;
        let mut offset = 0;
        for resource in plan.program().resources() {
            for q in 0..resource.state.qubits() {
                tableau.h(offset + q)?;
            }
            for &(a, b) in resource.state.edges() {
                tableau.cz(offset + a, offset + b)?;
            }
            for &q in resource.state.hadamards() {
                tableau.h(offset + q)?;
            }
            offset += resource.state.qubits();
        }
        Ok(tableau)
    }
    /// Sample independent attempts starting at a global shot ID.
    ///
    /// Versioned component streams come from `runtime::shot_seed`; partitioning
    /// the same range across calls preserves each record exactly. Each requested
    /// Pauli uses global photon IDs and must reference only unconsumed outputs.
    /// This sequential host implementation does not claim batched GPU frame sampling.
    pub fn sample(
        &self,
        plan: &CompiledFusionNetwork,
        start: u64,
        shots: usize,
        seed: u64,
        readouts: &[PauliProduct],
    ) -> Result<Vec<StabilizerShot>> {
        start
            .checked_add(u64::try_from(shots).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let fields = plan
            .outputs()
            .len()
            .checked_add(plan.fusions().len())
            .and_then(|n| n.checked_add(readouts.len()))
            .and_then(|n| n.checked_add(1))
            .ok_or(Error::Overflow)?;
        let entries = shots.checked_mul(fields).ok_or(Error::Overflow)?;
        for (resource, required, limit) in [
            ("stabilizer shots", shots, self.limits.max_shots),
            (
                "stabilizer result elements",
                entries,
                self.limits.max_result_elements,
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
        for product in readouts {
            if product
                .terms()
                .iter()
                .any(|(q, _)| !plan.outputs().iter().any(|p| p.0 == *q))
            {
                return Err(Error::Wires(
                    "readout references a consumed or absent photon".into(),
                ));
            }
        }
        let initial = self.prepare(plan)?;
        // Version-one engine namespace: independent of factories using the same seed.
        let seed = shot_seed(seed, 0, 0x5354_4142_494c_0001);
        let n = initial.qubits();
        let component_base = n.checked_mul(2).ok_or(Error::Overflow)?;
        let mut records = Vec::with_capacity(shots);
        for index in 0..shots {
            let shot_id = start + index as u64;
            let (mut state, lost, outcomes) =
                execute_trajectory(plan, &initial, shot_id, seed, PauliNoise::NONE, None)?;
            let pauli_expectations = readouts
                .iter()
                .map(|p| {
                    if p.terms().iter().any(|(q, _)| lost[*q]) {
                        Ok(0)
                    } else {
                        state.expectation(p)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let mut output_digits = Vec::with_capacity(plan.outputs().len());
            for (i, photon) in plan.outputs().iter().enumerate() {
                let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(
                    seed,
                    shot_id,
                    (component_base + plan.fusions().len() + i) as u64,
                ));
                output_digits.push(if lost[photon.0] {
                    2
                } else {
                    usize::from(
                        state
                            .measure(&product(&[(photon.0, Pauli::Z)]), rng.random())?
                            .negative,
                    )
                });
            }
            records.push(StabilizerShot {
                shot_id,
                outcomes,
                output_digits,
                pauli_expectations,
            });
        }
        Ok(records)
    }
}
fn product(terms: &[(usize, Pauli)]) -> PauliProduct {
    PauliProduct::new(terms.to_vec()).expect("distinct compiled operands")
}

/// Shared physical instrument for QEC boundary programs; `seed` is already namespaced.
pub(crate) fn execute_trajectory(
    plan: &CompiledFusionNetwork,
    initial: &Tableau,
    shot_id: u64,
    seed: u64,
    noise: PauliNoise,
    extra_loss: Option<&[bool]>,
) -> Result<(Tableau, Vec<bool>, Vec<FusionOutcome>)> {
    let mut state = initial.clone();
    let n = initial.qubits();
    let component_base = n * 2;
    let mut lost = vec![false; n];
    for (q, transmission) in plan.transmissions().iter().enumerate() {
        let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(seed, shot_id, (q * 2) as u64));
        if rng.random::<f64>() >= transmission.values()[0] || extra_loss.is_some_and(|mask| mask[q])
        {
            lost[q] = true;
            // A hidden Z result is a trajectory unraveling of the partial trace.
            state.measure(&product(&[(q, Pauli::Z)]), rng.random())?;
        }
    }
    let [px, py, pz] = noise.probabilities();
    for q in 0..n {
        let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(seed, shot_id, (q * 2 + 1) as u64));
        let u = rng.random::<f64>();
        if u < px + py {
            state.x(q)?;
        }
        if (px..px + py + pz).contains(&u) {
            state.z(q)?;
        }
    }
    let mut outcomes = Vec::with_capacity(plan.fusions().len());
    for (f, fusion) in plan.fusions().iter().enumerate() {
        let [a, b] = fusion.photons.map(|p| p.0);
        let mut rng =
            ChaCha8Rng::seed_from_u64(shot_seed(seed, shot_id, (component_base + f) as u64));
        let outcome = if lost[a] || lost[b] {
            // The coarse loss record discards information from either detector.
            for q in [a, b] {
                state.measure(&product(&[(q, Pauli::Z)]), rng.random())?;
            }
            FusionOutcome::Loss
        } else {
            let zz = state.measure(&product(&[(a, Pauli::Z), (b, Pauli::Z)]), rng.random())?;
            if zz.negative {
                let xx = state.measure(&product(&[(a, Pauli::X), (b, Pauli::X)]), rng.random())?;
                FusionOutcome::Success {
                    xx: if xx.negative {
                        Parity::Odd
                    } else {
                        Parity::Even
                    },
                }
            } else {
                let z = state.measure(&product(&[(a, Pauli::Z)]), rng.random())?;
                FusionOutcome::Failure { bit: z.negative }
            }
        };
        outcomes.push(outcome);
    }
    Ok((state, lost, outcomes))
}
