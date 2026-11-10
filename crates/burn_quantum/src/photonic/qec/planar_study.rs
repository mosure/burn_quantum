use super::{MemoryStudy, PlanarCode, PlanarCost, PlanarDecodeBudget, PlanarRecovery};
use crate::photonic::runtime::shot_seed;
use crate::photonic::stabilizer::{Pauli, PauliNoise, PauliProduct, Tableau, TableauLimits};
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
/// Data noise before each ideal syndrome round. Loss traces a physical site and
/// replaces it; local Pauli errors are unheralded. Syndrome/encoder gates are ideal.
#[derive(Clone, Copy, Debug)]
pub struct PlanarMemoryNoise {
    loss: f64,
    pauli: PauliNoise,
}
impl PlanarMemoryNoise {
    /// Validate independent per-round data loss and local X/Y/Z probabilities.
    pub fn new(loss: f64, pauli: PauliNoise) -> Result<Self> {
        if !loss.is_finite() || !(0.0..=1.).contains(&loss) {
            return Err(Error::InvalidValue("invalid planar data loss".into()));
        }
        Ok(Self { loss, pauli })
    }
    /// Convert equal-rail transmission per storage tick to per-round erasure.
    /// The tick count is caller-supplied physical dwell time. This is not a gate
    /// scheduling model, and it does not add attenuation a second time.
    pub fn storage(
        transmission: crate::photonic::RailTransmission,
        ticks: usize,
        pauli: PauliNoise,
    ) -> Result<Self> {
        let [a, b] = transmission.values();
        if a != b {
            return Err(Error::Unsupported(
                "planar erasures require equal rail attenuation".into(),
            ));
        }
        Self::new(1. - a.powf(ticks as f64), pauli)
    }
}
/// Finite logical memory experiment, retaining every attempted input and cost.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanarMemoryStudy {
    /// Counts of successful, logically failed, ambiguous and budget-limited inputs.
    pub outcomes: MemoryStudy,
    /// Completed syndrome rounds, including the final unsuccessful round.
    pub rounds: usize,
    /// Physical primitive costs across all attempts, excluding the external Choi reference.
    pub cost: PlanarCost,
    /// Actual erased and replaced data photons.
    pub replacements: usize,
}
impl PlanarCost {
    pub(super) fn add(&mut self, other: Self) {
        self.cnots += other.cnots;
        self.hadamards += other.hadamards;
        self.phases += other.phases;
        self.paulis += other.paulis;
        self.measurements += other.measurements;
        self.preparations += other.preparations;
        self.swaps += other.swaps;
    }
}
/// Run a genuinely encoded Bell memory with physical trace/replacement of lost
/// sites and actual ancilla check circuits. Corrections use only syndrome and loss
/// flags. Final logical X/Z correlations with an untouched reference score the run.
///
/// This bounded host reference uses ideal encoder, check gates and measurement;
/// local data noise is applied before each round. It is not a circuit-level
/// threshold estimate or a claim of a deterministic photonic implementation.
#[allow(clippy::too_many_arguments)]
pub fn run_planar_memory(
    code: &PlanarCode,
    rounds: usize,
    noise: PlanarMemoryNoise,
    start: u64,
    shots: usize,
    seed: u64,
    budget: PlanarDecodeBudget,
) -> Result<PlanarMemoryStudy> {
    if rounds == 0 || rounds > 1024 {
        return Err(Error::InvalidDimensions(
            "planar rounds must be in 1..=1024".into(),
        ));
    }
    start.checked_add(shots as u64).ok_or(Error::Overflow)?;
    let work = shots
        .checked_mul(rounds)
        .and_then(|v| v.checked_mul(code.qubits()))
        .ok_or(Error::Overflow)?;
    if work > 1 << 26 {
        return Err(Error::ResourceLimit {
            resource: "planar data-round attempts",
            required: work,
            limit: 1 << 26,
        });
    }
    let n = code.qubits();
    let mut initial = Tableau::zero(n + 2, TableauLimits::default())?;
    initial.h(n)?;
    initial.cnot(n, code.input_site())?;
    let mut encoding_cost = code.encode(&mut initial, 0)?;
    encoding_cost.preparations = n;
    let correlation = |p: PauliProduct, ref_p: Pauli| {
        PauliProduct::new(p.terms().iter().copied().chain([(n, ref_p)]).collect())
    };
    let xx = correlation(code.logical_x(0)?, Pauli::X)?;
    let zz = correlation(code.logical_z(0)?, Pauli::Z)?;
    let [px, py, pz] = noise.pauli.probabilities();
    let mut report = PlanarMemoryStudy {
        outcomes: MemoryStudy {
            attempts: shots,
            ..Default::default()
        },
        ..Default::default()
    };
    let domain = shot_seed(seed, 0, 0x504c_414e_4152_0001);
    for shot in 0..shots {
        let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(domain, start + shot as u64, 0));
        let mut state = initial.clone();
        report.cost.add(encoding_cost);
        let mut accepted = true;
        for _ in 0..rounds {
            let mut erased = vec![false; n];
            for (q, lost) in erased.iter_mut().enumerate() {
                *lost = rng.random::<f64>() < noise.loss;
                let u = rng.random::<f64>();
                if u < px + py {
                    state.x(q)?;
                }
                if (px..px + py + pz).contains(&u) {
                    state.z(q)?;
                }
            }
            let replacement = code.replace_erasures(&mut state, 0, &erased, || rng.random())?;
            report.replacements += replacement.preparations;
            report.cost.add(replacement);
            let (x, z, cost) = code.measure_syndrome(&mut state, 0, n + 1, || rng.random())?;
            report.cost.add(cost);
            report.rounds += 1;
            let correction = code.decode(&x, &z, &erased, budget)?;
            match &correction {
                PlanarRecovery::Correction { x, z } => {
                    report.cost.paulis += x.iter().chain(z).filter(|&&v| v).count();
                    code.correct(&mut state, 0, &correction)?;
                }
                PlanarRecovery::Ambiguous => {
                    report.outcomes.erased_logicals += 1;
                    accepted = false;
                    break;
                }
                PlanarRecovery::BudgetExceeded => {
                    report.outcomes.decoder_budget += 1;
                    accepted = false;
                    break;
                }
            }
        }
        if accepted {
            if state.expectation(&xx)? == 1 && state.expectation(&zz)? == 1 {
                report.outcomes.corrected += 1;
            } else {
                report.outcomes.logical_errors += 1;
            }
        }
    }
    Ok(report)
}
