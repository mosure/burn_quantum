use super::{FourStarTorus, MatchingDecode};
use crate::photonic::stabilizer::{PauliNoise, ReferenceSampler, StabilizerSimulator};
use crate::photonic::{CompiledFusionNetwork, runtime::shot_seed};
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
/// Independent physical photon attenuation and local Pauli faults for a closed network study.
#[derive(Clone, Copy, Debug)]
pub struct NetworkNoise {
    pub(super) loss: f64,
    pub(super) pauli: PauliNoise,
    pub(super) unboosted: bool,
}
impl NetworkNoise {
    /// Validate photon-loss probability. `unboosted=true` models the actual standard
    /// analyzer: XX is unavailable on ZZ-even failure. False is an explicit ideal
    /// complete-Bell reference, not an asserted deterministic linear-optical device.
    pub fn new(photon_loss: f64, pauli: PauliNoise, unboosted: bool) -> Result<Self> {
        if !photon_loss.is_finite() || !(0.0..=1.0).contains(&photon_loss) {
            return Err(Error::InvalidValue("invalid photon loss".into()));
        }
        Ok(Self {
            loss: photon_loss,
            pauli,
            unboosted,
        })
    }
}
/// Binomial Wilson interval for a fixed finite number of independent attempts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WilsonInterval {
    /// Lower probability endpoint.
    pub lower: f64,
    /// Upper probability endpoint.
    pub upper: f64,
}
impl WilsonInterval {
    /// Compute a Wilson score interval; 1.95996398454 gives approximately 95% coverage.
    pub fn new(events: usize, trials: usize, z: f64) -> Result<Self> {
        if trials == 0 || events > trials || !z.is_finite() || z <= 0. {
            return Err(Error::InvalidValue("invalid interval inputs".into()));
        }
        let n = trials as f64;
        let p = events as f64 / n;
        let d = 1. + z * z / n;
        let c = (p + z * z / (2. * n)) / d;
        let h = z * ((p * (1. - p) + z * z / (4. * n)) / n).sqrt() / d;
        Ok(Self {
            lower: (c - h).max(0.),
            upper: (c + h).min(1.),
        })
    }
}
/// Complete attempted-shot accounting for a toric syndrome-memory benchmark.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryStudy {
    /// All physical attempts, including photon loss and fusion failure.
    pub attempts: usize,
    /// Corrected and logically correct attempts.
    pub corrected: usize,
    /// Residual nontrivial homology after decoding (unheralded Pauli logical error).
    pub logical_errors: usize,
    /// Erased cycles make at least one logical membrane unknowable.
    pub erased_logicals: usize,
    /// Exact matching exceeded its explicit defect budget.
    pub decoder_budget: usize,
}
impl MemoryStudy {
    /// Unconditional unsuccessful-attempt fraction interval; no postselection denominator.
    pub fn unsuccessful_interval(&self, z: f64) -> Result<WilsonInterval> {
        let unsuccessful = self
            .attempts
            .checked_sub(self.corrected)
            .ok_or_else(|| Error::InvalidValue("corrected attempts exceed total".into()))?;
        WilsonInterval::new(unsuccessful, self.attempts, z)
    }
}
/// Nondeterministic host diagnostics, deliberately separate from seeded study records.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryStudyTimings {
    /// Reference tableau, observable and sampler construction (excludes caller routing).
    pub setup_seconds: f64,
    /// Clean/noisy frame sampling, photon-loss draws and detector-record assembly.
    pub sampling_seconds: f64,
    /// Syndrome validation, erasure checks, decoding and logical classification.
    pub decoding_seconds: f64,
}
/// Run an actual four-star resource/fusion parity model with photon-level loss and Pauli noise.
///
/// `plan` must be compiled from this network's recipe with ideal transmissions; the
/// independent photon loss is introduced here exactly once. Full commuting Bell
/// records are sampled first and then physically unavailable parities are erased.
/// This preserves the joint distribution of available records on a fully consumed
/// network. It does not expose counterfactual erased parity values to the decoder.
/// Logical failures are checked against coupled noiseless reference records only
/// after decoding; this benchmark does not claim a physical threshold.
#[allow(clippy::too_many_arguments)]
pub fn run_four_star_memory(
    network: &FourStarTorus,
    plan: &CompiledFusionNetwork,
    noise: NetworkNoise,
    start: u64,
    shots: usize,
    seed: u64,
    max_defects: usize,
    max_bits: usize,
) -> Result<MemoryStudy> {
    Ok(run_four_star_memory_profiled(
        network,
        plan,
        noise,
        start,
        shots,
        seed,
        max_defects,
        max_bits,
    )?
    .0)
}
/// Same deterministic study with separate wall-clock setup, sampling and decoder diagnostics.
/// Timing instrumentation never enters seeds, checkpoints or [`MemoryStudy`] equality.
#[allow(clippy::too_many_arguments)]
pub fn run_four_star_memory_profiled(
    network: &FourStarTorus,
    plan: &CompiledFusionNetwork,
    noise: NetworkNoise,
    start: u64,
    shots: usize,
    seed: u64,
    max_defects: usize,
    max_bits: usize,
) -> Result<(MemoryStudy, MemoryStudyTimings)> {
    if plan.program() != network.program() || !plan.outputs().is_empty() {
        return Err(Error::InvalidValue(
            "memory plan does not match closed network".into(),
        ));
    }
    if plan.transmissions().iter().any(|t| t.values() != [1., 1.]) {
        return Err(Error::Unsupported(
            "study loss must be specified once via NetworkNoise".into(),
        ));
    }
    let setup_start = Instant::now();
    let initial = StabilizerSimulator::default().prepare(plan)?;
    let observables = network.observables()?;
    let sampler = ReferenceSampler::compile(&initial, &observables, 1 << 24)?;
    let mut timings = MemoryStudyTimings {
        setup_seconds: setup_start.elapsed().as_secs_f64(),
        ..Default::default()
    };
    let sampling_start = Instant::now();
    let clean = sampler.sample(start, shots, seed, PauliNoise::NONE, max_bits)?;
    let noisy = sampler.sample(start, shots, seed, noise.pauli, max_bits)?;
    timings.sampling_seconds = sampling_start.elapsed().as_secs_f64();
    let fusions = plan.fusions().len();
    let mut report = MemoryStudy {
        attempts: shots,
        ..Default::default()
    };
    let seed = shot_seed(seed, 0, 0x4642_5143_4c4f_0001);
    for shot in 0..shots {
        let sampling_start = Instant::now();
        let mut lost = vec![false; plan.program().num_photons()];
        for (q, lost) in lost.iter_mut().enumerate() {
            let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(seed, start + shot as u64, q as u64));
            *lost = rng.random::<f64>() < noise.loss;
        }
        let mut xx = vec![false; fusions];
        let mut zz = xx.clone();
        let mut erased_x = xx.clone();
        let mut erased_z = xx.clone();
        let mut reference_x = xx.clone();
        let mut reference_z = xx.clone();
        for (i, fusion) in plan.fusions().iter().enumerate() {
            reference_x[i] = clean.bit(2 * i, shot)?;
            reference_z[i] = clean.bit(2 * i + 1, shot)?;
            let x = noisy.bit(2 * i, shot)?;
            let z = noisy.bit(2 * i + 1, shot)?;
            let erased = lost[fusion.photons[0].0] || lost[fusion.photons[1].0];
            erased_x[i] = erased || (noise.unboosted && !z);
            erased_z[i] = erased;
            xx[i] = !erased_x[i] && x;
            zz[i] = !erased_z[i] && z;
        }
        timings.sampling_seconds += sampling_start.elapsed().as_secs_f64();
        let decoding_start = Instant::now();
        let mut ambiguous = false;
        let mut budget = false;
        let mut failed = false;
        for (graph, observed, erased, truth) in [
            (network.dual(), &xx, &erased_x, &reference_x),
            (network.primal(), &zz, &erased_z, &reference_z),
        ] {
            if graph.syndrome(truth)?.iter().any(|&v| v) {
                return Err(Error::InvalidValue(
                    "resource sample violates geometric fusion checks".into(),
                ));
            }
            if graph.erasure_logicals(erased)?.iter().any(|&v| v) {
                ambiguous = true;
                continue;
            }
            match graph.decode(&graph.syndrome(observed)?, erased, max_defects)? {
                MatchingDecode::Correction(correction) => {
                    let residual: Vec<_> = observed
                        .iter()
                        .zip(truth.iter())
                        .zip(correction)
                        .map(|((&a, &b), c)| a ^ b ^ c)
                        .collect();
                    if graph.syndrome(&residual)?.iter().any(|&v| v) {
                        return Err(Error::InvalidValue(
                            "decoder did not close residual chain".into(),
                        ));
                    }
                    failed |= graph.logical_parities(&residual)?.iter().any(|&v| v);
                }
                MatchingDecode::BudgetExceeded => budget = true,
                MatchingDecode::Inconsistent => {
                    return Err(Error::InvalidValue(
                        "closed network gave inconsistent syndrome".into(),
                    ));
                }
            }
        }
        if ambiguous {
            report.erased_logicals += 1;
        } else if budget {
            report.decoder_budget += 1;
        } else if failed {
            report.logical_errors += 1;
        } else {
            report.corrected += 1;
        }
        timings.decoding_seconds += decoding_start.elapsed().as_secs_f64();
    }
    Ok((report, timings))
}
