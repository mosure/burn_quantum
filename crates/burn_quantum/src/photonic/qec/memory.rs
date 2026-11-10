//! Physical bare/encoded toric memory studies sharing actual resource and fusion records.
use super::{
    EncodedFusion, EncodedFusionRecovery, FourStarTorus, MatchingDecode, MemoryStudy, NetworkNoise,
    ShorEncodedResource, SixRingTorus, SyndromeGraph,
};
use crate::photonic::stabilizer::{
    Pauli, PauliNoise, PauliProduct, ReferenceSampler, StabilizerSimulator,
};
use crate::photonic::{
    CompiledFusionNetwork, FusionParities, FusionProgram, Parity, runtime::shot_seed,
};
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
/// Immutable resource/measurement/check correspondence, independent of physical routing.
#[derive(Clone, Debug)]
pub struct ToricFusionCode {
    program: FusionProgram,
    graphs: [SyndromeGraph; 2],
    records: [Vec<usize>; 2],
}
impl ToricFusionCode {
    /// Four-star convention with explicit local Hadamards retained by the resource recipes.
    pub fn four_star(network: &FourStarTorus) -> Self {
        let f = network.program().num_fusions();
        Self {
            program: network.program().clone(),
            graphs: [network.dual().clone(), network.primal().clone()],
            records: [
                (0..f).map(|i| 2 * i).collect(),
                (0..f).map(|i| 2 * i + 1).collect(),
            ],
        }
    }
    /// Published rotated six-ring convention with mixed XX/ZZ records in each graph.
    pub fn six_ring(network: &SixRingTorus) -> Self {
        Self {
            program: network.program().clone(),
            graphs: [
                network.graph(0).unwrap().clone(),
                network.graph(1).unwrap().clone(),
            ],
            records: [
                network.record_indices(0).unwrap().to_vec(),
                network.record_indices(1).unwrap().to_vec(),
            ],
        }
    }
}
/// Physical resource/fusion implementation of a toric memory code.
#[derive(Clone, Debug)]
pub struct PhysicalToricMemory {
    code: ToricFusionCode,
    program: FusionProgram,
    encoded: Option<Vec<EncodedFusion>>,
}
/// Explicit bounded reference work limits; exact matching remains bounded to 20 defects.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBudgets {
    /// Number of paired defects allowed before returning a counted budget outcome.
    pub max_defects: usize,
    /// Maximum qubits times measured physical Pauli products in the reference compiler.
    pub max_reference_terms: usize,
    /// Packed frame and record allocation limit per sampling call.
    pub max_frame_bits: usize,
}
impl Default for MemoryBudgets {
    fn default() -> Self {
        Self {
            max_defects: 18,
            max_reference_terms: 1 << 26,
            max_frame_bits: 1 << 26,
        }
    }
}
/// Complete memory accounting plus physical cost and detected inner-code events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalMemoryStudy {
    /// All attempts classified by logical correction, erasure, error or decoder budget.
    pub memory: MemoryStudy,
    /// Inconsistent inner Shor checks, conservatively converted to joint logical erasures.
    pub detected_inner_events: usize,
    /// Physical resource photons supplied in every attempt, before any loss.
    pub photons_per_attempt: usize,
    /// Actual two-photon destructive analyzer operations per attempt.
    pub fusions_per_attempt: usize,
}
impl PhysicalToricMemory {
    /// Direct unencoded physical program.
    pub fn bare(code: ToricFusionCode) -> Self {
        Self {
            program: code.program.clone(),
            code,
            encoded: None,
        }
    }
    /// Encode every actual resource vertex and replace every logical fusion by four
    /// physical fusions. Local logical-H frames use typed code orientations and pairing.
    pub fn shor(code: ToricFusionCode, max_photons: usize) -> Result<Self> {
        let required = code
            .program
            .num_photons()
            .checked_mul(4)
            .ok_or(Error::Overflow)?;
        let limit = max_photons.min(1 << 20);
        if required > limit {
            return Err(Error::ResourceLimit {
                resource: "encoded memory photons",
                required,
                limit,
            });
        }
        let mut program = FusionProgram::new();
        let mut blocks = vec![];
        for resource in code.program.resources() {
            blocks.extend(
                ShorEncodedResource::new(&resource.state)?.add_to(&mut program, resource.source)?,
            );
        }
        let mut encoded = vec![];
        for f in code.program.fusions() {
            encoded.push(EncodedFusion::append(
                &mut program,
                blocks[f.photons[0].0],
                blocks[f.photons[1].0],
                f.at,
            )?);
        }
        Ok(Self {
            code,
            program,
            encoded: Some(encoded),
        })
    }
    /// Actual physical recipe to compile and route; every supplied photon is counted.
    pub fn program(&self) -> &FusionProgram {
        &self.program
    }
    /// Number of inner encoded fusions (zero for a bare model).
    pub fn encoded_fusions(&self) -> usize {
        self.encoded.as_ref().map_or(0, Vec::len)
    }
    /// Exact reference-frame sampling with independent physical photon loss and X/Y/Z faults.
    /// Unboosted failure erasures are state-dependent. Inner detected errors become joint
    /// logical erasures; unknown inner logical errors remain for the outer decoder.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        plan: &CompiledFusionNetwork,
        noise: NetworkNoise,
        start: u64,
        shots: usize,
        seed: u64,
        budgets: MemoryBudgets,
    ) -> Result<PhysicalMemoryStudy> {
        if plan.program() != &self.program || !plan.outputs().is_empty() {
            return Err(Error::InvalidValue(
                "physical memory plan does not match a closed network".into(),
            ));
        }
        if plan.transmissions().iter().any(|t| t.values() != [1., 1.]) {
            return Err(Error::Unsupported(
                "specify memory photon loss exactly once through NetworkNoise".into(),
            ));
        }
        let initial = StabilizerSimulator::default().prepare(plan)?;
        let obs: Vec<_> = plan
            .program()
            .fusions()
            .iter()
            .flat_map(|f| {
                [Pauli::X, Pauli::Z]
                    .map(|p| PauliProduct::new(f.photons.iter().map(|q| (q.0, p)).collect()))
            })
            .collect::<Result<_>>()?;
        let sampler = ReferenceSampler::compile(&initial, &obs, budgets.max_reference_terms)?;
        let clean = sampler.sample(start, shots, seed, PauliNoise::NONE, budgets.max_frame_bits)?;
        let noisy = sampler.sample(start, shots, seed, noise.pauli, budgets.max_frame_bits)?;
        let mut report = PhysicalMemoryStudy {
            memory: MemoryStudy {
                attempts: shots,
                ..Default::default()
            },
            detected_inner_events: 0,
            photons_per_attempt: self.program.num_photons(),
            fusions_per_attempt: self.program.num_fusions(),
        };
        let loss_seed = shot_seed(seed, 0, 0x4642_5143_4c4f_0001);
        for shot in 0..shots {
            let lost: Vec<_> = (0..self.program.num_photons())
                .map(|q| {
                    let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(
                        loss_seed,
                        start + shot as u64,
                        q as u64,
                    ));
                    rng.random::<f64>() < noise.loss
                })
                .collect();
            let mut ideal_physical = vec![];
            let mut observed_physical = vec![];
            for (i, f) in self.program.fusions().iter().enumerate() {
                ideal_physical.push(FusionParities {
                    xx: Some(parity(clean.bit(2 * i, shot)?)),
                    zz: Some(parity(clean.bit(2 * i + 1, shot)?)),
                });
                let z = noisy.bit(2 * i + 1, shot)?;
                let erased = lost[f.photons[0].0] || lost[f.photons[1].0];
                observed_physical.push(FusionParities {
                    xx: (!erased && (!noise.unboosted || z))
                        .then_some(parity(noisy.bit(2 * i, shot)?)),
                    zz: (!erased).then_some(parity(z)),
                });
            }
            let (truth, observed) = if let Some(encoded) = &self.encoded {
                let mut truth = vec![];
                let mut observed = vec![];
                for f in encoded {
                    let clean =
                        f.recover_local_parities(f.physical_fusions().map(|i| ideal_physical[i]));
                    let EncodedFusionRecovery::Consistent(c) = clean else {
                        return Err(Error::InvalidValue(
                            "ideal encoded resource violates inner checks".into(),
                        ));
                    };
                    if c.xx.is_none() || c.zz.is_none() {
                        return Err(Error::InvalidValue(
                            "complete Bell data failed logical recovery".into(),
                        ));
                    }
                    truth.push(c);
                    match f
                        .recover_local_parities(f.physical_fusions().map(|i| observed_physical[i]))
                    {
                        EncodedFusionRecovery::Consistent(r) => observed.push(r),
                        EncodedFusionRecovery::Inconsistent => {
                            report.detected_inner_events += 1;
                            observed.push(FusionParities { xx: None, zz: None });
                        }
                    }
                }
                (truth, observed)
            } else {
                (ideal_physical, observed_physical)
            };
            let truth: Vec<_> = truth
                .iter()
                .flat_map(|r| [r.xx.unwrap() == Parity::Odd, r.zz.unwrap() == Parity::Odd])
                .collect();
            let observed: Vec<_> = observed
                .iter()
                .flat_map(|r| [r.xx, r.zz].map(|p| p.map(|p| p == Parity::Odd)))
                .collect();
            let mut ambiguous = false;
            let mut budget = false;
            let mut failed = false;
            for (graph, indices) in self.code.graphs.iter().zip(&self.code.records) {
                let reference: Vec<_> = indices.iter().map(|&i| truth[i]).collect();
                let record: Vec<_> = indices
                    .iter()
                    .map(|&i| observed[i].unwrap_or(false))
                    .collect();
                let erased: Vec<_> = indices.iter().map(|&i| observed[i].is_none()).collect();
                if graph.syndrome(&reference)?.iter().any(|&b| b) {
                    return Err(Error::InvalidValue(
                        "physical resource violates outer geometric check".into(),
                    ));
                }
                if graph.erasure_logicals(&erased)?.iter().any(|&b| b) {
                    ambiguous = true;
                    continue;
                }
                match graph.decode(&graph.syndrome(&record)?, &erased, budgets.max_defects)? {
                    MatchingDecode::Correction(correction) => {
                        let residual: Vec<_> = record
                            .iter()
                            .zip(&reference)
                            .zip(correction)
                            .map(|((&a, &b), c)| a ^ b ^ c)
                            .collect();
                        if graph.syndrome(&residual)?.iter().any(|&b| b) {
                            return Err(Error::InvalidValue(
                                "outer correction left syndrome".into(),
                            ));
                        }
                        failed |= graph.logical_parities(&residual)?.iter().any(|&b| b);
                    }
                    MatchingDecode::BudgetExceeded => budget = true,
                    MatchingDecode::Inconsistent => {
                        return Err(Error::InvalidValue(
                            "closed memory syndrome inconsistent".into(),
                        ));
                    }
                }
            }
            if ambiguous {
                report.memory.erased_logicals += 1;
            } else if budget {
                report.memory.decoder_budget += 1;
            } else if failed {
                report.memory.logical_errors += 1;
            } else {
                report.memory.corrected += 1;
            }
        }
        Ok(report)
    }
}
fn parity(bit: bool) -> Parity {
    if bit { Parity::Odd } else { Parity::Even }
}
