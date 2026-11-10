use super::{DecoderBudget, DecoderGraph, DetectorEdge, GraphLimits, GraphStatus};
use crate::photonic::{qec::SixRingTorus, runtime::shot_seed};
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
/// Fixed-shot, unconditional accounting. Limits and inconsistent syndromes are
/// separate unsuccessful attempts, never omitted from the denominator.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThresholdCounts {
    /// All attempted shots, including incomplete decoding.
    pub attempts: usize,
    /// Actual sampled attempts with a residual logical error after recovery.
    pub logical_failures: usize,
    /// Trials where at least one logical class is erased (different from error).
    pub ambiguous_erasures: usize,
    /// Attempts stopped by a declared decoder work bound.
    pub decoder_limits: usize,
    /// Attempts with detector syndromes inconsistent with the supplied graph.
    pub inconsistent: usize,
    /// Rank of the erased logical image across both channels (up to six for
    /// periodic six-ring, up to two for an open planar memory).
    pub logical_rank_histogram: [usize; 7],
    /// Sum of measured edge inspections across every attempted shot/channel.
    pub edge_visits: u64,
    /// Sum of actual union-find growth rounds across all attempts/channels.
    pub growth_rounds: u64,
}
impl ThresholdCounts {
    /// Erasure-only conditional mean over lost logical bits, summed over masks.
    /// This is not a failure estimator for runs containing additional Pauli noise.
    /// This is a Rao–Blackwell estimate; binomial intervals apply to the sampled
    /// `logical_failures` instead, not to these fractional conditional means.
    pub fn conditional_logical_failures(&self) -> f64 {
        self.logical_rank_histogram
            .iter()
            .enumerate()
            .map(|(rank, &n)| n as f64 * (1. - 2f64.powi(-(rank as i32))))
            .sum()
    }
    /// Logical errors plus work-limited and inconsistent attempts, without
    /// double-counting erased trials whose sampled logical correction succeeds.
    pub fn unsuccessful(&self) -> Result<usize> {
        let count = self
            .logical_failures
            .checked_add(self.decoder_limits)
            .and_then(|v| v.checked_add(self.inconsistent))
            .ok_or(Error::Overflow)?;
        if count > self.attempts {
            return Err(Error::InvalidValue("threshold failure accounting".into()));
        }
        Ok(count)
    }
    /// Aggregate independently indexed disjoint shot shards. Caller retains each
    /// shard's seed/start/range provenance and must not submit overlapping ranges.
    pub fn add(&mut self, other: &Self) -> Result<()> {
        let mut next = self.clone();
        for (a, b) in [
            (&mut next.attempts, other.attempts),
            (&mut next.logical_failures, other.logical_failures),
            (&mut next.ambiguous_erasures, other.ambiguous_erasures),
            (&mut next.decoder_limits, other.decoder_limits),
            (&mut next.inconsistent, other.inconsistent),
        ] {
            *a = a.checked_add(b).ok_or(Error::Overflow)?;
        }
        for (a, b) in next
            .logical_rank_histogram
            .iter_mut()
            .zip(other.logical_rank_histogram)
        {
            *a = a.checked_add(b).ok_or(Error::Overflow)?;
        }
        next.edge_visits = next
            .edge_visits
            .checked_add(other.edge_visits)
            .ok_or(Error::Overflow)?;
        next.growth_rounds = next
            .growth_rounds
            .checked_add(other.growth_rounds)
            .ok_or(Error::Overflow)?;
        *self = next;
        Ok(())
    }
}
/// Exact erasure-channel logical classification on the actual rotated six-ring
/// resource/fusion graph. This is a periodic bulk benchmark, not a finite memory
/// or a photon-loss model. XX and ZZ outcome erasures are independent in `run`.
#[derive(Clone, Debug)]
pub struct SixRingErasureModel {
    side: usize,
    graphs: [DecoderGraph; 2],
    records: [Vec<usize>; 2],
    record_count: usize,
}
impl SixRingErasureModel {
    /// Reuse and validate the actual network graph and full record partition.
    pub fn new(network: &SixRingTorus, limits: GraphLimits) -> Result<Self> {
        let graphs = [0, 1].map(|color| -> Result<_> {
            let graph = network.graph(color)?;
            for (resource, required, limit) in [
                (
                    "detector graph vertices",
                    graph.vertices(),
                    limits.max_detectors,
                ),
                (
                    "detector graph edges",
                    graph.edges().len(),
                    limits.max_edges,
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
            DecoderGraph::new(
                graph.vertices(),
                3,
                graph
                    .edges()
                    .iter()
                    .zip(graph.cuts())
                    .map(|(&(a, b), cut)| DetectorEdge {
                        a,
                        b,
                        logical: cut
                            .iter()
                            .enumerate()
                            .fold(0, |m, (j, &v)| m | ((v as u64) << j)),
                    })
                    .collect(),
                limits,
            )
        });
        let [a, b] = graphs;
        let records = [
            network.record_indices(0)?.to_vec(),
            network.record_indices(1)?.to_vec(),
        ];
        let record_count = network
            .program()
            .num_fusions()
            .checked_mul(2)
            .ok_or(Error::Overflow)?;
        let mut used = vec![false; record_count];
        for &record in records.iter().flatten() {
            if record >= record_count || used[record] {
                return Err(Error::Shape("six-ring record partition".into()));
            }
            used[record] = true;
        }
        if used.iter().any(|&v| !v) {
            return Err(Error::Shape("missing six-ring outcome record".into()));
        }
        Ok(Self {
            side: network.side(),
            graphs: [a?, b?],
            records,
            record_count,
        })
    }
    /// Even periodic side length in the published rotated lattice coordinates.
    pub fn side(&self) -> usize {
        self.side
    }
    /// Total XX/ZZ outcome coordinates, twice the number of actual fusions.
    pub fn records(&self) -> usize {
        self.record_count
    }
    /// Color-zero or color-one detector graph.
    pub fn graph(&self, color: usize) -> Result<&DecoderGraph> {
        self.graphs.get(color).ok_or(Error::Basis)
    }
    /// Map graph edges into the interleaved full fusion-outcome stream.
    pub fn record_indices(&self, color: usize) -> Result<&[usize]> {
        self.records
            .get(color)
            .map(Vec::as_slice)
            .ok_or(Error::Basis)
    }
    /// A fixed run; all shots execute even after enough failures are observed.
    /// Per-shot seeds use runtime::shot_seed with model namespace version one.
    pub fn run(
        &self,
        erasure: f64,
        start: u64,
        shots: usize,
        seed: u64,
        budget: DecoderBudget,
    ) -> Result<ThresholdCounts> {
        if !erasure.is_finite() || !(0.0..=1.0).contains(&erasure) {
            return Err(Error::InvalidValue("outcome erasure probability".into()));
        }
        self.run_with_masks(start, shots, seed, budget, |_, rng, mask| {
            for lost in mask {
                *lost = rng.random::<f64>() < erasure;
            }
            Ok(())
        })
    }
    /// Same exact graph analysis with a caller-supplied correlated full-record
    /// erasure mask (XX,ZZ for fusion0, then fusion1,...). The closure receives a
    /// deterministic per-shot RNG and must set every mask coordinate. Logical
    /// classes are sampled from a separate seed component after mask generation.
    /// This API does not invent a photon loss law; the caller records that model.
    pub fn run_with_masks<F>(
        &self,
        start: u64,
        shots: usize,
        seed: u64,
        budget: DecoderBudget,
        mut sample: F,
    ) -> Result<ThresholdCounts>
    where
        F: FnMut(u64, &mut ChaCha8Rng, &mut [bool]) -> Result<()>,
    {
        start
            .checked_add(u64::try_from(shots).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let model_seed = shot_seed(seed, 0, 0x5352_4552_4153_0001);
        let mut result = ThresholdCounts {
            attempts: shots,
            ..Default::default()
        };
        let mut mask = vec![false; self.record_count];
        let mut color_mask: [Vec<bool>; 2] =
            std::array::from_fn(|i| vec![false; self.records[i].len()]);
        for shot in start..start + shots as u64 {
            mask.fill(false);
            let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(model_seed, shot, 0));
            sample(shot, &mut rng, &mut mask)?;
            let mut rank = 0;
            let mut limited = false;
            for (color, erased) in color_mask.iter_mut().enumerate() {
                for (value, &record) in erased.iter_mut().zip(&self.records[color]) {
                    *value = mask[record];
                }
                let analysis = self.graphs[color].analyze_erasures(erased, budget)?;
                result.edge_visits = result
                    .edge_visits
                    .checked_add(analysis.work.edge_visits)
                    .ok_or(Error::Overflow)?;
                if analysis.status == GraphStatus::WorkLimit {
                    limited = true;
                    break;
                }
                rank += analysis.rank();
            }
            if limited {
                result.decoder_limits += 1;
                continue;
            }
            result.logical_rank_histogram[rank] += 1;
            if rank > 0 {
                result.ambiguous_erasures += 1;
                // Uniform logical classes are exactly the pushforward of the
                // independent fair erased outcomes through a rank-r binary map.
                let mut logical_rng = ChaCha8Rng::seed_from_u64(shot_seed(model_seed, shot, 1));
                if logical_rng.random::<u64>() & ((1u64 << rank) - 1) != 0 {
                    result.logical_failures += 1;
                }
            }
        }
        Ok(result)
    }
}

/// Fixed independent binary-fault graph study, with marginal logical-observable
/// counts in addition to the unconditional any-logical-error event.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNoiseStudy {
    /// All attempts and explicit decoder outcomes.
    pub counts: ThresholdCounts,
    /// Residual flips on each declared logical bit; a shot can increment several.
    /// These marginals are diagnostics, not additional unadjusted confidence tests.
    pub logical_failures: Vec<usize>,
}
impl DecoderGraph {
    /// Sample every independent graph coordinate with the same probability.
    /// The graph owner must identify each coordinate's physical location and
    /// validate its detector/logical signature independently. This runner passes
    /// only the syndrome and an empty erasure mask to the decoder; true faults
    /// enter the logical correctness check only after recovery is returned.
    pub fn run_independent_faults(
        &self,
        probability: f64,
        start: u64,
        shots: usize,
        seed: u64,
        budget: DecoderBudget,
    ) -> Result<GraphNoiseStudy> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(Error::InvalidValue(
                "independent graph fault probability".into(),
            ));
        }
        start
            .checked_add(u64::try_from(shots).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let model_seed = shot_seed(seed, 0, 0x4752_4150_4850_0001);
        let mut study = GraphNoiseStudy {
            counts: ThresholdCounts {
                attempts: shots,
                ..Default::default()
            },
            logical_failures: vec![0; self.logicals()],
        };
        let mut errors = vec![false; self.edges().len()];
        let erased = errors.clone();
        for shot in start..start + shots as u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(model_seed, shot, 0));
            for error in &mut errors {
                *error = rng.random::<f64>() < probability;
            }
            let syndrome = self.syndrome(&errors)?;
            let decoded = self.decode_union_find(&syndrome, &erased, budget)?;
            study.counts.edge_visits = study
                .counts
                .edge_visits
                .checked_add(decoded.work.edge_visits)
                .ok_or(Error::Overflow)?;
            study.counts.growth_rounds = study
                .counts
                .growth_rounds
                .checked_add(decoded.work.growth_rounds as u64)
                .ok_or(Error::Overflow)?;
            match decoded.status {
                GraphStatus::WorkLimit => study.counts.decoder_limits += 1,
                GraphStatus::Inconsistent => study.counts.inconsistent += 1,
                GraphStatus::Ambiguous => {
                    return Err(Error::InvalidValue(
                        "empty erasure mask produced ambiguity".into(),
                    ));
                }
                GraphStatus::Corrected => {
                    study.counts.logical_rank_histogram[0] += 1;
                    let residual = self.logical_flips(&errors)? ^ decoded.logical;
                    if residual != 0 {
                        study.counts.logical_failures += 1;
                    }
                    for (i, count) in study.logical_failures.iter_mut().enumerate() {
                        if residual >> i & 1 != 0 {
                            *count += 1;
                        }
                    }
                }
            }
        }
        Ok(study)
    }
}
