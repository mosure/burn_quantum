use super::{DecoderBudget, DecoderGraph, DetectorEdge, GraphLimits, GraphStatus, ThresholdCounts};
use crate::photonic::{
    qec::{ParityCode, PlanarCode},
    runtime::shot_seed,
};
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
/// Repeated noisy measurements of graphlike binary checks. Initial checks are
/// known +1, rounds0..R-1 follow data noise and have noisy readout, and layerR
/// is an ideal closing readout with no additional data-noise interval.
#[derive(Clone, Debug)]
pub struct RepeatedMeasurements {
    graph: DecoderGraph,
    rounds: usize,
    variables: usize,
    checks: usize,
}
impl RepeatedMeasurements {
    /// Every fault coordinate must touch at most two checks. The logical rows
    /// label final accumulated data-frame changes; measurement faults label zero.
    pub fn new(code: &ParityCode, rounds: usize, limits: GraphLimits) -> Result<Self> {
        if rounds == 0 || code.checks().is_empty() || code.logicals().len() > 64 {
            return Err(Error::InvalidDimensions(
                "repeated-check graph dimensions".into(),
            ));
        }
        let variables = code.variables();
        let checks = code.checks().len();
        let detectors = rounds
            .checked_add(1)
            .and_then(|r| r.checked_mul(checks))
            .ok_or(Error::Overflow)?;
        let count = rounds
            .checked_mul(variables.checked_add(checks).ok_or(Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        for (resource, required, limit) in [
            ("repeated-check detectors", detectors, limits.max_detectors),
            ("repeated-check edges", count, limits.max_edges),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        let mut endpoints = Vec::with_capacity(variables);
        for q in 0..variables {
            let check: Vec<_> = code
                .checks()
                .iter()
                .enumerate()
                .filter_map(|(i, row)| row[q].then_some(i))
                .collect();
            if check.len() > 2 {
                return Err(Error::Unsupported(
                    "hyperedge fault touches more than two detectors".into(),
                ));
            }
            endpoints.push(check);
        }
        let mut edges = Vec::with_capacity(count);
        for t in 0..rounds {
            for (q, check) in endpoints.iter().enumerate() {
                edges.push(DetectorEdge {
                    a: check.first().map_or(detectors, |v| t * checks + v),
                    b: check.get(1).map_or(detectors, |v| t * checks + v),
                    logical: code
                        .logicals()
                        .iter()
                        .enumerate()
                        .fold(0, |m, (i, row)| m | ((row[q] as u64) << i)),
                });
            }
        }
        for t in 0..rounds {
            for check in 0..checks {
                edges.push(DetectorEdge {
                    a: t * checks + check,
                    b: (t + 1) * checks + check,
                    logical: 0,
                });
            }
        }
        Ok(Self {
            graph: DecoderGraph::new(detectors, code.logicals().len(), edges, limits)?,
            rounds,
            variables,
            checks,
        })
    }
    /// Underlying spacetime graph with spatial-boundary endpoints.
    pub fn graph(&self) -> &DecoderGraph {
        &self.graph
    }
    /// Number of noisy measurement rounds (plus one ideal closing round).
    pub fn rounds(&self) -> usize {
        self.rounds
    }
    /// Number of data fault coordinates in each round.
    pub fn variables(&self) -> usize {
        self.variables
    }
    /// Number of check measurement records per round.
    pub fn checks(&self) -> usize {
        self.checks
    }
    /// Edge index for a data fault before noisy round `round`.
    pub fn data_edge(&self, round: usize, variable: usize) -> Result<usize> {
        if round >= self.rounds || variable >= self.variables {
            return Err(Error::Basis);
        }
        Ok(round * self.variables + variable)
    }
    /// Edge index for a wrong check outcome in noisy round `round`.
    pub fn measurement_edge(&self, round: usize, check: usize) -> Result<usize> {
        if round >= self.rounds || check >= self.checks {
            return Err(Error::Basis);
        }
        Ok(self.rounds * self.variables + round * self.checks + check)
    }
    /// Convert actual check histories into changes relative to the preceding
    /// layer. Requires R noisy layers and the independently ideal closing layer.
    pub fn detectors_from_history(&self, history: &[Vec<bool>]) -> Result<Vec<bool>> {
        if history.len() != self.rounds + 1 || history.iter().any(|row| row.len() != self.checks) {
            return Err(Error::Shape("repeated-check record history".into()));
        }
        let mut out = vec![false; self.graph.detectors()];
        for t in 0..=self.rounds {
            for c in 0..self.checks {
                out[t * self.checks + c] = history[t][c] ^ (t > 0 && history[t - 1][c]);
            }
        }
        Ok(out)
    }
}
/// Explicit phenomenological channel, not noisy physical syndrome-extraction
/// circuits or a photonic fusion model. Data X and Z faults are independent;
/// therefore their joint event is Y. A heralded data erasure replaces both bits
/// with independent fair bits at the same site. Measurement erasures independently
/// replace each missing outcome by a fair bit and expose its location.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhenomenologicalNoise {
    /// Independent X fault probability per non-erased data site per round.
    pub data_x: f64,
    /// Independent Z fault probability per non-erased data site per round.
    pub data_z: f64,
    /// Flip probability per available noisy check record.
    pub measurement: f64,
    /// Shared X/Z heralded depolarization probability per data site per round.
    pub data_erasure: f64,
    /// Heralded missing-record probability per noisy check measurement.
    pub measurement_erasure: f64,
}
impl PhenomenologicalNoise {
    /// Equal independent data-X, data-Z and measurement-error probabilities;
    /// no heralded erasures. In particular total data Pauli probability is 2p-p².
    pub fn symmetric(p: f64) -> Result<Self> {
        let out = Self {
            data_x: p,
            data_z: p,
            measurement: p,
            data_erasure: 0.,
            measurement_erasure: 0.,
        };
        out.validate()?;
        Ok(out)
    }
    fn validate(&self) -> Result<()> {
        if [
            self.data_x,
            self.data_z,
            self.measurement,
            self.data_erasure,
            self.measurement_erasure,
        ]
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            Err(Error::InvalidValue(
                "phenomenological noise probability".into(),
            ))
        } else {
            Ok(())
        }
    }
}
/// Two-channel open planar memory with complete finite-time histories and a
/// polynomial union-find decoder. No dense state or tableau is needed to sample
/// Pauli-frame trajectories; independent circuit/tableau tests establish the
/// detector map. This is not an inference of a photonic hardware threshold.
#[derive(Clone, Debug)]
pub struct PhenomenologicalMemory {
    distance: usize,
    channels: [RepeatedMeasurements; 2],
}
impl PhenomenologicalMemory {
    /// Construct from the existing unrotated [[d²+(d−1)²,1,d]] PlanarCode.
    pub fn new(code: &PlanarCode, rounds: usize, limits: GraphLimits) -> Result<Self> {
        Ok(Self {
            distance: code.distance(),
            channels: [
                RepeatedMeasurements::new(&code.x_error_code()?, rounds, limits)?,
                RepeatedMeasurements::new(&code.z_error_code()?, rounds, limits)?,
            ],
        })
    }
    /// Actual planar-code distance (not total graph size).
    pub fn distance(&self) -> usize {
        self.distance
    }
    /// 0 decodes X faults from Z checks; 1 decodes Z faults from X checks.
    pub fn channel(&self, index: usize) -> Result<&RepeatedMeasurements> {
        self.channels.get(index).ok_or(Error::Basis)
    }
    /// Fixed shot count with deterministic per-shot RNG. Every decoder limit or
    /// inconsistent detector record remains an unsuccessful trial. Ambiguous
    /// erasures are counted separately, but an actual logical error is determined
    /// from the sampled true frame and returned correction, not postselection.
    pub fn run(
        &self,
        noise: PhenomenologicalNoise,
        start: u64,
        shots: usize,
        seed: u64,
        budget: DecoderBudget,
    ) -> Result<ThresholdCounts> {
        noise.validate()?;
        start
            .checked_add(u64::try_from(shots).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let base = shot_seed(seed, 0, 0x5048_454e_4f4d_0001);
        let mut counts = ThresholdCounts {
            attempts: shots,
            ..Default::default()
        };
        let mut errors: [Vec<bool>; 2] =
            std::array::from_fn(|i| vec![false; self.channels[i].graph.edges().len()]);
        let mut erased = errors.clone();
        let rounds = self.channels[0].rounds;
        let variables = self.channels[0].variables;
        for shot in start..start + shots as u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(base, shot, 0));
            for t in 0..rounds {
                for q in 0..variables {
                    let lost = rng.random::<f64>() < noise.data_erasure;
                    for channel in 0..2 {
                        let e = t * variables + q;
                        erased[channel][e] = lost;
                        errors[channel][e] = rng.random::<f64>()
                            < if lost {
                                0.5
                            } else if channel == 0 {
                                noise.data_x
                            } else {
                                noise.data_z
                            };
                    }
                }
            }
            for channel in 0..2 {
                let c = &self.channels[channel];
                for e in rounds * variables..c.graph.edges().len() {
                    let lost = rng.random::<f64>() < noise.measurement_erasure;
                    erased[channel][e] = lost;
                    errors[channel][e] =
                        rng.random::<f64>() < if lost { 0.5 } else { noise.measurement };
                }
            }
            let mut failed = false;
            let mut rank = 0;
            let mut limit = false;
            let mut inconsistent = false;
            for channel in 0..2 {
                let graph = &self.channels[channel].graph;
                let syndrome = graph.syndrome(&errors[channel])?;
                let result = graph.decode_union_find(&syndrome, &erased[channel], budget)?;
                counts.edge_visits = counts
                    .edge_visits
                    .checked_add(result.work.edge_visits)
                    .ok_or(Error::Overflow)?;
                counts.growth_rounds = counts
                    .growth_rounds
                    .checked_add(result.work.growth_rounds as u64)
                    .ok_or(Error::Overflow)?;
                match result.status {
                    GraphStatus::WorkLimit => {
                        limit = true;
                        break;
                    }
                    GraphStatus::Inconsistent => {
                        inconsistent = true;
                        break;
                    }
                    _ => {}
                }
                rank += result.erased_logical_basis.len();
                failed |= graph.logical_flips(&errors[channel])? ^ result.logical != 0;
            }
            if limit {
                counts.decoder_limits += 1;
            } else if inconsistent {
                counts.inconsistent += 1;
            } else {
                counts.logical_rank_histogram[rank] += 1;
                if rank > 0 {
                    counts.ambiguous_erasures += 1;
                }
                if failed {
                    counts.logical_failures += 1;
                }
            }
        }
        Ok(counts)
    }
}
