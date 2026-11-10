use super::{Pauli, PauliProduct, Tableau};
use crate::photonic::runtime::shot_seed;
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
/// Independent per-qubit Pauli mixture, applied after resource preparation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(try_from = "[f64;3]", into = "[f64;3]")]
pub struct PauliNoise([f64; 3]);
impl PauliNoise {
    /// No Pauli faults.
    pub const NONE: Self = Self([0.; 3]);
    /// Probabilities of X, Y, Z; the remaining probability is identity.
    pub fn new(x: f64, y: f64, z: f64) -> Result<Self> {
        let p = [x, y, z];
        if p.iter().any(|v| !v.is_finite() || *v < 0.) || p.iter().sum::<f64>() > 1. {
            return Err(Error::InvalidValue("invalid Pauli mixture".into()));
        }
        Ok(Self(p))
    }
    /// X,Y,Z probabilities.
    pub fn probabilities(self) -> [f64; 3] {
        self.0
    }
}
impl TryFrom<[f64; 3]> for PauliNoise {
    type Error = Error;
    fn try_from(p: [f64; 3]) -> Result<Self> {
        Self::new(p[0], p[1], p[2])
    }
}
impl From<PauliNoise> for [f64; 3] {
    fn from(p: PauliNoise) -> Self {
        p.0
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Step {
    observable: PauliProduct,
    reference: bool,
    kickback: Option<PauliProduct>,
}
/// Bit-sliced measurement records; bits are negative Pauli eigenvalues.
#[derive(Clone, Debug)]
pub struct PackedMeasurements {
    rows: Vec<Vec<u64>>,
    shots: usize,
}
impl PackedMeasurements {
    /// Number of shots.
    pub fn shots(&self) -> usize {
        self.shots
    }
    /// Number of measurement records per shot.
    pub fn measurements(&self) -> usize {
        self.rows.len()
    }
    /// Checked access to one record.
    pub fn bit(&self, measurement: usize, shot: usize) -> Result<bool> {
        if shot >= self.shots || measurement >= self.rows.len() {
            return Err(Error::Basis);
        }
        Ok(self.rows[measurement][shot / 64] >> (shot % 64) & 1 != 0)
    }
    /// Packed record words by measurement; padding bits are always zero.
    pub fn rows(&self) -> &[Vec<u64>] {
        &self.rows
    }
}
/// Compile one reference tableau trajectory, then sample batches of bit-packed Pauli frames.
///
/// Supports a fixed ordered sequence of Pauli measurements, including noncommuting
/// ones. Adaptive changes to the measurement program require a different compiled
/// sampler. This host bit-sliced engine does not pretend to execute GPU kernels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceSampler {
    qubits: usize,
    steps: Vec<Step>,
}
impl ReferenceSampler {
    /// Precompute deterministic reference outcomes and random-measurement kickbacks.
    pub fn compile(
        initial: &Tableau,
        observables: &[PauliProduct],
        max_terms: usize,
    ) -> Result<Self> {
        let required = initial
            .qubits()
            .checked_mul(observables.len())
            .ok_or(Error::Overflow)?;
        if required > max_terms {
            return Err(Error::ResourceLimit {
                resource: "reference sampler terms",
                required,
                limit: max_terms,
            });
        }
        let mut state = initial.clone();
        let mut steps = Vec::with_capacity(observables.len());
        for observable in observables {
            let (record, kickback) = state.reference_measure(observable)?;
            steps.push(Step {
                observable: observable.clone(),
                reference: record.negative,
                kickback,
            });
        }
        Ok(Self {
            qubits: initial.qubits(),
            steps,
        })
    }
    /// Start a complete frame continuation. Only fixed measurement windows are streamed;
    /// live frame planes are retained in full, so no entanglement boundary is discarded.
    pub fn start(
        &self,
        start: u64,
        shots: usize,
        seed: u64,
        noise: PauliNoise,
        max_bits: usize,
    ) -> Result<ReferenceSession> {
        start
            .checked_add(u64::try_from(shots).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        check_bits(self.qubits, 0, shots, max_bits)?;
        let words = shots.div_ceil(64);
        let mut x = vec![vec![0u64; words]; self.qubits];
        let mut z = x.clone();
        let seed = shot_seed(seed, 0, 0x5245_4653_414d_0001);
        let [px, py, pz] = noise.0;
        if px + py + pz > 0. {
            for q in 0..self.qubits {
                for shot in 0..shots {
                    let mut rng =
                        ChaCha8Rng::seed_from_u64(shot_seed(seed, start + shot as u64, q as u64));
                    let draw = rng.random::<f64>();
                    let bit = 1u64 << (shot % 64);
                    if draw < px + py {
                        x[q][shot / 64] |= bit;
                    }
                    if draw >= px && draw < px + py + pz {
                        z[q][shot / 64] |= bit;
                    }
                }
            }
        }
        Ok(ReferenceSession {
            state: FrameCheckpoint {
                version: 1,
                qubits: self.qubits,
                program: self.steps.clone(),
                start,
                shots,
                seed,
                next: 0,
                x,
                z,
            },
        })
    }
    /// Restore a checkpoint after matching the entire compiled program, dimensions,
    /// cursor and padding. JSON deserialization alone never authorizes execution.
    pub fn resume(&self, checkpoint: FrameCheckpoint, max_bits: usize) -> Result<ReferenceSession> {
        if checkpoint.version != 1 {
            return Err(Error::Unsupported("unknown frame checkpoint schema".into()));
        }
        if checkpoint.qubits != self.qubits
            || checkpoint.program != self.steps
            || checkpoint.next > self.steps.len()
        {
            return Err(Error::InvalidValue(
                "frame checkpoint program/cursor mismatch".into(),
            ));
        }
        checkpoint
            .start
            .checked_add(u64::try_from(checkpoint.shots).map_err(|_| Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        check_bits(self.qubits, 0, checkpoint.shots, max_bits)?;
        let words = checkpoint.shots.div_ceil(64);
        for plane in [&checkpoint.x, &checkpoint.z] {
            if plane.len() != self.qubits
                || plane.iter().any(|row| {
                    row.len() != words
                        || (!checkpoint.shots.is_multiple_of(64)
                            && row
                                .last()
                                .is_some_and(|word| word >> (checkpoint.shots % 64) != 0))
                })
            {
                return Err(Error::Shape(
                    "invalid frame checkpoint plane/padding".into(),
                ));
            }
        }
        Ok(ReferenceSession { state: checkpoint })
    }
    /// Sample initial faults and all records, with stable per-shot RNG under sharding.
    /// The bit budget covers both frame planes, output rows and one random scratch row.
    pub fn sample(
        &self,
        start: u64,
        shots: usize,
        seed: u64,
        noise: PauliNoise,
        max_bits: usize,
    ) -> Result<PackedMeasurements> {
        check_bits(self.qubits, self.steps.len(), shots, max_bits)?;
        self.start(start, shots, seed, noise, max_bits)?
            .advance(self.steps.len(), max_bits)
    }
}
/// Versioned opaque complete frame state and fixed compiled program, suitable for serde.
/// Restore with `ReferenceSampler::resume`, which validates compatibility and allocation bounds.
/// Input byte sizes should also be bounded before deserializing untrusted persistence.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameCheckpoint {
    version: u32,
    qubits: usize,
    program: Vec<Step>,
    start: u64,
    shots: usize,
    seed: u64,
    next: usize,
    x: Vec<Vec<u64>>,
    z: Vec<Vec<u64>>,
}
/// A bounded measurement-window continuation retaining complete batched Pauli frames.
#[derive(Clone, Debug)]
pub struct ReferenceSession {
    state: FrameCheckpoint,
}
impl ReferenceSession {
    /// Next measurement index in the fixed program.
    pub fn position(&self) -> usize {
        self.state.next
    }
    /// Complete restart state; previously returned output rows are not retained.
    pub fn checkpoint(&self) -> FrameCheckpoint {
        self.state.clone()
    }
    /// Owned packed frame payload bytes, excluding immutable program and vector bookkeeping.
    pub fn frame_bytes(&self) -> usize {
        self.state
            .x
            .iter()
            .chain(&self.state.z)
            .map(|r| r.len() * 8)
            .sum()
    }
    /// Advance by at most the requested number of measurements. Returned rows begin at
    /// the old `position`; reaching the end yields a valid zero-row batch. The allocation
    /// budget is checked before mutating any frame or measurement cursor.
    pub fn advance(&mut self, measurements: usize, max_bits: usize) -> Result<PackedMeasurements> {
        let s = &mut self.state;
        let count = measurements.min(s.program.len() - s.next);
        check_bits(s.qubits, count, s.shots, max_bits)?;
        let words = s.shots.div_ceil(64);
        let mut rows = Vec::with_capacity(count);
        for i in s.next..s.next + count {
            let step = &s.program[i];
            let mut row = vec![if step.reference { u64::MAX } else { 0 }; words];
            for &(q, p) in step.observable.terms() {
                for (w, value) in row.iter_mut().enumerate() {
                    if p != Pauli::Z {
                        *value ^= s.z[q][w];
                    }
                    if p != Pauli::X {
                        *value ^= s.x[q][w];
                    }
                }
            }
            if let Some(kickback) = &step.kickback {
                let mut random = vec![0u64; words];
                for shot in 0..s.shots {
                    let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(
                        s.seed,
                        s.start + shot as u64,
                        (s.qubits + i) as u64,
                    ));
                    if rng.random() {
                        random[shot / 64] |= 1 << (shot % 64);
                    }
                }
                for (value, &r) in row.iter_mut().zip(&random) {
                    *value ^= r;
                }
                for &(q, p) in kickback.terms() {
                    for (w, &r) in random.iter().enumerate() {
                        if p != Pauli::Z {
                            s.x[q][w] ^= r;
                        }
                        if p != Pauli::X {
                            s.z[q][w] ^= r;
                        }
                    }
                }
            }
            if !s.shots.is_multiple_of(64)
                && let Some(last) = row.last_mut()
            {
                *last &= (1u64 << (s.shots % 64)) - 1;
            }
            rows.push(row);
        }
        s.next += count;
        Ok(PackedMeasurements {
            rows,
            shots: s.shots,
        })
    }
}
fn check_bits(qubits: usize, records: usize, shots: usize, max_bits: usize) -> Result<()> {
    let fields = qubits
        .checked_mul(2)
        .and_then(|x| x.checked_add(records))
        .and_then(|x| x.checked_add(1))
        .ok_or(Error::Overflow)?;
    let required = fields
        .checked_mul(shots.div_ceil(64))
        .and_then(|x| x.checked_mul(64))
        .ok_or(Error::Overflow)?;
    if required > max_bits {
        return Err(Error::ResourceLimit {
            resource: "reference sampler bits",
            required,
            limit: max_bits,
        });
    }
    Ok(())
}
