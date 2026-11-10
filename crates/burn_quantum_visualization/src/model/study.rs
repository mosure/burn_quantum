//! Bounded task outcomes accompanying a scientific trajectory.
use burn_quantum::{Error, Result};
use serde::{Deserialize, Serialize};

/// One finite outcome or resource count. Values are already in the displayed units.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyMetric {
    /// Short human-readable name.
    pub label: String,
    /// Measured value or explicitly named reference value.
    pub value: f64,
    /// Display suffix, such as `%`, `qubits` or `shots`.
    pub unit: String,
}
impl StudyMetric {
    /// Construct a metric; document validation checks bounds and finite values.
    pub fn new(label: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value,
            unit: unit.into(),
        }
    }
}
/// A held-out classification evaluation; matrix rows are true classes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassificationSummary {
    /// Names in confusion-matrix order.
    pub classes: Vec<String>,
    /// Counts indexed by true class then predicted class.
    pub confusion: Vec<Vec<usize>>,
    /// Training, validation and untouched test partition sizes.
    pub split_sizes: [usize; 3],
    /// Iteration selected using validation loss, before test evaluation.
    pub selected_iteration: usize,
    /// Recorded validation loss for each displayed training iterate.
    pub validation_loss: Vec<f64>,
}
/// Explicit classical three-literal CNF and its bounded quantum-search readout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SatSummary {
    /// Number of assignment bits, excluding clause work qubits.
    pub variables: usize,
    /// Signed, one-based literals; exactly three distinct variables per clause.
    pub clauses: Vec<Vec<i32>>,
    /// Independently verified assignment, or absent for the contradictory fixture.
    pub witness: Option<Vec<bool>>,
    /// Exact count for this deliberately small reference problem.
    pub satisfying_assignments: usize,
    /// Quantum marginal mass on satisfying assignments.
    pub success_probability: f64,
    /// Uniform random-assignment baseline.
    pub uniform_probability: f64,
}
/// Meaning of the photonic quantum-state representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhotonicSummary {
    /// Number of two-level logical qubits represented by resource photons.
    pub logical_qubits: usize,
    /// True when each state subsystem includes a third vacuum/loss level.
    pub includes_vacuum: bool,
    /// Number of seeded, unconditional trajectory attempts behind reported counts.
    pub attempts: usize,
    /// Successful delivered pairs in those attempts.
    pub delivered: usize,
    /// Attempts with photon loss/erasure.
    pub erased: usize,
}
/// Task context, independent references and resource costs for the Results view.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyReport {
    /// Brief task heading.
    pub title: String,
    /// What was computed and how its quality is measured.
    pub summary: String,
    /// Bounded key results and comparisons.
    pub metrics: Vec<StudyMetric>,
    /// Assumptions, provenance and scaling boundaries.
    pub notes: Vec<String>,
    /// Held-out classifier evaluation, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification: Option<ClassificationSummary>,
    /// Explicit 3SAT problem, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sat: Option<SatSummary>,
    /// Logical-versus-loss bookkeeping interpretation, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub photonic: Option<PhotonicSummary>,
}
impl StudyReport {
    /// Check bounded sizes, finite metrics, confusion counts and SAT witnesses.
    pub fn validate(&self, frames: usize) -> Result<()> {
        let invalid = || Error::InvalidValue("invalid experiment study report".into());
        if self.title.len() > 256
            || self.summary.len() > 4096
            || self.metrics.len() > 24
            || self.notes.len() > 16
            || self.notes.iter().any(|s| s.len() > 2048)
            || self
                .metrics
                .iter()
                .any(|m| m.label.len() > 128 || m.unit.len() > 32 || !m.value.is_finite())
        {
            return Err(invalid());
        }
        if let Some(c) = &self.classification {
            let n = c.classes.len();
            if !(2..=8).contains(&n)
                || c.classes.iter().any(|s| s.len() > 80)
                || c.confusion.len() != n
                || c.confusion.iter().any(|row| row.len() != n)
                || c.split_sizes.iter().any(|&s| s == 0 || s > 100_000)
                || c.confusion
                    .iter()
                    .flatten()
                    .try_fold(0usize, |a, b| a.checked_add(*b))
                    != Some(c.split_sizes[2])
                || c.selected_iteration >= frames
                || c.validation_loss.len() != frames
                || c.validation_loss.iter().any(|v| !v.is_finite() || *v < 0.)
            {
                return Err(invalid());
            }
        }
        if let Some(s) = &self.sat {
            if !(3..=12).contains(&s.variables)
                || s.clauses.len() > 32
                || s.clauses.is_empty()
                || s.clauses.iter().any(|c| {
                    c.len() != 3
                        || c.iter()
                            .any(|l| *l == 0 || l.unsigned_abs() as usize > s.variables)
                        || c[0].unsigned_abs() == c[1].unsigned_abs()
                        || c[0].unsigned_abs() == c[2].unsigned_abs()
                        || c[1].unsigned_abs() == c[2].unsigned_abs()
                })
                || !s.success_probability.is_finite()
                || !(0. ..=1.0005).contains(&s.success_probability)
                || !s.uniform_probability.is_finite()
            {
                return Err(invalid());
            }
            let satisfies = |bits: &[bool]| {
                s.clauses.iter().all(|c| {
                    c.iter()
                        .any(|&l| bits[l.unsigned_abs() as usize - 1] == (l > 0))
                })
            };
            let count = (0..1usize << s.variables)
                .filter(|&i| {
                    satisfies(
                        &(0..s.variables)
                            .map(|w| (i >> (s.variables - 1 - w)) & 1 != 0)
                            .collect::<Vec<_>>(),
                    )
                })
                .count();
            if count != s.satisfying_assignments
                || (count == 0 && s.success_probability > 1e-8)
                || (count == 1usize << s.variables && (s.success_probability - 1.).abs() > 5e-4)
                || (s.uniform_probability - count as f64 / (1usize << s.variables) as f64).abs()
                    > 1e-8
                || s.witness
                    .as_ref()
                    .is_some_and(|bits| bits.len() != s.variables || !satisfies(bits))
            {
                return Err(invalid());
            }
        }
        if let Some(p) = &self.photonic
            && (p.logical_qubits == 0
                || p.logical_qubits > 1024
                || p.attempts > 10_000_000
                || p.delivered
                    .checked_add(p.erased)
                    .is_none_or(|n| n > p.attempts))
        {
            return Err(invalid());
        }
        Ok(())
    }
}
