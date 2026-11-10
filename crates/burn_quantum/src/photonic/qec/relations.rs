use super::gf2::solve;
use crate::photonic::stabilizer::{Pauli, PauliProduct, Tableau};
use crate::photonic::{FusionOutcome, Parity};
use crate::{Error, Result};
/// A deterministic signed parity of commuting measurement outcomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParityRelation {
    /// Indices of measured Pauli products whose bits are XORed.
    pub terms: Vec<usize>,
    /// Expected XOR (true corresponds to a -1 product).
    pub negative: bool,
}
/// Inference from partially erased measurement records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParityInference {
    /// The observable eigenvalue is uniquely determined.
    Known(Parity),
    /// A relation exists, but the available records do not determine its parity.
    Erased,
    /// The target was not correlated with the declared measurement algebra.
    Unconstrained,
    /// Known records contradict the ideal resource checks (e.g. an uncorrected Pauli fault).
    Inconsistent,
}
/// Check group and output correlations derived from an actual pure stabilizer resource state.
///
/// This computes the intersection of the resource stabilizer and commuting measurement
/// groups by GF(2) elimination. Signed phases are checked against the tableau.
#[derive(Clone, Debug)]
pub struct MeasurementRelations {
    initial: Tableau,
    observables: Vec<PauliProduct>,
    commutations: Vec<Vec<bool>>,
    checks: Vec<ParityRelation>,
}
impl MeasurementRelations {
    /// Derive a complete independent check basis, bounded by n*m + m*m binary entries.
    pub fn derive(
        initial: &Tableau,
        observables: &[PauliProduct],
        max_entries: usize,
    ) -> Result<Self> {
        let m = observables.len();
        let required = m
            .checked_mul(initial.qubits().checked_add(m).ok_or(Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        if required > max_entries {
            return Err(Error::ResourceLimit {
                resource: "measurement relation entries",
                required,
                limit: max_entries,
            });
        }
        for (i, p) in observables.iter().enumerate() {
            if observables[..i].iter().any(|q| p.anticommutes(q)) {
                return Err(Error::Unsupported(
                    "relations require commuting measurements".into(),
                ));
            }
        }
        let mut matrix = vec![vec![false; m]; initial.qubits()];
        for (j, p) in observables.iter().enumerate() {
            for (i, v) in initial.commutations(p)?.into_iter().enumerate() {
                matrix[i][j] = v;
            }
        }
        let (_, kernel) =
            solve(&matrix, m, &vec![false; initial.qubits()]).expect("homogeneous system");
        let mut checks = vec![];
        for vector in kernel {
            checks.push(relation(
                initial,
                observables,
                &vector,
                &PauliProduct::new(vec![])?,
            )?);
        }
        Ok(Self {
            initial: initial.clone(),
            observables: observables.to_vec(),
            commutations: matrix,
            checks,
        })
    }
    /// Independent signed check rows.
    pub fn checks(&self) -> &[ParityRelation] {
        &self.checks
    }
    /// Ordered measurement observables.
    pub fn observables(&self) -> &[PauliProduct] {
        &self.observables
    }
    /// Append actual additional commuting measurements and rederive every signed
    /// check from the original resource state. Additional outcomes remain observed
    /// records; this never assumes their eigenvalues are positive or noiseless.
    pub fn extended(&self, additional: &[PauliProduct], max_entries: usize) -> Result<Self> {
        let m = self
            .observables
            .len()
            .checked_add(additional.len())
            .ok_or(Error::Overflow)?;
        let required = m
            .checked_mul(
                self.initial
                    .qubits()
                    .checked_add(m)
                    .ok_or(Error::Overflow)?,
            )
            .ok_or(Error::Overflow)?;
        if required > max_entries {
            return Err(Error::ResourceLimit {
                resource: "measurement relation entries",
                required,
                limit: max_entries,
            });
        }
        let mut observables = self.observables.clone();
        observables.extend_from_slice(additional);
        Self::derive(&self.initial, &observables, max_entries)
    }
    /// Derive a measurement parity correlated with a commuting output observable.
    pub fn correlation(&self, target: &PauliProduct) -> Result<Option<ParityRelation>> {
        if self.observables.iter().any(|p| p.anticommutes(target)) {
            return Err(Error::Unsupported(
                "target does not commute with measurements".into(),
            ));
        }
        let rhs = self.initial.commutations(target)?;
        let Some((solution, _)) = solve(&self.commutations, self.observables.len(), &rhs) else {
            return Ok(None);
        };
        Ok(Some(relation(
            &self.initial,
            &self.observables,
            &solution,
            target,
        )?))
    }
    /// Derive an independent complete basis of output stabilizers correlated with
    /// the measured records. Output sites must be distinct and unmeasured. The
    /// returned products commute, and each relation includes its exact Pauli sign.
    pub fn output_relations(
        &self,
        outputs: &[usize],
        max_entries: usize,
    ) -> Result<Vec<(PauliProduct, ParityRelation)>> {
        if outputs.iter().enumerate().any(|(i, &q)| {
            q >= self.initial.qubits()
                || outputs[..i].contains(&q)
                || self
                    .observables
                    .iter()
                    .any(|p| p.terms().iter().any(|&(site, _)| site == q))
        }) {
            return Err(Error::Wires(
                "invalid or measured output relation site".into(),
            ));
        }
        let m = self.observables.len();
        let width = m
            .checked_add(outputs.len().checked_mul(2).ok_or(Error::Overflow)?)
            .ok_or(Error::Overflow)?;
        let required = width
            .checked_mul(
                self.initial
                    .qubits()
                    .checked_add(width)
                    .ok_or(Error::Overflow)?,
            )
            .ok_or(Error::Overflow)?;
        if required > max_entries {
            return Err(Error::ResourceLimit {
                resource: "output relation entries",
                required,
                limit: max_entries,
            });
        }
        let mut matrix = self.commutations.clone();
        for &q in outputs {
            for p in [Pauli::X, Pauli::Z] {
                let signature = self
                    .initial
                    .commutations(&PauliProduct::new(vec![(q, p)])?)?;
                for (row, b) in matrix.iter_mut().zip(signature) {
                    row.push(b);
                }
            }
        }
        let (_, kernel) = solve(&matrix, width, &vec![false; self.initial.qubits()])
            .expect("homogeneous relations");
        let mut basis: Vec<(usize, Vec<bool>)> = vec![];
        let mut result = Vec::new();
        for vector in kernel {
            let original = &vector[m..];
            let mut reduced = original.to_vec();
            for (pivot, row) in &basis {
                if reduced[*pivot] {
                    for (b, &r) in reduced.iter_mut().zip(row) {
                        *b ^= r;
                    }
                }
            }
            let Some(pivot) = reduced.iter().position(|&v| v) else {
                continue;
            };
            basis.push((pivot, reduced));
            let product = PauliProduct::new(
                outputs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &q)| match (original[2 * i], original[2 * i + 1]) {
                        (false, false) => None,
                        (true, false) => Some((q, Pauli::X)),
                        (false, true) => Some((q, Pauli::Z)),
                        (true, true) => Some((q, Pauli::Y)),
                    })
                    .collect(),
            )?;
            // The joint nullspace vector already contains a measurement flow
            // for this output. Preserve that flow and check its canonical Pauli
            // sign directly instead of refactoring the full matrix per output.
            let relation = relation(&self.initial, &self.observables, &vector[..m], &product)?;
            result.push((product, relation));
        }
        Ok(result)
    }
    /// Infer an output eigenvalue, eliminating erased fusion parities with the full check group.
    /// Extra individual-Z failure measurements do not invalidate a relation after all erased
    /// factors have been eliminated. Physically lost output sites must be handled by the caller.
    pub fn infer(
        &self,
        records: &[Option<bool>],
        target: &PauliProduct,
    ) -> Result<ParityInference> {
        if records.len() != self.observables.len() {
            return Err(Error::Shape("measurement record count".into()));
        }
        let Some(correlation) = self.correlation(target)? else {
            return Ok(ParityInference::Unconstrained);
        };
        let unknown: Vec<_> = records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.is_none().then_some(i))
            .collect();
        let matrix: Vec<Vec<bool>> = self
            .checks
            .iter()
            .map(|c| unknown.iter().map(|i| c.terms.contains(i)).collect())
            .collect();
        let rhs: Vec<_> = self
            .checks
            .iter()
            .map(|c| {
                c.terms
                    .iter()
                    .fold(c.negative, |v, &i| v ^ records[i].unwrap_or(false))
            })
            .collect();
        let Some((solution, kernel)) = solve(&matrix, unknown.len(), &rhs) else {
            return Ok(ParityInference::Inconsistent);
        };
        let selected: Vec<_> = unknown
            .iter()
            .map(|i| correlation.terms.contains(i))
            .collect();
        if kernel.iter().any(|v| {
            v.iter()
                .zip(&selected)
                .fold(false, |s, (&a, &b)| s ^ (a && b))
        }) {
            return Ok(ParityInference::Erased);
        }
        let mut value = correlation
            .terms
            .iter()
            .fold(correlation.negative, |v, &i| {
                v ^ records[i].unwrap_or(false)
            });
        value ^= solution
            .iter()
            .zip(&selected)
            .fold(false, |s, (&a, &b)| s ^ (a && b));
        Ok(ParityInference::Known(if value {
            Parity::Odd
        } else {
            Parity::Even
        }))
    }
    /// Convert standard-fusion outcomes to records ordered XX,ZZ for every fusion.
    pub fn fusion_records(outcomes: &[FusionOutcome]) -> Vec<Option<bool>> {
        outcomes
            .iter()
            .flat_map(|o| {
                let p = o.parities();
                [
                    p.xx.map(|v| v == Parity::Odd),
                    p.zz.map(|v| v == Parity::Odd),
                ]
            })
            .collect()
    }
}
fn relation(
    initial: &Tableau,
    observables: &[PauliProduct],
    vector: &[bool],
    target: &PauliProduct,
) -> Result<ParityRelation> {
    let mut product = target.clone();
    let mut negative = false;
    let mut terms = vec![];
    for (j, &selected) in vector.iter().enumerate() {
        if selected {
            let (sign, next) = product.multiply_commuting(&observables[j])?;
            negative ^= sign;
            product = next;
            terms.push(j);
        }
    }
    let value = initial.expectation(&product)?;
    if value == 0 {
        return Err(Error::InvalidValue(
            "derived relation is not a resource stabilizer".into(),
        ));
    }
    negative ^= value < 0;
    Ok(ParityRelation { terms, negative })
}
