//! Checked adapters from signed physical measurement relations to fault decoders.
use super::{DecoderGraph, DetectorEdge, GraphLimits};
use crate::photonic::{
    qec::{MeasurementRelations, ParityCode, ParityRelation},
    stabilizer::PauliProduct,
};
use crate::{Error, Result};
/// Explicit bounded algebra/search limits. These are construction/certification
/// limits, distinct from the per-shot sparse graph decoding budget.
#[derive(Clone, Copy, Debug)]
pub struct RecordCodeLimits {
    /// Maximum actual measurement records, including explicitly appended closure.
    pub max_records: usize,
    /// Maximum XOR/pair-reduction attempts in graph-basis conversion.
    pub max_row_operations: usize,
}
impl Default for RecordCodeLimits {
    fn default() -> Self {
        Self {
            max_records: 4096,
            max_row_operations: 1 << 22,
        }
    }
}
/// Signed physical record model with an explicit subset of noisy coordinates.
/// Excluded coordinates must actually be measured under the model's stated
/// perfect-boundary assumption; their signs are never replaced by zero.
#[derive(Clone, Debug)]
pub struct RecordCode {
    records: usize,
    faults: Vec<usize>,
    checks: Vec<ParityRelation>,
    logicals: Vec<ParityRelation>,
    parity: ParityCode,
}
/// Graphlike representation obtained only by invertible check-row operations
/// and separating zero-fault-support consistency constraints. Logical labels
/// retain the exact original physical record coordinates.
#[derive(Clone, Debug)]
pub struct RecordGraph {
    graph: DecoderGraph,
    records: usize,
    checks: Vec<ParityRelation>,
    constraints: Vec<ParityRelation>,
    faults: Vec<usize>,
    row_operations: usize,
}
/// Exact bounded fault-coordinate distance, with an actual measurement-record witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordDistance {
    /// Minimum number of noisy record flips giving this undetected logical action.
    pub distance: usize,
    /// Original physical measurement indices, not compressed graph coordinates.
    pub records: Vec<usize>,
    /// Candidate fault subsets checked, including the returned witness.
    pub candidates: usize,
}
impl RecordCode {
    /// Validate unique noisy coordinates, derive every target's actual signed
    /// correlation, and retain the complete resource-derived check group.
    pub fn new(
        relations: &MeasurementRelations,
        targets: &[PauliProduct],
        fault_records: Vec<usize>,
        limits: RecordCodeLimits,
    ) -> Result<Self> {
        let records = relations.observables().len();
        if records > limits.max_records {
            return Err(Error::ResourceLimit {
                resource: "signed measurement records",
                required: records,
                limit: limits.max_records,
            });
        }
        if fault_records.is_empty()
            || targets.len() > 64
            || fault_records
                .iter()
                .enumerate()
                .any(|(i, &r)| r >= records || fault_records[..i].contains(&r))
        {
            return Err(Error::Shape("noisy record coordinate selection".into()));
        }
        let logicals = targets
            .iter()
            .map(|p| {
                relations.correlation(p)?.ok_or_else(|| {
                    Error::InvalidValue("unconstrained record-code logical target".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let checks = relations.checks().to_vec();
        let restrict = |row: &ParityRelation| {
            fault_records
                .iter()
                .map(|r| row.terms.contains(r))
                .collect()
        };
        let parity = ParityCode::new(
            fault_records.len(),
            checks.iter().map(restrict).collect(),
            logicals.iter().map(restrict).collect(),
        )?;
        Ok(Self {
            records,
            faults: fault_records,
            checks,
            logicals,
            parity,
        })
    }
    /// Complete binary fault code for independent algebraic/erasure oracles.
    pub fn parity_code(&self) -> &ParityCode {
        &self.parity
    }
    /// Ordered physical measurement indices of noisy coordinates.
    pub fn fault_records(&self) -> &[usize] {
        &self.faults
    }
    /// Every actual resource check, including deterministic Pauli signs.
    pub fn checks(&self) -> &[ParityRelation] {
        &self.checks
    }
    /// Exact signed measured parities for the selected output operators.
    pub fn logical_relations(&self) -> &[ParityRelation] {
        &self.logicals
    }
    /// Observable check violations, including measured closure records.
    pub fn syndrome(&self, records: &[bool]) -> Result<Vec<bool>> {
        validate_records(records, self.records)?;
        Ok(self.checks.iter().map(|r| evaluate(r, records)).collect())
    }
    /// Try a bounded weight-reducing row basis, rejecting remaining hyperedges.
    /// Successful conversion preserves the entire check space. This heuristic
    /// can reject a code that admits some other graph basis; rejection is never
    /// silently replaced by an approximate graph.
    pub fn try_graph(
        &self,
        limits: RecordCodeLimits,
        graph_limits: GraphLimits,
    ) -> Result<RecordGraph> {
        let mut work = 0;
        let mut basis: Vec<(usize, Row)> = Vec::new();
        let mut constraints = Vec::new();
        for check in &self.checks {
            let mut row = Row::from(check, self.records);
            for (pivot, prior) in &basis {
                charge(&mut work, limits)?;
                if row.bits[self.faults[*pivot]] {
                    row.xor(prior);
                }
            }
            if let Some(pivot) = self.faults.iter().position(|&r| row.bits[r]) {
                basis.push((pivot, row));
            } else {
                constraints.push(row.relation());
            }
        }
        let mut rows: Vec<_> = basis.into_iter().map(|(_, r)| r).collect();
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..rows.len() {
                for j in 0..rows.len() {
                    if i == j {
                        continue;
                    }
                    charge(&mut work, limits)?;
                    let old = self.faults.iter().filter(|&&r| rows[i].bits[r]).count();
                    let new = self
                        .faults
                        .iter()
                        .filter(|&&r| rows[i].bits[r] ^ rows[j].bits[r])
                        .count();
                    if new < old {
                        let prior = rows[j].clone();
                        rows[i].xor(&prior);
                        changed = true;
                    }
                }
            }
        }
        let mut checks: Vec<_> = rows.iter().map(Row::relation).collect();
        if checks.is_empty() {
            checks.push(ParityRelation {
                terms: vec![],
                negative: false,
            });
        }
        let boundary = checks.len();
        let mut edges = Vec::with_capacity(self.faults.len());
        for &record in &self.faults {
            let incidence: Vec<_> = checks
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.terms.contains(&record).then_some(i))
                .collect();
            if incidence.len() > 2 {
                return Err(Error::Unsupported(format!(
                    "record {record} touches {} derived checks; no graph approximation permitted",
                    incidence.len()
                )));
            }
            edges.push(DetectorEdge {
                a: incidence.first().copied().unwrap_or(boundary),
                b: incidence.get(1).copied().unwrap_or(boundary),
                logical: self
                    .logicals
                    .iter()
                    .enumerate()
                    .fold(0, |v, (i, r)| v | ((r.terms.contains(&record) as u64) << i)),
            });
        }
        Ok(RecordGraph {
            graph: DecoderGraph::new(checks.len(), self.logicals.len(), edges, graph_limits)?,
            records: self.records,
            checks,
            constraints,
            faults: self.faults.clone(),
            row_operations: work,
        })
    }
    /// Exhaust every smaller fault support before returning a minimum-distance
    /// witness. A work/weight limit is an error, never a claimed distance.
    pub fn minimum_distance(
        &self,
        logical: usize,
        max_weight: usize,
        max_candidates: usize,
    ) -> Result<RecordDistance> {
        if logical >= self.logicals.len() || max_weight == 0 || max_weight > self.faults.len() {
            return Err(Error::InvalidValue("record distance search inputs".into()));
        }
        let n = self.faults.len();
        let words = self.checks.len().div_ceil(64);
        let columns: Vec<Vec<u64>> = (0..n)
            .map(|q| {
                let mut c = vec![0; words];
                for (i, row) in self.parity.checks().iter().enumerate() {
                    if row[q] {
                        c[i / 64] |= 1 << (i % 64);
                    }
                }
                c
            })
            .collect();
        let mut work = 0;
        let mut selected = Vec::new();
        let mut syndrome = vec![0; words];
        let search = DistanceSearch {
            columns: &columns,
            logical: &self.parity.logicals()[logical],
            budget: max_candidates,
        };
        for weight in 1..=max_weight {
            if search.visit(0, weight, &mut selected, &mut syndrome, false, &mut work)? {
                return Ok(RecordDistance {
                    distance: weight,
                    records: selected.iter().map(|&q| self.faults[q]).collect(),
                    candidates: work,
                });
            }
        }
        Err(Error::ResourceLimit {
            resource: "record distance search weight",
            required: max_weight.checked_add(1).ok_or(Error::Overflow)?,
            limit: max_weight,
        })
    }
}
impl RecordGraph {
    /// Sparse decoder graph in the original noisy-coordinate order.
    pub fn graph(&self) -> &DecoderGraph {
        &self.graph
    }
    /// Original physical measurement indices corresponding to graph edges.
    pub fn fault_records(&self) -> &[usize] {
        &self.faults
    }
    /// Number of actual bounded basis-reduction attempts.
    pub fn row_operations(&self) -> usize {
        self.row_operations
    }
    /// Derive detector values from observed signed records. Violated zero-fault
    /// constraints reject records outside the declared perfect-closure model.
    pub fn syndrome(&self, records: &[bool]) -> Result<Vec<bool>> {
        validate_records(records, self.records)?;
        if self.constraints.iter().any(|r| evaluate(r, records)) {
            return Err(Error::InvalidValue(
                "records violate a perfect-coordinate consistency check".into(),
            ));
        }
        Ok(self.checks.iter().map(|r| evaluate(r, records)).collect())
    }
}
fn validate_records(records: &[bool], n: usize) -> Result<()> {
    if records.len() != n {
        Err(Error::Shape("actual measurement record count".into()))
    } else {
        Ok(())
    }
}
fn evaluate(r: &ParityRelation, records: &[bool]) -> bool {
    r.terms.iter().fold(r.negative, |v, &q| v ^ records[q])
}
fn charge(work: &mut usize, limits: RecordCodeLimits) -> Result<()> {
    *work = work.checked_add(1).ok_or(Error::Overflow)?;
    if *work > limits.max_row_operations {
        Err(Error::ResourceLimit {
            resource: "record graph row operations",
            required: *work,
            limit: limits.max_row_operations,
        })
    } else {
        Ok(())
    }
}
#[derive(Clone)]
struct Row {
    bits: Vec<bool>,
    negative: bool,
}
impl Row {
    fn from(r: &ParityRelation, n: usize) -> Self {
        let mut bits = vec![false; n];
        for &q in &r.terms {
            bits[q] ^= true;
        }
        Self {
            bits,
            negative: r.negative,
        }
    }
    fn xor(&mut self, other: &Self) {
        for (a, &b) in self.bits.iter_mut().zip(&other.bits) {
            *a ^= b;
        }
        self.negative ^= other.negative;
    }
    fn relation(&self) -> ParityRelation {
        ParityRelation {
            terms: self
                .bits
                .iter()
                .enumerate()
                .filter_map(|(i, &b)| b.then_some(i))
                .collect(),
            negative: self.negative,
        }
    }
}
struct DistanceSearch<'a> {
    columns: &'a [Vec<u64>],
    logical: &'a [bool],
    budget: usize,
}
impl DistanceSearch<'_> {
    fn visit(
        &self,
        start: usize,
        left: usize,
        selected: &mut Vec<usize>,
        syndrome: &mut [u64],
        parity: bool,
        work: &mut usize,
    ) -> Result<bool> {
        if left == 0 {
            *work = work.checked_add(1).ok_or(Error::Overflow)?;
            if *work > self.budget {
                return Err(Error::ResourceLimit {
                    resource: "record distance candidates",
                    required: *work,
                    limit: self.budget,
                });
            }
            return Ok(parity && syndrome.iter().all(|&v| v == 0));
        }
        for q in start..=self.columns.len() - left {
            selected.push(q);
            for (s, &c) in syndrome.iter_mut().zip(&self.columns[q]) {
                *s ^= c;
            }
            if self.visit(
                q + 1,
                left - 1,
                selected,
                syndrome,
                parity ^ self.logical[q],
                work,
            )? {
                return Ok(true);
            }
            for (s, &c) in syndrome.iter_mut().zip(&self.columns[q]) {
                *s ^= c;
            }
            selected.pop();
        }
        Ok(false)
    }
}
