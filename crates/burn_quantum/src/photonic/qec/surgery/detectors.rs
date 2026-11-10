//! Combined merge/split phenomenological detector model with random seam boundaries.
use super::*;
use crate::photonic::qec::threshold::{DecoderGraph, DetectorEdge, GraphLimits};
use crate::photonic::stabilizer::{Pauli, PauliProduct};
use crate::{Error, Result};
use std::collections::{BTreeSet, VecDeque};
/// Declared independent fault locations; no ideal encoder/check-gate hook errors are implied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurgeryFault {
    /// X or Z just before a merged/split round, numbered0..2d. Final checks are ideal.
    Data {
        /// Before this zero-based merge/split schedule round.
        round: usize,
        /// Relative site in the two-patch data register.
        site: usize,
        /// Independent X or Z fault; coincident coordinates compose to Y.
        pauli: Pauli,
    },
    /// Classical flipped sign in one nonfinal local-check measurement.
    Readout {
        /// Index in the complete record list, excluding noiseless final checks.
        record: usize,
    },
}
/// One recorded check; final=true is noiseless closure of the actual final data syndrome.
#[derive(Clone, Debug)]
pub struct SurgeryCheckRecord {
    /// Round in the complete schedule; 0..d merged,d..2d split,2d final.
    pub round: usize,
    /// Local check within its phase.
    pub check: usize,
    /// Measured relative data Pauli.
    pub observable: PauliProduct,
    /// Excludes this record from classical measurement-fault locations.
    pub final_check: bool,
}
/// Compilation budgets for the sparse transition model and distance certification.
#[derive(Clone, Copy, Debug)]
pub struct SurgeryDetectorLimits {
    /// Number of measured check signs, including ideal closure.
    pub max_records: usize,
    /// Number of data/readout fault locations.
    pub max_faults: usize,
    /// Sparse record/detector incidences visited while deriving all fault signatures.
    pub max_signature_work: usize,
    /// Edge visits in the four logical-distance searches.
    pub max_distance_work: usize,
}
impl Default for SurgeryDetectorLimits {
    fn default() -> Self {
        Self {
            max_records: 1 << 20,
            max_faults: 1 << 20,
            max_signature_work: 1 << 27,
            max_distance_work: 1 << 27,
        }
    }
}
/// Complete phase-transition detector model for ideal checks plus independent
/// round-boundary data X/Z and check-readout faults. Split gauge corrections are
/// deferred until after the ideal closing checks. All first-round seam gauges
/// and first split single-patch signs remain unknown; only justified relations
/// become detectors. This is a phenomenological model, not a circuit-noise model.
#[derive(Clone, Debug)]
pub struct SurgeryDetectorModel {
    plan: PlanarSurgery,
    records: Vec<SurgeryCheckRecord>,
    detectors: Vec<Vec<usize>>,
    faults: Vec<SurgeryFault>,
    graph: DecoderGraph,
    logical: Vec<PauliProduct>,
    distances: [usize; 4],
    witnesses: Vec<Vec<usize>>,
    work: usize,
}
impl SurgeryDetectorModel {
    /// Derive each physical fault signature, reject nongraphlike models, and
    /// certify shortest undetected paths for all four output-Choi observables.
    pub fn new(plan: PlanarSurgery, limits: SurgeryDetectorLimits) -> Result<Self> {
        let d = plan.code.distance();
        let n = 2 * plan.code.qubits();
        let m = plan.merged.len();
        let s = plan.split.len();
        let count = plan.measurements().checked_add(s).ok_or(Error::Overflow)?;
        let faults_count = (2 * d)
            .checked_mul(n)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(plan.measurements()))
            .ok_or(Error::Overflow)?;
        for (resource, required, limit) in [
            ("surgery detector records", count, limits.max_records),
            ("surgery fault locations", faults_count, limits.max_faults),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        let mut records = Vec::with_capacity(count);
        for round in 0..=2 * d {
            let checks = if round < d { &plan.merged } else { &plan.split };
            for (check, p) in checks.iter().enumerate() {
                records.push(SurgeryCheckRecord {
                    round,
                    check,
                    observable: p.clone(),
                    final_check: round == 2 * d,
                });
            }
        }
        let mut detectors = vec![];
        // First seam values are random. Only unchanged bulk/composite generators
        // have known initial values in the two input code spaces.
        for check in d..m {
            detectors.push(vec![check]);
        }
        for round in 1..d {
            for check in 0..m {
                detectors.push(vec![(round - 1) * m + check, round * m + check]);
            }
        }
        let split_start = d * m;
        for (check, p) in plan.split.iter().enumerate() {
            if plan.boundary.iter().any(|pair| pair.contains(&check)) {
                continue;
            }
            let prior = plan.merged.iter().position(|q| q == p).ok_or_else(|| {
                Error::InvalidValue("unchanged split check missing from merge".into())
            })?;
            detectors.push(vec![(d - 1) * m + prior, split_start + check]);
        }
        for &[a, b] in &plan.boundary {
            let composite = plan.split[a].multiply_commuting(&plan.split[b])?.1;
            let prior = plan
                .merged
                .iter()
                .position(|q| q == &composite)
                .ok_or_else(|| Error::InvalidValue("split composite missing from merge".into()))?;
            detectors.push(vec![(d - 1) * m + prior, split_start + a, split_start + b]);
        }
        // Includes ideal final closure, but never a known-zero assumption on
        // the individually random restored-boundary generators.
        for round in 1..=d {
            for check in 0..s {
                detectors.push(vec![
                    split_start + (round - 1) * s + check,
                    split_start + round * s + check,
                ]);
            }
        }
        // Every relation has a distinct greatest record index, so these rows
        // are independent. Their nullity equals the d random merge gauges and
        // d-1 random split gauges; no unproved known-zero boundary is inserted.
        let pivots: BTreeSet<_> = detectors
            .iter()
            .map(|row| *row.iter().max().unwrap())
            .collect();
        if pivots.len() != detectors.len() || records.len() - detectors.len() != 2 * d - 1 {
            return Err(Error::InvalidValue(
                "surgery detector relations are incomplete or dependent".into(),
            ));
        }
        let mut inverse = vec![vec![]; records.len()];
        for (i, row) in detectors.iter().enumerate() {
            for &r in row {
                inverse[r].push(i);
            }
        }
        let mut incidence = vec![vec![vec![]; 2]; n];
        for (i, record) in records.iter().enumerate() {
            for &(q, p) in record.observable.terms() {
                let color = match p {
                    Pauli::Z => 0,
                    Pauli::X => 1,
                    Pauli::Y => {
                        return Err(Error::Unsupported(
                            "surgery detector model must be CSS".into(),
                        ));
                    }
                };
                incidence[q][color].push(i);
            }
        }
        let (a, b, c) = match plan.basis {
            MergeBasis::ZZ => (
                plan.code.logical_z(0)?,
                plan.code.logical_z(plan.code.qubits())?,
                plan.code
                    .logical_x(0)?
                    .multiply_commuting(&plan.code.logical_x(plan.code.qubits())?)?
                    .1,
            ),
            MergeBasis::XX => (
                plan.code.logical_x(0)?,
                plan.code.logical_x(plan.code.qubits())?,
                plan.code
                    .logical_z(0)?
                    .multiply_commuting(&plan.code.logical_z(plan.code.qubits())?)?
                    .1,
            ),
        };
        let parity = a.multiply_commuting(&b)?.1;
        let logical = vec![a, b, c, parity];
        let mut faults = vec![];
        let mut edges = vec![];
        let mut work = 0usize;
        let mut add = |fault: SurgeryFault,
                       flips: Vec<usize>,
                       physical: Option<(usize, Pauli)>|
         -> Result<()> {
            let mut touched = BTreeSet::new();
            let mut label = 0u64;
            for &r in &flips {
                work = work
                    .checked_add(inverse[r].len() + 1)
                    .ok_or(Error::Overflow)?;
                if work > limits.max_signature_work {
                    return Err(Error::ResourceLimit {
                        resource: "surgery detector signature work",
                        required: work,
                        limit: limits.max_signature_work,
                    });
                }
                if r < d {
                    label ^= 1 << 3;
                }
                for &detector in &inverse[r] {
                    if !touched.insert(detector) {
                        touched.remove(&detector);
                    }
                }
            }
            if let Some((site, pauli)) = physical {
                let p = PauliProduct::new(vec![(site, pauli)])?;
                for (i, l) in logical.iter().enumerate() {
                    if p.anticommutes(l) {
                        label ^= 1 << i;
                    }
                }
            }
            let endpoints: Vec<_> = touched.into_iter().collect();
            let boundary = detectors.len();
            let (a, b) = match endpoints.as_slice() {
                [] => (boundary, boundary),
                [a] => (*a, boundary),
                [a, b] => (*a, *b),
                _ => {
                    return Err(Error::Unsupported(
                        "surgery fault has more than two detector endpoints".into(),
                    ));
                }
            };
            edges.push(DetectorEdge {
                a,
                b,
                logical: label,
            });
            faults.push(fault);
            Ok(())
        };
        for round in 0..2 * d {
            for (site, by_pauli) in incidence.iter().enumerate() {
                for (color, &pauli) in [Pauli::X, Pauli::Z].iter().enumerate() {
                    let flips = by_pauli[color]
                        .iter()
                        .copied()
                        .filter(|&r| records[r].round >= round)
                        .collect();
                    add(
                        SurgeryFault::Data { round, site, pauli },
                        flips,
                        Some((site, pauli)),
                    )?;
                }
            }
        }
        for (record, r) in records.iter().enumerate() {
            if !r.final_check {
                add(SurgeryFault::Readout { record }, vec![record], None)?;
            }
        }
        let graph = DecoderGraph::new(
            detectors.len(),
            4,
            edges,
            GraphLimits {
                max_detectors: limits.max_records,
                max_edges: limits.max_faults,
            },
        )?;
        let (mut distances, mut witnesses) = ([0; 4], vec![]);
        let mut distance_work = 0;
        for (i, distance) in distances.iter_mut().enumerate() {
            let witness =
                shortest_logical(&graph, i, &mut distance_work, limits.max_distance_work)?;
            *distance = witness.len();
            witnesses.push(witness);
        }
        if distances.iter().any(|&value| value < d) {
            return Err(Error::InvalidValue(format!(
                "surgery transition distance below d: {distances:?}"
            )));
        }
        Ok(Self {
            plan,
            records,
            detectors,
            faults,
            graph,
            logical,
            distances,
            witnesses,
            work,
        })
    }
    /// Checked doubled-seam schedule.
    pub fn plan(&self) -> &PlanarSurgery {
        &self.plan
    }
    /// Complete observed-record order, including ideal final checks.
    pub fn records(&self) -> &[SurgeryCheckRecord] {
        &self.records
    }
    /// Detector XOR supports; every ideal random branch yields zero for all rows.
    pub fn detectors(&self) -> &[Vec<usize>] {
        &self.detectors
    }
    /// Complete physical fault catalog in graph-edge order.
    pub fn faults(&self) -> &[SurgeryFault] {
        &self.faults
    }
    /// Graph with four output-Choi/record logical labels.
    pub fn graph(&self) -> &DecoderGraph {
        &self.graph
    }
    /// Data factors of the four Choi stabilizers. Label3 also XORs the recorded
    /// first seam parity; the untouched reference factors are implicit.
    pub fn logical_data_observables(&self) -> &[PauliProduct] {
        &self.logical
    }
    /// Minimum undetected physical/readout fault count for each logical label.
    pub fn logical_distances(&self) -> [usize; 4] {
        self.distances
    }
    /// Edge indices proving each reported space-time distance upper bound.
    pub fn distance_witnesses(&self) -> &[Vec<usize>] {
        &self.witnesses
    }
    /// Sparse signature incidences visited during model derivation.
    pub fn signature_work(&self) -> usize {
        self.work
    }
    /// Project raw signs, without substituting zero for unknown/erased records.
    pub fn detector_values(&self, records: &[bool]) -> Result<Vec<bool>> {
        if records.len() != self.records.len() {
            return Err(Error::Shape("surgery record count".into()));
        }
        Ok(self
            .detectors
            .iter()
            .map(|row| row.iter().fold(false, |v, &r| v ^ records[r]))
            .collect())
    }
    /// Exact raw-record flip mask for one declared fault, under reference-Pauli
    /// coupling and deferred split feed-forward. Useful for independent oracles.
    pub fn fault_records(&self, index: usize) -> Result<Vec<bool>> {
        let fault = self
            .faults
            .get(index)
            .ok_or_else(|| Error::InvalidValue("surgery fault index".into()))?;
        match fault {
            SurgeryFault::Readout { record } => {
                Ok((0..self.records.len()).map(|r| r == *record).collect())
            }
            SurgeryFault::Data { round, site, pauli } => {
                let p = PauliProduct::new(vec![(*site, *pauli)])?;
                Ok(self
                    .records
                    .iter()
                    .map(|r| r.round >= *round && p.anticommutes(&r.observable))
                    .collect())
            }
        }
    }
}
fn shortest_logical(
    graph: &DecoderGraph,
    logical: usize,
    work: &mut usize,
    budget: usize,
) -> Result<Vec<usize>> {
    let nodes = graph.detectors() + 1;
    let mut adjacency = vec![vec![]; nodes];
    let mut starts = BTreeSet::new();
    for (i, e) in graph.edges().iter().enumerate() {
        let flip = (e.logical >> logical) & 1 != 0;
        adjacency[e.a].push((e.b, flip, i));
        adjacency[e.b].push((e.a, flip, i));
        if flip {
            starts.insert(e.a);
            starts.insert(e.b);
        }
    }
    let mut best: Option<Vec<usize>> = None;
    for start in starts {
        let root = 2 * start;
        let target = root + 1;
        let mut parent = vec![None; 2 * nodes];
        let mut depth = vec![0; 2 * nodes];
        parent[root] = Some((root, usize::MAX));
        let mut queue = VecDeque::from([root]);
        while let Some(node) = queue.pop_front() {
            if node == target {
                break;
            }
            if best.as_ref().is_some_and(|p| depth[node] + 1 >= p.len()) {
                continue;
            }
            for &(next, flip, edge) in &adjacency[node / 2] {
                *work = work.checked_add(1).ok_or(Error::Overflow)?;
                if *work > budget {
                    return Err(Error::ResourceLimit {
                        resource: "surgery spacetime distance work",
                        required: *work,
                        limit: budget,
                    });
                }
                let next = 2 * next + ((node % 2) ^ usize::from(flip));
                if parent[next].is_none() {
                    parent[next] = Some((node, edge));
                    depth[next] = depth[node] + 1;
                    queue.push_back(next);
                }
            }
        }
        if parent[target].is_some() {
            let mut path = vec![];
            let mut node = target;
            while node != root {
                let (p, e) = parent[node].unwrap();
                path.push(e);
                node = p;
            }
            if best.as_ref().is_none_or(|p| path.len() < p.len()) {
                best = Some(path);
            }
        }
    }
    best.ok_or_else(|| Error::InvalidValue("surgery logical has no spacetime witness".into()))
}
