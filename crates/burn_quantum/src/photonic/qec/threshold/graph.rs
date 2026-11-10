use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// An independent binary fault coordinate with up to two detector endpoints.
/// Endpoint `detectors` denotes a virtual spatial boundary (no parity constraint).
/// `logical` is its flip mask on the declared logical observables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectorEdge {
    /// First detector endpoint, or the virtual boundary index.
    pub a: usize,
    /// Second detector endpoint, or the virtual boundary index.
    pub b: usize,
    /// XOR flip mask on the declared logical observables.
    pub logical: u64,
}
/// Allocation limits checked before graph construction and decoder allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphLimits {
    /// Maximum constrained detector vertices, excluding the virtual boundary.
    pub max_detectors: usize,
    /// Maximum independent binary fault coordinates per graph.
    pub max_edges: usize,
}
impl Default for GraphLimits {
    fn default() -> Self {
        Self {
            max_detectors: 1 << 20,
            max_edges: 1 << 24,
        }
    }
}
/// Work limits are outcomes, not silently discarded trials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderBudget {
    /// Maximum graph-edge inspections, including initialization and peeling.
    pub max_edge_visits: u64,
    /// Maximum synchronous half-edge growth rounds for the mixed decoder.
    pub max_growth_rounds: usize,
}
impl Default for DecoderBudget {
    fn default() -> Self {
        Self {
            max_edge_visits: 1 << 28,
            max_growth_rounds: 1 << 16,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Reproducible decoder effort, independent of host wall-clock timing.
pub struct DecoderWork {
    /// Number of edge inspections actually completed.
    pub edge_visits: u64,
    /// Number of synchronous half-edge growth steps.
    pub growth_rounds: usize,
}
impl DecoderWork {
    fn visit(&mut self, budget: DecoderBudget) -> bool {
        if self.edge_visits >= budget.max_edge_visits {
            return false;
        }
        self.edge_visits += 1;
        true
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// Decoder completion class; a returned correction can still have an unknown
/// logical error under a Pauli channel. Only the simulator can check that truth.
pub enum GraphStatus {
    /// Syndrome-consistent correction, with no intrinsic erasure ambiguity.
    Corrected,
    /// True erased coordinates support distinct logical classes.
    Ambiguous,
    /// The supplied syndrome cannot be explained by the allowed support.
    Inconsistent,
    /// A declared work bound prevented completion; do not use partial correction.
    WorkLimit,
}
/// Exact logical image of the erased cycle space. A rank-r erased component
/// makes 2^r logical classes equiprobable under the uniform erasure channel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErasureAnalysis {
    /// Completion class, including explicit work exhaustion.
    pub status: GraphStatus,
    /// Independent XOR basis, with distinct highest set bits.
    pub logical_basis: Vec<u64>,
    /// Completed work; never estimated from wall time.
    pub work: DecoderWork,
}
impl ErasureAnalysis {
    /// Dimension of the erased logical image; incomplete results have no basis.
    pub fn rank(&self) -> usize {
        self.logical_basis.len()
    }
    /// Logical observables whose values are not fixed by the available records.
    pub fn ambiguous_mask(&self) -> u64 {
        self.logical_basis.iter().fold(0, |a, b| a | b)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// Correction and erasure evidence. Correction is valid only for Corrected or
/// Ambiguous status; a work-limited or inconsistent result may contain partial work.
pub struct GraphDecode {
    /// Completion class, including explicit work exhaustion.
    pub status: GraphStatus,
    /// Physical fault coordinates in the selected recovery chain.
    pub correction: Vec<bool>,
    /// XOR flip mask on the declared logical observables.
    pub logical: u64,
    /// Logical image of genuinely erased edges, excluding grown Pauli support.
    pub erased_logical_basis: Vec<u64>,
    /// Completed work; never estimated from wall time.
    pub work: DecoderWork,
}
/// Sparse binary incidence graph, including parallel edges and relative cycles
/// ending on a spatial boundary. Memory is linear in vertices plus edges.
#[derive(Clone, Debug)]
pub struct DecoderGraph {
    detectors: usize,
    logicals: usize,
    edges: Vec<DetectorEdge>,
}
impl DecoderGraph {
    /// Validate sparse graph incidence, logical bit width and allocation limits.
    pub fn new(
        detectors: usize,
        logicals: usize,
        edges: Vec<DetectorEdge>,
        limits: GraphLimits,
    ) -> Result<Self> {
        if detectors == 0 || logicals > 64 {
            return Err(Error::InvalidDimensions("detector graph dimensions".into()));
        }
        for (resource, required, limit) in [
            ("detector graph vertices", detectors, limits.max_detectors),
            ("detector graph edges", edges.len(), limits.max_edges),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        detectors.checked_add(1).ok_or(Error::Overflow)?;
        let mask = if logicals == 64 {
            u64::MAX
        } else {
            (1u64 << logicals) - 1
        };
        if edges
            .iter()
            .any(|e| e.a > detectors || e.b > detectors || e.logical & !mask != 0)
        {
            return Err(Error::Shape("detector graph edge or logical mask".into()));
        }
        Ok(Self {
            detectors,
            logicals,
            edges,
        })
    }
    /// Number of constrained detector vertices; also the virtual boundary index.
    pub fn detectors(&self) -> usize {
        self.detectors
    }
    /// Number of declared logical observables (at most 64).
    pub fn logicals(&self) -> usize {
        self.logicals
    }
    /// Ordered independent fault coordinates and their incidence/labels.
    pub fn edges(&self) -> &[DetectorEdge] {
        &self.edges
    }
    fn validate_mask(&self, values: &[bool]) -> Result<()> {
        if values.len() != self.edges.len() {
            Err(Error::Shape("detector graph edge mask".into()))
        } else {
            Ok(())
        }
    }
    /// Detector parities generated by a complete binary fault vector.
    pub fn syndrome(&self, errors: &[bool]) -> Result<Vec<bool>> {
        self.validate_mask(errors)?;
        let mut out = vec![false; self.detectors];
        for (&fault, edge) in errors.iter().zip(&self.edges) {
            if fault {
                if edge.a < self.detectors {
                    out[edge.a] ^= true;
                }
                if edge.b < self.detectors {
                    out[edge.b] ^= true;
                }
            }
        }
        Ok(out)
    }
    /// Logical-observable flips generated by a binary fault vector.
    pub fn logical_flips(&self, errors: &[bool]) -> Result<u64> {
        self.validate_mask(errors)?;
        Ok(errors
            .iter()
            .zip(&self.edges)
            .fold(0, |value, (&fault, edge)| {
                value ^ if fault { edge.logical } else { 0 }
            }))
    }
    /// Exact cycle-space homology using union by size, path compression and XOR
    /// potentials. Boundary-to-boundary paths are cycles through the virtual node.
    /// No dense check matrix, quantum state, or measurement reference is constructed.
    pub fn analyze_erasures(
        &self,
        erased: &[bool],
        budget: DecoderBudget,
    ) -> Result<ErasureAnalysis> {
        self.validate_mask(erased)?;
        let mut dsu = Dsu::new(self.detectors + 1);
        let mut basis = [0u64; 64];
        let mut work = DecoderWork::default();
        for (&lost, edge) in erased.iter().zip(&self.edges) {
            if !work.visit(budget) {
                return Ok(ErasureAnalysis {
                    status: GraphStatus::WorkLimit,
                    logical_basis: vec![],
                    work,
                });
            }
            if lost && let Some(cycle) = dsu.join(edge.a, edge.b, edge.logical) {
                insert_basis(&mut basis, cycle);
            }
        }
        let logical_basis: Vec<_> = basis.into_iter().filter(|&v| v != 0).collect();
        Ok(ErasureAnalysis {
            status: if logical_basis.is_empty() {
                GraphStatus::Corrected
            } else {
                GraphStatus::Ambiguous
            },
            logical_basis,
            work,
        })
    }
}
fn insert_basis(basis: &mut [u64; 64], mut value: u64) {
    while value != 0 {
        let pivot = 63 - value.leading_zeros() as usize;
        if basis[pivot] == 0 {
            basis[pivot] = value;
            return;
        }
        value ^= basis[pivot];
    }
}
struct Dsu {
    parent: Vec<usize>,
    size: Vec<usize>,
    potential: Vec<u64>,
}
impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
            potential: vec![0; n],
        }
    }
    fn find(&mut self, q: usize) -> (usize, u64) {
        let mut root = q;
        let mut total = 0;
        while self.parent[root] != root {
            total ^= self.potential[root];
            root = self.parent[root];
        }
        let mut at = q;
        let mut prefix = 0;
        while self.parent[at] != at {
            let next = self.parent[at];
            let old = self.potential[at];
            self.parent[at] = root;
            self.potential[at] = total ^ prefix;
            prefix ^= old;
            at = next;
        }
        (root, total)
    }
    /// Returns the logical label of a newly closed cycle, or None for a tree edge.
    fn join(&mut self, a: usize, b: usize, label: u64) -> Option<u64> {
        let (mut ra, pa) = self.find(a);
        let (mut rb, pb) = self.find(b);
        let delta = pa ^ pb ^ label;
        if ra == rb {
            return Some(delta);
        }
        if self.size[ra] > self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[ra] = rb;
        self.potential[ra] = delta;
        self.size[rb] += self.size[ra];
        None
    }
}

impl DecoderGraph {
    /// Exact erasure-only correction by spanning-forest peeling. `Ambiguous`
    /// still returns a syndrome-consistent correction and the full logical image;
    /// it must not be interpreted as reliable recovery of erased logical data.
    pub fn decode_erasures(
        &self,
        syndrome: &[bool],
        erased: &[bool],
        budget: DecoderBudget,
    ) -> Result<GraphDecode> {
        self.decode(syndrome, erased, budget, false)
    }
    /// Unweighted union-find decoding: simultaneously grow all odd, non-boundary
    /// clusters by half an edge, fuse completed edges, then peel a spanning forest.
    /// This implements Algorithm1 of Delfosse–Nickerson (1709.06218v3) with path
    /// compression/union by size. Global edge sweeps give O((V+E)*rounds*alpha(V))
    /// work and O(V+E) storage, not the optimized boundary-list complexity of their
    /// Algorithm2. Unequal edge probabilities are not weighted by this decoder.
    pub fn decode_union_find(
        &self,
        syndrome: &[bool],
        erased: &[bool],
        budget: DecoderBudget,
    ) -> Result<GraphDecode> {
        self.decode(syndrome, erased, budget, true)
    }
    fn decode(
        &self,
        syndrome: &[bool],
        erased: &[bool],
        budget: DecoderBudget,
        grow: bool,
    ) -> Result<GraphDecode> {
        self.validate_mask(erased)?;
        if syndrome.len() != self.detectors {
            return Err(Error::Shape("detector syndrome width".into()));
        }
        let mut result = GraphDecode {
            status: GraphStatus::Corrected,
            correction: vec![false; self.edges.len()],
            logical: 0,
            erased_logical_basis: vec![],
            work: DecoderWork::default(),
        };
        let n = self.detectors + 1;
        let mut dsu = Dsu::new(n);
        let mut basis = [0; 64];
        let mut support = vec![0u8; self.edges.len()];
        let mut forest = Vec::with_capacity(n - 1);
        for (i, (&lost, e)) in erased.iter().zip(&self.edges).enumerate() {
            if !result.work.visit(budget) {
                result.status = GraphStatus::WorkLimit;
                return Ok(result);
            }
            if lost {
                support[i] = 2;
                if let Some(cycle) = dsu.join(e.a, e.b, e.logical) {
                    insert_basis(&mut basis, cycle);
                } else {
                    forest.push(i);
                }
            }
        }
        result.erased_logical_basis = basis.into_iter().filter(|&v| v != 0).collect();
        let mut parity = vec![false; n];
        let mut complete = Vec::new();
        loop {
            parity.fill(false);
            for (v, &bit) in syndrome.iter().enumerate() {
                if bit {
                    let root = dsu.find(v).0;
                    parity[root] ^= true;
                }
            }
            parity[dsu.find(self.detectors).0] = false;
            if !parity.iter().any(|&v| v) {
                break;
            }
            if !grow {
                result.status = GraphStatus::Inconsistent;
                return Ok(result);
            }
            if result.work.growth_rounds >= budget.max_growth_rounds {
                result.status = GraphStatus::WorkLimit;
                return Ok(result);
            }
            result.work.growth_rounds += 1;
            complete.clear();
            let mut progress = false;
            // Freeze parity for this entire synchronous growth round. Joining
            // clusters during this pass would introduce edge-order growth bias.
            for (i, e) in self.edges.iter().enumerate() {
                if !result.work.visit(budget) {
                    result.status = GraphStatus::WorkLimit;
                    return Ok(result);
                }
                if support[i] == 2 {
                    continue;
                }
                let a = dsu.find(e.a).0;
                let b = dsu.find(e.b).0;
                if a == b {
                    continue;
                }
                let increment = u8::from(parity[a]) + u8::from(parity[b]);
                if increment != 0 {
                    progress = true;
                    support[i] = (support[i] + increment).min(2);
                    if support[i] == 2 {
                        complete.push(i);
                    }
                }
            }
            if !progress {
                result.status = GraphStatus::Inconsistent;
                return Ok(result);
            }
            for &i in &complete {
                let e = self.edges[i];
                if dsu.join(e.a, e.b, e.logical).is_none() {
                    forest.push(i);
                }
            }
        }
        let mut adjacency = vec![Vec::new(); n];
        for i in forest {
            if !result.work.visit(budget) {
                result.status = GraphStatus::WorkLimit;
                return Ok(result);
            }
            let e = self.edges[i];
            adjacency[e.a].push((e.b, i));
            adjacency[e.b].push((e.a, i));
        }
        let mut visited = vec![false; n];
        let mut pending = syndrome.to_vec();
        pending.push(false);
        let mut order = Vec::with_capacity(n);
        let mut parent = vec![None; n];
        // Root the component touching a boundary at that boundary. Its parity is
        // unconstrained; all other roots must have even total detector syndrome.
        for root in std::iter::once(self.detectors).chain(0..self.detectors) {
            if visited[root] {
                continue;
            }
            visited[root] = true;
            let begin = order.len();
            order.push(root);
            let mut position = begin;
            while position < order.len() {
                let v = order[position];
                position += 1;
                for &(w, e) in &adjacency[v] {
                    if !result.work.visit(budget) {
                        result.status = GraphStatus::WorkLimit;
                        return Ok(result);
                    }
                    if !visited[w] {
                        visited[w] = true;
                        parent[w] = Some((v, e));
                        order.push(w);
                    }
                }
            }
            for &v in order[begin..].iter().rev() {
                if let Some((p, e)) = parent[v]
                    && pending[v]
                {
                    result.correction[e] = true;
                    result.logical ^= self.edges[e].logical;
                    pending[p] ^= true;
                }
            }
            if root != self.detectors && pending[root] {
                result.status = GraphStatus::Inconsistent;
                return Ok(result);
            }
        }
        if !result.erased_logical_basis.is_empty() {
            result.status = GraphStatus::Ambiguous;
        }
        Ok(result)
    }
}
