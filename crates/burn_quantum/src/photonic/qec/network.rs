use crate::photonic::stabilizer::{Pauli, PauliProduct};
use crate::photonic::{FusionProgram, NodeId, ResourceState};
use crate::{Error, Result};

/// A periodic syndrome graph with three toric cut parities and parallel physical edges.
#[derive(Clone, Debug)]
pub struct SyndromeGraph {
    vertices: usize,
    edges: Vec<(usize, usize)>,
    cuts: Vec<[bool; 3]>,
}
/// Exact minimum-cost matching result, bounded by the number of syndrome defects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchingDecode {
    /// Correction edges from shortest paths between optimally paired defects.
    Correction(Vec<bool>),
    /// The explicit defect-count budget prevents the exponential pairing solver.
    BudgetExceeded,
    /// Odd defect count or disconnected syndrome.
    Inconsistent,
}
impl SyndromeGraph {
    pub(super) fn from_edges(
        vertices: usize,
        edges: Vec<(usize, usize)>,
        cuts: Vec<[bool; 3]>,
    ) -> Result<Self> {
        if vertices == 0
            || edges.len() != cuts.len()
            || edges
                .iter()
                .any(|&(a, b)| a >= vertices || b >= vertices || a == b)
        {
            return Err(Error::Shape("invalid syndrome graph incidence".into()));
        }
        Ok(Self {
            vertices,
            edges,
            cuts,
        })
    }

    /// Number of parity-check vertices.
    pub fn vertices(&self) -> usize {
        self.vertices
    }
    /// Endpoints of each physical measurement-edge coordinate.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }
    /// Three wraparound cut indicators for every edge.
    pub fn cuts(&self) -> &[[bool; 3]] {
        &self.cuts
    }
    /// Logical cut ambiguities supported entirely on erased edges, including correlated masks.
    /// A spanning forest tracks cut parity; nontrivial erased cycles erase logical information.
    pub fn erasure_logicals(&self, erased: &[bool]) -> Result<[bool; 3]> {
        if erased.len() != self.edges.len() {
            return Err(Error::Shape("erased edge count".into()));
        }
        let mut parent: Vec<_> = (0..self.vertices).collect();
        let mut potential = vec![[false; 3]; self.vertices];
        let mut ambiguous = [false; 3];
        fn find(parent: &[usize], potential: &[[bool; 3]], mut q: usize) -> (usize, [bool; 3]) {
            let mut p = [false; 3];
            while parent[q] != q {
                for j in 0..3 {
                    p[j] ^= potential[q][j];
                }
                q = parent[q];
            }
            (q, p)
        }
        for (e, &lost) in erased.iter().enumerate() {
            if lost {
                let (a, b) = self.edges[e];
                let (ra, pa) = find(&parent, &potential, a);
                let (rb, pb) = find(&parent, &potential, b);
                let mut delta = [false; 3];
                for j in 0..3 {
                    delta[j] = pa[j] ^ pb[j] ^ self.cuts[e][j];
                }
                if ra == rb {
                    for j in 0..3 {
                        ambiguous[j] |= delta[j];
                    }
                } else {
                    parent[ra] = rb;
                    potential[ra] = delta;
                }
            }
        }
        Ok(ambiguous)
    }
    /// Compute endpoint defects for a measurement-error vector.
    pub fn syndrome(&self, errors: &[bool]) -> Result<Vec<bool>> {
        if errors.len() != self.edges.len() {
            return Err(Error::Shape("syndrome edge count".into()));
        }
        let mut s = vec![false; self.vertices];
        for (&e, &(a, b)) in errors.iter().zip(&self.edges) {
            if e {
                s[a] ^= true;
                s[b] ^= true;
            }
        }
        Ok(s)
    }
    /// Nontrivial toric homology of a closed residual chain.
    pub fn logical_parities(&self, errors: &[bool]) -> Result<[bool; 3]> {
        if errors.len() != self.edges.len() {
            return Err(Error::Shape("logical edge count".into()));
        }
        let mut l = [false; 3];
        for (&e, cut) in errors.iter().zip(&self.cuts) {
            if e {
                for j in 0..3 {
                    l[j] ^= cut[j];
                }
            }
        }
        Ok(l)
    }
    /// Dense check matrix, with an explicit entry budget for small independent oracles.
    pub fn checks(&self, max_entries: usize) -> Result<Vec<Vec<bool>>> {
        let required = self
            .vertices
            .checked_mul(self.edges.len())
            .ok_or(Error::Overflow)?;
        if required > max_entries {
            return Err(Error::ResourceLimit {
                resource: "syndrome matrix entries",
                required,
                limit: max_entries,
            });
        }
        let mut h = vec![vec![false; self.edges.len()]; self.vertices];
        for (i, &(a, b)) in self.edges.iter().enumerate() {
            h[a][i] ^= true;
            h[b][i] ^= true;
        }
        Ok(h)
    }
    /// Decode Pauli flips with erased edges assigned zero cost and other edges unit cost.
    /// Exact subset matching is bounded to at most 20 defects; excessive work is explicit.
    pub fn decode(
        &self,
        syndrome: &[bool],
        erased: &[bool],
        max_defects: usize,
    ) -> Result<MatchingDecode> {
        if syndrome.len() != self.vertices || erased.len() != self.edges.len() {
            return Err(Error::Shape("matching input shape".into()));
        }
        if max_defects > 20 {
            return Err(Error::InvalidValue(
                "matching defect budget exceeds 20".into(),
            ));
        }
        let defects: Vec<_> = syndrome
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| b.then_some(i))
            .collect();
        let n = defects.len();
        if !n.is_multiple_of(2) {
            return Ok(MatchingDecode::Inconsistent);
        }
        if n > max_defects {
            return Ok(MatchingDecode::BudgetExceeded);
        }
        if n == 0 {
            return Ok(MatchingDecode::Correction(vec![false; self.edges.len()]));
        }
        let mut paths = vec![vec![vec![]; n]; n];
        let mut costs = vec![vec![usize::MAX; n]; n];
        for (i, &source) in defects.iter().enumerate() {
            let mut distance = vec![usize::MAX; self.vertices];
            let mut previous = vec![None; self.vertices];
            let mut visited = vec![false; self.vertices];
            distance[source] = 0;
            for _ in 0..self.vertices {
                let Some(u) = (0..self.vertices)
                    .filter(|&v| !visited[v] && distance[v] != usize::MAX)
                    .min_by_key(|&v| (distance[v], v))
                else {
                    break;
                };
                visited[u] = true;
                for (e, &(a, b)) in self.edges.iter().enumerate() {
                    let next = if a == u {
                        Some(b)
                    } else if b == u {
                        Some(a)
                    } else {
                        None
                    };
                    if let Some(v) = next {
                        let d = distance[u] + usize::from(!erased[e]);
                        if d < distance[v] {
                            distance[v] = d;
                            previous[v] = Some((u, e));
                        }
                    }
                }
            }
            for j in i + 1..n {
                let mut v = defects[j];
                costs[i][j] = distance[v];
                while v != source {
                    let Some((u, e)) = previous[v] else {
                        return Ok(MatchingDecode::Inconsistent);
                    };
                    paths[i][j].push(e);
                    v = u;
                }
            }
        }
        let states = 1usize << n;
        let mut best = vec![usize::MAX; states];
        let mut choice = vec![0usize; states];
        best[0] = 0;
        for mask in 1..states {
            if !mask.count_ones().is_multiple_of(2) {
                continue;
            }
            let i = mask.trailing_zeros() as usize;
            let rest = mask ^ (1 << i);
            for (j, &cost) in costs[i].iter().enumerate().skip(i + 1) {
                if rest & (1 << j) != 0 {
                    let sub = rest ^ (1 << j);
                    if let Some(value) = best[sub].checked_add(cost)
                        && value < best[mask]
                    {
                        best[mask] = value;
                        choice[mask] = j;
                    }
                }
            }
        }
        if best[states - 1] == usize::MAX {
            return Ok(MatchingDecode::Inconsistent);
        }
        let mut correction = vec![false; self.edges.len()];
        let mut mask = states - 1;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            let j = choice[mask];
            for &e in &paths[i][j] {
                correction[e] ^= true;
            }
            mask ^= (1 << i) | (1 << j);
        }
        Ok(MatchingDecode::Correction(correction))
    }
}
/// Published four-star cubic edge/face incidence network with periodic boundaries.
///
/// A local H frame on edge resources converts the paper's XZ/ZX fusions to this
/// crate's XX/ZZ convention. Primal checks use ZZ, dual checks XX. Each consists
/// of 24 physical parity measurements. No threshold or calibrated device is implied.
#[derive(Clone, Debug)]
pub struct FourStarTorus {
    distance: usize,
    program: FusionProgram,
    primal: SyndromeGraph,
    dual: SyndromeGraph,
}
impl FourStarTorus {
    /// Build a distance-L periodic geometric reference, checking photon count before allocation.
    pub fn new(
        distance: usize,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        if distance < 2 {
            return Err(Error::InvalidDimensions(
                "four-star periodic distance must be >=2".into(),
            ));
        }
        let cells = distance.checked_pow(3).ok_or(Error::Overflow)?;
        let photons = cells.checked_mul(24).ok_or(Error::Overflow)?;
        let max_photons = max_photons.min(1 << 20);
        if photons > max_photons {
            return Err(Error::ResourceLimit {
                resource: "four-star photons",
                required: photons,
                limit: max_photons,
            });
        }
        let mut program = FusionProgram::new();
        let edge_state = ResourceState::four_star().with_local_hadamards(&[0, 1, 2, 3])?;
        let mut edge_ports = Vec::with_capacity(cells * 3);
        let mut face_ports = Vec::with_capacity(cells * 3);
        for _ in 0..cells * 3 {
            edge_ports.push(program.add_resource(edge_state.clone(), source)?);
        }
        for _ in 0..cells * 3 {
            face_ports.push(program.add_resource(ResourceState::four_star(), source)?);
        }
        let mut used = vec![0; cells * 3];
        let mut primal = SyndromeGraph {
            vertices: cells,
            edges: vec![],
            cuts: vec![],
        };
        let mut dual = primal.clone();
        for cell in 0..cells {
            let c = coords(cell, distance);
            for normal in 0..3 {
                let a = (normal + 1) % 3;
                let b = (normal + 2) % 3;
                let incident = [
                    (c, a),
                    (c, b),
                    (shift(c, b, 1, distance), a),
                    (shift(c, a, 1, distance), b),
                ];
                for (port, (base, axis)) in incident.into_iter().enumerate() {
                    let edge = idx(base, distance) * 3 + axis;
                    let q = edge_ports[edge][used[edge]];
                    used[edge] += 1;
                    program.fuse(face_ports[cell * 3 + normal][port], q, analyzer)?;
                    primal.edges.push((
                        cell,
                        idx(shift(c, normal, distance - 1, distance), distance),
                    ));
                    let mut cut = [false; 3];
                    cut[normal] = c[normal] == 0;
                    primal.cuts.push(cut);
                    dual.edges.push((
                        idx(base, distance),
                        idx(shift(base, axis, 1, distance), distance),
                    ));
                    let mut cut = [false; 3];
                    cut[axis] = base[axis] == distance - 1;
                    dual.cuts.push(cut);
                }
            }
        }
        debug_assert!(used.iter().all(|&n| n == 4));
        Ok(Self {
            distance,
            program,
            primal,
            dual,
        })
    }
    /// Periodic side length in cubic unit cells.
    pub fn distance(&self) -> usize {
        self.distance
    }
    /// Actual four-photon resource and destructive-fusion recipe.
    pub fn program(&self) -> &FusionProgram {
        &self.program
    }
    /// Cube checks on ZZ outcomes.
    pub fn primal(&self) -> &SyndromeGraph {
        &self.primal
    }
    /// Vertex checks on XX outcomes.
    pub fn dual(&self) -> &SyndromeGraph {
        &self.dual
    }
    /// Full ideal Bell parity sequence, ordered XX then ZZ per physical fusion.
    pub fn observables(&self) -> Result<Vec<PauliProduct>> {
        self.program
            .fusions()
            .iter()
            .flat_map(|f| {
                [Pauli::X, Pauli::Z]
                    .map(|p| PauliProduct::new(f.photons.iter().map(|q| (q.0, p)).collect()))
            })
            .collect()
    }
}
fn coords(i: usize, l: usize) -> [usize; 3] {
    [i / (l * l), (i / l) % l, i % l]
}
fn idx(c: [usize; 3], l: usize) -> usize {
    (c[0] * l + c[1]) * l + c[2]
}
fn shift(mut c: [usize; 3], axis: usize, step: usize, l: usize) -> [usize; 3] {
    c[axis] = (c[axis] + step) % l;
    c
}
