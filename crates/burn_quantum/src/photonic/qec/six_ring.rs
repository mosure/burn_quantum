//! Rotated six-ring FBQC lattice, Supplementary Note VC of Bartolucci et al. (2023).
use super::SyndromeGraph;
use crate::photonic::stabilizer::{Pauli, PauliProduct};
use crate::photonic::{FusionProgram, NodeId, ResourceState};
use crate::{Error, Result};
/// Periodic rotated six-ring network: one ring per cubic cell, fusions across faces,
/// and two-color check vertices joined by the diagonals of each fusion face.
/// This is the paper's rotated periodic convention, not its earlier BCC coordinates.
#[derive(Clone, Debug)]
pub struct SixRingTorus {
    side: usize,
    program: FusionProgram,
    graphs: [SyndromeGraph; 2],
    records: [Vec<usize>; 2],
}
impl SixRingTorus {
    /// Construct an even-sided periodic network. An odd side breaks the global
    /// two-color boundary condition and is rejected. There are 6*side³ physical photons.
    pub fn new(side: usize, source: NodeId, analyzer: NodeId, max_photons: usize) -> Result<Self> {
        if side < 2 || !side.is_multiple_of(2) {
            return Err(Error::InvalidDimensions(
                "rotated six-ring side must be even and >=2".into(),
            ));
        }
        let cells = side.checked_pow(3).ok_or(Error::Overflow)?;
        let photons = cells.checked_mul(6).ok_or(Error::Overflow)?;
        let limit = max_photons.min(1 << 20);
        if photons > limit {
            return Err(Error::ResourceLimit {
                resource: "six-ring photons",
                required: photons,
                limit,
            });
        }
        let mut program = FusionProgram::new();
        let mut rings = Vec::with_capacity(cells);
        for _ in 0..cells {
            rings.push(program.add_resource(ResourceState::six_ring(), source)?);
        }
        let mut edges: [Vec<(usize, usize)>; 2] = std::array::from_fn(|_| vec![]);
        let mut cuts: [Vec<[bool; 3]>; 2] = std::array::from_fn(|_| vec![]);
        let mut records: [Vec<usize>; 2] = std::array::from_fn(|_| vec![]);
        // Cycle(+z,+x,+y,-z,-x,-y), so opposed faces are opposite ring vertices.
        let positive = [1, 2, 0];
        for cell in 0..cells {
            let c = coords(cell, side);
            for axis in 0..3 {
                let mut neighbor = c;
                neighbor[axis] = (neighbor[axis] + 1) % side;
                let fusion = program.num_fusions();
                program.fuse(
                    rings[cell][positive[axis]],
                    rings[index(neighbor, side)][positive[axis] + 3],
                    analyzer,
                )?;
                let tangents: Vec<_> = (0..3).filter(|&a| a != axis).collect();
                let mut endpoints: [Vec<[usize; 3]>; 2] = std::array::from_fn(|_| vec![]);
                for a in 0..2 {
                    for b in 0..2 {
                        let mut corner = [0; 3];
                        corner[axis] = 1;
                        corner[tangents[0]] = a;
                        corner[tangents[1]] = b;
                        let vertices = std::array::from_fn::<_, 3, _>(|j| {
                            positive[j] + if corner[j] == 1 { 0 } else { 3 }
                        });
                        let p = corner_factor(vertices, positive[axis]);
                        endpoints[usize::from(p == Pauli::Z)]
                            .push(std::array::from_fn(|j| c[j] + corner[j]));
                    }
                }
                for (offset, pair) in endpoints.into_iter().enumerate() {
                    debug_assert_eq!(pair.len(), 2);
                    let color = pair[0].iter().sum::<usize>() % 2;
                    let a = pair[0].map(|v| v % side);
                    let b = pair[1].map(|v| v % side);
                    edges[color].push((index(a, side) / 2, index(b, side) / 2));
                    cuts[color].push(std::array::from_fn(|j| {
                        (pair[0][j] / side) != (pair[1][j] / side)
                    }));
                    records[color].push(2 * fusion + offset);
                }
            }
        }
        let [a, b] = edges;
        let [ca, cb] = cuts;
        Ok(Self {
            side,
            program,
            graphs: [
                SyndromeGraph::from_edges(cells / 2, a, ca)?,
                SyndromeGraph::from_edges(cells / 2, b, cb)?,
            ],
            records,
        })
    }
    /// Periodic cubic side in the explicitly rotated lattice coordinates.
    pub fn side(&self) -> usize {
        self.side
    }
    /// Actual resource-state and physical-fusion recipe.
    pub fn program(&self) -> &FusionProgram {
        &self.program
    }
    /// Immutable check graph for color 0 or 1; each bulk check has 12 incident records.
    pub fn graph(&self, color: usize) -> Result<&SyndromeGraph> {
        self.graphs.get(color).ok_or(Error::Basis)
    }
    /// Index in the full XX,ZZ-per-fusion stream for every edge of this color's graph.
    pub fn record_indices(&self, color: usize) -> Result<&[usize]> {
        self.records
            .get(color)
            .map(Vec::as_slice)
            .ok_or(Error::Basis)
    }
    /// Full commuting Bell observables, ordered XX then ZZ at each actual fusion.
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
fn corner_factor(vertices: [usize; 3], port: usize) -> Pauli {
    if vertices.iter().all(|q| q % 2 == vertices[0] % 2) {
        return Pauli::X;
    }
    let center = vertices
        .iter()
        .copied()
        .find(|q| vertices.contains(&((q + 1) % 6)) && vertices.contains(&((q + 5) % 6)))
        .expect("three selected cube faces are consecutive or alternating on the ring");
    if port == center { Pauli::X } else { Pauli::Z }
}
fn coords(i: usize, l: usize) -> [usize; 3] {
    [i / (l * l), (i / l) % l, i % l]
}
fn index(c: [usize; 3], l: usize) -> usize {
    (c[0] * l + c[1]) * l + c[2]
}
