//! Check-color-preserving rough/smooth surfaces of the six-ring fusion lattice.
use super::{BoundaryCertificate, BoundaryCertificateLimits, SixRingBoundaryNetwork};
use crate::photonic::{
    FusionProgram, NodeId, PhotonId, ResourceState,
    qec::{MeasurementRelations, SixRingPatch},
    stabilizer::{Pauli, PauliProduct},
};
use crate::{Error, Result};
/// Finite axis-aligned six-ring surface with actual color-preserving terminal measurements.
///
/// Both x walls retain color1 of the checkerboard bulk checks, and both y walls
/// retain color0. The physical X/Z basis therefore alternates with source parity
/// and face orientation. Time-end photons remain quantum outputs. Certificate
/// axes use an explicit local-H frame on output cells with odd x+y; no output
/// operation or closing syndrome measurement is silently added to the program.
#[derive(Clone, Debug)]
pub struct SixRingSurfaceNetwork {
    inner: SixRingBoundaryNetwork,
    hadamards: Vec<usize>,
}
impl SixRingSurfaceNetwork {
    /// One distance-labelled square patch propagated through the specified resource layers.
    /// Actual end-code and record distances must be obtained from their respective certificates.
    pub fn memory(
        distance: usize,
        rounds: usize,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        Self::build(distance, rounds, false, None, source, analyzer, max_photons)
    }
    /// Two patches separated by their color-preserving x boundaries throughout.
    pub fn separated(
        distance: usize,
        rounds: usize,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        Self::build(distance, rounds, true, None, source, analyzer, max_photons)
    }
    /// Open the shared interface for at least distance layers between nonempty end slabs.
    /// Every enabled seam bond is an actual XX/ZZ fusion. Layer count alone is not a threshold claim.
    #[allow(clippy::too_many_arguments)]
    pub fn parity_merge(
        distance: usize,
        pre_rounds: usize,
        merge_rounds: usize,
        post_rounds: usize,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        if pre_rounds == 0 || post_rounds == 0 || merge_rounds < distance {
            return Err(Error::InvalidValue(
                "surface merge requires nonempty end slabs and at least distance merge layers"
                    .into(),
            ));
        }
        let end = pre_rounds
            .checked_add(merge_rounds)
            .ok_or(Error::Overflow)?;
        let rounds = end.checked_add(post_rounds).ok_or(Error::Overflow)?;
        Self::with_window(
            distance,
            rounds,
            [pre_rounds, end],
            source,
            analyzer,
            max_photons,
        )
    }
    /// Explicit geometric seam window [start,end), including short-window negative controls.
    /// This constructor asserts neither sufficient repetition nor fault tolerance.
    pub fn with_window(
        distance: usize,
        rounds: usize,
        window: [usize; 2],
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        if window[0] == 0 || window[0] >= window[1] || window[1] >= rounds {
            return Err(Error::InvalidValue(
                "surface seam window must be strictly internal".into(),
            ));
        }
        Self::build(
            distance,
            rounds,
            true,
            Some(window),
            source,
            analyzer,
            max_photons,
        )
    }
    fn build(
        distance: usize,
        rounds: usize,
        paired: bool,
        window: Option<[usize; 2]>,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        if !(3..=31).contains(&distance)
            || distance.is_multiple_of(2)
            || !(1..=63).contains(&rounds)
        {
            return Err(Error::InvalidDimensions(
                "surface side must be odd3..31 and layers1..63".into(),
            ));
        }
        let nx = distance * if paired { 2 } else { 1 };
        let required = nx * distance * rounds * 6;
        let limit = max_photons.min(16384);
        if required > limit {
            return Err(Error::ResourceLimit {
                resource: "surface six-ring photons",
                required,
                limit,
            });
        }
        let mut centers = Vec::with_capacity(required / 6);
        for x in 0..nx {
            for y in 0..distance {
                for z in 0..rounds {
                    centers.push([x, y, z]);
                }
            }
        }
        let mut program = FusionProgram::new();
        let rings = centers
            .iter()
            .map(|_| program.add_resource(ResourceState::six_ring(), source))
            .collect::<Result<Vec<_>>>()?;
        let index = |c: [usize; 3]| (c[0] * distance + c[1]) * rounds + c[2];
        let positive = [1, 2, 0];
        let mut boundary = vec![];
        let mut ends: [[Vec<PhotonId>; 2]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
        for (cell, &c) in centers.iter().enumerate() {
            for axis in 0..3 {
                for sign in [1isize, -1] {
                    let port = positive[axis] + if sign < 0 { 3 } else { 0 };
                    let photon = rings[cell][port];
                    let mut next = c.map(|v| v as isize);
                    next[axis] += sign;
                    let crossing =
                        paired && axis == 0 && ((c[0] < distance) != (next[0] < distance as isize));
                    let cut =
                        crossing && !window.is_some_and(|[start, end]| start <= c[2] && c[2] < end);
                    let inside = next[0] >= 0
                        && next[0] < nx as isize
                        && next[1] >= 0
                        && next[1] < distance as isize
                        && next[2] >= 0
                        && next[2] < rounds as isize;
                    if inside && !cut {
                        if sign > 0 {
                            program.fuse(
                                photon,
                                rings[index(next.map(|v| v as usize))][positive[axis] + 3],
                                analyzer,
                            )?;
                        }
                    } else if axis == 2 {
                        ends[usize::from(sign > 0)][usize::from(paired && c[0] >= distance)]
                            .push(photon);
                    } else {
                        let basis = wall_basis(c, axis, sign > 0, port);
                        boundary.push(PauliProduct::new(vec![(photon.0, basis)])?);
                    }
                }
            }
        }
        for end in &mut ends {
            for block in end {
                block.sort_by_key(|q| q.0);
            }
        }
        let outputs = ends.iter().flatten().flatten().copied().collect::<Vec<_>>();
        let hadamards = outputs
            .iter()
            .filter(|q| {
                let c = centers[q.0 / 6];
                (c[0] + c[1]) % 2 == 1
            })
            .map(|q| q.0)
            .collect();
        let patch = SixRingPatch::from_parts(
            [nx, distance, rounds],
            program,
            boundary,
            outputs,
            max_photons,
        )?;
        Ok(Self {
            inner: SixRingBoundaryNetwork {
                patch,
                width: distance,
                depth: rounds,
                window,
                paired,
                centers,
                ends,
            },
            hadamards,
        })
    }
    /// Actual six-ring sources, destructive fusions, terminal measurements and retained outputs.
    pub fn patch(&self) -> &SixRingPatch {
        self.inner.patch()
    }
    /// Source coordinates in rotated integer x,y,z cell order; photon index is6*cell+port.
    pub fn resource_centers(&self) -> &[[usize; 3]] {
        self.inner.resource_centers()
    }
    /// Geometric patch side, subject to independent distance verification.
    pub fn distance_label(&self) -> usize {
        self.inner.width()
    }
    /// Number of source layers along time.
    pub fn rounds(&self) -> usize {
        self.inner.depth()
    }
    /// Enabled seam layer interval [start,end), orNone for a memory or separated pair.
    pub fn merge_window(&self) -> Option<[usize; 2]> {
        self.inner.merge_window()
    }
    /// Number of spatial end-code blocks, one or two.
    pub fn blocks(&self) -> usize {
        self.inner.blocks()
    }
    /// Actual photons in a specific time-end/block partition.
    pub fn end(&self, time_end: usize, block: usize) -> Result<&[PhotonId]> {
        self.inner.end(time_end, block)
    }
    /// Declared virtual local-H frame defining CSS coordinates at the outputs.
    pub fn hadamard_frame(&self) -> &[usize] {
        &self.hadamards
    }
    /// Complete signed ideal relation algebra in physical Pauli axes.
    pub fn ideal_relations(&self, max_entries: usize) -> Result<MeasurementRelations> {
        self.inner.ideal_relations(max_entries)
    }
    /// Derive complete signed channel, framed CSS checks and exact bounded physical support distances.
    pub fn certify(&self, limits: BoundaryCertificateLimits) -> Result<BoundaryCertificate> {
        self.inner.certify_in_frame(limits, &self.hadamards)
    }
}
fn wall_basis(c: [usize; 3], axis: usize, positive_face: bool, port: usize) -> Pauli {
    let mut corner = [0; 3];
    corner[axis] = usize::from(positive_face);
    let vertices =
        std::array::from_fn::<_, 3, _>(|j| [1, 2, 0][j] + if corner[j] == 0 { 3 } else { 0 });
    let factor = if vertices.iter().all(|q| q % 2 == vertices[0] % 2) {
        Pauli::X
    } else {
        let center = vertices
            .iter()
            .copied()
            .find(|q| vertices.contains(&((q + 1) % 6)) && vertices.contains(&((q + 5) % 6)))
            .expect("three ring corner ports are consecutive or alternating");
        if port == center { Pauli::X } else { Pauli::Z }
    };
    let keep = usize::from(axis == 0);
    if (c.iter().sum::<usize>() + corner.iter().sum::<usize>()) % 2 == keep {
        factor
    } else {
        match factor {
            Pauli::X => Pauli::Z,
            _ => Pauli::X,
        }
    }
}
