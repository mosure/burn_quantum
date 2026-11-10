//! Explicit mixed-basis staircase boundaries of the rotated six-ring network.
use super::super::SixRingPatch;
use crate::photonic::stabilizer::{Pauli, PauliProduct};
use crate::photonic::{FusionProgram, NodeId, PhotonId, ResourceState};
use crate::{Error, Result};
use std::collections::BTreeMap;
/// A derived finite six-ring memory or time-dependent seam network.
///
/// Source coordinates u,v,w are twice the unrotated BCC coordinates: all three
/// have the same parity. Rotated nearest-neighbor displacements are (1,1,-1),
/// (1,-1,1),(-1,1,1). Both u side walls measure X, both v walls Z. This validated
/// mixed-basis boundary is distinct from the publication's all-Z boundary drawing.
#[derive(Clone, Debug)]
pub struct SixRingBoundaryNetwork {
    pub(super) patch: SixRingPatch,
    pub(super) width: usize,
    pub(super) depth: usize,
    pub(super) window: Option<[usize; 2]>,
    pub(super) paired: bool,
    pub(super) centers: Vec<[usize; 3]>,
    pub(super) ends: [[Vec<PhotonId>; 2]; 2],
}
impl SixRingBoundaryNetwork {
    /// One encoded memory block, retaining every time-end port not removed by a side wall.
    pub fn memory(
        width: usize,
        depth: usize,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        Self::build(width, depth, false, None, source, analyzer, max_photons)
    }
    /// Two side-by-side memories, with the shared interface measured in X throughout.
    pub fn separated(
        width: usize,
        depth: usize,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        Self::build(width, depth, true, None, source, analyzer, max_photons)
    }
    /// Replace interface X measurements by actual XX/ZZ fusions during the window.
    /// Start must be odd, end even, and both strictly inside the time extent.
    /// These geometric conditions are not a substitute for the signed-channel certificate.
    pub fn parity_merge(
        width: usize,
        depth: usize,
        window: [usize; 2],
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        if window[0] == 0
            || window[0] >= window[1]
            || window[1] >= depth
            || window[0].is_multiple_of(2)
            || !window[1].is_multiple_of(2)
        {
            return Err(Error::InvalidValue(
                "six-ring seam requires internal odd start / even end".into(),
            ));
        }
        Self::build(
            width,
            depth,
            true,
            Some(window),
            source,
            analyzer,
            max_photons,
        )
    }
    fn build(
        width: usize,
        depth: usize,
        paired: bool,
        window: Option<[usize; 2]>,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        if !(3..=31).contains(&width) || width.is_multiple_of(2) || !(1..=63).contains(&depth) {
            return Err(Error::InvalidDimensions(
                "six-ring width must be odd 3..=31 and depth 1..=63".into(),
            ));
        }
        let umax = if paired { 2 * width + 1 } else { width };
        let count = (0..=umax)
            .map(|u| {
                (0..=width)
                    .map(|v| {
                        (0..=depth)
                            .filter(|&w| u % 2 == v % 2 && v % 2 == w % 2)
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum::<usize>();
        let required = count.checked_mul(6).ok_or(Error::Overflow)?;
        let limit = max_photons.min(4096);
        if required > limit {
            return Err(Error::ResourceLimit {
                resource: "boundary six-ring photons",
                required,
                limit,
            });
        }
        let mut centers = Vec::with_capacity(count);
        for u in 0..=umax {
            for v in 0..=width {
                for w in 0..=depth {
                    if u % 2 == v % 2 && v % 2 == w % 2 {
                        centers.push([u, v, w]);
                    }
                }
            }
        }
        let ids: BTreeMap<_, _> = centers.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut program = FusionProgram::new();
        let rings = centers
            .iter()
            .map(|_| program.add_resource(ResourceState::six_ring(), source))
            .collect::<Result<Vec<_>>>()?;
        let displacements = [[1isize, 1, -1], [1, -1, 1], [-1, 1, 1]];
        let positive = [1, 2, 0];
        let mut boundary = vec![];
        let mut ends: [[Vec<PhotonId>; 2]; 2] =
            std::array::from_fn(|_| std::array::from_fn(|_| vec![]));
        for (cell, &c) in centers.iter().enumerate() {
            for (axis, delta) in displacements.iter().enumerate() {
                for sign in [1isize, -1] {
                    let port = positive[axis] + if sign < 0 { 3 } else { 0 };
                    let photon = rings[cell][port];
                    let next = std::array::from_fn::<_, 3, _>(|j| c[j] as isize + sign * delta[j]);
                    let across_seam = paired && ((c[0] <= width) != (next[0] <= width as isize));
                    let merge_active = window.is_some_and(|[start, end]| {
                        let twice_mid = c[2] as isize + next[2];
                        2 * start as isize <= twice_mid && twice_mid <= 2 * end as isize
                    });
                    if across_seam && !merge_active {
                        boundary.push(PauliProduct::new(vec![(photon.0, Pauli::X)])?);
                        continue;
                    }
                    let neighbor = if next.iter().all(|&x| x >= 0) {
                        ids.get(&next.map(|x| x as usize))
                    } else {
                        None
                    };
                    if let Some(&other) = neighbor {
                        if sign > 0 {
                            program.fuse(photon, rings[other][positive[axis] + 3], analyzer)?;
                        }
                    } else if next[0] < 0 || next[0] > umax as isize {
                        boundary.push(PauliProduct::new(vec![(photon.0, Pauli::X)])?);
                    } else if next[1] < 0 || next[1] > width as isize {
                        boundary.push(PauliProduct::new(vec![(photon.0, Pauli::Z)])?);
                    } else {
                        let end = usize::from(next[2] > depth as isize);
                        let block = usize::from(paired && c[0] > width);
                        ends[end][block].push(photon);
                    }
                }
            }
        }
        for end in &mut ends {
            for group in end {
                group.sort_by_key(|q| q.0);
            }
        }
        let outputs = ends.iter().flatten().flatten().copied().collect();
        let patch = SixRingPatch::from_parts(
            [umax + 1, width + 1, depth + 1],
            program,
            boundary,
            outputs,
            max_photons,
        )?;
        Ok(Self {
            patch,
            width,
            depth,
            window,
            paired,
            centers,
            ends,
        })
    }
    /// Underlying actual fusion program and complete terminal-boundary executor.
    pub fn patch(&self) -> &SixRingPatch {
        &self.patch
    }
    /// Odd spatial width in doubled BCC source coordinates, not an asserted code distance.
    pub fn width(&self) -> usize {
        self.width
    }
    /// Maximum retained source time coordinate, starting at zero.
    pub fn depth(&self) -> usize {
        self.depth
    }
    /// Active interface-fusion window, or None for memory / separated patches.
    pub fn merge_window(&self) -> Option<[usize; 2]> {
        self.window
    }
    /// Number of separate spatial patches at each time end.
    pub fn blocks(&self) -> usize {
        if self.paired { 2 } else { 1 }
    }
    /// Ordered actual source positions; photon index is 6*source+ring_port.
    pub fn resource_centers(&self) -> &[[usize; 3]] {
        &self.centers
    }
    /// Photon identities in one time-end code block, excluding all terminal measurements.
    pub fn end(&self, time_end: usize, block: usize) -> Result<&[PhotonId]> {
        if time_end > 1 || block >= self.blocks() {
            return Err(Error::Basis);
        }
        Ok(&self.ends[time_end][block])
    }
}
