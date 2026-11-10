//! Local positive resource-corner detector construction for actual six-ring surfaces.
use super::{DecoderGraph, DetectorEdge, GraphLimits};
use crate::photonic::{
    qec::{MeasurementRelations, ParityRelation, fusion_surgery::SixRingSurfaceNetwork},
    stabilizer::{Pauli, PauliProduct},
};
use crate::{Error, Result};
use std::collections::BTreeMap;
/// Physical raw-record graph with explicitly appended ideal end-code readouts.
/// Each detector is a product of positive ring corner stabilizers. At a disabled
/// interface, incident corners are split into actual fusion-connected components.
/// All closing observables are returned for actual measurement; signs are not zeroed.
#[derive(Clone, Debug)]
pub struct CubicalRecordModel {
    observables: Vec<PauliProduct>,
    raw_records: usize,
    checks: Vec<ParityRelation>,
    endpoints: Vec<[[usize; 3]; 2]>,
    dims: [usize; 3],
    blocks: usize,
    merge_window: Option<[usize; 2]>,
    vertex_positions: Vec<[usize; 3]>,
}
impl CubicalRecordModel {
    /// Construct local detectors with linear storage and O(volume log volume)
    /// ordered-map work. The explicit
    /// budget counts 24 single-photon corner terms per supplied six-ring.
    /// This declares perfect nondestructive closing-check measurements in addition
    /// to the network's physical fusion and terminal single-photon measurements.
    pub fn new(network: &SixRingSurfaceNetwork, max_corner_terms: usize) -> Result<Self> {
        let cells = network.resource_centers();
        let required = cells.len().checked_mul(24).ok_or(Error::Overflow)?;
        if required > max_corner_terms {
            return Err(Error::ResourceLimit {
                resource: "six-ring corner terms",
                required,
                limit: max_corner_terms,
            });
        }
        let program = network.patch().program();
        let mut observables = network.patch().observables()?;
        let raw_records = observables.len();
        let mut partner = vec![None; program.num_photons()];
        for (i, f) in program.fusions().iter().enumerate() {
            let [a, b] = f.photons.map(|q| q.0);
            partner[a] = Some((b, i));
            partner[b] = Some((a, i));
        }
        let mut terminal = vec![None; program.num_photons()];
        for (i, p) in network.patch().boundary_measurements().iter().enumerate() {
            let [(q, pauli)] = p.terms() else {
                return Err(Error::Unsupported(
                    "cubical boundary must be single-photon Pauli".into(),
                ));
            };
            terminal[*q] = Some((*pauli, 2 * program.num_fusions() + i));
        }
        let mut vertices: BTreeMap<[usize; 3], Vec<Vec<(usize, Pauli)>>> = BTreeMap::new();
        for (cell, &c) in cells.iter().enumerate() {
            for bits in 0..8 {
                let corner = [bits & 1, (bits >> 1) & 1, (bits >> 2) & 1];
                let selected = std::array::from_fn::<_, 3, _>(|a| {
                    [1, 2, 0][a] + if corner[a] == 0 { 3 } else { 0 }
                });
                let terms = selected
                    .iter()
                    .map(|&p| (6 * cell + p, corner_factor(selected, p)))
                    .collect();
                let vertex = std::array::from_fn(|a| c[a] + corner[a]);
                vertices.entry(vertex).or_default().push(terms);
            }
        }
        let mut checks = Vec::new();
        let mut vertex_positions = Vec::new();
        for (position, corners) in vertices {
            let mut parent: Vec<_> = (0..corners.len()).collect();
            for (i, corner) in corners.iter().enumerate() {
                for &(q, p) in corner {
                    if let Some((other, _)) = partner[q]
                        && let Some(j) = corners.iter().position(|c| c.contains(&(other, p)))
                    {
                        let a = find(&parent, i);
                        let b = find(&parent, j);
                        parent[a] = b;
                    }
                }
            }
            let mut components: BTreeMap<usize, BTreeMap<usize, Pauli>> = BTreeMap::new();
            for (i, corner) in corners.into_iter().enumerate() {
                components
                    .entry(find(&parent, i))
                    .or_default()
                    .extend(corner);
            }
            for terms in components.into_values() {
                let mut records = Vec::new();
                let mut output = Vec::new();
                let mut valid = true;
                for (&q, &p) in &terms {
                    if let Some((other, fusion)) = partner[q] {
                        if terms.get(&other) != Some(&p) {
                            valid = false;
                            break;
                        }
                        if q < other {
                            records.push(2 * fusion + usize::from(p == Pauli::Z));
                        }
                    } else if let Some((measured, record)) = terminal[q] {
                        if measured != p {
                            valid = false;
                            break;
                        }
                        records.push(record);
                    } else {
                        output.push((q, p));
                    }
                }
                if !valid {
                    continue;
                }
                if !output.is_empty() {
                    records.push(observables.len());
                    observables.push(PauliProduct::new(output)?);
                }
                records.sort_unstable();
                checks.push(ParityRelation {
                    terms: records,
                    negative: false,
                });
                vertex_positions.push(position);
            }
        }
        let endpoints = observables[..raw_records]
            .iter()
            .map(|p| {
                let (q, basis) = p.terms()[0];
                let port = q % 6;
                let axis = match port % 3 {
                    1 => 0,
                    2 => 1,
                    _ => 2,
                };
                let tangents: Vec<_> = (0..3).filter(|&a| a != axis).collect();
                let mut ends = Vec::new();
                for a in 0..2 {
                    for b in 0..2 {
                        let mut corner = [0; 3];
                        corner[axis] = usize::from(port < 3);
                        corner[tangents[0]] = a;
                        corner[tangents[1]] = b;
                        let selected = std::array::from_fn::<_, 3, _>(|j| {
                            [1, 2, 0][j] + if corner[j] == 0 { 3 } else { 0 }
                        });
                        if corner_factor(selected, port) == basis {
                            ends.push(std::array::from_fn(|j| cells[q / 6][j] + corner[j]));
                        }
                    }
                }
                <Vec<[usize; 3]> as TryInto<[[usize; 3]; 2]>>::try_into(ends).map_err(|_| {
                    Error::InvalidValue(
                        "record does not have two geometric corner endpoints".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let dims = [
            network.distance_label() * network.blocks(),
            network.distance_label(),
            network.rounds(),
        ];
        let model = Self {
            observables,
            raw_records,
            checks,
            endpoints,
            dims,
            blocks: network.blocks(),
            merge_window: network.merge_window(),
            vertex_positions,
        };
        // Independent of labels, prove graphlikeness rather than discarding a
        // detector incidence. All raw records, including terminal bits, remain.
        model.graph_with_masks(&vec![0; raw_records], 0, GraphLimits::default())?;
        Ok(model)
    }
    /// Raw fusions/terminal observables followed by explicit closing observables.
    pub fn observables(&self) -> &[PauliProduct] {
        &self.observables
    }
    /// Number of original physical fusion/terminal outcome coordinates.
    pub fn raw_records(&self) -> usize {
        self.raw_records
    }
    /// Ideal nondestructive end-check observables added by this reference model.
    pub fn closure(&self) -> &[PauliProduct] {
        &self.observables[self.raw_records..]
    }
    /// Positive resource-derived detector relations, including observed closure bits.
    pub fn checks(&self) -> &[ParityRelation] {
        &self.checks
    }
    /// Global cube vertex for each detector (repeated for disconnected seam components).
    pub fn vertex_positions(&self) -> &[[usize; 3]] {
        &self.vertex_positions
    }
    /// Actual measured detector parities, never assumed closure eigenvalues.
    pub fn syndrome(&self, records: &[bool]) -> Result<Vec<bool>> {
        if records.len() != self.observables.len() {
            return Err(Error::Shape("cubical record count".into()));
        }
        Ok(self
            .checks
            .iter()
            .map(|c| c.terms.iter().fold(c.negative, |v, &r| v ^ records[r]))
            .collect())
    }
    /// Build exact sparse incidence with externally validated logical labels.
    /// Masks correspond to physical raw-record flips; closing outcomes are ideal.
    pub fn graph_with_masks(
        &self,
        masks: &[u64],
        logicals: usize,
        limits: GraphLimits,
    ) -> Result<DecoderGraph> {
        if masks.len() != self.raw_records {
            return Err(Error::Shape("cubical logical masks".into()));
        }
        let mut incidence = vec![Vec::new(); self.raw_records];
        for (i, c) in self.checks.iter().enumerate() {
            for &r in &c.terms {
                if r < self.raw_records {
                    incidence[r].push(i);
                }
            }
        }
        let boundary = self.checks.len();
        let edges = incidence
            .into_iter()
            .zip(masks)
            .map(|(nodes, &logical)| {
                if nodes.len() > 2 {
                    return Err(Error::Unsupported(
                        "cubical raw record is not graphlike".into(),
                    ));
                }
                Ok(DetectorEdge {
                    a: nodes.first().copied().unwrap_or(boundary),
                    b: nodes.get(1).copied().unwrap_or(boundary),
                    logical,
                })
            })
            .collect::<Result<_>>()?;
        DecoderGraph::new(boundary, logicals, edges, limits)
    }
    /// Derive labels for actual target operators from a bounded independent full
    /// resource relation calculation, requiring the exact same observable order.
    pub fn graph_from_relations(
        &self,
        relations: &MeasurementRelations,
        targets: &[PauliProduct],
        limits: GraphLimits,
    ) -> Result<DecoderGraph> {
        if relations.observables() != self.observables {
            return Err(Error::Shape("cubical relation observable order".into()));
        }
        if targets.len() > 64 {
            return Err(Error::InvalidDimensions("cubical logical count".into()));
        }
        let mut masks = vec![0; self.raw_records];
        for (i, p) in targets.iter().enumerate() {
            let r = relations
                .correlation(p)?
                .ok_or_else(|| Error::InvalidValue("uncorrelated cubical target".into()))?;
            for q in r.terms {
                if q < self.raw_records {
                    masks[q] ^= 1 << i;
                }
            }
        }
        self.graph_with_masks(&masks, targets.len(), limits)
    }
    /// Scalable two-color memory cut labels: color0 terminates on x walls,
    /// color1 on y walls. Only single rectangular memories are accepted.
    /// Small-model GF(2)/quantum tests verify equivalence to full Choi relations.
    pub fn memory_graph(&self, limits: GraphLimits) -> Result<DecoderGraph> {
        if self.blocks != 1 {
            return Err(Error::Unsupported(
                "memory cuts cannot label a surgical seam".into(),
            ));
        }
        let masks: Vec<_> = self
            .endpoints
            .iter()
            .map(|pair| {
                pair.iter().fold(0, |mask, v| {
                    let color = v.iter().sum::<usize>() % 2;
                    mask ^ if color == 0 && v[0] == self.dims[0] {
                        1
                    } else if color == 1 && v[1] == self.dims[1] {
                        2
                    } else {
                        0
                    }
                })
            })
            .collect();
        self.graph_with_masks(&masks, 2, limits)
    }
    /// Four geometric logical-flow cuts for an actual finite merge window.
    /// Three color0 rough components are the outer right x wall and the
    /// pre/post-merge seam; color1 contributes the upper y wall. The outer left
    /// x wall and lower y wall set the two reference components. Only missing
    /// detector endpoints contribute, so active seam vertices are not treated
    /// as boundaries. These four cuts span the full physical Choi-flow labels,
    /// checked independently against complete signed resource relations.
    pub fn surgery_graph(&self, limits: GraphLimits) -> Result<DecoderGraph> {
        let Some([start, end]) = self.merge_window else {
            return Err(Error::Unsupported(
                "surgery cuts require an active merge window".into(),
            ));
        };
        if self.blocks != 2 {
            return Err(Error::Unsupported(
                "surgery cuts require two patches".into(),
            ));
        }
        let mut incidence = vec![Vec::new(); self.raw_records];
        for (check, relation) in self.checks.iter().enumerate() {
            for &r in &relation.terms {
                if r < self.raw_records {
                    incidence[r].push(check);
                }
            }
        }
        let masks: Vec<_> = self
            .endpoints
            .iter()
            .enumerate()
            .map(|(r, pair)| {
                pair.iter().fold(0, |mask, v| {
                    if incidence[r].iter().any(|&c| self.vertex_positions[c] == *v) {
                        return mask;
                    }
                    let color = v.iter().sum::<usize>() % 2;
                    let bit = if color == 1 && v[1] == self.dims[1] {
                        8
                    } else if color == 0 && v[0] == self.dims[0] {
                        1
                    } else if color == 0 && v[0] == self.dims[0] / 2 && v[2] <= start {
                        2
                    } else if color == 0 && v[0] == self.dims[0] / 2 && v[2] >= end {
                        4
                    } else {
                        0
                    };
                    mask ^ bit
                })
            })
            .collect();
        self.graph_with_masks(&masks, 4, limits)
    }
}
fn find(parent: &[usize], mut q: usize) -> usize {
    while parent[q] != q {
        q = parent[q];
    }
    q
}
fn corner_factor(selected: [usize; 3], port: usize) -> Pauli {
    if selected.iter().all(|q| q % 2 == selected[0] % 2) {
        return Pauli::X;
    }
    let center = selected
        .iter()
        .copied()
        .find(|q| selected.contains(&((q + 1) % 6)) && selected.contains(&((q + 5) % 6)))
        .expect("cube corner ring stabilizer");
    if port == center { Pauli::X } else { Pauli::Z }
}
