use super::*;
use crate::photonic::stabilizer::{Pauli, PauliProduct};
use crate::{Error, Result};
use std::collections::VecDeque;
/// Exact shortest nontrivial CSS cycles in the merged graph, including boundaries.
/// Temporal repetition distance concerns independent seam-readout flips with ideal
/// endpoints; it is not a certificate for an unspecified circuit-level noise model.
#[derive(Clone, Debug, Default)]
pub struct SurgeryDistance {
    /// Minimum physical X weight with nontrivial protected logical action.
    pub x_distance: usize,
    /// Minimum physical Z weight with nontrivial protected logical action.
    pub z_distance: usize,
    /// Explicit shortest X witness, indexed in the two-patch data register.
    pub x_witness: Vec<usize>,
    /// Explicit shortest Z witness.
    pub z_witness: Vec<usize>,
    /// Number of repeated parity-check rounds.
    pub temporal_repetitions: usize,
    /// Actual bounded graph edge visits in both double-cover searches.
    pub search_work: usize,
}
impl PlanarSurgery {
    fn certify_distance_channel(
        &self,
        error: Pauli,
        logical: &PauliProduct,
        work: &mut usize,
        budget: usize,
    ) -> Result<Vec<usize>> {
        let check_pauli = if error == Pauli::X {
            Pauli::Z
        } else {
            Pauli::X
        };
        let checks: Vec<_> = self
            .merged
            .iter()
            .filter(|p| p.terms()[0].1 == check_pauli)
            .collect();
        let boundary = checks.len();
        let mut graph = vec![vec![]; boundary + 1];
        for q in 0..2 * self.code.qubits() {
            let endpoints: Vec<_> = checks
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.terms().iter().any(|&(s, _)| s == q).then_some(i))
                .collect();
            let (a, b) = match endpoints.as_slice() {
                [] => (boundary, boundary),
                [a] => (*a, boundary),
                [a, b] => (*a, *b),
                _ => {
                    return Err(Error::Unsupported(
                        "surgery checks are not graphlike".into(),
                    ));
                }
            };
            let bit = logical.terms().iter().any(|&(s, _)| s == q);
            graph[a].push((b, bit, q));
            graph[b].push((a, bit, q));
        }
        let mut best: Option<Vec<usize>> = None;
        for start in 0..graph.len() {
            let root = 2 * start;
            let target = root + 1;
            let mut parent = vec![None; 2 * graph.len()];
            parent[root] = Some((root, usize::MAX));
            let mut queue = VecDeque::from([root]);
            while let Some(node) = queue.pop_front() {
                if node == target {
                    break;
                }
                for &(next, flip, edge) in &graph[node / 2] {
                    *work = work.checked_add(1).ok_or(Error::Overflow)?;
                    if *work > budget {
                        return Err(Error::ResourceLimit {
                            resource: "surgery distance graph work",
                            required: *work,
                            limit: budget,
                        });
                    }
                    let next = 2 * next + ((node % 2) ^ usize::from(flip));
                    if parent[next].is_none() {
                        parent[next] = Some((node, edge));
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
        best.ok_or_else(|| Error::InvalidValue("merged logical has no physical witness".into()))
    }
    pub(super) fn certify_distance(&self, budget: usize) -> Result<SurgeryDistance> {
        let n = self.code.qubits();
        let mut x = self.code.logical_x(0)?;
        let mut z = self.code.logical_z(0)?;
        match self.basis {
            MergeBasis::ZZ => x = x.multiply_commuting(&self.code.logical_x(n)?)?.1,
            MergeBasis::XX => z = z.multiply_commuting(&self.code.logical_z(n)?)?.1,
        };
        let mut work = 0;
        let x_witness = self.certify_distance_channel(Pauli::X, &z, &mut work, budget)?;
        let z_witness = self.certify_distance_channel(Pauli::Z, &x, &mut work, budget)?;
        Ok(SurgeryDistance {
            x_distance: x_witness.len(),
            z_distance: z_witness.len(),
            x_witness,
            z_witness,
            temporal_repetitions: self.code.distance(),
            search_work: work,
        })
    }
}
