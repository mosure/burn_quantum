//! Exact unweighted logical distance in a binary incidence graph.
use super::DecoderGraph;
use crate::{Error, Result};
use serde::Serialize;
use std::collections::VecDeque;

/// Minimum undetected binary fault support that flips one selected logical bit.
/// This certifies only the supplied graph and its independent fault coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LogicalDistance {
    /// Number of distinct fault coordinates in the minimum support.
    pub distance: usize,
    /// Sorted fault-coordinate indices: their syndrome is zero and selected logical bit is one.
    pub edges: Vec<usize>,
    /// Setup edge inspections plus directed adjacency inspections across all BFS roots.
    pub edge_visits: usize,
}

impl DecoderGraph {
    /// Find an exact minimum undetected support flipping `logical`.
    ///
    /// Breadth-first search in the parity double cover finds the shortest odd
    /// closed walk. Relative boundary-to-boundary paths are cycles through the
    /// virtual boundary. Self-loops and parallel fault coordinates are retained.
    /// Every odd cycle touches a logical-labelled edge, so only its endpoints
    /// need be search roots. An arbitrary undetected support decomposes into
    /// cycles, one of which must be odd; this proves minimum support weight.
    ///
    /// The budget counts two setup inspections per original edge, followed by
    /// each directed adjacency inspection. Exhaustion returns no partial certificate.
    /// Vertex IDs are compressed to incident vertices, keeping memory O(edges)
    /// even when the graph declares many isolated detectors. Invalid logical
    /// indices and graphs with no such support return an explicit error.
    pub fn minimum_logical_distance(
        &self,
        logical: usize,
        max_edge_visits: usize,
    ) -> Result<LogicalDistance> {
        if logical >= self.logicals() {
            return Err(Error::InvalidValue(
                "logical distance bit outside graph".into(),
            ));
        }
        let no_witness =
            || Error::InvalidValue("graph has no undetected support for logical bit".into());
        let edges = self.edges();
        if edges.is_empty() {
            return Err(no_witness());
        }
        let setup_visits = edges.len().checked_mul(2).ok_or(Error::Overflow)?;
        if setup_visits > max_edge_visits {
            return Err(Error::ResourceLimit {
                resource: "logical distance edge visits",
                required: setup_visits,
                limit: max_edge_visits,
            });
        }
        let mut vertices = Vec::with_capacity(setup_visits);
        for edge in edges {
            vertices.extend([edge.a, edge.b]);
        }
        vertices.sort_unstable();
        vertices.dedup();
        let lifted = vertices.len().checked_mul(2).ok_or(Error::Overflow)?;
        let mut adjacent = vec![vec![]; vertices.len()];
        let mut starts = vec![false; vertices.len()];
        for (i, edge) in edges.iter().enumerate() {
            let a = vertices.binary_search(&edge.a).expect("collected endpoint");
            let b = vertices.binary_search(&edge.b).expect("collected endpoint");
            let flip = (edge.logical >> logical) & 1 != 0;
            adjacent[a].push((b, flip, i));
            if a != b {
                adjacent[b].push((a, flip, i));
            }
            if flip {
                starts[a] = true;
                starts[b] = true;
            }
        }
        let mut work = setup_visits;
        let mut best: Option<Vec<usize>> = None;
        for (start, &active) in starts.iter().enumerate() {
            if !active {
                continue;
            }
            if best.as_ref().is_some_and(|p| p.len() == 1) {
                break;
            }
            let root = 2 * start;
            let target = root + 1;
            let mut parent = vec![None; lifted];
            let mut depth = vec![0usize; lifted];
            parent[root] = Some((root, usize::MAX));
            let mut queue = VecDeque::from([root]);
            while let Some(node) = queue.pop_front() {
                if node == target {
                    break;
                }
                if best.as_ref().is_some_and(|p| depth[node] + 1 >= p.len()) {
                    continue;
                }
                for &(next, flip, edge) in &adjacent[node / 2] {
                    work = work.checked_add(1).ok_or(Error::Overflow)?;
                    if work > max_edge_visits {
                        return Err(Error::ResourceLimit {
                            resource: "logical distance edge visits",
                            required: work,
                            limit: max_edge_visits,
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
                let mut walk = vec![];
                let mut node = target;
                while node != root {
                    let (previous, edge) = parent[node].expect("connected BFS path");
                    walk.push(edge);
                    node = previous;
                }
                // A nonminimal rooted walk may contain a doubled stem. Binary
                // fault coordinates cancel in pairs; keep its actual support.
                walk.sort_unstable();
                let support = walk
                    .chunk_by(|a, b| a == b)
                    .filter_map(|group| (group.len() % 2 == 1).then_some(group[0]))
                    .collect::<Vec<_>>();
                if best.as_ref().is_none_or(|p| support.len() < p.len()) {
                    best = Some(support);
                }
            }
        }
        let support = best.ok_or_else(no_witness)?;
        Ok(LogicalDistance {
            distance: support.len(),
            edges: support,
            edge_visits: work,
        })
    }
}
