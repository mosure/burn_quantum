use burn_quantum::photonic::{CompiledFusionNetwork, NodeKind};
use serde::{Deserialize, Serialize};
/// A physical node and its drawing position. Coordinates are schematic, not optical lengths.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FabricNodeView {
    /// Stable fabric index.
    pub id: usize,
    /// Display label.
    pub label: String,
    /// Source, switch, or analyzer role.
    pub kind: String,
    /// Schematic scene coordinates; imported components must have magnitude at most 1e6.
    pub position: [f32; 3],
    /// Independent per-tick intensity transmissions for the two rails.
    pub storage: [f64; 2],
}
/// One actual directed fabric link.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FabricLinkView {
    /// Stable link index.
    pub id: usize,
    /// Source node.
    pub from: usize,
    /// Destination node.
    pub to: usize,
    /// Traversal duration.
    pub ticks: u64,
    /// Rail intensity transmissions.
    pub transmission: [f64; 2],
}
/// Actual reserved interval on a directed link.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteHopView {
    /// Link index.
    pub link: usize,
    /// Inclusive departure tick.
    pub departure: u64,
    /// Exclusive arrival tick.
    pub arrival: u64,
}
/// One routed photon, without invented path interpolation in the scientific record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhotonRouteView {
    /// Stable photon ID.
    pub photon: usize,
    /// Source node.
    pub from: usize,
    /// Analyzer node.
    pub to: usize,
    /// Source release tick.
    pub release: u64,
    /// Route arrival tick, before final synchronization wait.
    pub arrival: u64,
    /// Actual reserved hops.
    pub hops: Vec<RouteHopView>,
    /// Route transmission from listed links and intervening storage waits. Excludes
    /// pre-release storage and later synchronization/analyzer/detector attenuation.
    pub transmission: [f64; 2],
}
/// Scheduled destructive analyzer operation; no successful outcome is assumed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusionEvent {
    /// Scheduled tick.
    pub tick: u64,
    /// Analyzer node index.
    pub node: usize,
    /// Consumed photon IDs.
    pub photons: [usize; 2],
}
/// A supplied small resource, retaining stable photon ownership and its graph recipe.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceView {
    /// Fabric source node.
    pub source: usize,
    /// Stable photon IDs in local resource order.
    pub photons: Vec<usize>,
    /// Local graph edges before local Hadamards; these are entanglement, not optical links.
    pub edges: Vec<(usize, usize)>,
    /// Local graph vertices with a Hadamard after graph-state preparation.
    pub hadamards: Vec<usize>,
}
/// Exact compiled physical plan. This is scheduling evidence, not a fusion outcome trajectory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FabricSnapshot {
    /// Nodes in stable index order.
    pub nodes: Vec<FabricNodeView>,
    /// Directed links in stable index order.
    pub links: Vec<FabricLinkView>,
    /// Actual routed input photons.
    pub routes: Vec<PhotonRouteView>,
    /// Actual analyzer schedule.
    pub fusions: Vec<FusionEvent>,
    /// Unconsumed output IDs; vacuum remains possible under loss.
    pub outputs: Vec<usize>,
    /// Schedule completion tick.
    pub finish_tick: u64,
    /// Number of ideally supplied small resource states.
    pub resource_count: usize,
    /// Number of supplied photons, including consumed photons.
    pub photon_count: usize,
    /// Resource ownership/recipes. Older schema-one imports may omit this information.
    #[serde(default)]
    pub resources: Vec<ResourceView>,
}
impl FabricSnapshot {
    /// Copy an already checked compiled plan; drawing coordinates do not modify physics.
    pub fn from_compiled(plan: &CompiledFusionNetwork) -> Self {
        let mut offset = 0;
        let resources = plan
            .program()
            .resources()
            .iter()
            .map(|r| {
                let photons = (offset..offset + r.state.qubits()).collect();
                offset += r.state.qubits();
                ResourceView {
                    source: r.source.0,
                    photons,
                    edges: r.state.edges().to_vec(),
                    hadamards: r.state.hadamards().to_vec(),
                }
            })
            .collect();
        Self {
            resources,
            nodes: plan
                .fabric()
                .nodes()
                .iter()
                .enumerate()
                .map(|(id, node)| {
                    let x = match node.kind {
                        NodeKind::Source => -3.,
                        NodeKind::Fusion => 3.,
                        _ => 0.,
                    };
                    let same_role = plan
                        .fabric()
                        .nodes()
                        .iter()
                        .filter(|n| n.kind == node.kind)
                        .count();
                    let rank = plan.fabric().nodes()[..id]
                        .iter()
                        .filter(|n| n.kind == node.kind)
                        .count();
                    FabricNodeView {
                        id,
                        label: format!("{:?} {id}", node.kind),
                        kind: format!("{:?}", node.kind),
                        position: [x, (rank as f32 - (same_role - 1) as f32 * 0.5) * 2.2, 0.],
                        storage: node.storage.values(),
                    }
                })
                .collect(),
            links: plan
                .fabric()
                .links()
                .iter()
                .enumerate()
                .map(|(id, link)| FabricLinkView {
                    id,
                    from: link.from.0,
                    to: link.to.0,
                    ticks: link.ticks,
                    transmission: link.transmission.values(),
                })
                .collect(),
            routes: plan
                .routes()
                .iter()
                .map(|p| PhotonRouteView {
                    photon: p.photon.0,
                    from: p.route.from.0,
                    to: p.route.to.0,
                    release: p.route.release,
                    arrival: p.route.arrival,
                    transmission: p.route.transmission.values(),
                    hops: p
                        .route
                        .hops
                        .iter()
                        .map(|h| RouteHopView {
                            link: h.link,
                            departure: h.departure,
                            arrival: h.arrival,
                        })
                        .collect(),
                })
                .collect(),
            fusions: plan
                .fusions()
                .iter()
                .map(|f| FusionEvent {
                    tick: f.tick,
                    node: f.at.0,
                    photons: f.photons.map(|q| q.0),
                })
                .collect(),
            outputs: plan.outputs().iter().map(|q| q.0).collect(),
            finish_tick: plan.finish_tick(),
            resource_count: plan.program().resources().len(),
            photon_count: plan.program().num_photons(),
        }
    }
}

/// Scheduled photon location. This describes occupancy reservations, not a sampled survival event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhotonLocation {
    /// At a source, buffer or analyzer, including waiting between hops.
    Node(usize),
    /// On an actual link, with fraction of the reserved travel interval elapsed.
    Link {
        /// Link ID.
        link: usize,
        /// Fraction in [0,1).
        progress: f32,
    },
    /// The destructive measurement interval has finished.
    Consumed,
}
impl FabricSnapshot {
    /// Locate a scheduled photon, including source storage, routing waits and final synchronization.
    /// None means an older import did not retain source ownership for an unrouted output.
    pub fn photon_location(&self, photon: usize, tick: f64) -> Option<PhotonLocation> {
        if photon >= self.photon_count || !tick.is_finite() || tick < 0. {
            return None;
        }
        if self
            .fusions
            .iter()
            .any(|f| f.photons.contains(&photon) && tick >= f.tick as f64 + 1.)
        {
            return Some(PhotonLocation::Consumed);
        }
        if let Some(route) = self.routes.iter().find(|r| r.photon == photon) {
            let mut node = route.from;
            for hop in &route.hops {
                if tick < hop.departure as f64 {
                    return Some(PhotonLocation::Node(node));
                }
                let link = self.links.get(hop.link)?;
                if tick < hop.arrival as f64 {
                    return Some(PhotonLocation::Link {
                        link: hop.link,
                        progress: ((tick - hop.departure as f64)
                            / (hop.arrival - hop.departure) as f64)
                            as f32,
                    });
                }
                node = link.to;
            }
            Some(PhotonLocation::Node(node))
        } else {
            self.resources
                .iter()
                .find(|r| r.photons.contains(&photon))
                .map(|r| PhotonLocation::Node(r.source))
        }
    }
}
