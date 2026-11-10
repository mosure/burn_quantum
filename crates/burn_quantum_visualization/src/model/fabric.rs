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
}
impl FabricSnapshot {
    /// Copy an already checked compiled plan; drawing coordinates do not modify physics.
    pub fn from_compiled(plan: &CompiledFusionNetwork) -> Self {
        Self {
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
                    FabricNodeView {
                        id,
                        label: format!("{:?} {id}", node.kind),
                        kind: format!("{:?}", node.kind),
                        position: [x, 1.5 - id as f32, 0.],
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
