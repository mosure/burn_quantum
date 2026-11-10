//! Directed dual-rail transport graph with discrete-time exclusive link reservations.
mod packets;
mod rails;
mod routing;
use super::RailTransmission;
use crate::{Error, Result};
pub use packets::{FabricConstraints, PacketRequest, PacketScheduler, RoutePolicy};
pub use rails::{RailPacketScheduler, RailRoute, WavelengthConstraints};
pub(crate) use routing::Reservation;
pub use routing::{Route, RouteHop};
use serde::{Deserialize, Serialize};

/// Index of a node in a validated fabric. Bounds are checked when routing/compiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub usize);
/// Abstract node role, not a component-level optical device specification.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    /// Port supplying a complete ideal resource state at its declared birth tick.
    Source,
    /// Passive switching/transport junction.
    Switch,
    /// Standard two-input destructive fusion analyzer.
    Fusion,
}
/// Node with independent attenuation per stored tick. Legacy routing has unbounded
/// buffering; explicit [`FabricConstraints`] enable finite packet storage/configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FabricNode {
    /// Hardware role.
    pub kind: NodeKind,
    /// Intensity transmission per waiting tick for the two rails.
    pub storage: RailTransmission,
}
/// A directed link carries one synchronized dual-rail packet at a time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FabricLink {
    /// Source node.
    pub from: NodeId,
    /// Destination node.
    pub to: NodeId,
    /// Positive traversal duration. Occupancy is [departure, arrival).
    pub ticks: u64,
    /// Net rail transmissions over the traversal, including propagation/switch loss.
    pub transmission: RailTransmission,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FabricData {
    version: u32,
    nodes: Vec<FabricNode>,
    links: Vec<FabricLink>,
}
/// Validated transport model. It does not model coherent interference between links.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FabricData", into = "FabricData")]
pub struct Fabric(FabricData);
impl Fabric {
    /// Construct a directed fabric; at most 4096 nodes and 16384 links in this planner.
    pub fn new(nodes: Vec<FabricNode>, links: Vec<FabricLink>) -> Result<Self> {
        if nodes.is_empty() || nodes.len() > 4096 || links.len() > 16384 {
            return Err(Error::InvalidValue(
                "fabric exceeds node/link limits or is empty".into(),
            ));
        }
        for link in &links {
            if link.from.0 >= nodes.len()
                || link.to.0 >= nodes.len()
                || link.from == link.to
                || link.ticks == 0
            {
                return Err(Error::InvalidValue(
                    "invalid fabric link endpoints or duration".into(),
                ));
            }
        }
        Ok(Self(FabricData {
            version: 1,
            nodes,
            links,
        }))
    }
    /// Nodes in stable ID order.
    pub fn nodes(&self) -> &[FabricNode] {
        &self.0.nodes
    }
    /// Links in stable ID order, also the deterministic routing tie-break order.
    pub fn links(&self) -> &[FabricLink] {
        &self.0.links
    }
    /// Resolve a node ID, failing on out-of-range indices.
    pub fn node(&self, node: NodeId) -> Result<&FabricNode> {
        self.0
            .nodes
            .get(node.0)
            .ok_or_else(|| Error::InvalidValue("fabric node out of range".into()))
    }
}
impl TryFrom<FabricData> for Fabric {
    type Error = Error;
    fn try_from(v: FabricData) -> Result<Self> {
        if v.version != 1 {
            return Err(Error::Unsupported("unknown fabric schema version".into()));
        }
        Self::new(v.nodes, v.links)
    }
}
impl From<Fabric> for FabricData {
    fn from(v: Fabric) -> Self {
        v.0
    }
}
