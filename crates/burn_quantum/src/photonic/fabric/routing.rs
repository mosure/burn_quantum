use super::{Fabric, NodeId};
use crate::photonic::RailTransmission;
use crate::{Error, Result};
use serde::Serialize;

/// A scheduled traversal of one directed link.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RouteHop {
    /// Link ID in the fabric.
    pub link: usize,
    /// Inclusive departure tick.
    pub departure: u64,
    /// Exclusive arrival tick.
    pub arrival: u64,
}
/// Earliest-arrival route with all waiting and traversal attenuation accounted for.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Route {
    /// Starting node.
    pub from: NodeId,
    /// Destination node.
    pub to: NodeId,
    /// Packet release tick.
    pub release: u64,
    /// Arrival tick, before any synchronization wait at the destination.
    pub arrival: u64,
    /// Ordered link traversals.
    pub hops: Vec<RouteHop>,
    /// Product of rail transmissions along traversals and intermediate waits.
    pub transmission: RailTransmission,
}
#[derive(Clone, Debug)]
pub(crate) struct Reservation {
    link: usize,
    start: u64,
    end: u64,
}
impl Fabric {
    /// Route one packet on an otherwise unoccupied fabric before a bounded horizon.
    /// Earliest arrival is the objective; this is not maximum-transmission routing.
    pub fn route(&self, from: NodeId, to: NodeId, release: u64, horizon: u64) -> Result<Route> {
        self.route_reserved(from, to, release, horizon, &mut vec![])
    }
    pub(crate) fn route_reserved(
        &self,
        from: NodeId,
        to: NodeId,
        release: u64,
        horizon: u64,
        reservations: &mut Vec<Reservation>,
    ) -> Result<Route> {
        self.node(from)?;
        self.node(to)?;
        if release > horizon {
            return Err(Error::InvalidValue("route release exceeds horizon".into()));
        }
        let n = self.nodes().len();
        let mut distance = vec![None; n];
        let mut visited = vec![false; n];
        let mut previous: Vec<Option<(usize, RouteHop)>> = vec![None; n];
        distance[from.0] = Some(release);
        loop {
            let current = (0..n)
                .filter(|&i| !visited[i] && distance[i].is_some())
                .min_by_key(|&i| (distance[i].unwrap(), i));
            let Some(current) = current else {
                return Err(Error::Unsupported(
                    "no fabric route within the scheduling horizon".into(),
                ));
            };
            if current == to.0 {
                break;
            }
            visited[current] = true;
            for (id, link) in self
                .links()
                .iter()
                .enumerate()
                .filter(|(_, l)| l.from.0 == current)
            {
                let mut departure = distance[current].unwrap();
                let arrival = loop {
                    let Some(end) = departure
                        .checked_add(link.ticks)
                        .filter(|&end| end <= horizon)
                    else {
                        break None;
                    };
                    if let Some(blocked_until) = reservations
                        .iter()
                        .filter(|r| r.link == id && r.start < end && departure < r.end)
                        .map(|r| r.end)
                        .max()
                    {
                        departure = blocked_until;
                    } else {
                        break Some(end);
                    }
                };
                if let Some(arrival) = arrival
                    && distance[link.to.0].is_none_or(|old| arrival < old)
                {
                    distance[link.to.0] = Some(arrival);
                    previous[link.to.0] = Some((
                        current,
                        RouteHop {
                            link: id,
                            departure,
                            arrival,
                        },
                    ));
                }
            }
        }
        let mut hops = vec![];
        let mut current = to.0;
        while current != from.0 {
            let (parent, hop) = previous[current]
                .clone()
                .ok_or_else(|| Error::InvalidValue("broken route predecessor".into()))?;
            hops.push(hop);
            current = parent;
        }
        hops.reverse();
        let mut transmission = RailTransmission::IDEAL;
        let mut tick = release;
        for hop in &hops {
            let link = &self.links()[hop.link];
            transmission = transmission
                .then(
                    self.node(link.from)?
                        .storage
                        .over_ticks(hop.departure - tick),
                )
                .then(link.transmission);
            tick = hop.arrival;
        }
        // Commit reservations only after the complete route succeeds.
        reservations.extend(hops.iter().map(|h| Reservation {
            link: h.link,
            start: h.departure,
            end: h.arrival,
        }));
        Ok(Route {
            from,
            to,
            release,
            arrival: distance[to.0].unwrap(),
            hops,
            transmission,
        })
    }
}
