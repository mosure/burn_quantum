//! Bounded space-time routing with physical link and buffer occupancy.
use super::{Fabric, NodeId, Route, RouteHop};
use crate::photonic::RailTransmission;
use crate::{Error, Result};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Explicit objective within a fixed consumption deadline and finite capacities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RoutePolicy {
    /// Earliest feasible arrival, with stable node/link tie breaking.
    EarliestArrival,
    /// Maximize geometric mean rail transmission, including all waiting before consumption.
    /// This is an intensity-survival objective, not a state-dependent quantum fidelity estimate.
    MaximumSurvival,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Score(f64);
impl Eq for Score {}
impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
fn attenuation_cost(t: RailTransmission) -> f64 {
    let [a, b] = t.values();
    -(a.ln() + b.ln()) / 2.
}

/// Finite buffer/switch assumptions, separate from optical device noise.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FabricConstraints {
    /// Packet slots per node, including sources and analyzer input storage.
    pub buffers: Vec<usize>,
    /// Mandatory occupied ticks after entering each node, including source configuration.
    pub configure_ticks: Vec<u64>,
    /// Maximum visited space-time states per route. Bounds planning work independently of horizon.
    pub max_search_states: usize,
    /// Maximum retained occupancy intervals across all packets.
    pub max_reservations: usize,
}
impl FabricConstraints {
    fn validate(&self, fabric: &Fabric) -> Result<()> {
        if self.buffers.len() != fabric.nodes().len()
            || self.configure_ticks.len() != fabric.nodes().len()
            || self.max_search_states == 0
            || self.max_reservations == 0
        {
            return Err(Error::InvalidValue(
                "invalid finite fabric constraints".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
struct Booking {
    resource: usize,
    start: u64,
    end: u64,
}
/// One packet lifetime constraint; consumption occupies one tick at its destination.
#[derive(Clone, Copy, Debug)]
pub struct PacketRequest {
    /// Source port.
    pub from: NodeId,
    /// Destructive measurement destination.
    pub to: NodeId,
    /// Birth time, including source-buffer occupancy before release.
    pub birth: u64,
    /// Earliest departure time; source configuration may complete before release.
    pub release: u64,
    /// Exact measurement start; late arrival is rejected.
    pub consume: u64,
}
/// Incremental transactionally checked packet schedule on one immutable fabric.
#[derive(Clone, Debug)]
pub struct PacketScheduler {
    fabric: Fabric,
    constraints: FabricConstraints,
    bookings: Vec<Booking>,
}
impl PacketScheduler {
    /// Instantiate an empty finite schedule; a zero-slot node can be traversed without waiting.
    pub fn new(fabric: Fabric, constraints: FabricConstraints) -> Result<Self> {
        constraints.validate(&fabric)?;
        Ok(Self {
            fabric,
            constraints,
            bookings: vec![],
        })
    }
    /// Number of retained nonempty occupancy intervals.
    pub fn reservation_count(&self) -> usize {
        self.bookings.len()
    }
    fn capacity(&self, resource: usize) -> usize {
        if resource < self.fabric.links().len() {
            1
        } else {
            self.constraints.buffers[resource - self.fabric.links().len()]
        }
    }
    fn free(&self, resource: usize, start: u64, end: u64) -> bool {
        if start == end {
            return true;
        }
        let capacity = self.capacity(resource);
        if capacity == 0 {
            return false;
        }
        let overlaps: Vec<_> = self
            .bookings
            .iter()
            .filter(|b| b.resource == resource && b.start < end && start < b.end)
            .collect();
        let mut points = vec![start];
        points.extend(overlaps.iter().filter(|b| b.start > start).map(|b| b.start));
        points.into_iter().all(|t| {
            overlaps
                .iter()
                .filter(|b| b.start <= t && t < b.end)
                .count()
                < capacity
        })
    }
    fn buffer(&self, node: usize) -> usize {
        self.fabric.links().len() + node
    }
    fn commit(&mut self, bookings: Vec<Booking>) -> Result<()> {
        let mut candidate = self.clone();
        for b in bookings.into_iter().filter(|b| b.start < b.end) {
            if !candidate.free(b.resource, b.start, b.end) {
                return Err(Error::Unsupported(
                    "physical buffer/link capacity conflict".into(),
                ));
            }
            if candidate.bookings.len() >= candidate.constraints.max_reservations {
                return Err(Error::ResourceLimit {
                    resource: "packet reservations",
                    required: candidate.bookings.len() + 1,
                    limit: candidate.constraints.max_reservations,
                });
            }
            candidate.bookings.push(b);
        }
        self.bookings = candidate.bookings;
        Ok(())
    }
    /// Reserve an unconsumed output's source storage through the final schedule tick.
    pub fn store_output(&mut self, source: NodeId, birth: u64, finish: u64) -> Result<()> {
        self.fabric.node(source)?;
        if finish < birth {
            return Err(Error::InvalidValue("output precedes birth".into()));
        }
        self.commit(vec![Booking {
            resource: self.buffer(source.0),
            start: birth,
            end: finish,
        }])
    }
    /// Route a packet before its fixed consumption time, including finite waiting capacity.
    ///
    /// This time-expanded search allows later arrivals at a node when an earlier arrival
    /// cannot be stored. Failed search or capacity rejection leaves the scheduler unchanged.
    /// Transmission excludes pre-release and final synchronization storage, matching Route.
    pub fn route_at(&mut self, r: PacketRequest) -> Result<Route> {
        self.route_with_policy(r, RoutePolicy::EarliestArrival)
    }
    /// Route using the requested objective; budgets bound space-time search and relaxations.
    pub fn route_with_policy(&mut self, r: PacketRequest, policy: RoutePolicy) -> Result<Route> {
        self.fabric.node(r.from)?;
        self.fabric.node(r.to)?;
        if r.birth > r.release || r.release > r.consume {
            return Err(Error::InvalidValue("noncausal packet lifetime".into()));
        }
        let end = r.consume.checked_add(1).ok_or(Error::Overflow)?;
        let ready = r
            .birth
            .checked_add(self.constraints.configure_ticks[r.from.0])
            .ok_or(Error::Overflow)?
            .max(r.release);
        if ready > r.consume || !self.free(self.buffer(r.from.0), r.birth, ready) {
            return Err(Error::Unsupported(
                "source buffer/configuration unavailable".into(),
            ));
        }
        type State = (u64, usize);
        let initial = (ready, r.from.0);
        let score = match policy {
            RoutePolicy::EarliestArrival => ready as f64,
            RoutePolicy::MaximumSurvival => attenuation_cost(
                self.fabric.nodes()[r.from.0]
                    .storage
                    .over_ticks(ready - r.release),
            ),
        };
        let mut queue = BinaryHeap::from([Reverse((Score(score), initial))]);
        let mut visited = HashSet::new();
        let mut previous: HashMap<State, (State, Option<RouteHop>)> = HashMap::new();
        let mut queued = HashSet::from([initial]);
        let mut scores = HashMap::from([(initial, score)]);
        let mut relaxations = 1usize;
        let found = loop {
            let Some(Reverse((Score(score), state))) = queue.pop() else {
                return Err(Error::Unsupported(
                    "no finite-buffer route before consumption".into(),
                ));
            };
            let (time, node) = state;
            if !visited.insert(state) {
                continue;
            }
            if visited.len() > self.constraints.max_search_states {
                return Err(Error::ResourceLimit {
                    resource: "space-time route states",
                    required: visited.len(),
                    limit: self.constraints.max_search_states,
                });
            }
            if node == r.to.0
                && (policy == RoutePolicy::EarliestArrival || time == r.consume)
                && self.free(self.buffer(node), time, end)
            {
                break state;
            }
            let mut next = vec![];
            if time < r.consume && self.free(self.buffer(node), time, time + 1) {
                next.push((
                    (time + 1, node),
                    None,
                    attenuation_cost(self.fabric.nodes()[node].storage),
                ));
            }
            for (index, link) in self
                .fabric
                .links()
                .iter()
                .enumerate()
                .filter(|(_, l)| l.from.0 == node)
            {
                let Some(arrival) = time.checked_add(link.ticks).filter(|&t| t <= r.consume) else {
                    continue;
                };
                let Some(ready) = arrival
                    .checked_add(self.constraints.configure_ticks[link.to.0])
                    .filter(|&t| t <= r.consume)
                else {
                    continue;
                };
                if self.free(index, time, arrival)
                    && self.free(self.buffer(link.to.0), arrival, ready)
                {
                    next.push((
                        (ready, link.to.0),
                        Some(RouteHop {
                            link: index,
                            departure: time,
                            arrival,
                        }),
                        attenuation_cost(link.transmission)
                            + attenuation_cost(
                                self.fabric.nodes()[link.to.0]
                                    .storage
                                    .over_ticks(ready - arrival),
                            ),
                    ));
                }
            }
            for (state2, hop, cost) in next {
                let next_score = match policy {
                    RoutePolicy::EarliestArrival => state2.0 as f64,
                    RoutePolicy::MaximumSurvival => score + cost,
                };
                if !visited.contains(&state2)
                    && scores.get(&state2).is_none_or(|&old| next_score < old)
                {
                    queued.insert(state2);
                    relaxations += 1;
                    if relaxations > self.constraints.max_search_states {
                        return Err(Error::ResourceLimit {
                            resource: "space-time route states",
                            required: relaxations,
                            limit: self.constraints.max_search_states,
                        });
                    }
                    previous.insert(state2, (state, hop));
                    scores.insert(state2, next_score);
                    queue.push(Reverse((Score(next_score), state2)));
                }
            }
        };
        let mut hops = vec![];
        let mut current = found;
        while current != initial {
            let (parent, hop) = previous[&current].clone();
            if let Some(hop) = hop {
                hops.push(hop);
            }
            current = parent;
        }
        hops.reverse();
        let mut bookings = vec![];
        let mut location = r.from.0;
        let mut arrived = r.birth;
        let mut optical_tick = r.release;
        let mut transmission = RailTransmission::IDEAL;
        for hop in &hops {
            let link = &self.fabric.links()[hop.link];
            bookings.push(Booking {
                resource: self.buffer(location),
                start: arrived,
                end: hop.departure,
            });
            bookings.push(Booking {
                resource: hop.link,
                start: hop.departure,
                end: hop.arrival,
            });
            transmission = transmission
                .then(
                    self.fabric.nodes()[location]
                        .storage
                        .over_ticks(hop.departure - optical_tick),
                )
                .then(link.transmission);
            location = link.to.0;
            arrived = hop.arrival;
            optical_tick = hop.arrival;
        }
        bookings.push(Booking {
            resource: self.buffer(location),
            start: arrived,
            end,
        });
        self.commit(bookings)?;
        Ok(Route {
            from: r.from,
            to: r.to,
            release: r.release,
            arrival: hops.last().map_or(r.release, |h| h.arrival),
            hops,
            transmission,
        })
    }
    /// Atomically schedule both fusion operands at a fixed measurement tick.
    pub fn pair_at(&mut self, requests: [PacketRequest; 2]) -> Result<[Route; 2]> {
        self.pair_with_policy(requests, RoutePolicy::EarliestArrival)
    }
    /// Atomically reserve both operands with one explicit route objective.
    pub fn pair_with_policy(
        &mut self,
        requests: [PacketRequest; 2],
        policy: RoutePolicy,
    ) -> Result<[Route; 2]> {
        if requests[0].to != requests[1].to || requests[0].consume != requests[1].consume {
            return Err(Error::InvalidValue("fusion packets must meet".into()));
        }
        let mut candidate = self.clone();
        let a = candidate.route_with_policy(requests[0], policy)?;
        let b = candidate.route_with_policy(requests[1], policy)?;
        *self = candidate;
        Ok([a, b])
    }
}
