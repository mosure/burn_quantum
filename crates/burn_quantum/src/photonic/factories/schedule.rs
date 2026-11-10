use crate::{Error, Result};
use serde::Serialize;

/// A half-open occupancy interval on a capacity-limited physical resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Reservation {
    /// Caller-defined physical component ID (link, buffer, detector or switch).
    pub component: usize,
    /// Inclusive occupancy start.
    pub start: u64,
    /// Exclusive release, including explicitly requested reset latency.
    pub end: u64,
}
/// Monotonic interval scheduler with shared physical capacities and bounded history.
#[derive(Clone, Debug)]
pub struct CapacityCalendar {
    capacities: Vec<usize>,
    reservations: Vec<Reservation>,
    max_reservations: usize,
    watermark: u64,
}
impl CapacityCalendar {
    /// Each component has a strictly positive capacity.
    pub fn new(capacities: Vec<usize>, max_reservations: usize) -> Result<Self> {
        if capacities.is_empty() || capacities.len() > 65_536 || capacities.contains(&0) {
            return Err(Error::InvalidValue("invalid component capacities".into()));
        }
        Ok(Self {
            capacities,
            reservations: vec![],
            max_reservations,
            watermark: 0,
        })
    }
    /// Existing immutable reservations, in insertion order.
    pub fn reservations(&self) -> &[Reservation] {
        &self.reservations
    }
    /// Reserve all listed components simultaneously after all dependencies plus feed-forward.
    /// Duplicate component IDs are errors; one slot of each component is occupied.
    /// The earliest feasible interval is chosen; failure leaves the calendar unchanged.
    pub fn reserve(
        &mut self,
        components: &[usize],
        dependencies: &[u64],
        feed_forward: u64,
        duration: u64,
        reset: u64,
        deadline: u64,
    ) -> Result<(u64, u64)> {
        if components.is_empty() || duration == 0 {
            return Err(Error::InvalidValue("empty reservation or duration".into()));
        }
        let mut ids = components.to_vec();
        ids.sort_unstable();
        if ids.last().is_some_and(|&id| id >= self.capacities.len())
            || ids.windows(2).any(|w| w[0] == w[1])
        {
            return Err(Error::InvalidValue("invalid/repeated component ID".into()));
        }
        let required = self
            .reservations
            .len()
            .checked_add(ids.len())
            .ok_or(Error::Overflow)?;
        if required > self.max_reservations {
            return Err(Error::ResourceLimit {
                resource: "schedule reservations",
                required,
                limit: self.max_reservations,
            });
        }
        let occupied = duration.checked_add(reset).ok_or(Error::Overflow)?;
        let mut start = dependencies
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_add(feed_forward)
            .ok_or(Error::Overflow)?;
        start = start.max(self.watermark);
        loop {
            let end = start.checked_add(occupied).ok_or(Error::Overflow)?;
            if end > deadline {
                return Err(Error::InvalidValue("no capacity before deadline".into()));
            }
            let mut shift = None;
            for &component in &ids {
                let overlaps: Vec<_> = self
                    .reservations
                    .iter()
                    .filter(|r| r.component == component && r.start < end && start < r.end)
                    .collect();
                let mut boundaries = vec![start];
                boundaries.extend(overlaps.iter().filter(|r| r.start > start).map(|r| r.start));
                boundaries.sort_unstable();
                boundaries.dedup();
                for tick in boundaries {
                    let active: Vec<_> = overlaps
                        .iter()
                        .filter(|r| r.start <= tick && tick < r.end)
                        .collect();
                    if active.len() >= self.capacities[component] {
                        let next = active.iter().map(|r| r.end).min().unwrap();
                        shift = Some(shift.map_or(next, |old: u64| old.max(next)));
                        break;
                    }
                }
            }
            if let Some(next) = shift {
                start = next;
            } else {
                self.reservations
                    .extend(ids.into_iter().map(|component| Reservation {
                        component,
                        start,
                        end,
                    }));
                return Ok((start, start + duration));
            }
        }
    }
    /// Drop expired reservations once a causal watermark proves no earlier scheduling is needed.
    /// Future reservations cannot start before the monotonically advancing watermark.
    pub fn retire_before(&mut self, watermark: u64) {
        self.watermark = self.watermark.max(watermark);
        self.reservations.retain(|r| r.end > self.watermark);
    }
}
