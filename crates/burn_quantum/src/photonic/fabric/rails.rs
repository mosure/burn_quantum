//! Independent coherent rail reservations over explicitly independent wavelength channels.
use super::{
    Fabric, FabricConstraints, NodeId, PacketRequest, PacketScheduler, Route, RoutePolicy,
};
use crate::photonic::RailTransmission;
use crate::{Error, Result};
use serde::Serialize;
/// Wavelength-resolved physical capacities. A slot is a configured optical rail mode,
/// not an observed photon; both rails are reserved even in a one-photon superposition.
#[derive(Clone, Debug, Serialize)]
pub struct WavelengthConstraints {
    /// Stable distinct calibration labels; wavelengths do not interconvert in this model.
    pub wavelengths: Vec<String>,
    /// Independent link and buffer capacity/configuration for each wavelength channel.
    pub channels: Vec<FabricConstraints>,
    /// Maximum unsynchronized rail-arrival difference; larger skew is rejected.
    pub max_skew_ticks: u64,
    /// Deterministic routing objective inside the specified consumption deadline.
    pub policy: RoutePolicy,
}
/// Two optical paths carrying one encoded dual-rail input.
#[derive(Clone, Debug, Serialize)]
pub struct RailRoute {
    /// Routes in logical zero/one rail order; both channel reservations are real.
    pub rails: [Route; 2],
    /// Channel indices corresponding to the two routes.
    pub wavelengths: [usize; 2],
    /// Physical arrival skew before destination delay/synchronization.
    pub skew_ticks: u64,
    /// Equivalent phase-stable dual-rail attenuation after synchronizing the two paths.
    /// Like Route, this excludes common pre-release and final synchronization storage.
    pub packet: Route,
}
/// Transactional independent-rail scheduler with bounded per-wavelength state/history.
/// Paths and storage are assumed phase-stabilized and preserve optical coherence.
/// The executor must use matched wavelengths on corresponding analyzer input rails;
/// this scheduler supplies no frequency conversion or distinguishability approximation.
#[derive(Clone, Debug)]
pub struct RailPacketScheduler {
    fabric: Fabric,
    constraints: WavelengthConstraints,
    channels: Vec<PacketScheduler>,
}
impl RailPacketScheduler {
    /// Construct independent wavelength channels over the same calibrated topology.
    pub fn new(fabric: Fabric, constraints: WavelengthConstraints) -> Result<Self> {
        if constraints.wavelengths.is_empty()
            || constraints.wavelengths.len() > 64
            || constraints.wavelengths.len() != constraints.channels.len()
            || constraints
                .wavelengths
                .iter()
                .any(|s| s.is_empty() || s.len() > 128)
        {
            return Err(Error::InvalidValue(
                "invalid wavelength channel calibration".into(),
            ));
        }
        let mut names = constraints.wavelengths.clone();
        names.sort();
        if names.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::InvalidValue("duplicate wavelength channels".into()));
        }
        let channels = constraints
            .channels
            .iter()
            .map(|c| PacketScheduler::new(fabric.clone(), c.clone()))
            .collect::<Result<_>>()?;
        Ok(Self {
            fabric,
            constraints,
            channels,
        })
    }
    /// Route both rails atomically; source, destination and consumption times are shared.
    pub fn route_at(
        &mut self,
        request: PacketRequest,
        wavelengths: [usize; 2],
    ) -> Result<RailRoute> {
        if wavelengths.iter().any(|&w| w >= self.channels.len()) {
            return Err(Error::InvalidValue(
                "wavelength channel out of bounds".into(),
            ));
        }
        let mut candidate = self.clone();
        let a = candidate.channels[wavelengths[0]]
            .route_with_policy(request, self.constraints.policy)?;
        let b = candidate.channels[wavelengths[1]]
            .route_with_policy(request, self.constraints.policy)?;
        let arrival = a.arrival.max(b.arrival);
        let skew = a.arrival.abs_diff(b.arrival);
        if skew > self.constraints.max_skew_ticks {
            return Err(Error::Unsupported(
                "physical rail skew exceeds synchronization budget".into(),
            ));
        }
        let storage = self.fabric.node(request.to)?.storage;
        let zero = a
            .transmission
            .then(storage.over_ticks(arrival - a.arrival))
            .values()[0];
        let one = b
            .transmission
            .then(storage.over_ticks(arrival - b.arrival))
            .values()[1];
        let packet = Route {
            from: request.from,
            to: request.to,
            release: request.release,
            arrival,
            hops: vec![],
            transmission: RailTransmission::new(zero, one)?,
        };
        *self = candidate;
        Ok(RailRoute {
            rails: [a, b],
            wavelengths,
            skew_ticks: skew,
            packet,
        })
    }
    /// Atomically schedule both analyzer inputs with matched corresponding wavelengths.
    pub fn pair_at(
        &mut self,
        requests: [PacketRequest; 2],
        wavelengths: [usize; 2],
    ) -> Result<[RailRoute; 2]> {
        if requests[0].to != requests[1].to || requests[0].consume != requests[1].consume {
            return Err(Error::InvalidValue(
                "rail fusion operands do not meet".into(),
            ));
        }
        let mut candidate = self.clone();
        let a = candidate.route_at(requests[0], wavelengths)?;
        let b = candidate.route_at(requests[1], wavelengths)?;
        *self = candidate;
        Ok([a, b])
    }
    /// Keep both unconsumed output rail modes occupied at their source until completion.
    pub fn store_output(
        &mut self,
        source: NodeId,
        birth: u64,
        finish: u64,
        wavelengths: [usize; 2],
    ) -> Result<()> {
        if wavelengths.iter().any(|&w| w >= self.channels.len()) {
            return Err(Error::InvalidValue(
                "wavelength channel out of bounds".into(),
            ));
        }
        let mut candidate = self.clone();
        for wavelength in wavelengths {
            candidate.channels[wavelength].store_output(source, birth, finish)?;
        }
        *self = candidate;
        Ok(())
    }
    /// Total real optical-mode occupancy intervals retained across wavelength channels.
    pub fn reservation_count(&self) -> usize {
        self.channels
            .iter()
            .map(PacketScheduler::reservation_count)
            .sum()
    }
}
