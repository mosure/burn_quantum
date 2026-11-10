use super::{FusionProgram, PhotonId, SchedulePolicy, TimingOptions};
use crate::photonic::fabric::{FabricConstraints, PacketRequest, PacketScheduler, Reservation};
use crate::photonic::factories::CapacityCalendar;
use crate::photonic::{Fabric, NodeId, NodeKind, RailTransmission, Route};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Explicit bounds and detector assumptions for deterministic static compilation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompileOptions {
    /// Maximum supplied photons (IR size, not a dense-state allocation limit).
    pub max_photons: usize,
    /// Maximum fusions in a compiled network.
    pub max_fusions: usize,
    /// Inclusive scheduling horizon in integer ticks.
    pub max_ticks: u64,
    /// Attenuation before otherwise ideal PNR detection, applied once per consumed photon.
    pub detector_transmission: RailTransmission,
}
impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            max_photons: 4096,
            max_fusions: 2048,
            max_ticks: 1_000_000,
            detector_transmission: RailTransmission::IDEAL,
        }
    }
}
/// Transport of a stable photon ID to its consuming fusion.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoutedPhoton {
    /// Packet identity.
    pub photon: PhotonId,
    /// Scheduled route, including link reservations and transport attenuation.
    pub route: Route,
}
/// A scheduled standard fusion, after both input packets arrive.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ScheduledFusion {
    /// Consumed photons, in measurement target order.
    pub photons: [PhotonId; 2],
    /// Analyzer location.
    pub at: NodeId,
    /// Measurement start tick; subsequent release/reset follows the recorded TimingOptions.
    pub tick: u64,
}
/// Checked, immutable output of compilation; retain the program/fabric for provenance.
///
/// Only compilation constructs this type. Persist the validated source program
/// and fabric then recompile; untrusted serialized schedules are not executable.
#[derive(Clone, Debug, Serialize)]
pub struct CompiledFusionNetwork {
    version: u32,
    options: CompileOptions,
    timing: TimingOptions,
    constraints: Option<FabricConstraints>,
    pub(crate) program: FusionProgram,
    fabric: Fabric,
    pub(crate) transmissions: Vec<RailTransmission>,
    routes: Vec<RoutedPhoton>,
    pub(crate) fusions: Vec<ScheduledFusion>,
    pub(crate) outputs: Vec<PhotonId>,
    finish: u64,
}
impl CompiledFusionNetwork {
    /// Compilation limits and detector assumptions retained for provenance.
    pub fn options(&self) -> CompileOptions {
        self.options
    }
    /// Resource availability, dependency and scheduling provenance.
    pub fn timing(&self) -> &TimingOptions {
        &self.timing
    }
    /// Finite packet-buffer and switch constraints, when explicitly requested.
    pub fn constraints(&self) -> Option<&FabricConstraints> {
        self.constraints.as_ref()
    }
    /// Source program, including ideal resource descriptions.
    pub fn program(&self) -> &FusionProgram {
        &self.program
    }
    /// Exact fabric snapshot used for routing.
    pub fn fabric(&self) -> &Fabric {
        &self.fabric
    }
    /// Scheduled input routes in fusion/operand order.
    pub fn routes(&self) -> &[RoutedPhoton] {
        &self.routes
    }
    /// Scheduled destructive operations.
    pub fn fusions(&self) -> &[ScheduledFusion] {
        &self.fusions
    }
    /// Surviving, unconsumed IDs in output register order (which may physically be vacuum).
    pub fn outputs(&self) -> &[PhotonId] {
        &self.outputs
    }
    /// Accumulated transmissions per supplied photon, including storage and detection.
    pub fn transmissions(&self) -> &[RailTransmission] {
        &self.transmissions
    }
    /// End of the static execution schedule, including one tick per analyzer use.
    pub fn finish_tick(&self) -> u64 {
        self.finish
    }
}
pub(super) fn compile(
    program: &FusionProgram,
    fabric: &Fabric,
    options: CompileOptions,
) -> Result<CompiledFusionNetwork> {
    compile_timed(
        program,
        fabric,
        options,
        TimingOptions::serial(program.resources.len(), program.num_fusions()),
    )
}
pub(super) fn compile_timed(
    program: &FusionProgram,
    fabric: &Fabric,
    options: CompileOptions,
    timing: TimingOptions,
) -> Result<CompiledFusionNetwork> {
    compile_impl(program, fabric, options, timing, None)
}
pub(super) fn compile_finite(
    program: &FusionProgram,
    fabric: &Fabric,
    options: CompileOptions,
    timing: TimingOptions,
    constraints: FabricConstraints,
    fusion_ticks: &[u64],
) -> Result<CompiledFusionNetwork> {
    if fusion_ticks.len() != program.num_fusions() {
        return Err(Error::Shape(
            "one finite consumption tick per fusion is required".into(),
        ));
    }
    compile_impl(
        program,
        fabric,
        options,
        timing,
        Some((constraints, fusion_ticks)),
    )
}
fn compile_impl(
    program: &FusionProgram,
    fabric: &Fabric,
    options: CompileOptions,
    timing: TimingOptions,
    finite: Option<(FabricConstraints, &[u64])>,
) -> Result<CompiledFusionNetwork> {
    if timing.resource_births.len() != program.resources.len()
        || timing.dependencies.len() != program.num_fusions()
    {
        return Err(Error::Shape(
            "resource births/dependencies do not match program".into(),
        ));
    }
    if timing
        .resource_births
        .iter()
        .any(|&t| t > options.max_ticks)
    {
        return Err(Error::InvalidValue("resource born after horizon".into()));
    }
    for (index, deps) in timing.dependencies.iter().enumerate() {
        let mut ordered = deps.clone();
        ordered.sort_unstable();
        if ordered.iter().any(|&d| d >= index) || ordered.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::InvalidValue(
                "noncausal/repeated fusion dependency".into(),
            ));
        }
    }
    let count = program.num_photons();
    if count == 0 {
        return Err(Error::InvalidValue(
            "fusion program contains no resources".into(),
        ));
    }
    for (resource, required, limit) in [
        ("compiled photons", count, options.max_photons),
        (
            "compiled fusions",
            program.num_fusions(),
            options.max_fusions,
        ),
    ] {
        if required > limit {
            return Err(Error::ResourceLimit {
                resource,
                required,
                limit,
            });
        }
    }
    let mut sources = Vec::with_capacity(count);
    let mut births = Vec::with_capacity(count);
    for (instance, &birth) in program.resources.iter().zip(&timing.resource_births) {
        births.extend(std::iter::repeat_n(birth, instance.state.qubits()));
        if fabric.node(instance.source)?.kind != NodeKind::Source {
            return Err(Error::InvalidValue("resource port is not a source".into()));
        }
        sources.extend(std::iter::repeat_n(
            instance.source,
            instance.state.qubits(),
        ));
    }
    let mut transmissions = vec![RailTransmission::IDEAL; count];
    let mut consumed = vec![false; count];
    let mut routes = Vec::with_capacity(program.num_fusions() * 2);
    let mut fusions = Vec::with_capacity(program.num_fusions());
    let mut reservations: Vec<Reservation> = vec![];
    let constraints = finite.as_ref().map(|(c, _)| c.clone());
    let mut packets = constraints
        .clone()
        .map(|c| PacketScheduler::new(fabric.clone(), c))
        .transpose()?;
    let mut cursor = 0;
    let mut completion = Vec::<u64>::new();
    let mut analyzers =
        CapacityCalendar::new(vec![1; fabric.nodes().len()], program.num_fusions())?;
    for (index, request) in program.fusions.iter().enumerate() {
        let dependency_ready = timing.dependencies[index]
            .iter()
            .map(|&d| completion[d])
            .max()
            .unwrap_or(0);
        let control_ready = if timing.dependencies[index].is_empty() {
            0
        } else {
            dependency_ready
                .checked_add(timing.feed_forward_ticks)
                .ok_or(Error::Overflow)?
        };
        let mut release = control_ready
            .max(births[request.photons[0].0])
            .max(births[request.photons[1].0]);
        if timing.policy == SchedulePolicy::Serial {
            release = release.max(cursor);
        }
        if fabric.node(request.at)?.kind != NodeKind::Fusion {
            return Err(Error::InvalidValue(
                "fusion destination is not an analyzer".into(),
            ));
        }
        let mut pair = Vec::with_capacity(2);
        if let Some(packets) = &mut packets {
            let consume = finite.as_ref().unwrap().1[index];
            if consume < release || consume >= options.max_ticks {
                return Err(Error::InvalidValue(
                    "finite consumption outside causal horizon".into(),
                ));
            }
            pair.extend(packets.pair_at(request.photons.map(|photon| PacketRequest {
                from: sources[photon.0],
                to: request.at,
                birth: births[photon.0],
                release,
                consume,
            }))?);
        } else {
            for &photon in &request.photons {
                pair.push(fabric.route_reserved(
                    sources[photon.0],
                    request.at,
                    release,
                    options.max_ticks,
                    &mut reservations,
                )?);
            }
        }
        let arrival = finite.as_ref().map_or_else(
            || pair.iter().map(|r| r.arrival).max().unwrap(),
            |(_, ticks)| ticks[index],
        );
        let (tick, measured) = analyzers.reserve(
            &[request.at.0],
            &[arrival],
            0,
            1,
            timing.analyzer_reset_ticks,
            options.max_ticks,
        )?;
        if finite.is_some() && tick != arrival {
            return Err(Error::Unsupported(
                "finite fusion tick conflicts with analyzer reset".into(),
            ));
        }
        let next = measured
            .checked_add(timing.analyzer_reset_ticks)
            .ok_or(Error::Overflow)?;
        completion.push(measured);
        for (photon, route) in request.photons.into_iter().zip(pair) {
            transmissions[photon.0] = fabric
                .node(sources[photon.0])?
                .storage
                .over_ticks(release - births[photon.0])
                .then(route.transmission)
                .then(
                    fabric
                        .node(request.at)?
                        .storage
                        .over_ticks(tick - route.arrival),
                )
                .then(options.detector_transmission);
            consumed[photon.0] = true;
            routes.push(RoutedPhoton { photon, route });
        }
        fusions.push(ScheduledFusion {
            photons: request.photons,
            at: request.at,
            tick,
        });
        cursor = cursor.max(next);
    }
    cursor = cursor.max(births.iter().copied().max().unwrap_or(0));
    let outputs: Vec<_> = (0..count).filter(|&p| !consumed[p]).map(PhotonId).collect();
    for photon in &outputs {
        if let Some(packets) = &mut packets {
            packets.store_output(sources[photon.0], births[photon.0], cursor)?;
        }
        transmissions[photon.0] = fabric
            .node(sources[photon.0])?
            .storage
            .over_ticks(cursor - births[photon.0]);
    }
    Ok(CompiledFusionNetwork {
        version: 3,
        options,
        timing,
        constraints,
        program: program.clone(),
        fabric: fabric.clone(),
        transmissions,
        routes,
        fusions,
        outputs,
        finish: cursor,
    })
}
