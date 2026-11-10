//! Bounded fresh-resource retries with source heralding, finite buffers and delayed feedback.
use burn::tensor::DType;
use burn_quantum::{
    photonic::*,
    photonic::{
        fabric::FabricConstraints,
        factories::{CorrelatedFactoryBank, FactoryMode, ResourceFactory},
        program::TimingOptions,
        runtime::{JobIdentity, events::*},
        stabilizer::StabilizerSimulator,
    },
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use serde::{Deserialize, Serialize};
use std::time::Instant;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Operation {
    Generate,
    Measure,
    Feedback(bool),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct State {
    attempts: usize,
    photons: usize,
    birth: Option<u64>,
    success: bool,
    finish: u64,
    outcomes: Vec<FusionOutcome>,
    source_pairs: usize,
    source_emissions: usize,
    /// Reserved packet slots, including output storage until delayed feedback.
    /// Lost photons retain their reservation until the detector herald is known.
    buffers: Vec<(usize, u64, u64)>,
}
#[derive(Debug, Serialize)]
struct LatencyBin {
    ticks: u64,
    experiments: usize,
    probability: f64,
    exact_probability: f64,
}
/// This fixture emits all resources at one birth tick. Reconstruct the checked
/// buffer intervals from routed hops and extend retained outputs through feedback.
fn buffer_intervals(plan: &CompiledFusionNetwork, feedback: u64) -> Vec<(usize, u64, u64)> {
    let mut slots = vec![];
    let mut sources = vec![];
    for (resource, birth) in plan
        .program()
        .resources()
        .iter()
        .zip(&plan.timing().resource_births)
    {
        sources.extend(std::iter::repeat_n(
            (resource.source.0, *birth),
            resource.state.qubits(),
        ));
    }
    for routed in plan.routes() {
        let (mut node, mut arrival) = sources[routed.photon.0];
        for hop in &routed.route.hops {
            slots.push((node, arrival, hop.departure));
            node = plan.fabric().links()[hop.link].to.0;
            arrival = hop.arrival;
        }
        let consume = plan
            .fusions()
            .iter()
            .find(|f| f.photons.contains(&routed.photon))
            .unwrap()
            .tick
            + 1;
        slots.push((node, arrival, consume));
    }
    for output in plan.outputs() {
        let (node, birth) = sources[output.0];
        slots.push((node, birth, feedback));
    }
    slots
}
/// Exact three-round terminal distribution: exhausted source takes four ticks,
/// a reached analyzer takes six, and only a failed round retries fresh resources.
fn exact_latency(source_yield: f64, fusion_success: f64) -> [f64; 19] {
    let mut terminal = [0.; 19];
    let mut frontier = vec![(0, 1.)];
    for round in 0..3 {
        let mut next = vec![];
        for (tick, probability) in frontier {
            terminal[tick + 6] += probability * source_yield * fusion_success;
            for (duration, failure) in [
                (4, 1. - source_yield),
                (6, source_yield * (1. - fusion_success)),
            ] {
                if round == 2 {
                    terminal[tick + duration] += probability * failure;
                } else {
                    next.push((tick + duration, probability * failure));
                }
            }
        }
        frontier = next;
    }
    terminal
}
fn model() -> burn_quantum::Result<(FusionProgram, Fabric)> {
    let mut p = FusionProgram::new();
    let a = p.add_resource(ResourceState::bell_pair(), NodeId(0))?;
    let b = p.add_resource(ResourceState::bell_pair(), NodeId(1))?;
    p.fuse(a[1], b[0], NodeId(2))?;
    let fabric = Fabric::new(
        [NodeKind::Source, NodeKind::Source, NodeKind::Fusion]
            .map(|kind| FabricNode {
                kind,
                storage: RailTransmission::IDEAL,
            })
            .to_vec(),
        vec![
            FabricLink {
                from: NodeId(0),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::uniform(0.9)?,
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::uniform(0.8)?,
            },
        ],
    )?;
    Ok((p, fabric))
}
fn compile(
    p: &FusionProgram,
    f: &Fabric,
    birth: u64,
) -> burn_quantum::Result<CompiledFusionNetwork> {
    let mut timing = TimingOptions::serial(2, 1);
    timing.resource_births = vec![birth, birth];
    p.compile_finite(
        f,
        CompileOptions::default(),
        timing,
        FabricConstraints {
            buffers: vec![2, 2, 2],
            configure_ticks: vec![0; 3],
            max_search_states: 128,
            max_reservations: 32,
        },
        &[birth + 1],
    )
}
fn run() -> ExampleResult<(Report, Vec<LatencyBin>)> {
    let (p, f) = model()?;
    let plan = compile(&p, &f, 0)?;
    let dense =
        DenseFusionSimulator::default().execute(&plan, 1, &execution_device()?, DType::F32)?;
    let physical =
        values(dense.branches()[0].probability() + dense.branches()[1].probability())?[0];
    // A common outage correlates the two sources; conditional bright-mode attempts are independent.
    let source = |p| ResourceFactory::new(p, 2, 2, 1);
    let factory = CorrelatedFactoryBank::new(
        "shared-outage-0.2-brightness-0.9-v1".into(),
        vec![
            FactoryMode {
                probability: 0.2,
                sources: vec![source(0.)?, source(0.)?],
            },
            FactoryMode {
                probability: 0.8,
                sources: vec![source(0.9)?, source(0.9)?],
            },
        ],
    )?;
    let identity = JobIdentity {
        program: "bounded-bell-distribution-v1".into(),
        calibration: "shared-outage-0.2-brightness-0.9-links-0.9-0.8-feedback-2-v1".into(),
        executor: "event-tableau-v1".into(),
        seed: 73,
        attempts: 2048,
    };
    let mut accepted = 0;
    let mut photons = 0;
    let mut latency = 0;
    let mut rounds = 0;
    let mut source_pairs = 0;
    let mut source_emissions = 0;
    let mut buffer_ticks = 0;
    let mut peak_buffers = 0;
    let mut latencies = [0usize; 19];
    let mut host_latencies = vec![];
    for shot in 0..identity.attempts {
        let host_start = Instant::now();
        let initial = State {
            attempts: 0,
            photons: 0,
            birth: None,
            success: false,
            finish: 0,
            outcomes: vec![],
            source_pairs: 0,
            source_emissions: 0,
            buffers: vec![],
        };
        let mut machine = EventMachine::new(
            identity.clone(),
            shot,
            initial,
            vec![(0, Operation::Generate)],
            EventLimits {
                max_events: 9,
                max_pending: 1,
                max_checkpoint_bytes: 16384,
            },
        )?;
        let apply = |state: &mut State,
                     event: &Event<Operation>,
                     seed|
         -> burn_quantum::Result<Vec<(u64, Operation)>> {
            match event.operation {
                Operation::Generate => {
                    state.attempts += 1;
                    let generated = factory.generate(
                        seed,
                        shot,
                        state.attempts as u64,
                        event.tick,
                        event.tick + 2,
                    )?;
                    state.photons += generated
                        .sources
                        .iter()
                        .map(|r| r.photons_consumed)
                        .sum::<usize>();
                    state.source_emissions += generated
                        .sources
                        .iter()
                        .filter(|r| r.available_at.is_some())
                        .count();
                    state.birth = generated
                        .sources
                        .iter()
                        .all(|r| r.available_at.is_some())
                        .then_some(event.tick + 2);
                    state.source_pairs += usize::from(state.birth.is_some());
                    Ok(vec![match state.birth {
                        Some(t) => (t + 2, Operation::Measure),
                        None => (event.tick + 4, Operation::Feedback(false)),
                    }])
                }
                Operation::Measure => {
                    let plan = compile(
                        &p,
                        &f,
                        state.birth.ok_or_else(|| {
                            burn_quantum::Error::InvalidValue("missing heralded resources".into())
                        })?,
                    )?;
                    if plan.finish_tick() != event.tick {
                        return Err(burn_quantum::Error::InvalidValue(
                            "measurement precedes routes".into(),
                        ));
                    }
                    let sample = StabilizerSimulator::default()
                        .sample(&plan, shot, 1, seed, &[])?
                        .remove(0);
                    state
                        .buffers
                        .extend(buffer_intervals(&plan, event.tick + 2));
                    let outcome = sample.outcomes[0];
                    let success = matches!(outcome, FusionOutcome::Success { .. });
                    state.outcomes.push(outcome);
                    state.birth = None;
                    Ok(vec![(event.tick + 2, Operation::Feedback(success))])
                }
                Operation::Feedback(success) => {
                    state.finish = event.tick;
                    state.success = success;
                    Ok(if !success && state.attempts < 3 {
                        vec![(event.tick, Operation::Generate)]
                    } else {
                        vec![]
                    })
                }
            }
        };
        // Checkpoint with a pending factory outcome/measurement; restore full controller state.
        machine.advance(2, 1, apply)?;
        let bytes = machine.checkpoint()?;
        let mut replay = EventMachine::restore(&bytes, &identity, 16384, |s: &State| {
            if s.attempts <= 3 && s.photons == s.attempts * 4 {
                Ok(())
            } else {
                Err(burn_quantum::Error::InvalidValue(
                    "invalid retry state".into(),
                ))
            }
        })?;
        machine.advance(32, 16, apply)?;
        // Measured controller+simulation time includes checkpoint creation and
        // restoration, but excludes the duplicate replay evolution below.
        host_latencies.push(host_start.elapsed().as_secs_f64());
        for tick in 3..=32 {
            replay.advance(tick, 16, apply)?;
        }
        ensure(
            machine.state() == replay.state() && machine.completed() == replay.completed(),
            "window checkpoint replay",
        )?;
        ensure(
            machine.pending().next().is_none(),
            "bounded retry completed",
        )?;
        accepted += usize::from(machine.state().success);
        photons += machine.state().photons;
        latency += machine.state().finish;
        rounds += machine.state().attempts;
        source_pairs += machine.state().source_pairs;
        source_emissions += machine.state().source_emissions;
        latencies[machine.state().finish as usize] += 1;
        for tick in 0..machine.state().finish {
            let mut occupied = [0; 3];
            for &(node, start, end) in &machine.state().buffers {
                occupied[node] += usize::from(start <= tick && tick < end);
            }
            ensure(
                occupied.iter().all(|n| *n <= 2),
                "feedback output storage exceeds buffer capacity",
            )?;
            let total: usize = occupied.iter().sum();
            buffer_ticks += total;
            peak_buffers = peak_buffers.max(total);
        }
    }
    let count = identity.attempts as f64;
    let per_round = factory.joint_yield() * physical;
    let yield_expected = 1. - (1. - per_round).powi(3);
    let attempts_expected = yield_expected / per_round;
    let exact = exact_latency(factory.joint_yield(), physical);
    let expected_finish: f64 = exact.iter().enumerate().map(|(t, p)| t as f64 * p).sum();
    let bins: Vec<_> = latencies
        .iter()
        .zip(exact)
        .enumerate()
        .filter(|(_, (count, p))| **count > 0 || *p > 0.)
        .map(|(ticks, (&experiments, exact_probability))| LatencyBin {
            ticks: ticks as u64,
            experiments,
            probability: experiments as f64 / count,
            exact_probability,
        })
        .collect();
    ensure(
        (exact.iter().sum::<f64>() - 1.).abs() < 1e-12,
        "terminal duration distribution not complete",
    )?;
    ensure(
        bins.iter()
            .all(|b| (b.probability - b.exact_probability).abs() < 0.035),
        "latency distribution missed exact enumeration",
    )?;
    ensure(
        buffer_ticks == source_pairs * 10 && peak_buffers == 4,
        "routed reservation occupancy oracle",
    )?;
    ensure(
        (latency as f64 / count - expected_finish).abs() < 0.4,
        "mean latency missed exact renewal oracle",
    )?;
    let quantile = |q: f64| {
        let mut cumulative = 0;
        let rank = (q * count).ceil() as usize;
        latencies
            .iter()
            .position(|n| {
                cumulative += n;
                cumulative >= rank
            })
            .unwrap() as f64
    };
    host_latencies.sort_by(f64::total_cmp);
    let host_quantile = |q: f64| host_latencies[(q * count).ceil() as usize - 1];
    ensure(
        (accepted as f64 / count - yield_expected).abs() < 0.035,
        "bounded unconditional success",
    )?;
    ensure(
        (rounds as f64 / count - attempts_expected).abs() < 0.08,
        "fresh-resource attempt count",
    )?;
    Ok((
        Report::new(
            "fbqc_adaptive",
            73,
            0,
            &[
                ("dense_fusion_success", physical),
                ("bounded_success", accepted as f64 / count),
                ("expected_bounded_success", yield_expected),
                ("photons_per_experiment", photons as f64 / count),
                ("mean_finish_tick", latency as f64 / count),
                ("host_trajectory_shots", count),
                (
                    "source_yield_per_invocation",
                    source_emissions as f64 / (2 * rounds) as f64,
                ),
                (
                    "source_pair_yield_per_round",
                    source_pairs as f64 / rounds as f64,
                ),
                ("expected_source_pair_yield", factory.joint_yield()),
                (
                    "fusion_success_per_reached_analyzer",
                    accepted as f64 / source_pairs as f64,
                ),
                ("peak_reserved_buffer_slots", peak_buffers as f64),
                (
                    "reserved_buffer_slot_ticks_per_experiment",
                    buffer_ticks as f64 / count,
                ),
                (
                    "mean_reserved_buffer_slots",
                    buffer_ticks as f64 / latency as f64,
                ),
                ("latency_p50_ticks", quantile(0.5)),
                ("latency_p95_ticks", quantile(0.95)),
                ("latency_p99_ticks", quantile(0.99)),
                ("expected_finish_tick", expected_finish),
                (
                    "useful_output_pairs_per_tick",
                    accepted as f64 / latency as f64,
                ),
                (
                    "expected_useful_output_pairs_per_tick",
                    yield_expected / expected_finish,
                ),
                ("controller_p50_seconds", host_quantile(0.5)),
                ("controller_p95_seconds", host_quantile(0.95)),
                ("controller_p99_seconds", host_quantile(0.99)),
            ],
        )?,
        bins,
    ))
}
fn main() -> ExampleResult<()> {
    let (report, distribution) = run()?;
    let mut json = serde_json::to_value(report)?;
    json["latency_distribution"] = serde_json::to_value(distribution)?;
    json["occupancy_contract"] = serde_json::json!(
        "reserved network packet slots through delayed feedback; excludes source-internal preparation and in-flight links; one sequential experiment at a time; outputs leave at feedback"
    );
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn retries_use_fresh_photons_and_preserve_delayed_feedback() {
        super::run().unwrap();
    }
}
