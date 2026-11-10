//! Heralded factories, timed fabric compilation and resumable host Clifford trajectories.
//! A selected-device dense reference checks the ideal-factory physical probabilities.
use burn::tensor::DType;
use burn_quantum::{
    photonic::factories::ResourceFactory,
    photonic::program::TimingOptions,
    photonic::runtime::{JobIdentity, JobLedger, execute_shards},
    photonic::stabilizer::StabilizerSimulator,
    photonic::*,
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use std::sync::atomic::AtomicBool;
fn run() -> ExampleResult<Report> {
    let mut p = FusionProgram::new();
    let a = p.add_resource(ResourceState::bell_pair(), NodeId(0))?;
    let b = p.add_resource(ResourceState::bell_pair(), NodeId(1))?;
    p.fuse(a[1], b[0], NodeId(2))?;
    let ideal = RailTransmission::IDEAL;
    let fabric = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Source,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: ideal,
            },
        ],
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
    let plan = p.compile(&fabric, CompileOptions::default())?;
    let dense =
        DenseFusionSimulator::default().execute(&plan, 1, &execution_device()?, DType::F32)?;
    let reference =
        values(dense.branches()[0].probability() + dense.branches()[1].probability())?[0];
    ensure((reference - 0.36).abs() < 2e-5, "dense fusion reference")?;
    let factory = ResourceFactory::new(0.6, 2, 3, 3)?;
    let identity = JobIdentity {
        program: "bell-swap-0.9-0.8-v1".into(),
        calibration: "ideal-storage-independent-factory-0.6-v1".into(),
        executor: "host-tableau-v1".into(),
        seed: 47,
        attempts: 4096,
    };
    let mut ledger = JobLedger::<(bool, bool, usize, u64)>::new(identity, 4096)?;
    execute_shards(
        &mut ledger,
        4,
        128,
        &AtomicBool::new(false),
        |shot, seed| {
            let left = factory.generate(seed, shot, 0, 0, 9)?;
            let right = factory.generate(seed, shot, 1, 0, 9)?;
            let cost = left.photons_consumed + right.photons_consumed;
            match (left.available_at, right.available_at) {
                (Some(x), Some(y)) => {
                    let mut timing = TimingOptions::serial(2, 1);
                    timing.resource_births = vec![x, y];
                    let plan = p.compile_timed(&fabric, CompileOptions::default(), timing)?;
                    let result =
                        StabilizerSimulator::default().sample(&plan, shot, 1, seed, &[])?;
                    Ok((
                        true,
                        matches!(result[0].outcomes[0], FusionOutcome::Success { .. }),
                        cost,
                        plan.finish_tick(),
                    ))
                }
                _ => Ok((false, false, cost, left.finish_tick.max(right.finish_tick))),
            }
        },
    )?;
    let records = ledger.records().collect::<Vec<_>>();
    let available = records.iter().filter(|r| r.value.0).count() as f64 / 4096.;
    let success = records.iter().filter(|r| r.value.1).count() as f64 / 4096.;
    let cost = records.iter().map(|r| r.value.2).sum::<usize>() as f64 / 4096.;
    let expected_yield = factory.yield_probability().powi(2);
    ensure((available - expected_yield).abs() < 0.03, "factory yield")?;
    ensure(
        (success - expected_yield * reference).abs() < 0.03,
        "unconditional end-to-end success",
    )?;
    ensure(
        (cost - 4. * factory.expected_attempts()).abs() < 0.2,
        "source cost",
    )?;
    let checkpoint = serde_json::to_string(&ledger)?;
    let restored: JobLedger<(bool, bool, usize, u64)> = serde_json::from_str(&checkpoint)?;
    ensure(restored.completed() == 4096, "checkpoint completion")?;
    Report::new(
        "fbqc_factory_pipeline",
        47,
        0,
        &[
            ("dense_reference_success", reference),
            ("factory_availability", available),
            ("unconditional_success", success),
            ("photons_per_attempt", cost),
            ("host_trajectory_shots", 4096.),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn factories_routes_and_shards_preserve_unconditional_cost() {
        super::run().unwrap();
    }
}
