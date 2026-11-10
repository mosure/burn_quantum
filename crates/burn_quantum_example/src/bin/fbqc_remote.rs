//! Real process/TCP trajectory sharding with exact replay and a selected-device dense oracle.
use burn::tensor::DType;
use burn_quantum::{
    photonic::runtime::{network::*, *},
    photonic::stabilizer::{StabilizerShot, StabilizerSimulator},
    photonic::*,
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use std::{
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    process::{Child, Command, Stdio},
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};
const CAPABILITY: &str = "local-demonstration-worker-v1";
fn identity() -> JobIdentity {
    JobIdentity {
        program: "physical-bell-swap-v1".into(),
        calibration: "independent-0.9-0.8-links-ideal-storage-v1".into(),
        executor: "balanced-tableau-v1".into(),
        seed: 61,
        attempts: 1024,
    }
}
fn plan() -> burn_quantum::Result<CompiledFusionNetwork> {
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
    p.compile(&fabric, CompileOptions::default())
}
fn evaluate(
    plan: &CompiledFusionNetwork,
    id: u64,
    seed: u64,
    payload: &[u8],
) -> burn_quantum::Result<Vec<u8>> {
    if !payload.is_empty() {
        return Err(burn_quantum::Error::InvalidValue(
            "this immutable example accepts no mutable payload".into(),
        ));
    }
    let shot = StabilizerSimulator::default()
        .sample(plan, id, 1, seed, &[])?
        .remove(0);
    serde_json::to_vec(&shot).map_err(|e| burn_quantum::Error::InvalidValue(e.to_string()))
}
fn limits() -> NetworkLimits {
    NetworkLimits {
        max_batch_shots: 128,
        io_timeout: Duration::from_secs(10),
        deadline: Duration::from_secs(30),
        ..Default::default()
    }
}
fn worker() -> ExampleResult<()> {
    let plan = plan()?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    println!("WORKER_ADDRESS={}", listener.local_addr()?);
    std::io::stdout().flush()?;
    serve_worker(
        listener,
        identity(),
        CAPABILITY,
        limits(),
        4,
        &AtomicBool::new(false),
        |id, seed, payload| evaluate(&plan, id, seed, payload),
    )?;
    Ok(())
}
struct Workers(Vec<Child>);
impl Drop for Workers {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
fn run() -> ExampleResult<Report> {
    let plan = plan()?;
    let dense =
        DenseFusionSimulator::default().execute(&plan, 1, &execution_device()?, DType::F32)?;
    let expected =
        values(dense.branches()[0].probability() + dense.branches()[1].probability())?[0];
    ensure(
        (expected - 0.36).abs() < 2e-5,
        "selected-device physical oracle",
    )?;
    let mut children = Workers(vec![]);
    let mut readers = vec![];
    let mut endpoints = vec![];
    for _ in 0..2 {
        let mut command = Command::new(std::env::current_exe()?);
        if cfg!(test) {
            command.args(["--exact", "tests::worker_entry", "--nocapture"]);
        } else {
            command.arg("--worker");
        }
        let mut child = command
            .env("BURN_QUANTUM_REMOTE_EXAMPLE_WORKER", "1")
            .stdout(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().unwrap();
        children.0.push(child);
        let (tx, rx) = std::sync::mpsc::channel();
        readers.push(std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some((_, address)) = line.split_once("WORKER_ADDRESS=") {
                    let _ = tx.send(address.to_owned());
                }
            }
        }));
        endpoints.push(WorkerEndpoint {
            address: rx.recv_timeout(Duration::from_secs(10))?.parse()?,
            capability: CAPABILITY.into(),
        });
    }
    let started = Instant::now();
    let mut remote = JobLedger::new(identity(), 1024)?;
    execute_remote(
        &mut remote,
        &endpoints,
        &[],
        limits(),
        &AtomicBool::new(false),
    )?;
    let elapsed = started.elapsed().as_secs_f64();
    for child in &mut children.0 {
        ensure(child.wait()?.success(), "worker exit")?;
    }
    for reader in readers {
        reader.join().map_err(|_| "worker stdout reader panic")?;
    }
    let mut local = JobLedger::new(identity(), 1024)?;
    execute_shards(&mut local, 1, 128, &AtomicBool::new(false), |id, seed| {
        evaluate(&plan, id, seed, &[])
    })?;
    ensure(
        remote.records().eq(local.records()),
        "byte-for-byte process replay",
    )?;
    let bytes = remote.records().map(|r| r.value.len()).sum::<usize>();
    let shots: Vec<StabilizerShot> = remote
        .records()
        .map(|r| serde_json::from_slice(&r.value))
        .collect::<Result<_, _>>()?;
    let success = shots
        .iter()
        .filter(|s| matches!(s.outcomes[0], FusionOutcome::Success { .. }))
        .count() as f64
        / 1024.;
    ensure(
        (success - expected).abs() < 0.05,
        "unconditional sampled success",
    )?;
    let checkpoint = serde_json::to_vec(&remote)?;
    let restored: JobLedger<Vec<u8>> = serde_json::from_slice(&checkpoint)?;
    ensure(
        restored.records().eq(remote.records()),
        "remote checkpoint replay",
    )?;
    Report::new(
        "fbqc_remote",
        61,
        0,
        &[
            ("dense_reference_success", expected),
            ("sampled_success", success),
            ("host_trajectory_shots", 1024.),
            ("worker_processes", 2.),
            ("result_payload_bytes", bytes as f64),
            ("remote_seconds", elapsed),
        ],
    )
}
fn main() -> ExampleResult<()> {
    if std::env::args().any(|a| a == "--worker") {
        worker()
    } else {
        run()?.print()
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn worker_entry() {
        if std::env::var_os("BURN_QUANTUM_REMOTE_EXAMPLE_WORKER").is_some() {
            super::worker().unwrap();
        }
    }
    #[test]
    fn real_process_shards_match_local_photonic_trajectories() {
        super::run().unwrap();
    }
}
