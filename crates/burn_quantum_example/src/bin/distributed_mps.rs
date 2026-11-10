//! Actual multi-process ownership of one non-Clifford entangled MPS, with recovery.
use burn::tensor::DType;
use burn_quantum::{
    DenseSimulator, Register, StateVector, Unitary,
    tensor_network::{distributed::*, *},
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use std::{
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    process::{Child, Command, Stdio},
    sync::atomic::AtomicBool,
    time::Duration,
};
const CAP: &str = "local-mps-demonstration-v1";
fn identity() -> StateIdentity {
    StateIdentity {
        program: "ghz-t-h-cnot-six-sites-v1".into(),
        calibration: "ideal-exact-schmidt-v1".into(),
        run: "local-process-oracle-v1".into(),
    }
}
fn network() -> StateNetworkLimits {
    StateNetworkLimits {
        deadline: Duration::from_secs(60),
        ..Default::default()
    }
}
fn worker() -> ExampleResult<()> {
    let start: usize = std::env::var("MPS_WORKER_START")?.parse()?;
    let count: usize = std::env::var("MPS_WORKER_COUNT")?.parse()?;
    let fail = std::env::var("MPS_FAIL_COMMIT_ONCE").is_ok();
    let site = SiteData {
        shape: [1, 2, 1],
        values: vec![1.0.into(), 0.0.into()],
    };
    let mut owner = ShardWorker::new(
        ShardCheckpoint {
            identity: identity(),
            start,
            total: 6,
            epoch: 0,
            center: 0,
            report: Default::default(),
            sites: vec![site; count],
        },
        Default::default(),
    )?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    println!("WORKER_ADDRESS={}", listener.local_addr()?);
    std::io::stdout().flush()?;
    let mut injected = false;
    serve_state_worker(
        listener,
        &mut owner,
        CAP,
        network(),
        10_000,
        &AtomicBool::new(false),
        |cp| {
            // A failed durable write must leave the owner prepared. The coordinator
            // has already durably selected COMMIT and must reconcile, never rerun gates.
            if fail && !injected && cp.epoch() == 1 && !cp.is_prepared() {
                injected = true;
                return Err(burn_quantum::Error::InvalidValue(
                    "injected worker persistence failure".into(),
                ));
            }
            Ok(())
        },
    )?;
    Ok(())
}
struct Workers(Vec<Child>);
impl Drop for Workers {
    fn drop(&mut self) {
        for c in &mut self.0 {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}
fn spawn(count: usize, inject: bool) -> ExampleResult<(Workers, Vec<WorkerEndpoint>)> {
    let mut workers = Workers(vec![]);
    let mut endpoints = vec![];
    for owner in 0..count {
        let mut command = Command::new(std::env::current_exe()?);
        if cfg!(test) {
            command.args(["--exact", "tests::worker_entry", "--nocapture"]);
        } else {
            command.arg("--worker");
        }
        command
            .env("MPS_WORKER_START", (owner * 6 / count).to_string())
            .env("MPS_WORKER_COUNT", (6 / count).to_string());
        if inject && owner == 1 {
            command.env("MPS_FAIL_COMMIT_ONCE", "1");
        }
        let mut child = command.stdout(Stdio::piped()).spawn()?;
        let stdout = child.stdout.take().unwrap();
        workers.0.push(child);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some((_, address)) = line.split_once("WORKER_ADDRESS=") {
                    let _ = tx.send(address.to_owned());
                }
            }
        });
        endpoints.push(WorkerEndpoint {
            address: rx.recv_timeout(Duration::from_secs(10))?.parse()?,
            capability: CAP.into(),
        });
    }
    Ok((workers, endpoints))
}
fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let engine = DenseSimulator::default();
    let mut oracle =
        StateVector::zero(Register::qubits(6)?, 1, &device, DType::F32, engine.limits)?;
    let gates = vec![
        (vec![0], Unitary::h()),
        (vec![0, 1], Unitary::cnot()),
        (vec![1, 2], Unitary::cnot()),
        (vec![2, 3], Unitary::cnot()),
        (vec![3, 4], Unitary::cnot()),
        (vec![4, 5], Unitary::cnot()),
        (vec![5], Unitary::t()),
        (vec![2], Unitary::h()),
        (vec![2, 3], Unitary::cnot()),
    ];
    for (w, g) in &gates {
        oracle = engine.apply(&oracle, w, g)?;
    }
    let expected_re = values(oracle.amplitudes().real())?;
    let expected_im = values(oracle.amplitudes().imag())?;
    let mut statistics = vec![];
    let mut max_error = 0.0f64;
    let mut recovered = 0;
    for count in [1, 3] {
        let (_workers, endpoints) = spawn(count, count == 3)?;
        let mut remote = DistributedMps::connect(
            identity(),
            endpoints.clone(),
            &device,
            DType::F32,
            Default::default(),
            network(),
        )?;
        let mut durable = remote.checkpoint();
        let mut persist = |cp: &CoordinatorCheckpoint| {
            durable = cp.clone();
            Ok(())
        };
        for (j, (w, g)) in gates.iter().enumerate() {
            let result = if w.len() == 1 {
                remote.apply_single(w[0], g, &mut persist)
            } else {
                remote.apply_adjacent(w[0], g, &mut persist)
            };
            if let Err(error) = result {
                ensure(
                    count == 3 && j == 0,
                    "only injected first-commit failure is expected",
                )?;
                ensure(
                    remote.needs_reconciliation(),
                    "uncertain commit must block reads",
                )?;
                ensure(
                    remote.amplitude(&[0; 6]).is_err(),
                    "read cannot observe mixed epochs",
                )?;
                let cp: CoordinatorCheckpoint =
                    serde_json::from_slice(&serde_json::to_vec(&remote.checkpoint())?)?;
                remote = DistributedMps::restore(
                    cp,
                    &identity(),
                    endpoints.clone(),
                    &device,
                    DType::F32,
                    Default::default(),
                    network(),
                )?;
                remote.reconcile(&mut persist)?;
                ensure(!remote.needs_reconciliation(), "journal reconciliation")?;
                recovered += 1;
                eprintln!("reconciled expected failure: {error}");
            }
        }
        let cp = remote.export()?;
        let local = MpsState::restore(&cp, &device, DType::F32, Default::default())?;
        let dense = local.to_dense(engine.limits)?;
        let re = values(dense.amplitudes().real())?;
        let im = values(dense.amplitudes().imag())?;
        for i in 0..64 {
            max_error = max_error
                .max((re[i] - expected_re[i]).abs())
                .max((im[i] - expected_im[i]).abs());
        }
        let zero = remote.amplitude(&[0; 6])?;
        ensure(
            (zero.re - expected_re[0]).abs() < 3e-6 && (zero.im - expected_im[0]).abs() < 3e-6,
            "streamed amplitude",
        )?;
        let norm = remote.expectation_product(&[])?;
        ensure(
            (norm.re - 1.).abs() < 1e-5 && norm.im.abs() < 1e-5,
            "streamed global norm",
        )?;
        let ops = vec![
            (0, Unitary::z().matrix().clone()),
            (5, Unitary::z().matrix().clone()),
        ];
        ensure(
            (remote.expectation_product(&ops)? - local.expectation_product(&ops)?).norm() < 3e-6,
            "streamed entangled correlation",
        )?;
        ensure(
            remote.statistics().peak_gate_sites == 2,
            "at most two site tensors per distributed gate",
        )?;
        statistics.push(remote.statistics());
    }
    ensure(
        max_error < 4e-6,
        "one/three-owner complete dense channel amplitudes",
    )?;
    ensure(recovered == 1, "injected commit recovered exactly once")?;
    Report::new(
        "distributed_mps",
        0,
        9,
        &[
            ("maximum_amplitude_error", max_error),
            ("recovered_uncertain_commits", recovered as f64),
            ("one_worker_wire_bytes", statistics[0].bytes as f64),
            ("three_workers_wire_bytes", statistics[1].bytes as f64),
            ("one_worker_rpc_calls", statistics[0].calls as f64),
            ("three_workers_rpc_calls", statistics[1].calls as f64),
        ],
    )
}
fn main() -> ExampleResult<()> {
    if std::env::args().any(|a| a == "--worker") {
        return worker();
    }
    run()?.print()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn worker_entry() {
        if std::env::var("MPS_WORKER_START").is_ok() {
            super::worker().unwrap();
        }
    }
    #[test]
    fn distributed_entanglement_and_recovery() {
        super::run().unwrap();
    }
}
