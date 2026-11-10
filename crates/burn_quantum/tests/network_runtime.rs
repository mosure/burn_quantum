//! Real loopback worker protocol, failure, replay and process-isolation tests.
use burn_quantum::photonic::runtime::network::*;
use burn_quantum::photonic::runtime::{JobIdentity, JobLedger, execute_shards};
use std::net::TcpListener;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
fn identity() -> JobIdentity {
    JobIdentity {
        program: "network-fixture-v1".into(),
        calibration: "ideal-v1".into(),
        executor: "deterministic-v1".into(),
        seed: 47,
        attempts: 64,
    }
}
fn limits() -> NetworkLimits {
    NetworkLimits {
        max_batch_shots: 8,
        io_timeout: Duration::from_millis(500),
        deadline: Duration::from_secs(10),
        ..Default::default()
    }
}
fn evaluate(id: u64, seed: u64, payload: &[u8]) -> burn_quantum::Result<Vec<u8>> {
    if payload != b"fixture" {
        return Err(burn_quantum::Error::InvalidValue("payload".into()));
    }
    Ok((seed ^ id).to_be_bytes().to_vec())
}
#[test]
fn actual_tcp_workers_match_local_ids_and_resume_checkpoints() {
    let listeners: Vec<_> = (0..3)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let endpoints: Vec<_> = listeners
        .iter()
        .map(|s| WorkerEndpoint {
            address: s.local_addr().unwrap(),
            capability: "local-test-capability".into(),
        })
        .collect();
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let handles: Vec<_> = listeners
            .into_iter()
            .map(|listener| {
                let stop = &stop;
                scope.spawn(move || {
                    serve_worker(
                        listener,
                        identity(),
                        "local-test-capability",
                        limits(),
                        64,
                        stop,
                        evaluate,
                    )
                    .unwrap()
                })
            })
            .collect();
        let mut job = JobLedger::new(identity(), 64).unwrap();
        let partial =
            request_batch(&endpoints[0], &identity(), &[0, 1, 2], b"fixture", limits()).unwrap();
        job.merge(identity(), partial).unwrap();
        let saved = serde_json::to_string(&job).unwrap();
        let mut restored: JobLedger<Vec<u8>> = serde_json::from_str(&saved).unwrap();
        execute_remote(
            &mut restored,
            &endpoints,
            b"fixture",
            limits(),
            &AtomicBool::new(false),
        )
        .unwrap();
        let mut local = JobLedger::new(identity(), 64).unwrap();
        execute_shards(&mut local, 2, 8, &AtomicBool::new(false), |id, seed| {
            evaluate(id, seed, b"fixture")
        })
        .unwrap();
        assert_eq!(
            restored.records().collect::<Vec<_>>(),
            local.records().collect::<Vec<_>>()
        );
        stop.store(true, std::sync::atomic::Ordering::Release);
        for h in handles {
            h.join().unwrap();
        }
    });
}
#[test]
fn rejected_capabilities_and_timeouts_do_not_invent_results() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut endpoint = WorkerEndpoint {
        address: listener.local_addr().unwrap(),
        capability: "wrong".into(),
    };
    let handle = std::thread::spawn(move || {
        serve_worker(
            listener,
            identity(),
            "right",
            limits(),
            2,
            &AtomicBool::new(false),
            evaluate,
        )
        .unwrap()
    });
    assert!(request_batch(&endpoint, &identity(), &[0], b"fixture", limits()).is_err());
    endpoint.capability = "right".into();
    let mut wrong = identity();
    wrong.calibration = "changed".into();
    assert!(request_batch(&endpoint, &wrong, &[0], b"fixture", limits()).is_err());
    assert_eq!(handle.join().unwrap(), 2);
    // A listener that never accepts is a real stalled peer, with a bounded deadline.
    let stalled = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = WorkerEndpoint {
        address: stalled.local_addr().unwrap(),
        capability: "right".into(),
    };
    let mut job = JobLedger::new(identity(), 64).unwrap();
    let brief = NetworkLimits {
        io_timeout: Duration::from_millis(20),
        deadline: Duration::from_millis(100),
        wave_attempts: 1,
        ..limits()
    };
    assert!(
        execute_remote(
            &mut job,
            &[endpoint],
            b"fixture",
            brief,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    assert_eq!(job.completed(), 0);
}
#[test]
fn worker_process_entry() {
    if std::env::var_os("BURN_QUANTUM_TEST_WORKER").is_none() {
        return;
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    println!("WORKER_ADDRESS={}", listener.local_addr().unwrap());
    use std::io::Write;
    std::io::stdout().flush().unwrap();
    serve_worker(
        listener,
        identity(),
        "process-capability",
        limits(),
        8,
        &AtomicBool::new(false),
        evaluate,
    )
    .unwrap();
}
#[test]
fn separate_worker_process_uses_the_same_deterministic_protocol() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    struct Guard(std::process::Child);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let mut child = Guard(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "worker_process_entry", "--nocapture"])
            .env("BURN_QUANTUM_TEST_WORKER", "1")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let stdout = child.0.stdout.take().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            if let Some(value) = line.split_once("WORKER_ADDRESS=").map(|(_, value)| value) {
                let _ = send.send(value.to_owned());
            }
        }
    });
    let address = receive
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .parse()
        .unwrap();
    let endpoint = WorkerEndpoint {
        address,
        capability: "process-capability".into(),
    };
    let mut job = JobLedger::new(identity(), 64).unwrap();
    execute_remote(
        &mut job,
        &[endpoint],
        b"fixture",
        limits(),
        &AtomicBool::new(false),
    )
    .unwrap();
    for record in job.records() {
        assert_eq!(
            record.value,
            evaluate(
                record.id,
                burn_quantum::photonic::runtime::shot_seed(47, record.id, 0),
                b"fixture"
            )
            .unwrap()
        );
    }
    assert!(child.0.wait().unwrap().success());
    reader.join().unwrap();
}

#[test]
fn worker_cancellation_is_observed_between_shots_in_one_batch() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = WorkerEndpoint {
        address: listener.local_addr().unwrap(),
        capability: "cancel-test".into(),
    };
    let stop = AtomicBool::new(false);
    let calls = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let server = scope.spawn(|| {
            serve_worker(
                listener,
                identity(),
                "cancel-test",
                limits(),
                1,
                &stop,
                |_, _, _| {
                    calls.fetch_add(1, Ordering::AcqRel);
                    stop.store(true, Ordering::Release);
                    Ok(vec![1])
                },
            )
            .unwrap()
        });
        // The request is atomic: no partial results or invented success are returned.
        assert!(request_batch(&endpoint, &identity(), &[0, 1, 2], b"fixture", limits()).is_err());
        assert_eq!(server.join().unwrap(), 1);
    });
    assert_eq!(calls.load(Ordering::Acquire), 1);
}
