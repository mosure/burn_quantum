//! Bounded asynchronous jobs: provenance, replay, queue pressure, expiry and cancellation.
use burn_quantum::{
    Error,
    photonic::runtime::{JobIdentity, jobs::*},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
fn capabilities() -> JobCapabilities {
    JobCapabilities {
        programs: vec!["circuit-h-z-v1".into()],
        compiler: "compiler-v1".into(),
        physical_model: "model-v1".into(),
        executor: "test-v1".into(),
        calibration: "cal-v1".into(),
        calibration_expires_at: 200,
        readouts: vec!["z".into()],
        precision_bits: vec![32],
        max_attempts: 100,
        max_parameters: 1,
    }
}
fn request() -> SimulationRequest {
    SimulationRequest {
        compiler: "compiler-v1".into(),
        physical_model: "model-v1".into(),
        version: 1,
        identity: JobIdentity {
            program: "circuit-h-z-v1".into(),
            calibration: "cal-v1".into(),
            executor: "test-v1".into(),
            seed: 123,
            attempts: 24,
        },
        bindings: vec![0.2],
        readouts: vec!["z".into()],
        precision_bits: 32,
        error_budget: 0.,
        max_result_bytes: 1 << 18,
        timeout_ms: 1000,
    }
}
fn shot(seed: u64) -> RawSimulationShot {
    RawSimulationShot {
        raw: vec![seed],
        accepted: seed & 1 == 0,
        values: vec![if seed & 1 == 0 { 1. } else { -1. }],
        photons: 2,
        latency_ticks: 7,
    }
}
fn wait(service: &LocalSimulationJobs, id: JobId) -> JobStatus {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let status = service.status(id).unwrap();
        if !matches!(status, JobStatus::Running | JobStatus::Queued) {
            return status;
        }
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}
#[test]
fn immutable_jobs_replay_across_worker_counts_and_retain_raw_unaccepted_records() {
    let mut records = vec![];
    for workers in [1, 3] {
        let service = LocalSimulationJobs::new(
            capabilities(),
            JobServiceLimits {
                workers,
                ..Default::default()
            },
            |_, _, seed, _| Ok(shot(seed)),
        )
        .unwrap();
        let r = request();
        service.submit(JobId(10), r.clone(), 100).unwrap();
        service.submit(JobId(10), r.clone(), 100).unwrap();
        let mut changed = r;
        changed.bindings[0] = 0.3;
        assert!(service.submit(JobId(10), changed, 100).is_err());
        assert_eq!(wait(&service, JobId(10)), JobStatus::Completed);
        let result = service.result(JobId(10)).unwrap();
        assert_eq!(result.execution_kind, "local_simulator");
        assert_eq!(result.shots.len(), 24);
        assert!(result.shots.iter().any(|(_, s)| !s.accepted));
        assert!(result.encoded_bytes > 0);
        records.push(result.shots);
        service.forget(JobId(10)).unwrap();
        assert!(service.status(JobId(10)).is_err());
    }
    assert_eq!(records[0], records[1]);
}
#[test]
fn capability_calibration_and_result_budgets_fail_explicitly() {
    let service = LocalSimulationJobs::new(capabilities(), Default::default(), |_, _, seed, _| {
        Ok(shot(seed))
    })
    .unwrap();
    for kind in 0..5 {
        let mut r = request();
        match kind {
            0 => r.identity.calibration = "different".into(),
            1 => r.readouts = vec!["unknown".into()],
            2 => r.precision_bits = 64,
            3 => r.bindings[0] = f64::NAN,
            _ => r.error_budget = -1.,
        };
        assert!(service.submit(JobId(kind), r, 100).is_err());
    }
    assert!(service.submit(JobId(9), request(), 200).is_err());
    let mut r = request();
    r.max_result_bytes = 100;
    service.submit(JobId(10), r, 100).unwrap();
    assert!(matches!(wait(&service, JobId(10)), JobStatus::Failed(_)));
    assert!(service.result(JobId(10)).unwrap().encoded_bytes <= 100);
}
#[test]
fn queued_cancellation_backpressure_and_cooperative_timeout() {
    let entered = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let e = entered.clone();
    let r = release.clone();
    let service = LocalSimulationJobs::new(
        capabilities(),
        JobServiceLimits {
            workers: 1,
            queued: 1,
            retained: 4,
            ..Default::default()
        },
        move |_, _, seed, cancel| {
            e.store(true, Ordering::Release);
            while !r.load(Ordering::Acquire) && !cancel.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            Ok(shot(seed))
        },
    )
    .unwrap();
    let mut first = request();
    first.timeout_ms = 5;
    service.submit(JobId(1), first, 100).unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !entered.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    service.submit(JobId(2), request(), 100).unwrap();
    assert!(service.submit(JobId(3), request(), 100).is_err());
    service.cancel(JobId(2)).unwrap();
    assert_eq!(service.status(JobId(2)).unwrap(), JobStatus::Cancelled);
    assert!(service.result(JobId(2)).unwrap().shots.is_empty());
    std::thread::sleep(Duration::from_millis(10));
    release.store(true, Ordering::Release);
    assert_eq!(wait(&service, JobId(1)), JobStatus::TimedOut);
    assert_eq!(service.result(JobId(1)).unwrap().shots.len(), 1);
}
#[test]
fn evaluator_failure_keeps_completed_attempt_ids() {
    let service = LocalSimulationJobs::new(capabilities(), Default::default(), |_, id, seed, _| {
        if id == 3 {
            Err(Error::InvalidValue("physical execution failed".into()))
        } else {
            Ok(shot(seed))
        }
    })
    .unwrap();
    service.submit(JobId(5), request(), 100).unwrap();
    assert!(matches!(wait(&service, JobId(5)), JobStatus::Failed(_)));
    assert_eq!(
        service
            .result(JobId(5))
            .unwrap()
            .shots
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}
#[test]
fn final_callback_crossing_calibration_expiry_keeps_record_but_fails_lease() {
    let service = LocalSimulationJobs::new(
        capabilities(),
        JobServiceLimits {
            workers: 1,
            ..Default::default()
        },
        |_, _, seed, _| {
            std::thread::sleep(Duration::from_millis(1050));
            Ok(shot(seed))
        },
    )
    .unwrap();
    let mut r = request();
    r.identity.attempts = 1;
    r.timeout_ms = 2500;
    service.submit(JobId(70), r, 199).unwrap();
    assert!(
        matches!(wait(&service,JobId(70)),JobStatus::Failed(message) if message.contains("calibration expired"))
    );
    let result = service.result(JobId(70)).unwrap();
    assert_eq!(result.shots.len(), 1);
    assert_eq!(result.shots[0].0, 0);
}

#[test]
fn cancellation_observed_by_final_callback_retains_record_and_stays_cancelled() {
    let entered = Arc::new(AtomicBool::new(false));
    let marker = entered.clone();
    let service = LocalSimulationJobs::new(
        capabilities(),
        JobServiceLimits {
            workers: 1,
            ..Default::default()
        },
        move |_, _, seed, cancel| {
            marker.store(true, Ordering::Release);
            while !cancel.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            Ok(shot(seed))
        },
    )
    .unwrap();
    let mut r = request();
    r.identity.attempts = 1;
    service.submit(JobId(99), r, 100).unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while !entered.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    service.cancel(JobId(99)).unwrap();
    assert_eq!(wait(&service, JobId(99)), JobStatus::Cancelled);
    let result = service.result(JobId(99)).unwrap();
    assert_eq!(result.shots.len(), 1);
    assert_eq!(result.shots[0].0, 0);
    assert_eq!(result.shots[0].1.photons, 2);
}
