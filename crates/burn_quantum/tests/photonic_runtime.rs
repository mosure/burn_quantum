//! Independent factory, scheduling and restart/sharding contracts.
use burn_quantum::photonic::{factories::*, runtime::*};
use std::sync::atomic::{AtomicBool, Ordering};
fn identity() -> JobIdentity {
    JobIdentity {
        program: "fixture-v1".into(),
        calibration: "ideal-v1".into(),
        executor: "test-v1".into(),
        seed: 123,
        attempts: 200,
    }
}
#[test]
fn shards_resume_replay_and_cancel_without_changing_attempts() {
    let mut serial = JobLedger::new(identity(), 200).unwrap();
    let cancelled = AtomicBool::new(false);
    execute_shards(&mut serial, 1, 17, &cancelled, |_, seed| Ok(seed)).unwrap();
    let mut parallel = JobLedger::new(identity(), 200).unwrap();
    execute_shards(&mut parallel, 4, 13, &cancelled, |id, seed| {
        if id == 12 {
            cancelled.store(true, Ordering::Release);
        }
        Ok(seed)
    })
    .unwrap();
    assert_eq!(parallel.completed(), 13);
    let json = serde_json::to_string(&parallel).unwrap();
    let mut resumed: JobLedger<u64> = serde_json::from_str(&json).unwrap();
    cancelled.store(false, Ordering::Release);
    execute_shards(&mut resumed, 3, 21, &cancelled, |_, seed| Ok(seed)).unwrap();
    assert_eq!(
        serial.records().collect::<Vec<_>>(),
        resumed.records().collect::<Vec<_>>()
    );
    resumed
        .merge(identity(), serial.records().collect())
        .unwrap();
    assert_eq!(resumed.completed(), 200);
    let before = serde_json::to_string(&resumed).unwrap();
    assert!(
        resumed
            .merge(identity(), vec![ShotRecord { id: 0, value: 0 }])
            .is_err()
    );
    assert_eq!(serde_json::to_string(&resumed).unwrap(), before);
    let mut bad = identity();
    bad.calibration = "changed".into();
    assert!(resumed.merge(bad, vec![]).is_err());
    assert!(
        serde_json::from_str::<JobLedger<u64>>(&json.replace("\"version\":1", "\"version\":2"))
            .is_err()
    );
}
#[test]
fn failed_wave_is_atomic_and_worker_panics_are_errors() {
    let mut job = JobLedger::<u64>::new(identity(), 200).unwrap();
    let cancel = AtomicBool::new(false);
    assert!(
        execute_shards(&mut job, 3, 9, &cancel, |id, seed| if id == 2 {
            Err(burn_quantum::Error::InvalidValue("worker".into()))
        } else {
            Ok(seed)
        })
        .is_err()
    );
    assert_eq!(job.completed(), 0);
    assert!(
        execute_shards(&mut job, 3, 9, &cancel, |id, seed| {
            assert!(id != 1 && id != 7, "worker panic");
            Ok(seed)
        })
        .is_err()
    );
    assert_eq!(job.completed(), 0);
}
#[test]
fn heralded_factory_counts_exhausted_attempts_and_matches_geometric_yield() {
    let factory = ResourceFactory::new(0.3, 6, 2, 4).unwrap();
    assert!((factory.yield_probability() - (1. - 0.7f64.powi(4))).abs() < 1e-12);
    let mut successes = 0;
    let mut attempts = 0;
    for shot in 0..20_000 {
        let r = factory.generate(8, shot, 7, 10, 18).unwrap();
        successes += usize::from(r.available_at.is_some());
        attempts += r.attempts;
        assert_eq!(r.photons_consumed, 6 * r.attempts);
        assert_eq!(r.finish_tick, 10 + 2 * r.attempts as u64);
    }
    assert!((successes as f64 / 20_000. - factory.yield_probability()).abs() < 0.015);
    assert!((attempts as f64 / 20_000. - factory.expected_attempts()).abs() < 0.035);
    let failed = ResourceFactory::new(0., 6, 2, 4)
        .unwrap()
        .generate(8, 1, 7, 0, 8)
        .unwrap();
    assert_eq!(failed.available_at, None);
    assert_eq!(failed.photons_consumed, 24);
    assert!(factory.generate(8, 0, 0, 10, 17).is_err());
    assert!(ResourceFactory::new(f64::NAN, 6, 2, 4).is_err());
}
#[test]
fn calendar_respects_capacity_feedback_reset_and_watermarks_atomically() {
    let mut c = CapacityCalendar::new(vec![2, 1], 8).unwrap();
    assert_eq!(c.reserve(&[0], &[0], 0, 5, 0, 100).unwrap(), (0, 5));
    assert_eq!(c.reserve(&[0], &[0], 0, 8, 0, 100).unwrap(), (0, 8));
    assert_eq!(c.reserve(&[0, 1], &[0], 0, 3, 2, 100).unwrap(), (5, 8));
    assert_eq!(c.reserve(&[1], &[8], 1, 2, 0, 100).unwrap(), (10, 12));
    let before = c.reservations().to_vec();
    assert!(c.reserve(&[1], &[10], 0, 10, 0, 15).is_err());
    assert_eq!(before, c.reservations());
    c.retire_before(12);
    c.retire_before(0);
    assert_eq!(c.reserve(&[0], &[], 0, 1, 0, 100).unwrap(), (12, 13));
    assert!(c.reserve(&[0, 0], &[], 0, 1, 0, 100).is_err());
}

#[test]
fn timed_compilation_uses_births_parallel_analyzers_and_feedback() {
    use burn_quantum::photonic::program::{SchedulePolicy, TimingOptions};
    use burn_quantum::photonic::*;
    let fabric = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: RailTransmission::uniform(0.5).unwrap(),
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: RailTransmission::IDEAL,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: RailTransmission::IDEAL,
            },
        ],
        vec![
            FabricLink {
                from: NodeId(0),
                to: NodeId(1),
                ticks: 1,
                transmission: RailTransmission::IDEAL,
            },
            FabricLink {
                from: NodeId(0),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::IDEAL,
            },
        ],
    )
    .unwrap();
    let mut p = FusionProgram::new();
    let a = p
        .add_resource(ResourceState::bell_pair(), NodeId(0))
        .unwrap();
    let b = p
        .add_resource(ResourceState::bell_pair(), NodeId(0))
        .unwrap();
    let c = p
        .add_resource(ResourceState::bell_pair(), NodeId(0))
        .unwrap();
    p.fuse(a[0], a[1], NodeId(1)).unwrap();
    p.fuse(b[0], b[1], NodeId(2)).unwrap();
    let mut timing = TimingOptions {
        resource_births: vec![5, 5, 7],
        dependencies: vec![vec![], vec![]],
        policy: SchedulePolicy::Concurrent,
        feed_forward_ticks: 2,
        analyzer_reset_ticks: 1,
    };
    let plan = p
        .compile_timed(&fabric, CompileOptions::default(), timing.clone())
        .unwrap();
    assert_eq!(plan.fusions()[0].tick, 7);
    assert_eq!(plan.fusions()[1].tick, 7);
    assert_eq!(plan.finish_tick(), 9);
    assert_eq!(plan.transmissions()[c[0].0].values(), [0.25; 2]);
    // The second packet waits one tick at its source for its exclusive link.
    assert_eq!(plan.transmissions()[a[1].0].values(), [0.5; 2]);
    timing.dependencies[1] = vec![0];
    let dependent = p
        .compile_timed(&fabric, CompileOptions::default(), timing.clone())
        .unwrap();
    assert_eq!(dependent.routes()[2].route.release, 10);
    assert_eq!(dependent.fusions()[1].tick, 12);
    timing.dependencies[0] = vec![1];
    assert!(
        p.compile_timed(&fabric, CompileOptions::default(), timing)
            .is_err()
    );
}
