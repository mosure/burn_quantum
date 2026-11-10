//! Causal frontier invariants, rollback and complete-state replay.
use burn_quantum::{
    Error,
    photonic::runtime::{JobIdentity, events::*},
};
use serde::{Deserialize, Serialize};
fn identity() -> JobIdentity {
    JobIdentity {
        program: "causal-feedback-v1".into(),
        calibration: "explicit-source-controller-v1".into(),
        executor: "events-v1".into(),
        seed: 19,
        attempts: 4,
    }
}
fn limits() -> EventLimits {
    EventLimits {
        max_events: 128,
        max_pending: 8,
        max_checkpoint_bytes: 16384,
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct State {
    frame: Vec<bool>,
    draws: Vec<u64>,
    control: Vec<u64>,
}
fn machine() -> EventMachine<State, u64> {
    EventMachine::new(
        identity(),
        0,
        State {
            frame: vec![true, false],
            draws: vec![],
            control: vec![],
        },
        vec![(2, 0), (4, 1)],
        limits(),
    )
    .unwrap()
}
fn apply(
    state: &mut State,
    event: &Event<u64>,
    seed: u64,
) -> burn_quantum::Result<Vec<(u64, u64)>> {
    state.draws.push(seed);
    state.frame[(event.operation % 2) as usize] ^= seed & 1 == 1;
    state.control.push(event.tick);
    Ok(if event.operation < 4 {
        vec![(event.tick + 3, event.operation + 2)]
    } else {
        vec![]
    })
}
#[test]
fn window_restart_preserves_state_pending_feedback_and_randomness() {
    let mut whole = machine();
    whole.advance(40, 128, apply).unwrap();
    let mut windows = machine();
    windows.advance(3, 2, apply).unwrap();
    assert_eq!(windows.pending().count(), 2);
    assert_eq!(windows.watermark(), 3);
    let bytes = windows.checkpoint().unwrap();
    let mut windows = EventMachine::restore(&bytes, &identity(), 16384, |s: &State| {
        if s.frame.len() == 2 {
            Ok(())
        } else {
            Err(Error::Shape("frame".into()))
        }
    })
    .unwrap();
    for tick in [4, 7, 8, 40] {
        windows.advance(tick, 2, apply).unwrap();
    }
    assert_eq!(whole.state(), windows.state());
    assert_eq!(whole.completed(), windows.completed());
    assert_eq!(whole.checkpoint().unwrap(), windows.checkpoint().unwrap());
}
#[test]
fn failed_transition_is_atomic_and_queue_limit_does_not_drop_events() {
    let mut m = machine();
    let before = m.checkpoint().unwrap();
    assert!(
        m.advance(4, 1, |s, e, _| {
            s.frame.clear();
            Ok(vec![(e.tick - 1, 1)])
        })
        .is_err()
    );
    assert_eq!(m.checkpoint().unwrap(), before);
    assert!(
        m.advance(4, 1, |s, e, _| {
            s.frame.clear();
            Ok(vec![(e.tick + 1, 1); 9])
        })
        .is_err()
    );
    assert_eq!(m.checkpoint().unwrap(), before);
    m.advance(100, 1, apply).unwrap();
    assert_eq!(m.watermark(), 2);
    assert!(m.advance(1, 1, apply).is_err());
}
#[test]
fn checkpoint_rejects_changed_identity_corrupt_frontier_and_external_budget() {
    let m = machine();
    let bytes = m.checkpoint().unwrap();
    let mut wrong = identity();
    wrong.calibration = "different".into();
    assert!(EventMachine::<State, u64>::restore(&bytes, &wrong, 16384, |_| Ok(())).is_err());
    assert!(
        EventMachine::<State, u64>::restore(&bytes, &identity(), bytes.len() - 1, |_| Ok(()))
            .is_err()
    );
    let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    v["pending"][0]["cause"] = serde_json::json!(1);
    assert!(
        EventMachine::<State, u64>::restore(
            &serde_json::to_vec(&v).unwrap(),
            &identity(),
            16384,
            |_| Ok(())
        )
        .is_err()
    );
    let mut l = limits();
    l.max_checkpoint_bytes = 20;
    let m = EventMachine::<State, u64>::new(identity(), 0, m.state().clone(), vec![], l).unwrap();
    assert!(m.checkpoint().is_err());
}

#[test]
fn restored_same_tick_frontier_cannot_reorder_completed_event_ids() {
    let mut m = EventMachine::new(
        identity(),
        0,
        machine().state().clone(),
        vec![(2, 0u64), (2, 1)],
        limits(),
    )
    .unwrap();
    m.advance(2, 1, |_, _, _| Ok(vec![])).unwrap();
    let bytes = m.checkpoint().unwrap();
    assert!(EventMachine::<State, u64>::restore(&bytes, &identity(), 16384, |_| Ok(())).is_ok());
    let mut invalid: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    invalid["completed"][0]["id"] = serde_json::json!(1);
    invalid["pending"][0]["id"] = serde_json::json!(0);
    assert!(
        EventMachine::<State, u64>::restore(
            &serde_json::to_vec(&invalid).unwrap(),
            &identity(),
            16384,
            |_| Ok(())
        )
        .is_err()
    );
}
