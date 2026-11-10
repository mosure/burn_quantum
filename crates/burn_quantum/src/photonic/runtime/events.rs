//! Bounded causal event frontiers with transactional callbacks and complete-state checkpoints.
//!
//! A window advances time; it never discards caller state, entanglement, queued photons or
//! pending feedback. This executor has one state owner. Partitioning that state across
//! machines requires a separate, model-specific causal-cut proof.
use super::{JobIdentity, shot_seed};
use crate::{Error, Result};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, BTreeSet};

/// Bounds count generated events over the entire trajectory, not just a window.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventLimits {
    /// Maximum events generated, including executed events retained for provenance.
    pub max_events: usize,
    /// Maximum simultaneously pending events (backpressure produces an error).
    pub max_pending: usize,
    /// Maximum serialized checkpoint bytes, checked before decoding and while encoding.
    pub max_checkpoint_bytes: usize,
}
impl EventLimits {
    fn validate(self) -> Result<()> {
        if self.max_events == 0
            || self.max_events > 1_048_576
            || self.max_pending == 0
            || self.max_pending > self.max_events
            || self.max_checkpoint_bytes == 0
        {
            return Err(Error::InvalidValue("invalid event budgets".into()));
        }
        Ok(())
    }
}
/// Event IDs are stable within one job/shot and independent of window boundaries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event<E> {
    /// Monotonically allocated identity; same-time events execute in ID order.
    pub id: u64,
    /// Physical/control readiness tick.
    pub tick: u64,
    /// Causal parent event, or None for an initially supplied event.
    pub cause: Option<u64>,
    /// Typed model operation.
    pub operation: E,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot<S, E> {
    version: u32,
    identity: JobIdentity,
    shot: u64,
    limits: EventLimits,
    watermark: u64,
    next_id: u64,
    state: S,
    pending: Vec<Event<E>>,
    completed: Vec<Event<E>>,
}
/// Complete model state plus pending frontier and audit history.
///
/// `S` must include all correlations, source queues and controller/decoder state.
/// Callers bound live state and event payload allocations. This executor bounds
/// counts and serialized bytes; it cannot infer heap size from a generic type.
/// `S::clone` must isolate mutable state (deep copy or copy-on-write). Shared
/// interior-mutability aliases violate the transactional rollback contract.
pub struct EventMachine<S, E> {
    snapshot: Snapshot<S, E>,
    queue: BTreeMap<(u64, u64), Event<E>>,
}
impl<S: Clone + Serialize + DeserializeOwned, E: Clone + Serialize + DeserializeOwned>
    EventMachine<S, E>
{
    /// Initialize one shot with explicit roots and caller-validated complete state.
    pub fn new(
        identity: JobIdentity,
        shot: u64,
        state: S,
        initial: Vec<(u64, E)>,
        limits: EventLimits,
    ) -> Result<Self> {
        limits.validate()?;
        identity.validate()?;
        if shot >= identity.attempts
            || initial.len() > limits.max_pending
            || initial.len() > limits.max_events
        {
            return Err(Error::InvalidValue(
                "event initial shot/queue exceeds budget".into(),
            ));
        }
        let queue = initial
            .into_iter()
            .enumerate()
            .map(|(id, (tick, operation))| {
                (
                    (tick, id as u64),
                    Event {
                        id: id as u64,
                        tick,
                        cause: None,
                        operation,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        Ok(Self {
            snapshot: Snapshot {
                version: 1,
                identity,
                shot,
                limits,
                watermark: 0,
                next_id: queue.len() as u64,
                state,
                pending: vec![],
                completed: vec![],
            },
            queue,
        })
    }
    /// Full retained state; advancing a window does not drop modes or frames.
    pub fn state(&self) -> &S {
        &self.snapshot.state
    }
    /// Committed causal audit in execution order.
    pub fn completed(&self) -> &[Event<E>] {
        &self.snapshot.completed
    }
    /// Pending events ordered by physical readiness and stable ID.
    pub fn pending(&self) -> impl Iterator<Item = &Event<E>> {
        self.queue.values()
    }
    /// No subsequently inserted event may precede this tick.
    pub fn watermark(&self) -> u64 {
        self.snapshot.watermark
    }
    /// Advance a bounded number of events no later than `through`.
    ///
    /// Each callback receives a versioned event seed and a candidate state. It returns
    /// future events, including feedback delays or fresh-resource retries. Errors,
    /// backward time, and queue overflow roll back that event's state and frontier.
    /// Earlier committed events remain. Callbacks must not perform external side effects
    /// or mutate shared state through aliases retained by the original snapshot.
    pub fn advance<F>(&mut self, through: u64, max_steps: usize, mut apply: F) -> Result<usize>
    where
        F: FnMut(&mut S, &Event<E>, u64) -> Result<Vec<(u64, E)>>,
    {
        if through < self.snapshot.watermark || max_steps == 0 {
            return Err(Error::InvalidValue(
                "invalid causal window/step budget".into(),
            ));
        }
        let mut steps = 0;
        while steps < max_steps {
            let Some((&key, event)) = self.queue.first_key_value() else {
                break;
            };
            if event.tick > through {
                break;
            }
            let mut candidate = self.snapshot.state.clone();
            let domain = shot_seed(
                self.snapshot.identity.seed,
                self.snapshot.shot,
                0x4556_454e_5453_0001,
            );
            let seed = shot_seed(domain, event.id, 0);
            let emitted = apply(&mut candidate, event, seed)?;
            let total = self
                .snapshot
                .next_id
                .checked_add(emitted.len() as u64)
                .ok_or(Error::Overflow)?;
            let pending = self.queue.len() - 1 + emitted.len();
            if total > self.snapshot.limits.max_events as u64
                || pending > self.snapshot.limits.max_pending
            {
                return Err(Error::InvalidValue(
                    "event generation or pending queue budget exhausted".into(),
                ));
            }
            if emitted.iter().any(|(tick, _)| *tick < event.tick) {
                return Err(Error::InvalidValue(
                    "event emitted before its causal parent".into(),
                ));
            }
            let event = self.queue.remove(&key).unwrap();
            for (tick, operation) in emitted {
                let id = self.snapshot.next_id;
                self.snapshot.next_id += 1;
                self.queue.insert(
                    (tick, id),
                    Event {
                        id,
                        tick,
                        cause: Some(event.id),
                        operation,
                    },
                );
            }
            self.snapshot.state = candidate;
            self.snapshot.watermark = event.tick;
            self.snapshot.completed.push(event);
            steps += 1;
        }
        if self
            .queue
            .first_key_value()
            .is_none_or(|(key, _)| key.0 > through)
        {
            self.snapshot.watermark = through;
        }
        Ok(steps)
    }
    /// Encode a complete checkpoint without building an unbounded intermediate JSON string.
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        struct Bounded {
            bytes: Vec<u8>,
            limit: usize,
        }
        impl std::io::Write for Bounded {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                if b.len() > self.limit.saturating_sub(self.bytes.len()) {
                    return Err(std::io::Error::other("event checkpoint budget"));
                }
                self.bytes.extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut snapshot = self.snapshot.clone();
        snapshot.pending = self.queue.values().cloned().collect();
        let mut out = Bounded {
            bytes: vec![],
            limit: snapshot.limits.max_checkpoint_bytes,
        };
        serde_json::to_writer(&mut out, &snapshot)
            .map_err(|e| Error::InvalidValue(e.to_string()))?;
        Ok(out.bytes)
    }
    /// Restore only the expected immutable model and validate caller state before execution.
    ///
    /// `max_bytes` is an out-of-band admission limit; a serialized checkpoint cannot
    /// authorize its own unbounded allocation. State validation includes quantum-state
    /// and controller invariants appropriate to the application's representation.
    pub fn restore<F>(
        bytes: &[u8],
        expected: &JobIdentity,
        max_bytes: usize,
        validate_state: F,
    ) -> Result<Self>
    where
        F: FnOnce(&S) -> Result<()>,
    {
        if bytes.len() > max_bytes {
            return Err(Error::InvalidValue(
                "event checkpoint admission budget".into(),
            ));
        }
        let mut s: Snapshot<S, E> =
            serde_json::from_slice(bytes).map_err(|e| Error::InvalidValue(e.to_string()))?;
        s.limits.validate()?;
        s.identity.validate()?;
        if s.version != 1
            || &s.identity != expected
            || s.shot >= s.identity.attempts
            || bytes.len() > s.limits.max_checkpoint_bytes
            || s.next_id > s.limits.max_events as u64
            || s.pending.len() > s.limits.max_pending
            || s.pending.len() + s.completed.len() != s.next_id as usize
        {
            return Err(Error::InvalidValue(
                "event checkpoint identity/count mismatch".into(),
            ));
        }
        let mut known = BTreeMap::new();
        let mut ids = BTreeSet::new();
        let mut previous = None;
        for event in &s.completed {
            let key = (event.tick, event.id);
            if previous.is_some_and(|p| p >= key) || event.tick > s.watermark {
                return Err(Error::InvalidValue("invalid committed event order".into()));
            }
            Self::validate_event(event, &known, &mut ids, s.next_id)?;
            known.insert(event.id, event.tick);
            previous = Some(key);
        }
        let mut queue = BTreeMap::new();
        for event in s.pending.drain(..) {
            if event.tick < s.watermark || previous.is_some_and(|key| key >= (event.tick, event.id))
            {
                return Err(Error::InvalidValue(
                    "pending event precedes the committed frontier".into(),
                ));
            }
            Self::validate_event(&event, &known, &mut ids, s.next_id)?;
            queue.insert((event.tick, event.id), event);
        }
        validate_state(&s.state)?;
        Ok(Self { snapshot: s, queue })
    }
    fn validate_event(
        event: &Event<E>,
        known: &BTreeMap<u64, u64>,
        ids: &mut BTreeSet<u64>,
        next_id: u64,
    ) -> Result<()> {
        if event.id >= next_id
            || !ids.insert(event.id)
            || event.cause.is_some_and(|id| {
                id >= event.id || known.get(&id).is_none_or(|&tick| tick > event.tick)
            })
        {
            return Err(Error::InvalidValue(
                "invalid event identity/causal parent".into(),
            ));
        }
        Ok(())
    }
}
