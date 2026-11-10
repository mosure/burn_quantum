//! Asynchronous immutable local-simulator jobs with bounded queues and raw records.
//!
//! This service never selects hardware or silently falls back from a provider.
//! Cancellation/deadlines are cooperative between shots; a running evaluator is
//! not forcibly interrupted. The same request type can be checked by a future
//! provider adapter, but local results are explicitly labeled simulator results.
use super::{JobIdentity, shot_seed};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// Executor-declared local capabilities and immutable calibration validity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobCapabilities {
    /// Versioned executor implementation fingerprint.
    pub executor: String,
    /// Supported immutable program fingerprints.
    pub programs: Vec<String>,
    /// Compiler fingerprint accepted by this service.
    pub compiler: String,
    /// Physical model fingerprint accepted by this service.
    pub physical_model: String,
    /// Exact physical-model/calibration fingerprint accepted by this service.
    pub calibration: String,
    /// Wall-clock expiry supplied by caller as Unix seconds; checked at submit.
    pub calibration_expires_at: u64,
    /// Supported named readouts (no implicit substitutes).
    pub readouts: Vec<String>,
    /// Supported numerical precisions, a nonempty subset of 32/64.
    pub precision_bits: Vec<u8>,
    /// Maximum attempts per request, including rejected/herald-failed shots.
    pub max_attempts: u64,
    /// Maximum classical scalar bindings per request.
    pub max_parameters: usize,
}
/// Immutable request; seed and stable attempt IDs are independent of scheduling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationRequest {
    /// Semantic request schema, currently 1.
    pub version: u32,
    /// Program/calibration/executor/seed/attempt count.
    pub identity: JobIdentity,
    /// Immutable compiler version/content fingerprint.
    pub compiler: String,
    /// Immutable physical-model fingerprint.
    pub physical_model: String,
    /// Finite classical parameter/input bindings, frozen at submission.
    pub bindings: Vec<f64>,
    /// Ordered readout names, all explicitly supported by the executor.
    pub readouts: Vec<String>,
    /// Numerical precision in bits, 32 or 64 (evaluator must support the request).
    pub precision_bits: u8,
    /// Explicit requested absolute approximation bound, finite in `[0,1]`.
    pub error_budget: f64,
    /// Maximum total retained encoded raw-record bytes for this job.
    pub max_result_bytes: usize,
    /// Wall-clock budget from submission, including time queued.
    pub timeout_ms: u64,
}
/// One actual attempted experiment; keep raw outcomes even when unaccepted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawSimulationShot {
    /// Detector/measurement record, whose schema belongs to `identity.executor`.
    pub raw: Vec<u64>,
    /// Explicit scientific acceptance, separate from execution completion.
    pub accepted: bool,
    /// Named-request readouts in request order; finite even on failure records.
    pub values: Vec<f64>,
    /// Physical source/ancilla photons consumed by this attempt.
    pub photons: u64,
    /// Modeled physical elapsed ticks, including generation/waiting/control.
    pub latency_ticks: u64,
}
/// Stable service-local handle. Requests using the same id are immutable/idempotent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct JobId(pub u64);
/// Execution status; failed/cancelled/expired jobs retain completed raw attempts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Waiting in the bounded FIFO.
    Queued,
    /// Evaluator is active.
    Running,
    /// Every requested attempt completed, independent of scientific acceptance.
    Completed,
    /// Cooperative user cancellation.
    Cancelled,
    /// Submission-relative deadline expired.
    TimedOut,
    /// Evaluator, panic or resource-budget failure, with a diagnostic.
    Failed(String),
}
/// Honest simulator result envelope; acceptance never changes attempted-shot IDs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationResult {
    /// Immutable originating request.
    pub request: SimulationRequest,
    /// Always `local_simulator`; no provider/hardware claim.
    pub execution_kind: String,
    /// Current lifecycle state.
    pub status: JobStatus,
    /// Complete `(attempt_id, raw record)` pairs, ordered by ID.
    pub shots: Vec<(u64, RawSimulationShot)>,
    /// Retained encoded bytes, checked before every append.
    pub encoded_bytes: usize,
    /// Elapsed host wall time at this snapshot (not physical latency).
    pub wall_seconds: f64,
}
/// Queue/concurrency/memory limits independent of requested physical costs.
#[derive(Clone, Copy, Debug)]
pub struct JobServiceLimits {
    /// Fixed number of simulator worker threads.
    pub workers: usize,
    /// Maximum simultaneously queued jobs (running jobs excluded).
    pub queued: usize,
    /// Maximum retained job identities/results, including terminal jobs.
    pub retained: usize,
    /// Maximum result bytes allowed for one request.
    pub result_bytes: usize,
}
impl Default for JobServiceLimits {
    fn default() -> Self {
        Self {
            workers: 2,
            queued: 64,
            retained: 256,
            result_bytes: 16 << 20,
        }
    }
}
struct Entry {
    request: SimulationRequest,
    status: JobStatus,
    shots: Vec<(u64, RawSimulationShot)>,
    bytes: usize,
    submitted: Instant,
    calibration_lease: Duration,
    cancel: Arc<AtomicBool>,
}
struct State {
    entries: BTreeMap<JobId, Entry>,
    queue: VecDeque<JobId>,
    shutdown: bool,
}
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}
/// Asynchronous local executor. Drop requests cancellation and joins workers;
/// evaluator callbacks must terminate cooperatively for bounded shutdown.
pub struct LocalSimulationJobs {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    capabilities: JobCapabilities,
    limits: JobServiceLimits,
}
impl LocalSimulationJobs {
    /// Launch a bounded pool around a real local simulator evaluator. The callback
    /// receives immutable request, attempt id, stable seed, and cancellation flag.
    pub fn new<F>(
        capabilities: JobCapabilities,
        limits: JobServiceLimits,
        evaluate: F,
    ) -> Result<Self>
    where
        F: Fn(&SimulationRequest, u64, u64, &AtomicBool) -> Result<RawSimulationShot>
            + Send
            + Sync
            + 'static,
    {
        if capabilities.programs.is_empty()
            || capabilities.programs.iter().any(|p| p.is_empty())
            || capabilities.compiler.is_empty()
            || capabilities.physical_model.is_empty()
            || capabilities.executor.is_empty()
            || capabilities.calibration.is_empty()
            || capabilities.max_attempts == 0
            || capabilities.readouts.is_empty()
            || capabilities.precision_bits.is_empty()
            || capabilities
                .precision_bits
                .iter()
                .any(|p| ![32, 64].contains(p))
            || limits.workers == 0
            || limits.workers > 256
            || limits.queued == 0
            || limits.retained == 0
            || limits.result_bytes == 0
        {
            return Err(Error::InvalidValue(
                "invalid asynchronous job service limits".into(),
            ));
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                entries: BTreeMap::new(),
                queue: VecDeque::new(),
                shutdown: false,
            }),
            wake: Condvar::new(),
        });
        let evaluate = Arc::new(evaluate);
        let mut threads = Vec::with_capacity(limits.workers);
        for _ in 0..limits.workers {
            let shared = shared.clone();
            let evaluate = evaluate.clone();
            threads.push(std::thread::spawn(move || worker(shared, evaluate)));
        }
        Ok(Self {
            shared,
            threads,
            capabilities,
            limits,
        })
    }
    /// Capability negotiation is explicit; this executor is always a local simulator.
    pub fn capabilities(&self) -> &JobCapabilities {
        &self.capabilities
    }
    /// Submit after checking provenance, calibration freshness, readouts and limits.
    /// `now_unix_seconds` is explicit for reproducible tests; providers should use
    /// an authenticated wall clock. Repeating an id with identical content is safe.
    pub fn submit(
        &self,
        id: JobId,
        request: SimulationRequest,
        now_unix_seconds: u64,
    ) -> Result<()> {
        request.identity.validate()?;
        let c = &self.capabilities;
        if !c.programs.contains(&request.identity.program)
            || request.compiler != c.compiler
            || request.physical_model != c.physical_model
            || request.version != 1
            || request.identity.calibration != c.calibration
            || request.identity.executor != c.executor
            || now_unix_seconds >= c.calibration_expires_at
            || request.identity.attempts > c.max_attempts
            || request.bindings.len() > c.max_parameters
            || request.bindings.iter().any(|x| !x.is_finite())
            || request.readouts.is_empty()
            || request.readouts.iter().any(|r| !c.readouts.contains(r))
            || !c.precision_bits.contains(&request.precision_bits)
            || !request.error_budget.is_finite()
            || !(0.0..=1.).contains(&request.error_budget)
            || request.max_result_bytes == 0
            || request.max_result_bytes > self.limits.result_bytes
            || request.timeout_ms == 0
        {
            return Err(Error::InvalidValue(
                "simulation request fails capability/calibration/budget contract".into(),
            ));
        }
        let mut state = self.shared.state.lock().map_err(poison)?;
        if let Some(existing) = state.entries.get(&id) {
            return if existing.request == request {
                Ok(())
            } else {
                Err(Error::InvalidValue(
                    "job id reused with different immutable request".into(),
                ))
            };
        }
        if state.entries.len() >= self.limits.retained || state.queue.len() >= self.limits.queued {
            return Err(Error::ResourceLimit {
                resource: "asynchronous job queue/retention",
                required: state.entries.len() + 1,
                limit: self.limits.retained,
            });
        }
        state.entries.insert(
            id,
            Entry {
                request,
                status: JobStatus::Queued,
                shots: vec![],
                bytes: 0,
                submitted: Instant::now(),
                calibration_lease: Duration::from_secs(c.calibration_expires_at - now_unix_seconds),
                cancel: Arc::new(AtomicBool::new(false)),
            },
        );
        state.queue.push_back(id);
        self.shared.wake.notify_one();
        Ok(())
    }
    /// Read current status without waiting on evaluator execution.
    pub fn status(&self, id: JobId) -> Result<JobStatus> {
        let s = self.shared.state.lock().map_err(poison)?;
        Ok(s.entries.get(&id).ok_or_else(unknown)?.status.clone())
    }
    /// Snapshot raw completed attempts, including terminal partial jobs.
    pub fn result(&self, id: JobId) -> Result<SimulationResult> {
        let s = self.shared.state.lock().map_err(poison)?;
        let e = s.entries.get(&id).ok_or_else(unknown)?;
        Ok(SimulationResult {
            request: e.request.clone(),
            execution_kind: "local_simulator".into(),
            status: e.status.clone(),
            shots: e.shots.clone(),
            encoded_bytes: e.bytes,
            wall_seconds: e.submitted.elapsed().as_secs_f64(),
        })
    }
    /// Cancel a queued job immediately or an active job at the next shot boundary.
    pub fn cancel(&self, id: JobId) -> Result<()> {
        let mut s = self.shared.state.lock().map_err(poison)?;
        let e = s.entries.get_mut(&id).ok_or_else(unknown)?;
        e.cancel.store(true, Ordering::Release);
        if e.status == JobStatus::Queued {
            e.status = JobStatus::Cancelled;
            s.queue.retain(|v| *v != id);
        }
        Ok(())
    }
    /// Release a terminal result so retention limits can admit future jobs.
    pub fn forget(&self, id: JobId) -> Result<()> {
        let mut s = self.shared.state.lock().map_err(poison)?;
        let e = s.entries.get(&id).ok_or_else(unknown)?;
        if matches!(e.status, JobStatus::Queued | JobStatus::Running) {
            return Err(Error::InvalidValue(
                "cannot forget active simulation job".into(),
            ));
        }
        s.entries.remove(&id);
        Ok(())
    }
}
impl Drop for LocalSimulationJobs {
    fn drop(&mut self) {
        if let Ok(mut s) = self.shared.state.lock() {
            s.shutdown = true;
            for e in s.entries.values() {
                e.cancel.store(true, Ordering::Release);
            }
            self.shared.wake.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
fn poison(e: impl std::fmt::Display) -> Error {
    Error::InvalidValue(format!("job service lock: {e}"))
}
fn unknown() -> Error {
    Error::InvalidValue("unknown simulation job".into())
}
fn worker<F>(shared: Arc<Shared>, evaluate: Arc<F>)
where
    F: Fn(&SimulationRequest, u64, u64, &AtomicBool) -> Result<RawSimulationShot>
        + Send
        + Sync
        + 'static,
{
    loop {
        let (id, request, cancel, start, lease) = {
            let Ok(mut s) = shared.state.lock() else {
                return;
            };
            while s.queue.is_empty() && !s.shutdown {
                s = match shared.wake.wait(s) {
                    Ok(v) => v,
                    Err(_) => return,
                };
            }
            if s.shutdown {
                return;
            }
            let id = s.queue.pop_front().unwrap();
            let e = s.entries.get_mut(&id).unwrap();
            e.status = JobStatus::Running;
            (
                id,
                e.request.clone(),
                e.cancel.clone(),
                e.submitted,
                e.calibration_lease,
            )
        };
        let timeout = Duration::from_millis(request.timeout_ms);
        let mut terminal = JobStatus::Completed;
        for attempt in 0..request.identity.attempts {
            if cancel.load(Ordering::Acquire) {
                terminal = JobStatus::Cancelled;
                break;
            }
            if start.elapsed() >= lease {
                terminal =
                    JobStatus::Failed("calibration expired during queued/running job".into());
                break;
            }
            if start.elapsed() >= timeout {
                terminal = JobStatus::TimedOut;
                break;
            }
            let answer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                evaluate(
                    &request,
                    attempt,
                    shot_seed(request.identity.seed, attempt, 0),
                    &cancel,
                )
            }));
            let answer = match answer {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    terminal = JobStatus::Failed(e.to_string());
                    break;
                }
                Err(_) => {
                    terminal = JobStatus::Failed("simulator evaluator panicked".into());
                    break;
                }
            };
            if answer.values.len() != request.readouts.len()
                || answer.values.iter().any(|v| !v.is_finite())
            {
                terminal = JobStatus::Failed("invalid raw simulation readout".into());
                break;
            }
            // Bound raw vector lengths before encoding, then bound exact encoded size.
            if answer.raw.len() > request.max_result_bytes / 8
                || answer.values.len() > request.max_result_bytes / 8
            {
                terminal = JobStatus::Failed("raw simulation record budget".into());
                break;
            }
            let size = match serde_json::to_vec(&(attempt, &answer)) {
                Ok(v) => v.len(),
                Err(e) => {
                    terminal = JobStatus::Failed(e.to_string());
                    break;
                }
            };
            let Ok(mut s) = shared.state.lock() else {
                return;
            };
            let e = s.entries.get_mut(&id).unwrap();
            if size > request.max_result_bytes.saturating_sub(e.bytes) {
                terminal = JobStatus::Failed("retained simulation record budget".into());
                break;
            }
            e.bytes += size;
            e.shots.push((attempt, answer));
            if cancel.load(Ordering::Acquire) {
                terminal = JobStatus::Cancelled;
                break;
            }
            // Keep an actual completed experiment even if its callback crossed the
            // wall deadline. Mark the partial/late result honestly and stop work.
            if start.elapsed() >= lease {
                terminal =
                    JobStatus::Failed("calibration expired during queued/running job".into());
                break;
            }
            if start.elapsed() >= timeout {
                terminal = JobStatus::TimedOut;
                break;
            }
        }
        if let Ok(mut s) = shared.state.lock() {
            let entry = s.entries.get_mut(&id).unwrap();
            // Linearize completion against cancel(), which holds this same lock.
            entry.status =
                if terminal == JobStatus::Completed && entry.cancel.load(Ordering::Acquire) {
                    JobStatus::Cancelled
                } else {
                    terminal
                };
        } else {
            return;
        }
    }
}
