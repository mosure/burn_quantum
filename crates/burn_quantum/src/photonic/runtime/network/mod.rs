//! Versioned bounded TCP worker protocol for deterministic shot jobs.
//!
//! Transport is plain TCP for trusted networks or an authenticated encrypted tunnel.
//! The capability string authorizes a configured worker; it is not transport encryption.
//! Workers execute a preconfigured callback, never shell commands received over the wire.
use super::{JobIdentity, JobLedger, ShotRecord, shot_seed};
use crate::transport::{self as wire, DeadlineStream};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Explicit connection, frame, batch and aggregate retention limits.
#[derive(Clone, Copy, Debug)]
pub struct NetworkLimits {
    /// Maximum encoded request/reply length, excluding the length prefix.
    pub max_frame_bytes: usize,
    /// Maximum attempted shots in one network request.
    pub max_batch_shots: usize,
    /// Timeout for one connection's socket reads/writes.
    pub io_timeout: Duration,
    /// Total wall-time budget for one execute/serve invocation, checked cooperatively.
    pub deadline: Duration,
    /// Maximum retained payload bytes in the coordinator ledger.
    pub max_result_bytes: usize,
    /// Retry budget for a failed wave, including its first attempt.
    pub wave_attempts: usize,
}
impl Default for NetworkLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 1 << 20,
            max_batch_shots: 1024,
            io_timeout: Duration::from_secs(10),
            deadline: Duration::from_secs(300),
            max_result_bytes: 1 << 26,
            wave_attempts: 2,
        }
    }
}
impl NetworkLimits {
    fn validate(self) -> Result<()> {
        if !(128..=1 << 26).contains(&self.max_frame_bytes)
            || self.max_batch_shots == 0
            || self.max_batch_shots > 1_048_576
            || self.io_timeout.is_zero()
            || self.deadline.is_zero()
            || self.wave_attempts == 0
            || self.wave_attempts > 32
        {
            return Err(Error::InvalidValue("invalid network limits".into()));
        }
        Ok(())
    }
}
/// Explicit remote worker and its configured authorization capability.
#[derive(Clone)]
pub struct WorkerEndpoint {
    /// Resolved address; no implicit discovery or fallback.
    pub address: SocketAddr,
    /// Opaque caller-supplied capability, sent only over the selected trusted transport.
    pub capability: String,
}
impl std::fmt::Debug for WorkerEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerEndpoint")
            .field("address", &self.address)
            .field("capability", &"[redacted]")
            .finish()
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u32,
    capability: String,
    identity: JobIdentity,
    ids: Vec<u64>,
    payload: Vec<u8>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    version: u32,
    identity: JobIdentity,
    records: Vec<ShotRecord<Vec<u8>>>,
    error: Option<String>,
}

fn io(error: impl std::fmt::Display) -> Error {
    Error::InvalidValue(format!("shot transport: {error}"))
}
fn capability_valid(value: &str) -> bool {
    !value.is_empty() && value.len() <= 4096
}

/// Serve a bounded number of requests for one immutable job/model configuration.
///
/// Shutdown and deadlines are cooperative between requests and shots. A running callback is
/// not forcibly interrupted. Replayed requests repeat the same deterministic shots.
/// Callbacks receive `(shot_id, derived_seed, payload)` and must validate their payload.
/// Invalid requests are rejected without calling the evaluator. The supplied listener
/// controls exposure; the example/test listeners bind only loopback ephemeral ports.
pub fn serve_worker<F>(
    listener: TcpListener,
    expected: JobIdentity,
    capability: &str,
    limits: NetworkLimits,
    max_requests: usize,
    stop: &AtomicBool,
    evaluate: F,
) -> Result<usize>
where
    F: Fn(u64, u64, &[u8]) -> Result<Vec<u8>>,
{
    limits.validate()?;
    expected.validate()?;
    if !capability_valid(capability) || max_requests == 0 {
        return Err(Error::InvalidValue(
            "invalid worker capability/request budget".into(),
        ));
    }
    listener.set_nonblocking(true).map_err(io)?;
    let started = Instant::now();
    let mut completed = 0;
    while completed < max_requests
        && !stop.load(Ordering::Acquire)
        && started.elapsed() < limits.deadline
    {
        let (stream, _) = match listener.accept() {
            Ok(value) => value,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            Err(e) => return Err(io(e)),
        };
        let end = Instant::now()
            .checked_add(
                limits
                    .io_timeout
                    .min(limits.deadline.saturating_sub(started.elapsed())),
            )
            .ok_or(Error::Overflow)?;
        let mut stream = DeadlineStream::new(stream, end)?;
        completed += 1;
        // Malformed/disconnected requests consume a request slot and cannot reach evaluation.
        let request: Request = match wire::read(&mut stream, limits.max_frame_bytes) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let result = evaluate_request(
            &request, &expected, capability, limits, end, stop, &evaluate,
        );
        let response = match result {
            Ok(records) => Response {
                version: 1,
                identity: expected.clone(),
                records,
                error: None,
            },
            Err(_) => Response {
                version: 1,
                identity: expected.clone(),
                records: vec![],
                error: Some("worker rejected request or evaluation failed".into()),
            },
        };
        // Client cancellation/disconnection must not crash the service.
        let _ = wire::write(&mut stream, &response, limits.max_frame_bytes);
    }
    Ok(completed)
}
fn evaluate_request<F>(
    r: &Request,
    expected: &JobIdentity,
    capability: &str,
    limits: NetworkLimits,
    end: Instant,
    stop: &AtomicBool,
    evaluate: &F,
) -> Result<Vec<ShotRecord<Vec<u8>>>>
where
    F: Fn(u64, u64, &[u8]) -> Result<Vec<u8>>,
{
    if r.version != 1
        || r.capability != capability
        || &r.identity != expected
        || r.ids.is_empty()
        || r.ids.len() > limits.max_batch_shots
        || r.ids.iter().any(|&i| i >= expected.attempts)
    {
        return Err(Error::InvalidValue(
            "request capability/model/shot mismatch".into(),
        ));
    }
    let mut sorted = r.ids.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|x| x[0] == x[1]) {
        return Err(Error::InvalidValue("duplicate requested shot".into()));
    }
    let mut bytes = 0usize;
    let mut records = Vec::with_capacity(r.ids.len());
    for &id in &r.ids {
        if stop.load(Ordering::Acquire) || Instant::now() >= end {
            return Err(Error::InvalidValue(
                "worker batch cancelled or deadline exhausted".into(),
            ));
        }
        let value = evaluate(id, shot_seed(expected.seed, id, 0), &r.payload)?;
        bytes = bytes.checked_add(value.len()).ok_or(Error::Overflow)?;
        if bytes > limits.max_frame_bytes {
            return Err(Error::InvalidValue("worker result payload limit".into()));
        }
        records.push(ShotRecord { id, value });
    }
    Ok(records)
}
/// Submit one validated batch and reject incomplete, duplicate or foreign replies.
pub fn request_batch(
    endpoint: &WorkerEndpoint,
    identity: &JobIdentity,
    ids: &[u64],
    payload: &[u8],
    limits: NetworkLimits,
) -> Result<Vec<ShotRecord<Vec<u8>>>> {
    limits.validate()?;
    identity.validate()?;
    if !capability_valid(&endpoint.capability)
        || ids.is_empty()
        || ids.len() > limits.max_batch_shots
        || payload.len() > limits.max_frame_bytes
        || ids.iter().any(|&id| id >= identity.attempts)
    {
        return Err(Error::InvalidValue("invalid request budget/IDs".into()));
    }
    let mut expected = ids.to_vec();
    expected.sort_unstable();
    if expected.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::InvalidValue("duplicate request IDs".into()));
    }
    let end = Instant::now()
        .checked_add(limits.io_timeout)
        .ok_or(Error::Overflow)?;
    let stream = TcpStream::connect_timeout(&endpoint.address, limits.io_timeout).map_err(io)?;
    let mut stream = DeadlineStream::new(stream, end)?;
    wire::write(
        &mut stream,
        &Request {
            version: 1,
            capability: endpoint.capability.clone(),
            identity: identity.clone(),
            ids: ids.to_vec(),
            payload: payload.to_vec(),
        },
        limits.max_frame_bytes,
    )?;
    let response: Response = wire::read(&mut stream, limits.max_frame_bytes)?;
    let mut received: Vec<_> = response.records.iter().map(|r| r.id).collect();
    received.sort_unstable();
    if response.version != 1
        || &response.identity != identity
        || response.error.is_some()
        || received != expected
    {
        return Err(Error::InvalidValue(
            "incomplete or mismatched worker reply".into(),
        ));
    }
    Ok(response.records)
}
/// Execute missing shots across explicit worker endpoints with atomic waves and bounded retries.
///
/// Same job/shot IDs derive identical seeds across processes. On exhausted retries,
/// completed prior waves remain; the failed wave can be resumed from its checkpoint.
/// Endpoints must host the exact job identity, calibration and deterministic callback.
pub fn execute_remote(
    ledger: &mut JobLedger<Vec<u8>>,
    endpoints: &[WorkerEndpoint],
    payload: &[u8],
    limits: NetworkLimits,
    cancelled: &AtomicBool,
) -> Result<()> {
    limits.validate()?;
    if endpoints.is_empty() || endpoints.len() > 256 {
        return Err(Error::InvalidValue("invalid worker count".into()));
    }
    let started = Instant::now();
    let mut retained = ledger.records().try_fold(0usize, |n, r| {
        n.checked_add(r.value.len()).ok_or(Error::Overflow)
    })?;
    if retained > limits.max_result_bytes {
        return Err(Error::InvalidValue("retained result budget".into()));
    }
    let max_wave = limits
        .max_batch_shots
        .checked_mul(endpoints.len())
        .ok_or(Error::Overflow)?;
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
        if started.elapsed() >= limits.deadline {
            return Err(Error::InvalidValue("remote job deadline exhausted".into()));
        }
        let ids = ledger.missing(max_wave);
        if ids.is_empty() {
            return Ok(());
        }
        let mut result = None;
        for _ in 0..limits.wave_attempts {
            if cancelled.load(Ordering::Acquire) {
                return Ok(());
            }
            let remaining = limits.deadline.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            let limits = NetworkLimits {
                io_timeout: limits.io_timeout.min(remaining),
                ..limits
            };
            let reply: Result<Vec<ShotRecord<Vec<u8>>>> = std::thread::scope(|scope| {
                let handles: Vec<_> = ids
                    .chunks(limits.max_batch_shots)
                    .zip(endpoints)
                    .map(|(chunk, endpoint)| {
                        let identity = ledger.identity();
                        scope.spawn(move || {
                            request_batch(endpoint, identity, chunk, payload, limits)
                        })
                    })
                    .collect();
                let joined: Vec<_> = handles.into_iter().map(|h| h.join()).collect();
                let mut records = Vec::with_capacity(ids.len());
                for r in joined {
                    records.extend(r.map_err(|_| {
                        Error::InvalidValue("network coordinator worker panic".into())
                    })??);
                }
                Ok(records)
            });
            match reply {
                Ok(records) => {
                    result = Some(records);
                    break;
                }
                Err(_) => continue,
            }
        }
        let records = result.ok_or_else(|| {
            Error::InvalidValue("worker wave failed or timed out; checkpoint can resume".into())
        })?;
        let extra = records.iter().try_fold(0usize, |n, r| {
            n.checked_add(r.value.len()).ok_or(Error::Overflow)
        })?;
        let next = retained.checked_add(extra).ok_or(Error::Overflow)?;
        if next > limits.max_result_bytes {
            return Err(Error::InvalidValue("aggregate result payload limit".into()));
        }
        ledger.merge(ledger.identity().clone(), records)?;
        retained = next;
    }
}
