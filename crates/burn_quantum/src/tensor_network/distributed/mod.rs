//! Distributed ownership of MPS site tensors and versioned boundary updates.
//!
//! Workers retain contiguous tensor partitions; a coordinator transfers only
//! the sites needed by a gate or center move. Every cut retains its Schmidt
//! index. This is exact distributed state ownership, not circuit cutting or
//! independent-shot parallelism. Entanglement still controls communication.
//!
//! A single-writer two-phase journal blocks reads after uncertain commits until
//! reconciliation. It is not a consensus system: recovery needs the coordinator
//! journal and every worker's persisted checkpoint. Use a trusted network or an
//! authenticated tunnel; the capability token is not encryption.
mod coordinator;
mod protocol;
mod worker;
use super::{MpsLimits, SiteData, TruncationReport};
pub use crate::photonic::runtime::network::WorkerEndpoint;
use crate::{Error, Result};
pub use coordinator::{CoordinatorCheckpoint, DistributedMps, TransferStatistics};
use serde::{Deserialize, Serialize};
use std::time::Duration;
pub use worker::{ShardCheckpoint, ShardWorker, WorkerCheckpoint, serve_state_worker};

/// Immutable execution provenance, checked on every request and restore.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateIdentity {
    /// Caller-supplied program content fingerprint.
    pub program: String,
    /// Caller-supplied immutable model/calibration fingerprint.
    pub calibration: String,
    /// Caller-chosen run identity, preventing accidental cross-job replay.
    pub run: String,
}
impl StateIdentity {
    fn validate(&self) -> Result<()> {
        if [&self.program, &self.calibration, &self.run]
            .iter()
            .any(|s| s.is_empty() || s.len() > 4096)
        {
            return Err(Error::InvalidValue(
                "invalid distributed state identity".into(),
            ));
        }
        Ok(())
    }
}
/// Explicit local/TCP resource limits and cooperative deadlines.
#[derive(Clone, Copy, Debug)]
pub struct StateNetworkLimits {
    /// Maximum serialized request or response (also bounds checkpoint transfer).
    pub max_frame_bytes: usize,
    /// Connection plus absolute read/write budget for one RPC.
    pub io_timeout: Duration,
    /// Server invocation lifetime, including idle time.
    pub deadline: Duration,
    /// Upper bound on state-owning workers.
    pub max_workers: usize,
}
impl Default for StateNetworkLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 8 << 20,
            io_timeout: Duration::from_secs(15),
            deadline: Duration::from_secs(300),
            max_workers: 128,
        }
    }
}
impl StateNetworkLimits {
    fn validate(&self) -> Result<()> {
        if !(256..=64 << 20).contains(&self.max_frame_bytes)
            || self.io_timeout.is_zero()
            || self.deadline.is_zero()
            || self.max_workers == 0
        {
            return Err(Error::InvalidValue(
                "invalid distributed state network limits".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Metadata {
    pub start: usize,
    pub count: usize,
    pub total: usize,
    pub epoch: u64,
    pub center: usize,
    pub report: TruncationReport,
    pub shapes: Vec<[usize; 3]>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Update {
    pub previous: u64,
    pub next: u64,
    pub center: usize,
    pub report: TruncationReport,
    pub sites: Vec<(usize, SiteData)>,
}
fn err(e: impl std::fmt::Display) -> Error {
    Error::InvalidValue(format!("MPS state transport: {e}"))
}
