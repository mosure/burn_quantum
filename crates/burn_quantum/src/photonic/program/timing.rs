//! Timed resource availability and classical dependencies for static fusion networks.
use serde::{Deserialize, Serialize};

/// Scheduling policy for fusions whose quantum inputs are disjoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SchedulePolicy {
    /// Preserve original serial execution semantics.
    Serial,
    /// Permit independent fusions on different analyzers to overlap.
    Concurrent,
}
/// Availability and control metadata supplied after factory heralding.
///
/// This schedules a fixed network; conditional topology changes and retries require
/// a new checked program. An exhausted factory must not be represented as a birth.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingOptions {
    /// One birth time per resource in creation order.
    pub resource_births: Vec<u64>,
    /// Earlier fusion indices whose results must be available, one vector per fusion.
    pub dependencies: Vec<Vec<usize>>,
    /// Static concurrency policy.
    pub policy: SchedulePolicy,
    /// Latency after dependency measurement completion before releasing inputs.
    pub feed_forward_ticks: u64,
    /// Analyzer reset occupancy after its one-tick measurement.
    pub analyzer_reset_ticks: u64,
}
impl TimingOptions {
    /// Original zero-birth, serial policy with no additional latency/reset.
    pub fn serial(resources: usize, fusions: usize) -> Self {
        Self {
            resource_births: vec![0; resources],
            dependencies: vec![vec![]; fusions],
            policy: SchedulePolicy::Serial,
            feed_forward_ticks: 0,
            analyzer_reset_ticks: 0,
        }
    }
}
