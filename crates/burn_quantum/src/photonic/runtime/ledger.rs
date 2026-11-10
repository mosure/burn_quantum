use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Caller-supplied immutable experiment identity. Fingerprints must cover actual semantics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobIdentity {
    /// Semantic program fingerprint, not a mutable file path.
    pub program: String,
    /// Physical model/calibration fingerprint.
    pub calibration: String,
    /// Compiler and backend version fingerprint.
    pub executor: String,
    /// Global seed; derivation scheme is versioned by the checkpoint schema.
    pub seed: u64,
    /// Total attempted shots, including losses and failures.
    pub attempts: u64,
}
impl JobIdentity {
    pub(super) fn validate(&self) -> Result<()> {
        if [&self.program, &self.calibration, &self.executor]
            .iter()
            .any(|s| s.is_empty() || s.len() > 4096)
        {
            return Err(Error::InvalidValue(
                "job requires bounded, nonempty provenance fingerprints".into(),
            ));
        }
        Ok(())
    }
}
/// One result with a stable global attempt ID.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShotRecord<T> {
    /// Global attempt ID, in the job's half-open attempt range.
    pub id: u64,
    /// Backend-specific outcome, including explicit failures/loss.
    pub value: T,
}
/// Checked append-only result ledger with idempotent retry reconciliation.
#[derive(Clone, Debug)]
pub struct JobLedger<T> {
    identity: JobIdentity,
    max_records: usize,
    results: BTreeMap<u64, T>,
    next_missing: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint<T> {
    version: u32,
    identity: JobIdentity,
    max_records: usize,
    records: Vec<ShotRecord<T>>,
}
impl<T: Clone + PartialEq> JobLedger<T> {
    /// Bound the retained results before accepting a job.
    pub fn new(identity: JobIdentity, max_records: usize) -> Result<Self> {
        identity.validate()?;
        if identity.attempts > max_records as u64 {
            return Err(Error::InvalidValue(
                "job attempts exceed retained-record budget".into(),
            ));
        }
        Ok(Self {
            identity,
            max_records,
            results: BTreeMap::new(),
            next_missing: 0,
        })
    }
    /// Immutable request identity.
    pub fn identity(&self) -> &JobIdentity {
        &self.identity
    }
    /// Completed count, including physically unsuccessful attempts.
    pub fn completed(&self) -> usize {
        self.results.len()
    }
    /// Results ordered by stable shot identity.
    pub fn records(&self) -> impl Iterator<Item = ShotRecord<T>> + '_ {
        self.results.iter().map(|(&id, value)| ShotRecord {
            id,
            value: value.clone(),
        })
    }
    /// Up to `limit` missing IDs; does not allocate a full job-sized range.
    pub fn missing(&self, limit: usize) -> Vec<u64> {
        (self.next_missing..self.identity.attempts)
            .filter(|id| !self.results.contains_key(id))
            .take(limit)
            .collect()
    }
    /// Atomically merge a worker batch. Identical retries are idempotent; conflicts fail.
    pub fn merge(&mut self, identity: JobIdentity, records: Vec<ShotRecord<T>>) -> Result<()> {
        if identity != self.identity {
            return Err(Error::InvalidValue(
                "job/calibration identity mismatch".into(),
            ));
        }
        if records.len() > self.max_records {
            return Err(Error::InvalidValue(
                "worker batch exceeds result budget".into(),
            ));
        }
        let mut pending = BTreeMap::new();
        for record in records {
            if record.id >= self.identity.attempts {
                return Err(Error::InvalidValue("shot ID outside job".into()));
            }
            if self
                .results
                .get(&record.id)
                .or_else(|| pending.get(&record.id))
                .is_some_and(|old| *old != record.value)
            {
                return Err(Error::InvalidValue(
                    "conflicting result for repeated shot ID".into(),
                ));
            }
            pending.insert(record.id, record.value);
        }
        self.results.extend(pending);
        while self.next_missing < self.identity.attempts
            && self.results.contains_key(&self.next_missing)
        {
            self.next_missing += 1;
        }
        Ok(())
    }
}
impl<T: Clone + PartialEq + Serialize> Serialize for JobLedger<T> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        Checkpoint {
            version: 1,
            identity: self.identity.clone(),
            max_records: self.max_records,
            records: self.records().collect(),
        }
        .serialize(serializer)
    }
}
impl<'de, T: Clone + PartialEq + Deserialize<'de>> Deserialize<'de> for JobLedger<T> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let c = Checkpoint::<T>::deserialize(deserializer)?;
        if c.version != 1 {
            return Err(serde::de::Error::custom(
                "unsupported job checkpoint version",
            ));
        }
        let mut job =
            Self::new(c.identity.clone(), c.max_records).map_err(serde::de::Error::custom)?;
        job.merge(c.identity, c.records)
            .map_err(serde::de::Error::custom)?;
        Ok(job)
    }
}
