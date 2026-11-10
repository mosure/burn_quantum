use super::{protocol::*, *};
use crate::tensor_network::state::validate_gauge;
use crate::transport::{self, DeadlineStream};
use std::{
    net::TcpListener,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
/// A contiguous partition, including both open Schmidt indices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardCheckpoint {
    /// Immutable provenance.
    pub identity: StateIdentity,
    /// Global index of the first owned site.
    pub start: usize,
    /// Total sites across all workers.
    pub total: usize,
    /// Last fully committed global generation.
    pub epoch: u64,
    /// Global canonical center.
    pub center: usize,
    /// Global approximation report.
    pub report: TruncationReport,
    /// Contiguous locally owned tensors only.
    pub sites: Vec<SiteData>,
}
/// Durable worker journal, including an uncommitted prepared update and last ACK.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerCheckpoint {
    version: u32,
    state: ShardCheckpoint,
    pending: Option<Update>,
    last: Option<Update>,
}
impl WorkerCheckpoint {
    /// Committed generation, useful to persist journals and monitor recovery.
    pub fn epoch(&self) -> u64 {
        self.state.epoch
    }
    /// Whether a prepared transaction still awaits a commit/abort decision.
    pub fn is_prepared(&self) -> bool {
        self.pending.is_some()
    }
}
/// Single-writer state owner. No arbitrary evaluator/code is accepted remotely.
#[derive(Clone, Debug)]
pub struct ShardWorker {
    checkpoint: WorkerCheckpoint,
    limits: MpsLimits,
}
impl ShardWorker {
    /// Validate initial state and canonical gauge at every owned site.
    pub fn new(state: ShardCheckpoint, limits: MpsLimits) -> Result<Self> {
        validate(&state, limits)?;
        Ok(Self {
            checkpoint: WorkerCheckpoint {
                version: 1,
                state,
                pending: None,
                last: None,
            },
            limits,
        })
    }
    /// Full portable journal for crash recovery. Persist before acknowledging mutations.
    pub fn checkpoint(&self) -> WorkerCheckpoint {
        self.checkpoint.clone()
    }
    /// Recover prepared/committed transactions; persisted data must be authenticated.
    pub fn restore(
        checkpoint: WorkerCheckpoint,
        expected: &StateIdentity,
        limits: MpsLimits,
    ) -> Result<Self> {
        if checkpoint.version != 1 || checkpoint.state.identity != *expected {
            return Err(err("worker checkpoint identity/version differs"));
        }
        validate(&checkpoint.state, limits)?;
        let s = Self { checkpoint, limits };
        if let Some(p) = &s.checkpoint.pending {
            s.candidate(p)?;
        }
        if let Some(last) = &s.checkpoint.last {
            let committed = &s.checkpoint.state;
            if last.next != committed.epoch
                || last.previous.checked_add(1) != Some(last.next)
                || last.center != committed.center
                || last.report != committed.report
            {
                return Err(err("worker last transaction invalid"));
            }
            let mut seen = std::collections::BTreeSet::new();
            for (index, site) in &last.sites {
                site.validate(limits)?;
                if *index < committed.start
                    || !seen.insert(*index)
                    || committed.sites.get(*index - committed.start) != Some(site)
                {
                    return Err(err("worker committed replay cache differs from state"));
                }
            }
        }
        Ok(s)
    }
    fn metadata(&self) -> Metadata {
        let s = &self.checkpoint.state;
        Metadata {
            start: s.start,
            count: s.sites.len(),
            total: s.total,
            epoch: s.epoch,
            center: s.center,
            report: s.report,
            shapes: s.sites.iter().map(|x| x.shape).collect(),
        }
    }
    fn candidate(&self, u: &Update) -> Result<ShardCheckpoint> {
        let s = &self.checkpoint.state;
        if u.previous != s.epoch || u.previous.checked_add(1) != Some(u.next) || u.center >= s.total
        {
            return Err(err("stale or invalid state transaction"));
        }
        let mut c = s.clone();
        let mut seen = std::collections::BTreeSet::new();
        for (i, site) in &u.sites {
            if *i < s.start || *i >= s.start + s.sites.len() || !seen.insert(*i) {
                return Err(err("update crosses ownership or duplicates a site"));
            }
            site.validate(self.limits)?;
            c.sites[*i - s.start] = site.clone();
        }
        c.epoch = u.next;
        c.center = u.center;
        c.report = u.report;
        validate(&c, self.limits)?;
        Ok(c)
    }
    fn handle(&mut self, command: Command) -> Result<Reply> {
        match command {
            Command::Inspect => {
                if self.checkpoint.pending.is_some() {
                    return Err(err("state has prepared transaction; reconcile first"));
                }
                Ok(Reply::Meta(self.metadata()))
            }
            Command::Read { epoch, indices } => {
                let s = &self.checkpoint.state;
                if self.checkpoint.pending.is_some() || epoch != s.epoch {
                    return Err(err("read blocked by stale epoch/prepared transaction"));
                }
                if indices.len() > s.sites.len() {
                    return Err(err("read index budget"));
                }
                let mut out = Vec::with_capacity(indices.len());
                for i in indices {
                    if i < s.start || i >= s.start + s.sites.len() {
                        return Err(err("read crosses ownership"));
                    }
                    out.push((i, s.sites[i - s.start].clone()));
                }
                Ok(Reply::Sites(out))
            }
            Command::Prepare(u) => {
                if let Some(p) = &self.checkpoint.pending {
                    return if *p == u {
                        Ok(Reply::Ack)
                    } else {
                        Err(err("conflicting prepared transaction"))
                    };
                }
                if self.checkpoint.last.as_ref() == Some(&u) {
                    return Ok(Reply::Ack);
                }
                self.candidate(&u)?;
                self.checkpoint.pending = Some(u);
                Ok(Reply::Ack)
            }
            Command::Commit { epoch } => {
                if self
                    .checkpoint
                    .last
                    .as_ref()
                    .is_some_and(|u| u.next == epoch)
                    && self.checkpoint.state.epoch == epoch
                {
                    return Ok(Reply::Ack);
                }
                let u = self
                    .checkpoint
                    .pending
                    .as_ref()
                    .filter(|p| p.next == epoch)
                    .ok_or_else(|| err("no matching prepared transaction"))?
                    .clone();
                self.checkpoint.state = self.candidate(&u)?;
                self.checkpoint.last = Some(u);
                self.checkpoint.pending = None;
                Ok(Reply::Ack)
            }
            Command::Abort { epoch } => {
                if self.checkpoint.state.epoch >= epoch {
                    return Err(err("cannot abort committed generation"));
                }
                if self
                    .checkpoint
                    .pending
                    .as_ref()
                    .is_some_and(|u| u.next != epoch)
                {
                    return Err(err("abort transaction differs"));
                }
                self.checkpoint.pending = None;
                Ok(Reply::Ack)
            }
        }
    }
}
fn validate(s: &ShardCheckpoint, limits: MpsLimits) -> Result<()> {
    limits.validate()?;
    s.identity.validate()?;
    super::super::limit("MPS sites", s.total, limits.max_sites)?;
    if s.sites.is_empty()
        || s.total == 0
        || s.center >= s.total
        || s.start
            .checked_add(s.sites.len())
            .is_none_or(|end| end > s.total)
    {
        return Err(err("invalid shard range"));
    }
    super::super::state::validate_report(s.report, limits)?;
    let mut count = 0usize;
    for (j, site) in s.sites.iter().enumerate() {
        site.validate(limits)?;
        count = count
            .checked_add(site.values.len())
            .ok_or(Error::Overflow)?;
        let i = s.start + j;
        if (i == 0 && site.shape[0] != 1)
            || (i + 1 == s.total && site.shape[2] != 1)
            || (j > 0 && s.sites[j - 1].shape[2] != site.shape[0])
        {
            return Err(err("shard bond mismatch"));
        }
        validate_gauge(site, i.cmp(&s.center), 2e-5)?;
    }
    super::super::limit("MPS shard elements", count, limits.max_state_elements)
}
/// Serve a bounded number of requests. `persist` must durably save the complete
/// candidate checkpoint before returning Ok if crash recovery is required.
/// Persistence failure rolls back in-memory mutation; reads remain available.
/// Socket and shutdown deadlines are cooperative; SVD never runs on a worker.
#[allow(clippy::too_many_arguments)]
pub fn serve_state_worker<P>(
    listener: TcpListener,
    worker: &mut ShardWorker,
    capability: &str,
    network: StateNetworkLimits,
    max_requests: usize,
    stop: &AtomicBool,
    mut persist: P,
) -> Result<usize>
where
    P: FnMut(&WorkerCheckpoint) -> Result<()>,
{
    network.validate()?;
    if capability.is_empty() || capability.len() > 4096 || max_requests == 0 {
        return Err(err("invalid state server limits/capability"));
    }
    listener.set_nonblocking(true).map_err(err)?;
    let end = Instant::now()
        .checked_add(network.deadline)
        .ok_or(Error::Overflow)?;
    let mut served = 0;
    while served < max_requests && !stop.load(Ordering::Relaxed) && Instant::now() < end {
        let (stream, _) = match listener.accept() {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }
            Err(e) => return Err(err(e)),
        };
        let connection_end = Instant::now()
            .checked_add(network.io_timeout)
            .ok_or(Error::Overflow)?
            .min(end);
        let mut stream = DeadlineStream::new(stream, connection_end)?;
        let request = match transport::read::<Request>(&mut stream, network.max_frame_bytes) {
            Ok(r) => r,
            Err(_) => {
                served += 1;
                continue;
            }
        };
        let reply = if request.version != 1
            || request.identity != worker.checkpoint.state.identity
            || request.capability != capability
        {
            Reply::Rejected("identity, capability or version mismatch".into())
        } else {
            let mutation = matches!(
                request.command,
                Command::Prepare(_) | Command::Commit { .. } | Command::Abort { .. }
            );
            let mut candidate = worker.clone();
            match candidate.handle(request.command) {
                Err(e) => Reply::Rejected(e.to_string()),
                Ok(reply) => {
                    if mutation {
                        match persist(&candidate.checkpoint) {
                            Ok(()) => {
                                *worker = candidate;
                                reply
                            }
                            Err(e) => Reply::Rejected(e.to_string()),
                        }
                    } else {
                        reply
                    }
                }
            }
        };
        let response = Response {
            version: 1,
            identity: worker.checkpoint.state.identity.clone(),
            reply,
        };
        // A disconnected client may have an uncertain commit. Retain the journal
        // and keep serving so its idempotent reconciliation can finish.
        let _ = transport::write(&mut stream, &response, network.max_frame_bytes);
        served += 1;
    }
    Ok(served)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn identity() -> StateIdentity {
        StateIdentity {
            program: "unit-test".into(),
            calibration: "ideal".into(),
            run: "recovery".into(),
        }
    }
    fn owner() -> ShardWorker {
        let site = SiteData {
            shape: [1, 2, 1],
            values: vec![1.0.into(), 0.0.into()],
        };
        ShardWorker::new(
            ShardCheckpoint {
                identity: identity(),
                start: 0,
                total: 2,
                epoch: 0,
                center: 0,
                report: Default::default(),
                sites: vec![site.clone(), site],
            },
            Default::default(),
        )
        .unwrap()
    }
    fn update() -> Update {
        Update {
            previous: 0,
            next: 1,
            center: 1,
            report: Default::default(),
            sites: vec![(
                0,
                SiteData {
                    shape: [1, 2, 1],
                    values: vec![0.0.into(), 1.0.into()],
                },
            )],
        }
    }
    #[test]
    fn prepared_restart_blocks_reads_then_commit_replays_exactly() {
        let mut worker = owner();
        let update = update();
        worker.handle(Command::Prepare(update.clone())).unwrap();
        assert!(worker.handle(Command::Inspect).is_err());
        assert!(
            worker
                .handle(Command::Read {
                    epoch: 0,
                    indices: vec![0]
                })
                .is_err()
        );
        let saved = serde_json::to_vec(&worker.checkpoint()).unwrap();
        let cp = serde_json::from_slice(&saved).unwrap();
        let mut resumed = ShardWorker::restore(cp, &identity(), Default::default()).unwrap();
        resumed.handle(Command::Prepare(update)).unwrap();
        resumed.handle(Command::Commit { epoch: 1 }).unwrap();
        let committed = resumed.checkpoint();
        resumed.handle(Command::Commit { epoch: 1 }).unwrap();
        assert_eq!(resumed.checkpoint(), committed);
        assert!(resumed.handle(Command::Abort { epoch: 1 }).is_err());
        assert!(
            resumed
                .handle(Command::Read {
                    epoch: 0,
                    indices: vec![0]
                })
                .is_err()
        );
        let Reply::Sites(v) = resumed
            .handle(Command::Read {
                epoch: 1,
                indices: vec![0],
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(v[0].1.values, vec![0.0.into(), 1.0.into()]);
    }
    #[test]
    fn abort_conflicting_retries_and_checkpoint_validation() {
        let mut worker = owner();
        let before = worker.checkpoint();
        let change = update();
        worker.handle(Command::Prepare(change.clone())).unwrap();
        let mut conflict = change;
        conflict.center = 0;
        assert!(worker.handle(Command::Prepare(conflict)).is_err());
        worker.handle(Command::Abort { epoch: 1 }).unwrap();
        assert_eq!(worker.checkpoint(), before);
        worker.handle(Command::Abort { epoch: 1 }).unwrap();
        let mut corrupted = before;
        corrupted.state.sites[0].values[0] = 2.0.into();
        assert!(ShardWorker::restore(corrupted, &identity(), Default::default()).is_err());
        let mut bad = identity();
        bad.calibration = "new".into();
        assert!(ShardWorker::restore(worker.checkpoint(), &bad, Default::default()).is_err());
        let mut invalid = update();
        invalid.sites[0].0 = 2;
        assert!(worker.handle(Command::Prepare(invalid)).is_err());
    }
    #[test]
    fn committed_replay_cache_must_match_current_tensor_generation() {
        let mut worker = owner();
        worker.handle(Command::Prepare(update())).unwrap();
        worker.handle(Command::Commit { epoch: 1 }).unwrap();
        let committed = worker.checkpoint();
        assert!(ShardWorker::restore(committed.clone(), &identity(), Default::default()).is_ok());
        let mut wrong_tensor = committed.clone();
        wrong_tensor.last.as_mut().unwrap().sites[0].1.values[0] = 2.0.into();
        assert!(ShardWorker::restore(wrong_tensor, &identity(), Default::default()).is_err());
        let mut wrong_center = committed.clone();
        wrong_center.last.as_mut().unwrap().center = 0;
        assert!(ShardWorker::restore(wrong_center, &identity(), Default::default()).is_err());
        let mut wrong_report = committed.clone();
        wrong_report.last.as_mut().unwrap().report.max_norm_drift = 0.01;
        assert!(ShardWorker::restore(wrong_report, &identity(), Default::default()).is_err());
        let mut duplicate = committed;
        let site = duplicate.last.as_ref().unwrap().sites[0].clone();
        duplicate.last.as_mut().unwrap().sites.push(site);
        assert!(ShardWorker::restore(duplicate, &identity(), Default::default()).is_err());
    }
}
