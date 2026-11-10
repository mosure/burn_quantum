use super::{protocol::*, *};
use crate::tensor_network::{
    MpsCheckpoint, MpsState,
    tensor::{data, environment_step, from, mm, move_left, move_right, reshape},
};
use crate::{ComplexTensor, Matrix, Unitary};
use burn::tensor::{DType, Device};
use num_complex::Complex64;
/// Actual serialized payload and RPC counts, excluding TCP/IP framing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransferStatistics {
    /// Successfully decoded request/reply payload bytes, including length prefixes.
    pub bytes: usize,
    /// Successfully acknowledged remote calls.
    pub calls: usize,
    /// Largest number of tensor sites held temporarily for a gate/center move.
    pub peak_gate_sites: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Decision {
    Preparing,
    Commit,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Journal {
    decision: Decision,
    updates: Vec<Update>,
}
/// Coordinator metadata/journal. Contains at most the current edited sites,
/// never a copy of the complete distributed state. Authenticate persisted data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinatorCheckpoint {
    version: u32,
    identity: StateIdentity,
    metadata: Vec<Metadata>,
    journal: Option<Journal>,
    statistics: TransferStatistics,
}
/// Single-writer coordinator with disjoint ownership, preserved Schmidt cuts,
/// and recoverable two-phase boundary updates. No consensus/automatic failover.
pub struct DistributedMps {
    checkpoint: CoordinatorCheckpoint,
    endpoints: Vec<WorkerEndpoint>,
    device: Device,
    dtype: DType,
    limits: MpsLimits,
    network: StateNetworkLimits,
    // Restored approximate local gauges require a global streaming norm check.
    normalization_pending: bool,
}
impl DistributedMps {
    /// Attach to initialized owners; verify contiguous partitions, canonical
    /// center, global generation and every inter-worker Schmidt dimension.
    pub fn connect(
        identity: StateIdentity,
        endpoints: Vec<WorkerEndpoint>,
        device: &Device,
        dtype: DType,
        limits: MpsLimits,
        network: StateNetworkLimits,
    ) -> Result<Self> {
        identity.validate()?;
        limits.validate()?;
        network.validate()?;
        crate::backend::supported_dtype(dtype)?;
        super::super::limit("MPS workers", endpoints.len(), network.max_workers)?;
        let mut metadata = Vec::new();
        let mut statistics = TransferStatistics::default();
        for endpoint in &endpoints {
            let (reply, bytes) = call(endpoint, &identity, Command::Inspect, network)?;
            let Reply::Meta(meta) = reply else {
                return Err(err("expected worker metadata"));
            };
            metadata.push(meta);
            statistics.bytes += bytes;
            statistics.calls += 1;
        }
        validate_metadata(&metadata, limits)?;
        let mut state = Self {
            checkpoint: CoordinatorCheckpoint {
                version: 1,
                identity,
                metadata,
                journal: None,
                statistics,
            },
            endpoints,
            device: device.clone(),
            dtype,
            limits,
            network,
            normalization_pending: true,
        };
        state.validate_normalization()?;
        Ok(state)
    }
    /// Restore a durable coordinator journal, then call `reconcile` before reads.
    /// Endpoints must be in the same ownership order as the original job. Without
    /// an outstanding journal, restoration immediately streams a global norm check
    /// from the live owners; pending journals defer that check until reconciliation.
    pub fn restore(
        checkpoint: CoordinatorCheckpoint,
        expected: &StateIdentity,
        endpoints: Vec<WorkerEndpoint>,
        device: &Device,
        dtype: DType,
        limits: MpsLimits,
        network: StateNetworkLimits,
    ) -> Result<Self> {
        limits.validate()?;
        network.validate()?;
        expected.validate()?;
        crate::backend::supported_dtype(dtype)?;
        if checkpoint.version != 1
            || checkpoint.identity != *expected
            || checkpoint.metadata.len() != endpoints.len()
        {
            return Err(err("coordinator checkpoint identity/owners differ"));
        }
        super::super::limit("MPS workers", endpoints.len(), network.max_workers)?;
        validate_metadata(&checkpoint.metadata, limits)?;
        if let Some(journal) = &checkpoint.journal {
            if journal.updates.len() != endpoints.len() {
                return Err(err("journal owner count differs"));
            }
            let first = &journal.updates[0];
            let mut next = checkpoint.metadata.clone();
            for (i, u) in journal.updates.iter().enumerate() {
                if u.previous != checkpoint.metadata[i].epoch
                    || u.previous.checked_add(1) != Some(u.next)
                    || u.center != first.center
                    || u.report != first.report
                {
                    return Err(err("journal generations differ"));
                }
                let mut seen = std::collections::BTreeSet::new();
                for (index, site) in &u.sites {
                    site.validate(limits)?;
                    let m = &mut next[i];
                    if *index < m.start || *index >= m.start + m.count || !seen.insert(*index) {
                        return Err(err("journal ownership differs"));
                    }
                    m.shapes[*index - m.start] = site.shape;
                }
                next[i].epoch = u.next;
                next[i].center = u.center;
                next[i].report = u.report;
            }
            validate_metadata(&next, limits)?;
        }
        let mut state = Self {
            checkpoint,
            endpoints,
            device: device.clone(),
            dtype,
            limits,
            network,
            normalization_pending: true,
        };
        if state.checkpoint.journal.is_none() {
            state.validate_normalization()?;
        }
        Ok(state)
    }
    /// Durable, credential-free metadata and outstanding decision.
    pub fn checkpoint(&self) -> CoordinatorCheckpoint {
        self.checkpoint.clone()
    }
    /// Communication diagnostics.
    pub fn statistics(&self) -> TransferStatistics {
        self.checkpoint.statistics
    }
    /// Global committed generation as last reconciled by this coordinator.
    pub fn epoch(&self) -> u64 {
        self.checkpoint.metadata[0].epoch
    }
    /// Whether an interrupted transaction or restored-state norm check blocks access.
    pub fn needs_reconciliation(&self) -> bool {
        self.checkpoint.journal.is_some() || self.normalization_pending
    }
    fn ready(&self) -> Result<()> {
        if self.needs_reconciliation() {
            Err(err("reconcile pending distributed state transaction first"))
        } else {
            Ok(())
        }
    }
    fn rpc(&mut self, owner: usize, command: Command) -> Result<Reply> {
        let (reply, bytes) = call(
            &self.endpoints[owner],
            &self.checkpoint.identity,
            command,
            self.network,
        )?;
        self.checkpoint.statistics.bytes = self
            .checkpoint
            .statistics
            .bytes
            .checked_add(bytes)
            .ok_or(Error::Overflow)?;
        self.checkpoint.statistics.calls = self
            .checkpoint
            .statistics
            .calls
            .checked_add(1)
            .ok_or(Error::Overflow)?;
        Ok(reply)
    }
    fn owner(&self, site: usize) -> Result<usize> {
        self.checkpoint
            .metadata
            .iter()
            .position(|m| site >= m.start && site < m.start + m.count)
            .ok_or_else(|| err("site outside distributed chain"))
    }
    fn shape(&self, index: usize) -> Result<[usize; 3]> {
        let owner = self.owner(index)?;
        let metadata = &self.checkpoint.metadata[owner];
        Ok(metadata.shapes[index - metadata.start])
    }
    fn read_site(&mut self, index: usize) -> Result<SiteData> {
        self.ready()?;
        self.read_site_unchecked(index)
    }
    // Used only after transaction readiness or during a global restore check.
    fn read_site_unchecked(&mut self, index: usize) -> Result<SiteData> {
        let owner = self.owner(index)?;
        let reply = self.rpc(
            owner,
            Command::Read {
                epoch: self.epoch(),
                indices: vec![index],
            },
        )?;
        let Reply::Sites(mut sites) = reply else {
            return Err(err("expected site reply"));
        };
        if sites.len() != 1 || sites[0].0 != index {
            return Err(err("site reply identity differs"));
        }
        let (_, site) = sites.remove(0);
        site.validate(self.limits)?;
        let m = &self.checkpoint.metadata[owner];
        if site.shape != m.shapes[index - m.start] {
            return Err(err("site shape differs from committed metadata"));
        }
        Ok(site)
    }
    /// Resolve a persisted decision idempotently. Before COMMIT, recovery aborts
    /// all prepares; after the durable COMMIT decision, recovery commits all owners.
    /// A missing worker leaves the journal pending and further reads blocked.
    pub fn reconcile<P>(&mut self, persist: &mut P) -> Result<()>
    where
        P: FnMut(&CoordinatorCheckpoint) -> Result<()>,
    {
        let Some(journal) = self.checkpoint.journal.clone() else {
            return if self.normalization_pending {
                self.validate_normalization()
            } else {
                Ok(())
            };
        };
        // A prior persistence call can have written the commit decision and then
        // failed. Never turn that uncertainty into an abort: first establish a
        // successfully durable COMMIT before any worker can receive Commit.
        if journal.decision == Decision::Commit {
            persist(&self.checkpoint)?;
        }
        for (i, u) in journal.updates.iter().enumerate() {
            let command = match journal.decision {
                Decision::Preparing => Command::Abort { epoch: u.next },
                Decision::Commit => Command::Commit { epoch: u.next },
            };
            if !matches!(self.rpc(i, command)?, Reply::Ack) {
                return Err(err("transaction acknowledgement differs"));
            }
        }
        let mut finished = self.checkpoint.clone();
        if journal.decision == Decision::Commit {
            for (i, u) in journal.updates.iter().enumerate() {
                let meta = &mut finished.metadata[i];
                for (index, site) in &u.sites {
                    meta.shapes[*index - meta.start] = site.shape;
                }
                meta.epoch = u.next;
                meta.center = u.center;
                meta.report = u.report;
            }
        }
        finished.journal = None;
        persist(&finished)?;
        self.checkpoint = finished;
        if self.normalization_pending {
            self.validate_normalization()?;
        }
        Ok(())
    }
    fn transaction<P>(
        &mut self,
        sites: Vec<(usize, SiteData)>,
        center: usize,
        report: TruncationReport,
        persist: &mut P,
    ) -> Result<()>
    where
        P: FnMut(&CoordinatorCheckpoint) -> Result<()>,
    {
        self.ready()?;
        let epoch = self.epoch();
        let next = epoch.checked_add(1).ok_or(Error::Overflow)?;
        let mut updates = vec![
            Update {
                previous: epoch,
                next,
                center,
                report,
                sites: Vec::new()
            };
            self.endpoints.len()
        ];
        for (i, site) in sites {
            let owner = self.owner(i)?;
            updates[owner].sites.push((i, site));
        }
        let mut candidate = self.checkpoint.clone();
        candidate.journal = Some(Journal {
            decision: Decision::Preparing,
            updates: updates.clone(),
        });
        // Reuse restore's full journal/bond/limit checks before contacting a worker.
        Self::restore(
            candidate.clone(),
            &candidate.identity,
            self.endpoints.clone(),
            &self.device,
            self.dtype,
            self.limits,
            self.network,
        )?;
        persist(&candidate)?;
        self.checkpoint = candidate;
        for (i, u) in updates.iter().enumerate() {
            if !matches!(self.rpc(i, Command::Prepare(u.clone()))?, Reply::Ack) {
                return Err(err("prepare acknowledgement differs"));
            }
        }
        let mut decision = self.checkpoint.clone();
        decision.journal.as_mut().unwrap().decision = Decision::Commit;
        self.checkpoint = decision;
        persist(&self.checkpoint)?;
        self.reconcile(persist)
    }
    fn move_center<P>(&mut self, target: usize, persist: &mut P) -> Result<()>
    where
        P: FnMut(&CoordinatorCheckpoint) -> Result<()>,
    {
        self.ready()?;
        self.owner(target)?;
        while self.checkpoint.metadata[0].center != target {
            let center = self.checkpoint.metadata[0].center;
            let right = center < target;
            let i = if right { center } else { center - 1 };
            let a = self
                .read_site(i)?
                .tensor(&self.device, self.dtype, self.limits)?;
            let b = self
                .read_site(i + 1)?
                .tensor(&self.device, self.dtype, self.limits)?;
            self.checkpoint.statistics.peak_gate_sites =
                self.checkpoint.statistics.peak_gate_sites.max(2);
            let (a, b) = if right {
                move_right(&a, &b, self.limits)?
            } else {
                move_left(&a, &b, self.limits)?
            };
            self.transaction(
                vec![(i, SiteData::read(&a)?), (i + 1, SiteData::read(&b)?)],
                if right { i + 1 } else { i },
                self.checkpoint.metadata[0].report,
                persist,
            )?;
        }
        Ok(())
    }
    /// Apply an adjacent two-site unitary. Only two site tensors cross the wire;
    /// canonical-center moves also exchange two sites, preserving all cut bonds.
    /// `persist` is called before prepare, before commit, and after reconciliation.
    pub fn apply_adjacent<P>(&mut self, left: usize, gate: &Unitary, persist: &mut P) -> Result<()>
    where
        P: FnMut(&CoordinatorCheckpoint) -> Result<()>,
    {
        self.ready()?;
        self.owner(left)?;
        self.owner(left.checked_add(1).ok_or(Error::Overflow)?)?;
        let dimension = self.shape(left)?[1]
            .checked_mul(self.shape(left + 1)?[1])
            .ok_or(Error::Overflow)?;
        if dimension != gate.matrix().dimension() {
            return Err(err("distributed gate dimension differs"));
        }
        self.limits
            .tensor(crate::error::checked_product([dimension, dimension])?)?;
        self.move_center(left, persist)?;
        let a = self
            .read_site(left)?
            .tensor(&self.device, self.dtype, self.limits)?;
        let b = self
            .read_site(left + 1)?
            .tensor(&self.device, self.dtype, self.limits)?;
        if a.dims()[1].checked_mul(b.dims()[1]) != Some(gate.matrix().dimension()) {
            return Err(err("distributed gate dimension differs"));
        }
        let mut local = MpsState {
            sites: vec![a, b],
            center: 0,
            limits: self.limits,
            report: self.checkpoint.metadata[0].report,
        };
        local.block(0, 2, gate.matrix(), None)?;
        self.checkpoint.statistics.peak_gate_sites =
            self.checkpoint.statistics.peak_gate_sites.max(2);
        self.transaction(
            vec![
                (left, SiteData::read(&local.sites[0])?),
                (left + 1, SiteData::read(&local.sites[1])?),
            ],
            left + 1,
            local.report,
            persist,
        )
    }
    /// Apply a one-site unitary after moving the canonical center exactly.
    pub fn apply_single<P>(&mut self, site: usize, gate: &Unitary, persist: &mut P) -> Result<()>
    where
        P: FnMut(&CoordinatorCheckpoint) -> Result<()>,
    {
        self.ready()?;
        let dimension = self.shape(site)?[1];
        if dimension != gate.matrix().dimension() {
            return Err(err("distributed gate dimension differs"));
        }
        self.limits
            .tensor(crate::error::checked_product([dimension, dimension])?)?;
        self.move_center(site, persist)?;
        let a = self
            .read_site(site)?
            .tensor(&self.device, self.dtype, self.limits)?;
        if a.dims()[1] != gate.matrix().dimension() {
            return Err(err("distributed gate dimension differs"));
        }
        let mut local = MpsState {
            sites: vec![a],
            center: 0,
            limits: self.limits,
            report: self.checkpoint.metadata[0].report,
        };
        local.block(0, 1, gate.matrix(), None)?;
        self.checkpoint.statistics.peak_gate_sites =
            self.checkpoint.statistics.peak_gate_sites.max(1);
        self.transaction(
            vec![(site, SiteData::read(&local.sites[0])?)],
            site,
            local.report,
            persist,
        )
    }
    /// Stream a product-operator expectation through owners, retaining one site
    /// and a bond-sized environment. No complete state is assembled.
    pub fn expectation_product(&mut self, operators: &[(usize, Matrix)]) -> Result<Complex64> {
        self.ready()?;
        self.expectation_product_unchecked(operators)
    }
    fn validate_normalization(&mut self) -> Result<()> {
        if self.checkpoint.journal.is_some() {
            return Err(err("reconcile before checking global norm"));
        }
        super::super::state::validate_norm(self.expectation_product_unchecked(&[])?)?;
        self.normalization_pending = false;
        Ok(())
    }
    fn expectation_product_unchecked(
        &mut self,
        operators: &[(usize, Matrix)],
    ) -> Result<Complex64> {
        for (i, (index, matrix)) in operators.iter().enumerate() {
            self.limits.tensor(crate::error::checked_product([
                matrix.dimension(),
                matrix.dimension(),
            ])?)?;
            self.owner(*index)?;
            if operators[..i].iter().any(|(j, _)| j == index) {
                return Err(err("duplicate product operator"));
            }
        }
        let mut env = from(&[1.0.into()], [1, 1], &self.device, self.dtype)?;
        for i in 0..self.checkpoint.metadata[0].total {
            let site =
                self.read_site_unchecked(i)?
                    .tensor(&self.device, self.dtype, self.limits)?;
            let matrix = operators
                .iter()
                .find(|(index, _)| *index == i)
                .map(|(_, m)| m);
            env = environment_step(&env, &site, matrix, self.limits)?;
        }
        Ok(data(&env)?[0])
    }
    /// Stream a basis amplitude, retaining a single Schmidt-boundary vector.
    pub fn amplitude(&mut self, digits: &[usize]) -> Result<Complex64> {
        self.ready()?;
        if digits.len() != self.checkpoint.metadata[0].total {
            return Err(err("basis digits differ"));
        }
        let mut env = from(&[1.0.into()], [1, 1], &self.device, self.dtype)?;
        for (i, &digit) in digits.iter().enumerate() {
            let site = self
                .read_site(i)?
                .tensor(&self.device, self.dtype, self.limits)?;
            let [l, d, r] = site.dims();
            if digit >= d {
                return Err(Error::Basis);
            }
            let selected = ComplexTensor {
                re: site.re.slice_dim(1, digit..digit + 1),
                im: site.im.slice_dim(1, digit..digit + 1),
            };
            env = mm(&env, &reshape(&selected, [l, r]));
        }
        Ok(data(&env)?[0])
    }
    /// Explicit complete export for checkpoint archival or bounded dense oracles.
    /// This method, unlike gates/readouts, intentionally gathers all site tensors.
    pub fn export(&mut self) -> Result<MpsCheckpoint> {
        self.ready()?;
        let mut sites = Vec::new();
        for i in 0..self.checkpoint.metadata[0].total {
            sites.push(self.read_site(i)?);
        }
        let cp = MpsCheckpoint {
            version: 1,
            sites,
            center: self.checkpoint.metadata[0].center,
            report: self.checkpoint.metadata[0].report,
        };
        MpsState::restore(&cp, &self.device, self.dtype, self.limits)?;
        Ok(cp)
    }
}
fn validate_metadata(metadata: &[Metadata], limits: MpsLimits) -> Result<()> {
    if metadata.is_empty() {
        return Err(err("no state owners"));
    }
    let first = &metadata[0];
    if first.total == 0 || first.center >= first.total {
        return Err(err("invalid global center/count"));
    }
    super::super::limit("MPS sites", first.total, limits.max_sites)?;
    super::super::state::validate_report(first.report, limits)?;
    let mut next = 0usize;
    let mut bond = 1;
    let mut elements = 0usize;
    for m in metadata {
        if m.start != next
            || m.count == 0
            || m.shapes.len() != m.count
            || m.total != first.total
            || m.epoch != first.epoch
            || m.center != first.center
            || m.report != first.report
        {
            return Err(err("workers disagree on generation/layout"));
        }
        for &[l, d, r] in &m.shapes {
            if l != bond || d < 2 || l == 0 || r == 0 {
                return Err(err("worker Schmidt boundary differs"));
            }
            super::super::limit("MPS bond", l.max(r), limits.max_bond)?;
            let n = crate::error::checked_product([l, d, r])?;
            limits.tensor(n)?;
            elements = elements.checked_add(n).ok_or(Error::Overflow)?;
            bond = r;
        }
        next = next.checked_add(m.count).ok_or(Error::Overflow)?;
    }
    if next != first.total || bond != 1 {
        return Err(err("workers do not cover an open chain"));
    }
    super::super::limit("MPS distributed state", elements, limits.max_state_elements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_network::distributed::{ShardCheckpoint, ShardWorker, serve_state_worker};
    use std::{
        net::TcpListener,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };
    #[test]
    fn ambiguous_commit_persistence_never_reverts_to_abort_and_restart_reconciles() {
        let device = Device::flex();
        let limits = MpsLimits::default();
        let network = StateNetworkLimits::default();
        let identity = StateIdentity {
            program: "h".into(),
            calibration: "ideal".into(),
            run: "ambiguous-durable-commit".into(),
        };
        let state = MpsState::basis(&[2], &[0], &device, DType::F32, limits)
            .unwrap()
            .checkpoint()
            .unwrap();
        let mut owner = ShardWorker::new(
            ShardCheckpoint {
                identity: identity.clone(),
                start: 0,
                total: 1,
                epoch: 0,
                center: 0,
                report: state.report,
                sites: state.sites,
            },
            limits,
        )
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = WorkerEndpoint {
            address: listener.local_addr().unwrap(),
            capability: "unit-test-capability".into(),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let shutdown = stop.clone();
        let server = std::thread::spawn(move || {
            serve_state_worker(
                listener,
                &mut owner,
                "unit-test-capability",
                network,
                64,
                &shutdown,
                |_| Ok(()),
            )
            .unwrap();
            owner.checkpoint()
        });
        let mut coordinator = DistributedMps::connect(
            identity.clone(),
            vec![endpoint.clone()],
            &device,
            DType::F32,
            limits,
            network,
        )
        .unwrap();
        let mut disk = None;
        let mut injected = false;
        let mut ambiguous = |checkpoint: &CoordinatorCheckpoint| {
            disk = Some(checkpoint.clone());
            if !injected
                && checkpoint
                    .journal
                    .as_ref()
                    .is_some_and(|j| j.decision == Decision::Commit)
            {
                injected = true;
                return Err(err("write completed but durability acknowledgement failed"));
            }
            Ok(())
        };
        assert!(
            coordinator
                .apply_single(0, &Unitary::h(), &mut ambiguous)
                .is_err()
        );
        assert!(coordinator.needs_reconciliation());
        assert!(matches!(
            coordinator.checkpoint.journal.as_ref().unwrap().decision,
            Decision::Commit
        ));
        assert!(coordinator.amplitude(&[0]).is_err());
        assert!(
            coordinator
                .reconcile(&mut |_| Err(err("durability still unavailable")))
                .is_err()
        );
        // No worker Commit may precede a successfully persisted commit decision.
        assert!(call(&endpoint, &identity, Command::Inspect, network).is_err());
        let saved = serde_json::to_vec(&disk.unwrap()).unwrap();
        let mut resumed = DistributedMps::restore(
            serde_json::from_slice(&saved).unwrap(),
            &identity,
            vec![endpoint],
            &device,
            DType::F32,
            limits,
            network,
        )
        .unwrap();
        resumed.reconcile(&mut |_| Ok(())).unwrap();
        assert_eq!(resumed.epoch(), 1);
        assert!(!resumed.needs_reconciliation());
        assert!(
            (resumed.amplitude(&[0]).unwrap().re - std::f64::consts::FRAC_1_SQRT_2).abs() < 2e-6
        );
        assert!(
            (resumed.amplitude(&[1]).unwrap().re - std::f64::consts::FRAC_1_SQRT_2).abs() < 2e-6
        );
        stop.store(true, Ordering::Release);
        let checkpoint = server.join().unwrap();
        assert_eq!(checkpoint.epoch(), 1);
        ShardWorker::restore(checkpoint, &identity, limits).unwrap();
    }
    #[test]
    fn accumulated_owner_norm_error_rejects_attach_and_blocks_failed_recovery() {
        let device = Device::flex();
        let limits = MpsLimits {
            max_sites: 64,
            max_bond: 1,
            max_tensor_elements: 2,
            max_state_elements: 128,
            ..Default::default()
        };
        let network = StateNetworkLimits::default();
        let identity = StateIdentity {
            program: "identity".into(),
            calibration: "ideal".into(),
            run: "global-norm".into(),
        };
        let mut checkpoint = MpsState::basis(&[2; 64], &[0; 64], &device, DType::F32, limits)
            .unwrap()
            .checkpoint()
            .unwrap();
        for site in checkpoint.sites.iter_mut().skip(1) {
            site.values[0] *= 1.000009;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut endpoints = Vec::new();
        let mut servers = Vec::new();
        let mut metadata = Vec::new();
        for start in [0, 32] {
            let sites = checkpoint.sites[start..start + 32].to_vec();
            metadata.push(Metadata {
                start,
                count: 32,
                total: 64,
                epoch: 0,
                center: 0,
                report: checkpoint.report,
                shapes: sites.iter().map(|s| s.shape).collect(),
            });
            // Individually valid local gauges cannot establish full-chain normalization.
            let mut worker = ShardWorker::new(
                ShardCheckpoint {
                    identity: identity.clone(),
                    start,
                    total: 64,
                    epoch: 0,
                    center: 0,
                    report: checkpoint.report,
                    sites,
                },
                limits,
            )
            .unwrap();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            endpoints.push(WorkerEndpoint {
                address: listener.local_addr().unwrap(),
                capability: "norm-test".into(),
            });
            let shutdown = stop.clone();
            servers.push(std::thread::spawn(move || {
                serve_state_worker(
                    listener,
                    &mut worker,
                    "norm-test",
                    network,
                    1024,
                    &shutdown,
                    |_| Ok(()),
                )
                .unwrap()
            }));
        }
        assert!(
            matches!(DistributedMps::connect(identity.clone(),endpoints.clone(),&device,DType::F32,limits,network),Err(Error::InvalidValue(message)) if message.contains("global norm"))
        );
        let saved = CoordinatorCheckpoint {
            version: 1,
            identity: identity.clone(),
            metadata,
            journal: Some(Journal {
                decision: Decision::Preparing,
                updates: vec![
                    Update {
                        previous: 0,
                        next: 1,
                        center: 0,
                        report: checkpoint.report,
                        sites: vec![]
                    };
                    2
                ],
            }),
            statistics: TransferStatistics::default(),
        };
        let mut recovered = DistributedMps::restore(
            saved,
            &identity,
            endpoints.clone(),
            &device,
            DType::F32,
            limits,
            network,
        )
        .unwrap();
        assert!(recovered.needs_reconciliation());
        assert!(recovered.amplitude(&[0; 64]).is_err());
        assert!(
            matches!(recovered.reconcile(&mut |_|Ok(())),Err(Error::InvalidValue(message)) if message.contains("global norm"))
        );
        assert!(recovered.needs_reconciliation());
        assert!(recovered.expectation_product(&[]).is_err());
        assert!(
            recovered
                .apply_single(0, &Unitary::h(), &mut |_| Ok(()))
                .is_err()
        );
        assert!(
            recovered
                .apply_adjacent(0, &Unitary::cnot(), &mut |_| Ok(()))
                .is_err()
        );
        assert!(
            DistributedMps::restore(
                recovered.checkpoint(),
                &identity,
                endpoints,
                &device,
                DType::F32,
                limits,
                network
            )
            .is_err()
        );
        stop.store(true, Ordering::Relaxed);
        for server in servers {
            server.join().unwrap();
        }
    }
}
