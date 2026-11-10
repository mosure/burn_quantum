//! Finite six-ring fusion boxes with explicit terminal boundaries and output correlations.
use super::{MeasurementRelations, ParityInference};
use crate::photonic::stabilizer::{
    Pauli, PauliNoise, PauliProduct, StabilizerSimulator, execute_trajectory,
};
use crate::photonic::{
    CompiledFusionNetwork, FusionOutcome, FusionProgram, NodeId, PhotonId, ResourceState,
    runtime::shot_seed,
};
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
/// Physical operation at an exterior face port of a finite ring lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchBoundary {
    /// Retain the photon as a quantum output, without terminal measurement.
    Output,
    /// Destructively measure the specified physical Pauli on that boundary photon.
    Measure(Pauli),
}
/// A single independent loss event that removes a declared set of physical photons.
/// Overlapping bursts combine by union, retaining correlations between records.
#[derive(Clone, Debug)]
pub struct PhotonLossBurst {
    photons: Vec<PhotonId>,
    probability: f64,
}
impl PhotonLossBurst {
    /// Validate a nonempty set of distinct photon identities and event probability.
    pub fn new(photons: Vec<PhotonId>, probability: f64) -> Result<Self> {
        if photons.is_empty()
            || photons
                .iter()
                .enumerate()
                .any(|(i, q)| photons[..i].contains(q))
        {
            return Err(Error::Wires("invalid photon loss burst".into()));
        }
        if !probability.is_finite() || !(0.0..=1.).contains(&probability) {
            return Err(Error::InvalidValue("invalid loss burst probability".into()));
        }
        Ok(Self {
            photons,
            probability,
        })
    }
}
/// One complete physical attempt. All losses and failures remain in the result.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchShot {
    /// Stable global shot identifier.
    pub shot_id: u64,
    /// Actual standard-fusion outcomes, including resolved 00/11 failures.
    pub fusion_outcomes: Vec<FusionOutcome>,
    /// Fusion XX/ZZ followed by physical boundary readouts; lost values are None.
    pub records: Vec<Option<bool>>,
    /// Presence flags in the patch's retained quantum output order.
    pub output_present: Vec<bool>,
    /// Ideal-code inference using only observed records and signed checks.
    /// Unheralded faults can make a known inference wrong; compare with expectations.
    pub inferred: Vec<ParityInference>,
    /// Actual conditional-trajectory output expectation for each requested target.
    /// Zero if any factor is on a physically lost output.
    pub expectations: Vec<i8>,
}
/// Open rectangular box of the published rotated six-ring bulk connectivity.
///
/// Internal face ports undergo actual standard fusions. Exterior ports undergo
/// explicit one-photon Pauli measurements or remain quantum outputs. This general
/// finite geometry does not assert that every boundary choice defines a distance-d
/// surface code; signed checks and output correlations are derived from its state.
#[derive(Clone, Debug)]
pub struct SixRingPatch {
    dimensions: [usize; 3],
    program: FusionProgram,
    boundary: Vec<PauliProduct>,
    outputs: Vec<PhotonId>,
}
impl SixRingPatch {
    /// Internal checked geometry entry point shared by derived boundary builders.
    pub(crate) fn from_parts(
        dimensions: [usize; 3],
        program: FusionProgram,
        boundary: Vec<PauliProduct>,
        mut outputs: Vec<PhotonId>,
        max_photons: usize,
    ) -> Result<Self> {
        let n = program.num_photons();
        let limit = max_photons.min(16384);
        if n == 0 || dimensions.contains(&0) {
            return Err(Error::InvalidDimensions(
                "empty finite fusion geometry".into(),
            ));
        }
        if n > limit {
            return Err(Error::ResourceLimit {
                resource: "open six-ring photons",
                required: n,
                limit,
            });
        }
        let mut used = vec![false; n];
        for q in program
            .fusions()
            .iter()
            .flat_map(|f| f.photons.iter().map(|q| q.0))
            .chain(
                boundary
                    .iter()
                    .map(|p| p.terms().first().map_or(n, |(q, _)| *q)),
            )
            .chain(outputs.iter().map(|q| q.0))
        {
            if q >= n || used[q] {
                return Err(Error::Wires(
                    "finite geometry reuses or omits a photon identity".into(),
                ));
            }
            used[q] = true;
        }
        if used.iter().any(|v| !v) || boundary.iter().any(|p| p.terms().len() != 1) {
            return Err(Error::Wires(
                "finite geometry must consume or retain every photon exactly once".into(),
            ));
        }
        outputs.sort_by_key(|q| q.0);
        Ok(Self {
            dimensions,
            program,
            boundary,
            outputs,
        })
    }
    /// Face order is +x,+y,+z,-x,-y,-z. Every cell contains an actual six-ring
    /// resource; maximum photons is checked before allocation.
    pub fn new(
        dimensions: [usize; 3],
        faces: [PatchBoundary; 6],
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        if dimensions.contains(&0) {
            return Err(Error::InvalidDimensions(
                "empty six-ring patch dimension".into(),
            ));
        }
        let cells = dimensions
            .iter()
            .try_fold(1usize, |v, &n| v.checked_mul(n).ok_or(Error::Overflow))?;
        let photons = cells.checked_mul(6).ok_or(Error::Overflow)?;
        let limit = max_photons.min(4096);
        if photons > limit {
            return Err(Error::ResourceLimit {
                resource: "open six-ring photons",
                required: photons,
                limit,
            });
        }
        let mut program = FusionProgram::new();
        let mut rings = Vec::with_capacity(cells);
        for _ in 0..cells {
            rings.push(program.add_resource(ResourceState::six_ring(), source)?);
        }
        let index = |c: [usize; 3]| (c[0] * dimensions[1] + c[1]) * dimensions[2] + c[2];
        let positive = [1, 2, 0];
        let mut boundary = Vec::new();
        let mut outputs = Vec::new();
        for x in 0..dimensions[0] {
            for y in 0..dimensions[1] {
                for z in 0..dimensions[2] {
                    let c = [x, y, z];
                    let cell = index(c);
                    for axis in 0..3 {
                        if c[axis] + 1 < dimensions[axis] {
                            let mut next = c;
                            next[axis] += 1;
                            program.fuse(
                                rings[cell][positive[axis]],
                                rings[index(next)][positive[axis] + 3],
                                analyzer,
                            )?;
                        }
                        for negative in [false, true] {
                            let face = axis + if negative { 3 } else { 0 };
                            if (negative && c[axis] == 0)
                                || (!negative && c[axis] + 1 == dimensions[axis])
                            {
                                let photon =
                                    rings[cell][positive[axis] + if negative { 3 } else { 0 }];
                                match faces[face] {
                                    PatchBoundary::Output => outputs.push(photon),
                                    PatchBoundary::Measure(pauli) => {
                                        boundary.push(PauliProduct::new(vec![(photon.0, pauli)])?)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        outputs.sort_by_key(|p| p.0);
        Ok(Self {
            dimensions,
            program,
            boundary,
            outputs,
        })
    }
    /// Explicit Bell-link strip: +x/+y boundary X, -x/-y boundary Z, and both
    /// z ends retained. Each ring becomes a length-three cluster wire under these
    /// terminal measurements. This has no asserted increasing error distance.
    pub fn bell_strip(
        length: usize,
        source: NodeId,
        analyzer: NodeId,
        max_photons: usize,
    ) -> Result<Self> {
        Self::new(
            [1, 1, length],
            [
                PatchBoundary::Measure(Pauli::X),
                PatchBoundary::Measure(Pauli::X),
                PatchBoundary::Output,
                PatchBoundary::Measure(Pauli::Z),
                PatchBoundary::Measure(Pauli::Z),
                PatchBoundary::Output,
            ],
            source,
            analyzer,
            max_photons,
        )
    }
    /// Cell dimensions in the rotated bulk coordinates.
    pub fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }
    /// Internal resource/fusion program; boundary measurements are explicit below.
    pub fn program(&self) -> &FusionProgram {
        &self.program
    }
    /// Ordered terminal physical boundary measurements, following all fusion records.
    pub fn boundary_measurements(&self) -> &[PauliProduct] {
        &self.boundary
    }
    /// Retained quantum output identities. Boundary photons are excluded.
    pub fn outputs(&self) -> &[PhotonId] {
        &self.outputs
    }
    /// Full XX/ZZ fusion observables followed by physical boundary Pauli observables.
    pub fn observables(&self) -> Result<Vec<PauliProduct>> {
        let mut observables: Vec<_> = self
            .program
            .fusions()
            .iter()
            .flat_map(|f| {
                [Pauli::X, Pauli::Z]
                    .map(|p| PauliProduct::new(f.photons.iter().map(|q| (q.0, p)).collect()))
            })
            .collect::<Result<_>>()?;
        observables.extend(self.boundary.clone());
        Ok(observables)
    }
    fn matching_plan(&self, plan: &CompiledFusionNetwork) -> Result<()> {
        if plan.program() != &self.program {
            return Err(Error::InvalidValue(
                "compiled plan differs from finite patch".into(),
            ));
        }
        Ok(())
    }
    /// Complete signed check basis and available output correlation algebra.
    pub fn relations(
        &self,
        plan: &CompiledFusionNetwork,
        max_entries: usize,
    ) -> Result<MeasurementRelations> {
        self.matching_plan(plan)?;
        let initial = StabilizerSimulator::default().prepare(plan)?;
        MeasurementRelations::derive(&initial, &self.observables()?, max_entries)
    }
    /// Execute actual standard fusions and terminal boundary instruments. Balanced
    /// compiled attenuation, optional correlated loss bursts and independent local
    /// Pauli faults are applied once. No ideal complete-Bell substitution is used.
    /// Requested correlations are evaluated before any retained output is measured.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        plan: &CompiledFusionNetwork,
        start: u64,
        shots: usize,
        seed: u64,
        pauli: PauliNoise,
        bursts: &[PhotonLossBurst],
        targets: &[PauliProduct],
        max_entries: usize,
    ) -> Result<Vec<PatchShot>> {
        self.matching_plan(plan)?;
        start.checked_add(shots as u64).ok_or(Error::Overflow)?;
        let fields = self
            .program
            .num_photons()
            .checked_mul(3)
            .and_then(|v| v.checked_add(targets.len() * 2))
            .ok_or(Error::Overflow)?;
        let required = shots.checked_mul(fields).ok_or(Error::Overflow)?;
        if required > max_entries {
            return Err(Error::ResourceLimit {
                resource: "finite patch result entries",
                required,
                limit: max_entries,
            });
        }
        if bursts
            .iter()
            .any(|b| b.photons.iter().any(|q| q.0 >= self.program.num_photons()))
        {
            return Err(Error::Wires("loss burst references absent photon".into()));
        }
        if targets.iter().any(|p| {
            p.terms()
                .iter()
                .any(|&(q, _)| !self.outputs.iter().any(|p| p.0 == q))
        }) {
            return Err(Error::Wires(
                "patch target is not on retained outputs".into(),
            ));
        }
        let initial = StabilizerSimulator::default().prepare(plan)?;
        let relations = MeasurementRelations::derive(&initial, &self.observables()?, max_entries)?;
        let domain = shot_seed(seed, 0, 0x4f50_454e_4642_0001);
        let mut result = Vec::with_capacity(shots);
        for index in 0..shots {
            let shot_id = start + index as u64;
            let mut burst_loss = vec![false; initial.qubits()];
            for (i, burst) in bursts.iter().enumerate() {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(shot_seed(domain, shot_id, 0x1000_0000 + i as u64));
                if rng.random::<f64>() < burst.probability {
                    for q in &burst.photons {
                        burst_loss[q.0] = true;
                    }
                }
            }
            let (mut state, lost, fusion_outcomes) =
                execute_trajectory(plan, &initial, shot_id, domain, pauli, Some(&burst_loss))?;
            let mut records = MeasurementRelations::fusion_records(&fusion_outcomes);
            for (i, p) in self.boundary.iter().enumerate() {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(shot_seed(domain, shot_id, 0x2000_0000 + i as u64));
                let q = p.terms()[0].0;
                if lost[q] {
                    records.push(None);
                } else {
                    records.push(Some(state.measure(p, rng.random())?.negative));
                }
            }
            let mut expectations = Vec::with_capacity(targets.len());
            let mut inferred = Vec::with_capacity(targets.len());
            for p in targets {
                if p.terms().iter().any(|&(q, _)| lost[q]) {
                    expectations.push(0);
                    inferred.push(ParityInference::Erased);
                } else {
                    expectations.push(state.expectation(p)?);
                    inferred.push(relations.infer(&records, p)?);
                }
            }
            result.push(PatchShot {
                shot_id,
                fusion_outcomes,
                records,
                output_present: self.outputs.iter().map(|q| !lost[q.0]).collect(),
                inferred,
                expectations,
            });
        }
        Ok(result)
    }
}
