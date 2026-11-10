use crate::photonic::stabilizer::{Measurement, Pauli, PauliNoise, PauliProduct, Tableau};
use crate::photonic::{FusionOutcome, Parity, runtime::shot_seed};
use crate::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Physical sampling and an explicitly conditioned ideal channel reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SixRingSampling {
    /// Sample all source, loss and standard-analyzer outcomes; stop on rejection.
    Physical,
    /// Force only odd-ZZ analyzer branches, retaining random XX/trim/check records.
    /// Requires an ideal model; each consumed ring contributes log herald weight -ln(4).
    /// This is not a successful unconditional sample or a repeat-until-success protocol.
    ConditionedReference,
}
/// Actual primitive cost up to completion or physical rejection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpticalSurgeryCost {
    /// Heralded six-ring source invocations, including unavailable sources.
    pub ring_attempts: usize,
    /// Successfully emitted six-ring resources.
    pub rings_emitted: usize,
    /// All six photons per emitted ring, including trimmed and subsequently lost photons.
    pub resource_photons: usize,
    /// Single fresh |0> syndrome photons.
    pub syndrome_photons: usize,
    /// Attempted physical Z measurements deleting ring vertices 4 and 5.
    pub trim_measurements: usize,
    /// Actual destructive standard Bell analyzers invoked.
    pub fusions: usize,
    /// Final syndrome Z readouts (zero if the instrument aborts earlier).
    pub syndrome_measurements: usize,
    /// Local Hadamards, including graph-to-Choi and basis changes.
    pub hadamards: usize,
    /// Local S/S-dagger optical phase operations for Y-basis checks.
    pub phases: usize,
    /// Applied X/Z byproduct corrections; Y uses two operations.
    pub paulis: usize,
}
impl OpticalSurgeryCost {
    pub(super) fn add(&mut self, other: Self) {
        self.ring_attempts += other.ring_attempts;
        self.rings_emitted += other.rings_emitted;
        self.resource_photons += other.resource_photons;
        self.syndrome_photons += other.syndrome_photons;
        self.trim_measurements += other.trim_measurements;
        self.fusions += other.fusions;
        self.syndrome_measurements += other.syndrome_measurements;
        self.hadamards += other.hadamards;
        self.phases += other.phases;
        self.paulis += other.paulis;
    }
}
/// A physical check attempt never turns a missing record into a known parity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpticalCheckFailure {
    /// The heralded ring source emitted nothing; previous operations remain consumed.
    MissingResource,
    /// A ring-trimming photon did not produce a measurement record.
    MissingTrim,
    /// One of the two photons of an attempted fusion was lost or not detected.
    FusionLoss,
    /// Standard analyzer returned its resolved even-ZZ computational-basis failure.
    FusionRejected,
    /// The final destructive syndrome measurement was missing.
    MissingSyndrome,
}
/// Physical record of one resource-derived CNOT, even if it did not complete.
#[derive(Clone, Debug)]
pub struct OpticalGateRecord {
    /// Abstract data/ancilla carriers before gate teleportation, in control/target order.
    pub logical_sites: [usize; 2],
    /// Whether the six-photon resource was emitted.
    pub emitted: bool,
    /// Z results on resource ports 4 and 5; None means not reached or missing.
    pub trims: [Option<bool>; 2],
    /// Actual analyzer records, in input-control/input-target order.
    pub fusions: Vec<FusionOutcome>,
    /// First physical rejection, if any.
    pub failure: Option<OpticalCheckFailure>,
}
/// Complete local instrument record and conditional output stored at original logical indices.
#[derive(Clone, Debug)]
pub struct OpticalCheckAttempt {
    /// Some only if all gate teleportations and the final readout completed.
    pub measurement: Option<Measurement>,
    /// First missing/rejected event; no retry or input rollback is performed.
    pub failure: Option<OpticalCheckFailure>,
    /// Every attempted six-ring gate, including its rejected branch.
    pub gates: Vec<OpticalGateRecord>,
    /// Actual physical resources and measurements consumed.
    pub cost: OpticalSurgeryCost,
    /// Some only for the expressly conditioned ideal reference, not physical sampling.
    pub log_herald_weight: Option<f64>,
}
/// Six-ring gate teleportation implementing a weight-at-most-six Pauli projector.
///
/// The resource is a ring 0-1-2-3-4-5-0. Z deletion of 4,5 and H on 0,2
/// produces the CNOT Choi state: input ports 0,3 and output ports 1,2.
/// Measured-photon survival includes transmission and detection, identically on
/// the two rails; output storage is ideal. Independent Pauli defects act on the
/// emitted ring. Dark counts, asymmetric filtering and unheralded output loss
/// require a different instrument and are not approximated by this model.
#[derive(Clone, Copy, Debug)]
pub struct SixRingCheckInstrument {
    availability: f64,
    measured_survival: f64,
    signal_survival: f64,
    source_noise: PauliNoise,
}
impl SixRingCheckInstrument {
    /// Ideal sources/detection, with the actual 50% standard-analyzer success rate.
    pub const IDEAL: Self = Self {
        availability: 1.,
        measured_survival: 1.,
        signal_survival: 1.,
        source_noise: PauliNoise::NONE,
    };
    /// Checked balanced optical model. Final syndrome detection uses measured_survival.
    pub fn new(
        availability: f64,
        measured_survival: f64,
        signal_survival: f64,
        source_noise: PauliNoise,
    ) -> Result<Self> {
        if [availability, measured_survival, signal_survival]
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.).contains(p))
        {
            return Err(Error::InvalidValue(
                "invalid six-ring instrument probability".into(),
            ));
        }
        Ok(Self {
            availability,
            measured_survival,
            signal_survival,
            source_noise,
        })
    }
    /// Ideal all-gate-herald probability for a weight-w check, excluding its +/- result.
    /// The measured loss factors include four ring measurements, two signal photons
    /// per CNOT, and one final syndrome detection. Pauli defects do not alter this
    /// herald probability, but can corrupt the returned parity and surviving data.
    pub fn herald_probability(&self, weight: usize) -> Result<f64> {
        check_weight(weight)?;
        Ok((self.availability
            * self.measured_survival.powi(4)
            * self.signal_survival.powi(2)
            * 0.25)
            .powi(weight as i32)
            * self.measured_survival)
    }
    pub(super) fn validate_sampling(&self, sampling: SixRingSampling) -> Result<()> {
        if sampling == SixRingSampling::ConditionedReference
            && (self.availability != 1.
                || self.measured_survival != 1.
                || self.signal_survival != 1.
                || self.source_noise.probabilities() != [0.; 3])
        {
            return Err(Error::Unsupported(
                "conditioned six-ring reference requires an ideal model".into(),
            ));
        }
        Ok(())
    }
    /// Execute an actual nondestructive Pauli measurement on arbitrary entangled input.
    ///
    /// Workspace is seven distinct exclusively owned scratch indices: a syndrome
    /// photon then six resource slots. It is reset before use; none may be in the
    /// observable. Consumed carrier slots are relabeled after teleportation, not
    /// physically swapped: gate records identify the replacement resource outputs.
    /// On rejection the partially measured input must not be reused as its original state.
    pub fn measure(
        &self,
        state: &mut Tableau,
        observable: &PauliProduct,
        workspace: [usize; 7],
        sampling: SixRingSampling,
        seed: u64,
    ) -> Result<OpticalCheckAttempt> {
        let weight = observable.terms().len();
        check_weight(weight)?;
        self.validate_sampling(sampling)?;
        if workspace.iter().enumerate().any(|(i, &q)| {
            q >= state.qubits()
                || workspace[..i].contains(&q)
                || observable.terms().iter().any(|&(r, _)| r == q)
        }) || observable.terms().iter().any(|&(q, _)| q >= state.qubits())
        {
            return Err(Error::Wires("invalid six-ring scratch/data indices".into()));
        }
        let random = state.expectation(observable)? == 0;
        let mut rng = ChaCha8Rng::seed_from_u64(shot_seed(seed, 0, 0x3652_494e_4753_0001));
        let mut cost = OpticalSurgeryCost {
            syndrome_photons: 1,
            ..Default::default()
        };
        reset(state, workspace[0], &mut rng)?;
        for &(q, p) in observable.terms() {
            basis(state, q, p, false, &mut cost)?;
        }
        let mut gates = Vec::with_capacity(weight);
        for &(q, _) in observable.terms() {
            let gate = self.cnot(
                state,
                [q, workspace[0]],
                workspace[1..].try_into().unwrap(),
                sampling,
                &mut rng,
                &mut cost,
            )?;
            let failure = gate.failure;
            gates.push(gate);
            if failure.is_some() {
                return Ok(OpticalCheckAttempt {
                    measurement: None,
                    failure,
                    gates,
                    cost,
                    log_herald_weight: None,
                });
            }
        }
        cost.syndrome_measurements += 1;
        if sampling == SixRingSampling::Physical && rng.random::<f64>() >= self.measured_survival {
            state.measure(&pauli(&[(workspace[0], Pauli::Z)])?, rng.random())?;
            return Ok(OpticalCheckAttempt {
                measurement: None,
                failure: Some(OpticalCheckFailure::MissingSyndrome),
                gates,
                cost,
                log_herald_weight: None,
            });
        }
        let negative = state
            .measure(&pauli(&[(workspace[0], Pauli::Z)])?, rng.random())?
            .negative;
        for &(q, p) in observable.terms() {
            basis(state, q, p, true, &mut cost)?;
        }
        Ok(OpticalCheckAttempt {
            measurement: Some(Measurement { negative, random }),
            failure: None,
            gates,
            cost,
            log_herald_weight: (sampling == SixRingSampling::ConditionedReference)
                .then_some(-(weight as f64) * 4f64.ln()),
        })
    }
    fn cnot(
        &self,
        state: &mut Tableau,
        data: [usize; 2],
        r: [usize; 6],
        sampling: SixRingSampling,
        rng: &mut ChaCha8Rng,
        cost: &mut OpticalSurgeryCost,
    ) -> Result<OpticalGateRecord> {
        let mut record = OpticalGateRecord {
            logical_sites: data,
            emitted: false,
            trims: [None; 2],
            fusions: vec![],
            failure: None,
        };
        cost.ring_attempts += 1;
        if sampling == SixRingSampling::Physical && rng.random::<f64>() >= self.availability {
            record.failure = Some(OpticalCheckFailure::MissingResource);
            return Ok(record);
        }
        record.emitted = true;
        cost.rings_emitted += 1;
        cost.resource_photons += 6;
        // Internal CZs specify the supplied six-ring state. They are not falsely
        // counted as deterministic online optical entangling gates.
        for q in r {
            reset(state, q, rng)?;
            state.h(q)?;
        }
        for i in 0..6 {
            state.cz(r[i], r[(i + 1) % 6])?;
        }
        let [px, py, pz] = self.source_noise.probabilities();
        for q in r {
            let u = rng.random::<f64>();
            if u < px + py {
                state.x(q)?;
            }
            if (px..px + py + pz).contains(&u) {
                state.z(q)?;
            }
        }
        for (i, port) in [4, 5].into_iter().enumerate() {
            cost.trim_measurements += 1;
            let lost = sampling == SixRingSampling::Physical
                && rng.random::<f64>() >= self.measured_survival;
            let bit = state
                .measure(&pauli(&[(r[port], Pauli::Z)])?, rng.random())?
                .negative;
            if lost {
                record.failure = Some(OpticalCheckFailure::MissingTrim);
                return Ok(record);
            }
            record.trims[i] = Some(bit);
        }
        for (i, port) in [3, 0].into_iter().enumerate() {
            if record.trims[i].unwrap() {
                state.z(r[port])?;
                cost.paulis += 1;
            }
        }
        for port in [0, 2] {
            state.h(r[port])?;
            cost.hadamards += 1;
        }
        let mut xx = [false; 2];
        for (i, port) in [0, 3].into_iter().enumerate() {
            cost.fusions += 1;
            let [a, b] = [data[i], r[port]];
            let lost = sampling == SixRingSampling::Physical
                && (rng.random::<f64>() >= self.signal_survival
                    || rng.random::<f64>() >= self.measured_survival);
            if lost {
                for q in [a, b] {
                    state.measure(&pauli(&[(q, Pauli::Z)])?, rng.random())?;
                }
                record.fusions.push(FusionOutcome::Loss);
                record.failure = Some(OpticalCheckFailure::FusionLoss);
                return Ok(record);
            }
            let zz = state.measure(
                &pauli(&[(a, Pauli::Z), (b, Pauli::Z)])?,
                sampling == SixRingSampling::ConditionedReference || rng.random(),
            )?;
            if !zz.negative {
                let bit = state
                    .measure(&pauli(&[(a, Pauli::Z)])?, rng.random())?
                    .negative;
                record.fusions.push(FusionOutcome::Failure { bit });
                record.failure = Some(OpticalCheckFailure::FusionRejected);
                return Ok(record);
            }
            xx[i] = state
                .measure(&pauli(&[(a, Pauli::X), (b, Pauli::X)])?, rng.random())?
                .negative;
            record.fusions.push(FusionOutcome::Success {
                xx: if xx[i] { Parity::Odd } else { Parity::Even },
            });
        }
        // CNOT propagates the two Bell byproducts into its output frame.
        state.x(r[1])?;
        cost.paulis += 1;
        if xx[0] ^ xx[1] {
            state.z(r[1])?;
            cost.paulis += 1;
        }
        if xx[1] {
            state.z(r[2])?;
            cost.paulis += 1;
        }
        // Swap storage labels so callers retain logical indices. These three
        // tableau CNOTs are a state-index permutation, not optical operations.
        for (old, new) in [(data[0], r[1]), (data[1], r[2])] {
            state.cnot(old, new)?;
            state.cnot(new, old)?;
            state.cnot(old, new)?;
        }
        Ok(record)
    }
}
fn check_weight(weight: usize) -> Result<()> {
    if !(1..=6).contains(&weight) {
        Err(Error::Unsupported(
            "six-ring checks require weight 1..=6".into(),
        ))
    } else {
        Ok(())
    }
}
fn pauli(terms: &[(usize, Pauli)]) -> Result<PauliProduct> {
    PauliProduct::new(terms.to_vec())
}
fn reset(state: &mut Tableau, q: usize, rng: &mut ChaCha8Rng) -> Result<()> {
    if state
        .measure(&pauli(&[(q, Pauli::Z)])?, rng.random())?
        .negative
    {
        state.x(q)?;
    }
    Ok(())
}
fn basis(
    state: &mut Tableau,
    q: usize,
    p: Pauli,
    undo: bool,
    cost: &mut OpticalSurgeryCost,
) -> Result<()> {
    match p {
        Pauli::Z => {}
        Pauli::X => {
            state.h(q)?;
            cost.hadamards += 1;
        }
        Pauli::Y => {
            if undo {
                state.h(q)?;
                state.s(q)?;
            } else {
                for _ in 0..3 {
                    state.s(q)?;
                }
                state.h(q)?;
            }
            cost.hadamards += 1;
            cost.phases += 1;
        }
    }
    Ok(())
}
