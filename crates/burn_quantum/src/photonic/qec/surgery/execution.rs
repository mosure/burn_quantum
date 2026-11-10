use super::*;
use crate::photonic::stabilizer::{Measurement, Pauli, Tableau};
use crate::{Error, Result};
/// Check phase; feed-forward occurs after Split round zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurgeryPhase {
    /// Joined-boundary checks, including the new seam gauges.
    Merge,
    /// Restored independent patch checks.
    Split,
}
/// One local check requested from an ideal or physical instrument.
#[derive(Clone, Debug)]
pub struct SurgeryMeasurement {
    /// Merge or restored-patch phase.
    pub phase: SurgeryPhase,
    /// Zero-based round within that phase.
    pub round: usize,
    /// Index in merged_checks or split_checks.
    pub check: usize,
    /// Actual global data indices after mapping the two patch offsets.
    pub observable: PauliProduct,
}
/// Complete successful measurement transcript and physical split correction.
#[derive(Clone, Debug)]
pub struct SurgeryRecord {
    /// Logical XX or ZZ negative-eigenvalue bit before any logical feed-forward.
    pub parity: bool,
    /// One complete generator-sign vector per merged round.
    pub merged: Vec<Vec<bool>>,
    /// One original-code sign vector per split round; rounds after zero are corrected.
    pub split: Vec<Vec<bool>>,
    /// Actual physical Pauli frame applied after the first split round.
    pub correction: PauliProduct,
    /// Ordered check outcomes. No detector failure or erased record is silently dropped.
    pub outcomes: Vec<(SurgeryMeasurement, Measurement)>,
    /// Random ideal Pauli projections; this does not count optical herald weights.
    pub random_measurements: usize,
}
/// Physical attempt stopped after a rejected/erased local-check instrument.
#[derive(Clone, Debug)]
pub struct SurgeryAbort {
    /// Completed check outcomes before the failed instrument.
    pub outcomes: Vec<(SurgeryMeasurement, Measurement)>,
    /// Instrument that did not return a valid nondestructive parity.
    pub at: SurgeryMeasurement,
}
/// A rejected physical attempt leaves its partially evolved input consumed/mutated.
#[derive(Clone, Debug)]
pub enum SurgeryAttempt {
    /// Every local check succeeded and the final patch gauges were restored.
    Complete(SurgeryRecord),
    /// A physical check failed; the partially evolved data must not be retried.
    Aborted(SurgeryAbort),
}
/// Measurement-only logical CNOT: ZZ(control,ancilla), XX(ancilla,target), Z(ancilla).
#[derive(Clone, Debug)]
pub struct SurgeryCnotRecord {
    /// The smooth and rough parity-measurement transcripts.
    pub surgeries: [SurgeryRecord; 2],
    /// Physical destructive Z outcomes for all ancillary patch data.
    pub ancilla_readout: Vec<bool>,
    /// Ancillary logical Z outcome.
    pub ancilla_parity: bool,
    /// Applied Z-control / X-target correction in physical coordinates.
    pub correction: PauliProduct,
}
pub(super) fn apply_pauli(state: &mut Tableau, p: &PauliProduct) -> Result<()> {
    for &(q, pauli) in p.terms() {
        match pauli {
            Pauli::X => state.x(q)?,
            Pauli::Z => state.z(q)?,
            Pauli::Y => {
                state.x(q)?;
                state.z(q)?;
            }
        }
    }
    Ok(())
}
impl PlanarSurgery {
    /// Remap a relative two-patch product to disjoint actual patch locations.
    pub fn remap(&self, p: &PauliProduct, offsets: [usize; 2]) -> Result<PauliProduct> {
        let n = self.code.qubits();
        if offsets[0].abs_diff(offsets[1]) < n {
            return Err(Error::Wires("overlapping surgery patches".into()));
        }
        PauliProduct::new(
            p.terms()
                .iter()
                .map(|&(q, p)| {
                    if q >= 2 * n {
                        return Err(Error::Wires("surgery product outside data register".into()));
                    }
                    Ok((offsets[q / n].checked_add(q % n).ok_or(Error::Overflow)?, p))
                })
                .collect::<Result<_>>()?,
        )
    }
    /// Paired seam correction restoring both patch gauges without changing either
    /// protected merged logical. The first boundary value is fixed to zero, making
    /// the frame deterministic. Nonideal syndromes need a noise-aware decoder.
    pub fn split_correction(&self, signs: &[bool]) -> Result<PauliProduct> {
        if signs.len() != self.split.len() {
            return Err(Error::Shape("surgery split record width".into()));
        }
        for (i, &sign) in signs.iter().enumerate() {
            if sign && !self.boundary.iter().any(|p| p.contains(&i)) {
                return Err(Error::Unsupported(
                    "nonideal bulk surgery syndrome requires decoder".into(),
                ));
            }
        }
        let mut correction = PauliProduct::new(vec![])?;
        let mut bit = false;
        for (i, &[a, b]) in self.boundary.iter().enumerate() {
            if signs[a] != signs[b] {
                return Err(Error::Unsupported(
                    "inconsistent opposing surgery checks require decoder".into(),
                ));
            }
            bit ^= signs[a];
            if bit {
                correction = correction.multiply_commuting(&self.seam_checks()[i + 1])?.1;
            }
        }
        Ok(correction)
    }
    /// Ideal local-projector reference. Callback bits are consumed only for random
    /// projections. The encoded Choi tests establish the channel on arbitrary inputs.
    pub fn execute(
        &self,
        state: &mut Tableau,
        offsets: [usize; 2],
        mut bit: impl FnMut() -> bool,
    ) -> Result<SurgeryRecord> {
        match self.execute_with(state, offsets, |state, request| {
            let random = state.expectation(&request.observable)? == 0;
            Ok(Some(state.measure(&request.observable, random && bit())?))
        })? {
            SurgeryAttempt::Complete(record) => Ok(record),
            SurgeryAttempt::Aborted(_) => unreachable!(),
        }
    }
    /// Execute through actual local instruments. None denotes a physical abort:
    /// partial outcomes are returned and the input state is never rolled back or
    /// reused as though no measurement occurred. This ideal-code controller rejects
    /// inconsistent noisy syndromes; noisy decoding is a separate declared model.
    pub fn execute_with(
        &self,
        state: &mut Tableau,
        offsets: [usize; 2],
        mut measure: impl FnMut(&mut Tableau, &SurgeryMeasurement) -> Result<Option<Measurement>>,
    ) -> Result<SurgeryAttempt> {
        let n = self.code.qubits();
        if offsets
            .iter()
            .any(|o| o.checked_add(n).is_none_or(|end| end > state.qubits()))
        {
            return Err(Error::Wires("surgery patch outside tableau".into()));
        }
        for p in &self.split {
            if state.expectation(&self.remap(p, offsets)?)? != 1 {
                return Err(Error::Unsupported(
                    "surgery input must be in both positive code spaces".into(),
                ));
            }
        }
        let mut outcomes = vec![];
        let mut merged = vec![];
        let mut split = vec![];
        let mut correction = PauliProduct::new(vec![])?;
        for phase in [SurgeryPhase::Merge, SurgeryPhase::Split] {
            let checks = if phase == SurgeryPhase::Merge {
                &self.merged
            } else {
                &self.split
            };
            for round in 0..self.code.distance() {
                let mut signs = vec![];
                for (check, p) in checks.iter().enumerate() {
                    let request = SurgeryMeasurement {
                        phase,
                        round,
                        check,
                        observable: self.remap(p, offsets)?,
                    };
                    let Some(outcome) = measure(state, &request)? else {
                        return Ok(SurgeryAttempt::Aborted(SurgeryAbort {
                            outcomes,
                            at: request,
                        }));
                    };
                    if state.expectation(&request.observable)?
                        != if outcome.negative { -1 } else { 1 }
                    {
                        return Err(Error::InvalidValue(
                            "instrument did not produce its declared Pauli eigenstate".into(),
                        ));
                    }
                    signs.push(outcome.negative);
                    outcomes.push((request, outcome));
                }
                if phase == SurgeryPhase::Merge {
                    if round == 0 {
                        if signs[self.code.distance()..].iter().any(|&s| s) {
                            return Err(Error::Unsupported(
                                "nonideal merged bulk syndrome requires decoder".into(),
                            ));
                        }
                    } else if signs != merged[0] {
                        return Err(Error::Unsupported(
                            "temporal surgery syndrome requires decoder".into(),
                        ));
                    }
                    merged.push(signs);
                } else {
                    if round == 0 {
                        correction = self.remap(&self.split_correction(&signs)?, offsets)?;
                        apply_pauli(state, &correction)?;
                    } else if signs.iter().any(|&s| s) {
                        return Err(Error::Unsupported(
                            "postsplit syndrome requires decoder".into(),
                        ));
                    }
                    split.push(signs);
                }
            }
        }
        let parity = merged[0][..self.code.distance()]
            .iter()
            .fold(false, |a, &b| a ^ b);
        let random_measurements = outcomes.iter().filter(|(_, m)| m.random).count();
        Ok(SurgeryAttempt::Complete(SurgeryRecord {
            parity,
            merged,
            split,
            correction,
            outcomes,
            random_measurements,
        }))
    }
    /// Complete ideal measurement-only logical CNOT, preserving external entanglement.
    /// Patches are [control, fresh encoded |+> ancilla, target], all disjoint. The
    /// ancilla is destroyed by physical single-site Z measurements. Each parity
    /// stage has d merge and d split rounds, with no transversal data-data gate.
    pub fn cnot(
        distance: usize,
        state: &mut Tableau,
        patches: [usize; 3],
        limits: SurgeryLimits,
        mut bit: impl FnMut() -> bool,
    ) -> Result<SurgeryCnotRecord> {
        let smooth = Self::new(distance, MergeBasis::ZZ, limits)?;
        let rough = Self::new(distance, MergeBasis::XX, limits)?;
        let n = smooth.code.qubits();
        if patches
            .iter()
            .enumerate()
            .any(|(i, p)| patches[..i].iter().any(|q| p.abs_diff(*q) < n))
        {
            return Err(Error::Wires("CNOT surgery patches overlap".into()));
        }
        for &offset in &patches {
            if offset.checked_add(n).is_none_or(|end| end > state.qubits()) {
                return Err(Error::Wires("CNOT surgery patch outside tableau".into()));
            }
            for check in smooth
                .code
                .x_checks(offset)?
                .iter()
                .chain(&smooth.code.z_checks(offset)?)
            {
                if state.expectation(check)? != 1 {
                    return Err(Error::Unsupported(
                        "CNOT surgery input must be in positive code space".into(),
                    ));
                }
            }
        }
        if state.expectation(&smooth.code.logical_x(patches[1])?)? != 1 {
            return Err(Error::InvalidValue(
                "surgery ancilla must be encoded |+>".into(),
            ));
        }
        let a = smooth.execute(state, [patches[0], patches[1]], &mut bit)?;
        let b = rough.execute(state, [patches[1], patches[2]], &mut bit)?;
        let mut ancilla_readout = vec![];
        for q in 0..n {
            let p = PauliProduct::new(vec![(patches[1] + q, Pauli::Z)])?;
            let random = state.expectation(&p)? == 0;
            ancilla_readout.push(state.measure(&p, random && bit())?.negative);
        }
        let ancilla_parity = smooth
            .code
            .logical_z(0)?
            .terms()
            .iter()
            .fold(false, |v, (q, _)| v ^ ancilla_readout[*q]);
        let mut terms = vec![];
        if b.parity {
            terms.extend(smooth.code.logical_z(patches[0])?.terms());
        }
        if a.parity ^ ancilla_parity {
            terms.extend(smooth.code.logical_x(patches[2])?.terms());
        }
        let correction = PauliProduct::new(terms)?;
        apply_pauli(state, &correction)?;
        Ok(SurgeryCnotRecord {
            surgeries: [a, b],
            ancilla_readout,
            ancilla_parity,
            correction,
        })
    }
}
