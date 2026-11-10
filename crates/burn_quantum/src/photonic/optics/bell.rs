//! Optical Bell instruments derived from Fock mode transformations and detector records.
use super::{
    Detector, FockBasis, Interferometer, ModeId, PhotonPacket, PhotonSector, WavepacketLayout,
};
use crate::photonic::{DualRail, Parity};
use crate::{ComplexTensor, DenseSimulator, DensityMatrix, Error, Matrix, ResourceLimits, Result};
use burn::tensor::Tensor;
use nalgebra::{DMatrix, linalg::SymmetricEigen};
use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2};

/// Declared classical readout, which can be wrong under imperfect calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum BellRecord {
    /// Click pattern declares both Bell parities. Detector noise can cause false heralds.
    Resolved {
        /// Declared XX eigenvalue.
        xx: Parity,
        /// Declared ZZ eigenvalue.
        zz: Parity,
    },
    /// Standard analyzer resolves individual identical Z values but no XX parity.
    ZFailure {
        /// False means 00; true means 11 in the ideal model.
        bit: bool,
    },
    /// Boosted four-photon result is ambiguous between Phi+ and Phi- (ZZ even).
    AmbiguousEven,
    /// Incomplete, saturated or corrupt photon-count record; never a usable Bell result.
    Rejected,
}
/// A calibrated destructive POVM effect with its declared result.
#[derive(Clone, Debug, serde::Serialize)]
pub struct BellEffect {
    record: BellRecord,
    effect: Matrix,
    square_root: Matrix,
}
impl BellEffect {
    /// Classical declaration attached to this effect.
    pub fn record(&self) -> BellRecord {
        self.record
    }
    /// Positive effect on two encoded dual-rail sites (dimension nine).
    pub fn matrix(&self) -> &Matrix {
        &self.effect
    }
}
/// Branch with absolute probability and a subnormalized surviving density matrix.
#[derive(Clone, Debug)]
pub struct OpticalBellBranch {
    record: BellRecord,
    probability: Tensor<1>,
    state: Option<DensityMatrix>,
}
impl OpticalBellBranch {
    /// Reported record, including explicit ambiguous/rejected attempts.
    pub fn record(&self) -> BellRecord {
        self.record
    }
    /// Unconditional per-batch probability, retaining input gradients.
    pub fn probability(&self) -> Tensor<1> {
        self.probability.clone()
    }
    /// Surviving state with trace equal to absolute branch weight.
    pub fn unnormalized_state(&self) -> Option<&DensityMatrix> {
        self.state.as_ref()
    }
    /// Explicit normalization with synchronous validation and quotient gradients.
    pub fn conditional_state(&self, min_probability: f64) -> Result<Option<DensityMatrix>> {
        crate::error::tolerance(min_probability)?;
        if crate::complex::read(self.probability())?
            .iter()
            .any(|p| !p.is_finite() || *p <= min_probability)
        {
            return Err(Error::InvalidValue(
                "impossible optical Bell conditioning".into(),
            ));
        }
        self.state
            .as_ref()
            .map(|s| {
                let [b, n, _] = s.entries().dims();
                let p = self.probability().reshape([b, 1, 1]);
                DensityMatrix::from_entries(
                    s.register().clone(),
                    ComplexTensor::new(s.entries().real() / p.clone(), s.entries().imag() / p)?,
                    ResourceLimits {
                        max_state_elements: b * n * n,
                        ..Default::default()
                    },
                )
            })
            .transpose()
    }
}
/// Loss parameters for the first Grice boost, with two supplied Phi+ ancillary photons.
/// Detectors resolve up to four photons with zero dark counts. Any missing photon rejects.
#[derive(Clone, Debug)]
pub struct GriceConfig {
    /// Balanced-rail transmissions for the two signal photons before the analyzer.
    pub signal_transmissions: [f64; 2],
    /// Balanced-rail transmissions for the two ancillary photons before the analyzer.
    pub ancilla_transmissions: [f64; 2],
    /// Efficiencies of the eight output rail detectors; all four counts are required.
    pub detector_efficiencies: [f64; 8],
    /// Caller-supplied model/calibration fingerprint.
    pub calibration: String,
}
impl Default for GriceConfig {
    fn default() -> Self {
        Self {
            signal_transmissions: [1.; 2],
            ancilla_transmissions: [1.; 2],
            detector_efficiencies: [1.; 8],
            calibration: "ideal-grice-first-level-v1".into(),
        }
    }
}
/// Complete destructive instrument derived from a specified optical recipe.
/// Construction/lifting is host F64; branch tensor execution uses the caller's Burn device.
#[derive(Clone, Debug, serde::Serialize)]
pub struct OpticalBellInstrument {
    effects: Vec<BellEffect>,
    calibration: String,
    supplied_ancillas: usize,
}
impl OpticalBellInstrument {
    /// Four spatial rail modes and two balanced Ry splitters. Internal wavepacket
    /// overlap squared is in `[0,1]`; each detector has its own noisy response.
    /// All modes/labels are traced after detection, deriving the remote-state instrument.
    pub fn standard(
        visibility: f64,
        detectors: [Detector; 4],
        calibration: String,
    ) -> Result<Self> {
        probability(visibility)?;
        validate_calibration(&calibration)?;
        let layout = WavepacketLayout::new(4, 2)?;
        let basis = FockBasis::new(8, PhotonSector::UpTo(2), 45)?;
        let first = layout
            .lift(&Interferometer::beam_splitter(
                4,
                ModeId(0),
                ModeId(2),
                FRAC_PI_2,
            )?)?
            .lift(&basis, ResourceLimits::default())?;
        let second = layout
            .lift(&Interferometer::beam_splitter(
                4,
                ModeId(1),
                ModeId(3),
                FRAC_PI_2,
            )?)?
            .lift(&basis, ResourceLimits::default())?;
        let d = basis.size();
        let mut amplitude = vec![vec![Complex64::new(0., 0.); 9]; d];
        for a in 0..3 {
            for b in 0..3 {
                let mut packets = vec![];
                if a < 2 {
                    packets.push(PhotonPacket::new(a, vec![1.0.into(), 0.0.into()])?);
                }
                if b < 2 {
                    packets.push(PhotonPacket::new(
                        2 + b,
                        vec![visibility.sqrt().into(), (1. - visibility).sqrt().into()],
                    )?);
                }
                let input = super::wavepacket::packet_amplitudes(&basis, layout, &packets)?;
                let middle = matrix_vector(&first, &input);
                let output = matrix_vector(&second, &middle);
                for (row, value) in output.into_iter().enumerate() {
                    amplitude[row][a * 3 + b] = value;
                }
            }
        }
        let records = vec![
            resolved(Parity::Even, Parity::Odd),
            resolved(Parity::Odd, Parity::Odd),
            BellRecord::ZFailure { bit: false },
            BellRecord::ZFailure { bit: true },
            BellRecord::Rejected,
        ];
        let mut effects = vec![vec![Complex64::new(0., 0.); 81]; records.len()];
        for (row, occupation) in layout.spatial_occupations(&basis)?.iter().enumerate() {
            let responses: Vec<_> = detectors
                .iter()
                .zip(occupation)
                .map(|(d, &n)| d.response(n))
                .collect::<Result<_>>()?;
            let mut weights = [0.; 5];
            for a in 0..responses[0].len().min(4) {
                for b in 0..responses[1].len().min(4) {
                    for c in 0..responses[2].len().min(4) {
                        for e in 0..responses[3].len().min(4) {
                            let p = responses[0][a]
                                * responses[1][b]
                                * responses[2][c]
                                * responses[3][e];
                            weights[standard_record([a, b, c, e])] += p;
                        }
                    }
                }
            }
            for (effect, &weight) in effects.iter_mut().zip(&weights) {
                add_outer(effect, &amplitude[row], weight);
            }
        }
        Self::from_effects(records, effects, calibration, 0)
    }
    /// The first Grice boost: Eq. (3) of Wein et al., arXiv:1509.00088v2,
    /// independently on each rail, with an ideal supplied Phi+ ancillary pair.
    /// Pattern labels are derived from Bell-basis amplitudes, not a success coin.
    pub fn grice(config: GriceConfig) -> Result<Self> {
        validate_calibration(&config.calibration)?;
        for &p in config
            .signal_transmissions
            .iter()
            .chain(&config.ancilla_transmissions)
            .chain(&config.detector_efficiencies)
        {
            probability(p)?;
        }
        let basis = FockBasis::new(8, PhotonSector::Fixed(4), 330)?;
        let lifted = Self::grice_interferometer()?.lift(&basis, ResourceLimits::default())?;
        let d = basis.size();
        let mut amplitudes = vec![vec![Complex64::new(0., 0.); 9]; d];
        for a in 0..2 {
            for b in 0..2 {
                for ancillary_rail in 0..2 {
                    let mut occupation = vec![0; 8];
                    for mode in [a, 2 + b, 4 + ancillary_rail, 6 + ancillary_rail] {
                        occupation[mode] += 1;
                    }
                    let input = basis.index(&occupation)?;
                    for (row, amplitude) in amplitudes.iter_mut().enumerate() {
                        amplitude[a * 3 + b] += lifted.entries()[row * d + input] * FRAC_1_SQRT_2;
                    }
                }
            }
        }
        let records = vec![
            resolved(Parity::Even, Parity::Odd),
            resolved(Parity::Odd, Parity::Odd),
            resolved(Parity::Even, Parity::Even),
            resolved(Parity::Odd, Parity::Even),
            BellRecord::AmbiguousEven,
            BellRecord::Rejected,
        ];
        let mut effects = vec![vec![Complex64::new(0., 0.); 81]; 6];
        let survival = config
            .signal_transmissions
            .iter()
            .chain(&config.ancilla_transmissions)
            .product::<f64>();
        for (occupation, amplitude) in basis.occupations().iter().zip(&amplitudes) {
            let bell = [
                (amplitude[1] + amplitude[3]) * FRAC_1_SQRT_2,
                (amplitude[1] - amplitude[3]) * FRAC_1_SQRT_2,
                (amplitude[0] + amplitude[4]) * FRAC_1_SQRT_2,
                (amplitude[0] - amplitude[4]) * FRAC_1_SQRT_2,
            ];
            let supported: Vec<_> = bell
                .iter()
                .enumerate()
                .filter(|(_, v)| v.norm_sqr() > 1e-12)
                .map(|(i, _)| i)
                .collect();
            let index = match supported.as_slice() {
                [] => continue,
                [only] => *only,
                [2, 3] => 4,
                _ => {
                    return Err(Error::InvalidValue(
                        "Grice recipe produced unexpected ambiguous Bell support".into(),
                    ));
                }
            };
            let detection = occupation
                .iter()
                .zip(&config.detector_efficiencies)
                .map(|(&n, &eta)| eta.powi(n as i32))
                .product::<f64>();
            add_outer(&mut effects[index], amplitude, survival * detection);
        }
        // Rejection includes lost signal/ancillary photons, missed detections and vacuum inputs.
        for i in 0..9 {
            effects[5][i * 9 + i] = 1.0.into();
        }
        let (retained, rejected) = effects.split_at_mut(5);
        for effect in retained {
            for (target, value) in rejected[0].iter_mut().zip(effect) {
                *target -= *value;
            }
        }
        Self::from_effects(records, effects, config.calibration, 2)
    }
    /// Eight-mode passive unitary for the first Grice boost; paired indices are rails.
    pub fn grice_interferometer() -> Result<Interferometer> {
        let i = Complex64::new(0., 1.);
        let block = [
            1.0.into(),
            i,
            i,
            (-1.0).into(),
            i,
            1.0.into(),
            (-1.0).into(),
            i,
            i,
            (-1.0).into(),
            1.0.into(),
            i,
            (-1.0).into(),
            i,
            i,
            1.0.into(),
        ];
        let mut entries = vec![Complex64::new(0., 0.); 64];
        for a in 0..4 {
            for b in 0..4 {
                for rail in 0..2 {
                    entries[(2 * a + rail) * 8 + 2 * b + rail] = block[a * 4 + b] / 2.;
                }
            }
        }
        Interferometer::new(Matrix::new(8, entries)?)
    }
    /// Immutable effects in classical-record order; their sum is identity on dimension nine.
    pub fn effects(&self) -> &[BellEffect] {
        &self.effects
    }
    /// Caller-supplied calibration fingerprint retained by the compiled instrument.
    pub fn calibration(&self) -> &str {
        &self.calibration
    }
    /// Extra ideally supplied ancillary photons per attempt (zero or two).
    pub fn supplied_ancillas(&self) -> usize {
        self.supplied_ancillas
    }
    /// Apply the complete destructive instrument, tracing both inputs in every outcome.
    /// Noise may make a resolved declaration an incorrect parity report; effects are authoritative.
    pub fn branches(
        &self,
        engine: &DenseSimulator,
        state: &DensityMatrix,
        sites: [usize; 2],
    ) -> Result<Vec<OpticalBellBranch>> {
        DualRail::validate(state.register())?;
        state.register().local_size(&sites)?;
        let keep: Vec<_> = (0..state.register().wires())
            .filter(|w| !sites.contains(w))
            .collect();
        self.effects
            .iter()
            .map(|effect| {
                let measured = state.left_right(engine, &sites, &effect.square_root)?;
                let probability = measured.trace();
                let state = if keep.is_empty() {
                    None
                } else {
                    Some(measured.reduced(&keep, engine.limits)?)
                };
                Ok(OpticalBellBranch {
                    record: effect.record,
                    probability,
                    state,
                })
            })
            .collect()
    }
    pub(super) fn from_effects(
        records: Vec<BellRecord>,
        effects: Vec<Vec<Complex64>>,
        calibration: String,
        supplied_ancillas: usize,
    ) -> Result<Self> {
        for entry in 0..81 {
            let sum: Complex64 = effects.iter().map(|e| e[entry]).sum();
            let target = if entry / 9 == entry % 9 { 1. } else { 0. };
            if (sum - target).norm() > 1e-9 {
                return Err(Error::InvalidValue(
                    "optical Bell effects are not complete".into(),
                ));
            }
        }
        let effects = records
            .into_iter()
            .zip(effects)
            .map(|(record, data)| {
                let square_root = positive_root(&data)?;
                Ok(BellEffect {
                    record,
                    effect: Matrix::new(9, data)?,
                    square_root,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            effects,
            calibration,
            supplied_ancillas,
        })
    }
}
fn resolved(xx: Parity, zz: Parity) -> BellRecord {
    BellRecord::Resolved { xx, zz }
}
fn standard_record([a, b, c, d]: [usize; 4]) -> usize {
    if a + b + c + d != 2 {
        return 4;
    }
    if b + d == 0 {
        return 2;
    }
    if a + c == 0 {
        return 3;
    }
    if a + b == 2 || c + d == 2 { 0 } else { 1 }
}
fn add_outer(effect: &mut [Complex64], row: &[Complex64], weight: f64) {
    for a in 0..9 {
        for b in 0..9 {
            effect[a * 9 + b] += row[a].conj() * row[b] * weight;
        }
    }
}
fn matrix_vector(matrix: &Matrix, input: &[Complex64]) -> Vec<Complex64> {
    let n = matrix.dimension();
    (0..n)
        .map(|row| {
            input
                .iter()
                .enumerate()
                .map(|(col, &x)| matrix.entries()[row * n + col] * x)
                .sum()
        })
        .collect()
}
fn positive_root(data: &[Complex64]) -> Result<Matrix> {
    let eigen = SymmetricEigen::new(DMatrix::from_row_slice(9, 9, data));
    if eigen
        .eigenvalues
        .iter()
        .any(|v| !v.is_finite() || *v < -1e-9)
    {
        return Err(Error::InvalidValue("optical effect is not positive".into()));
    }
    let mut values = vec![Complex64::new(0., 0.); 81];
    for a in 0..9 {
        for b in 0..9 {
            for k in 0..9 {
                values[a * 9 + b] += eigen.eigenvectors[(a, k)]
                    * eigen.eigenvectors[(b, k)].conj()
                    * eigen.eigenvalues[k].max(0.).sqrt();
            }
        }
    }
    Matrix::new(9, values)
}
fn probability(p: f64) -> Result<()> {
    if p.is_finite() && (0.0..=1.).contains(&p) {
        Ok(())
    } else {
        Err(Error::InvalidValue(
            "optical probability outside [0,1]".into(),
        ))
    }
}
fn validate_calibration(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 4096 {
        Err(Error::InvalidValue(
            "invalid optical calibration identity".into(),
        ))
    } else {
        Ok(())
    }
}

pub(super) fn grice_records() -> Vec<BellRecord> {
    vec![
        resolved(Parity::Even, Parity::Odd),
        resolved(Parity::Odd, Parity::Odd),
        resolved(Parity::Even, Parity::Even),
        resolved(Parity::Odd, Parity::Even),
        BellRecord::AmbiguousEven,
        BellRecord::Rejected,
    ]
}
