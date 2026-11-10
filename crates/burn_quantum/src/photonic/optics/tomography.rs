//! Executable passive-optical tomography and finite-shot mixed-state overlap estimation.
use super::{Detector, FockDensity, Interferometer, ModeId, PhotonSector};
use crate::{ComplexTensor, Error, ResourceLimits, Result};
use burn::tensor::{Tensor, TensorData};
use nalgebra::{DMatrix, linalg::SymmetricEigen};
use num_complex::Complex64;

/// Physically executable passive-optical measurement setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpticalTomographySetting {
    /// Direct number-resolving detection of all modes, including vacuum.
    Number,
    /// Balanced Ry splitter on the pair, optionally preceded by phase pi/2 on `a`.
    Coherence {
        /// First optical mode.
        a: usize,
        /// Second optical mode.
        b: usize,
        /// True selects the imaginary quadrature.
        imaginary: bool,
    },
}
/// Bounded complete tomography for photon-number-superselected vacuum/single-photon states.
#[derive(Clone, Debug)]
pub struct OpticalTomography {
    shots: usize,
    max_modes: usize,
    max_attempts: usize,
    failure_probability: f64,
}
/// Finite-shot reconstructed state and explicit physical measurement accounting.
#[derive(Clone, Debug)]
pub struct OpticalTomographyReport {
    /// Positive normalized estimate; PSD projection creates finite-sample bias.
    pub state: FockDensity,
    /// Complete count vectors, in the same lexicographic threshold-record order for each setting.
    pub histograms: Vec<Vec<usize>>,
    /// Executed measurement settings in order.
    pub settings: Vec<OpticalTomographySetting>,
    /// Total attempted preparations; no-click records are included.
    pub attempts: usize,
    /// Frobenius displacement introduced by positivity/trace reconstruction.
    pub projection_correction: f64,
    /// Conservative Hoeffding/union-bound upper bound on trace distance from the true state.
    /// Assumes independent preparations, ideal known measurement optics and the declared sector.
    pub trace_distance_bound: f64,
    /// Upper bound on probability that the trace-distance guarantee fails.
    pub failure_probability: f64,
}
/// Squared Uhlmann fidelity estimated from two independent optical tomographies.
#[derive(Clone, Debug)]
pub struct OpticalOverlapEstimate {
    /// Plug-in squared fidelity of reconstructed states; biased at finite shots.
    pub squared_fidelity: f64,
    /// Conservative absolute fidelity error bound, clipped to one.
    pub absolute_error_bound: f64,
    /// Physical preparation attempts for both states and every measurement setting.
    pub attempts: usize,
    /// Union bound on failure of the two tomography guarantees.
    pub failure_probability: f64,
    /// Complete first-state tomography evidence.
    pub left: OpticalTomographyReport,
    /// Complete independent second-state tomography evidence.
    pub right: OpticalTomographyReport,
}
impl OpticalTomography {
    /// Declare physical shots per setting, mode/work limits and confidence failure probability.
    /// At most eight optical modes are supported; every loss/no-click outcome is retained.
    pub fn new(
        shots_per_setting: usize,
        max_modes: usize,
        max_attempts: usize,
        failure_probability: f64,
    ) -> Result<Self> {
        if shots_per_setting < 2
            || !(1..=8).contains(&max_modes)
            || max_attempts == 0
            || !failure_probability.is_finite()
            || failure_probability <= 0.
            || failure_probability >= 1.
        {
            return Err(Error::InvalidValue(
                "invalid optical tomography protocol".into(),
            ));
        }
        Ok(Self {
            shots: shots_per_setting,
            max_modes,
            max_attempts,
            failure_probability,
        })
    }
    /// Inspect the actual optics/readout plan: one number basis and two settings per mode pair.
    pub fn settings(&self, modes: usize) -> Result<Vec<OpticalTomographySetting>> {
        if modes == 0 || modes > self.max_modes {
            return Err(Error::ResourceLimit {
                resource: "tomography modes",
                required: modes,
                limit: self.max_modes,
            });
        }
        let mut settings = vec![OpticalTomographySetting::Number];
        for a in 0..modes {
            for b in a + 1..modes {
                for imaginary in [false, true] {
                    settings.push(OpticalTomographySetting::Coherence { a, b, imaginary });
                }
            }
        }
        Ok(settings)
    }
    /// Run the declared passive transformations and independent PNR shot experiments.
    /// Only batch one, sectors Fixed(1)/UpTo(1), and no vacuum/photon coherences are identifiable.
    pub fn reconstruct(
        &self,
        state: &FockDensity,
        seed: u64,
        limits: ResourceLimits,
    ) -> Result<OpticalTomographyReport> {
        let modes = state.basis().modes();
        let settings = self.settings(modes)?;
        self.check_attempts(settings.len(), 1)?;
        validate_input(state)?;
        let d = state.basis().size();
        let mut histogram = Vec::with_capacity(settings.len());
        let mut data = vec![Complex64::new(0., 0.); d * d];
        let indices = (0..modes)
            .map(|m| {
                let mut o = vec![0; modes];
                o[m] = 1;
                state.basis().index(&o)
            })
            .collect::<Result<Vec<_>>>()?;
        let vacuum = state.basis().index(&vec![0; modes]).ok();
        let mut vacuum_probability = 0.;
        for (number, &setting) in settings.iter().enumerate() {
            let measured = match setting {
                OpticalTomographySetting::Number => state.clone(),
                OpticalTomographySetting::Coherence { a, b, imaginary } => {
                    let rho = if imaginary {
                        state.phase_shift(
                            ModeId(a),
                            Tensor::from_data(
                                TensorData::new(vec![std::f64::consts::FRAC_PI_2], [1]),
                                (&state.entries().device(), state.entries().dtype()),
                            ),
                            limits,
                        )?
                    } else {
                        state.clone()
                    };
                    rho.apply(
                        &Interferometer::beam_splitter(
                            modes,
                            ModeId(a),
                            ModeId(b),
                            std::f64::consts::FRAC_PI_2,
                        )?,
                        limits,
                    )?
                }
            };
            let distribution =
                measured.detect(&vec![Detector::ideal(1)?; modes], 1 << modes, limits)?;
            let counts = distribution
                .sample_counts(
                    self.shots,
                    crate::photonic::runtime::shot_seed(seed, number as u64, 0x544f_4d4f),
                    self.max_attempts,
                )?
                .remove(0)
                .counts;
            let probability = |mode: Option<usize>| -> f64 {
                let mut record = vec![0; modes];
                if let Some(m) = mode {
                    record[m] = 1;
                }
                let index = distribution
                    .records()
                    .iter()
                    .position(|r| r == &record)
                    .expect("complete detector alphabet");
                counts[index] as f64 / self.shots as f64
            };
            match setting {
                OpticalTomographySetting::Number => {
                    vacuum_probability = probability(None);
                    if let Some(v) = vacuum {
                        data[v * d + v] = vacuum_probability.into();
                    }
                    for (m, &index) in indices.iter().enumerate() {
                        data[index * d + index] = probability(Some(m)).into();
                    }
                }
                OpticalTomographySetting::Coherence { a, b, imaginary } => {
                    let difference = (probability(Some(b)) - probability(Some(a))) / 2.;
                    let ia = indices[a];
                    let ib = indices[b];
                    if imaginary {
                        data[ia * d + ib].im = -difference;
                        data[ib * d + ia].im = difference;
                    } else {
                        data[ia * d + ib].re = difference;
                        data[ib * d + ia].re = difference;
                    }
                }
            }
            histogram.push(counts);
        }
        let raw = data.clone();
        let photon = DMatrix::from_fn(modes, modes, |a, b| data[indices[a] * d + indices[b]]);
        let eigen = SymmetricEigen::new(photon);
        let positive = eigen.eigenvalues.map(|x| x.max(0.));
        let total = positive.iter().sum::<f64>();
        let survival = 1. - vacuum_probability;
        if !total.is_finite() || total < 0. || survival > 0. && total == 0. {
            return Err(Error::InvalidValue(
                "invalid optical tomography reconstruction".into(),
            ));
        }
        for a in 0..modes {
            for b in 0..modes {
                let value = if total == 0. {
                    Complex64::new(0., 0.)
                } else {
                    (0..modes)
                        .map(|k| {
                            eigen.eigenvectors[(a, k)]
                                * eigen.eigenvectors[(b, k)].conj()
                                * positive[k]
                                * survival
                                / total
                        })
                        .sum()
                };
                data[indices[a] * d + indices[b]] = value;
            }
        }
        let correction = data
            .iter()
            .zip(&raw)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let radius = ((2. * settings.len() as f64 * (modes + 1) as f64 / self.failure_probability)
            .ln()
            / (2. * self.shots as f64))
            .sqrt();
        let entry_factor = (1 + modes + 2 * modes * (modes - 1)) as f64;
        let bound = (0.5 * (d as f64).sqrt() * (entry_factor.sqrt() * radius + correction)).min(1.);
        let re = Tensor::from_data(
            TensorData::new(data.iter().map(|z| z.re).collect(), [1, d, d]),
            (&state.entries().device(), state.entries().dtype()),
        );
        let im = Tensor::from_data(
            TensorData::new(data.iter().map(|z| z.im).collect(), [1, d, d]),
            (&state.entries().device(), state.entries().dtype()),
        );
        let estimate =
            FockDensity::from_entries(state.basis().clone(), ComplexTensor::new(re, im)?, limits)?;
        Ok(OpticalTomographyReport {
            state: estimate,
            histograms: histogram,
            attempts: settings.len() * self.shots,
            settings,
            projection_correction: correction,
            trace_distance_bound: bound,
            failure_probability: self.failure_probability,
        })
    }
    /// Estimate mixed-state fidelity from actual independent photon-count experiments.
    /// Spectral work uses reconstructed density matrices only, as classical post-processing.
    pub fn overlap(
        &self,
        left: &FockDensity,
        right: &FockDensity,
        seed: u64,
        limits: ResourceLimits,
    ) -> Result<OpticalOverlapEstimate> {
        if left.basis() != right.basis() {
            return Err(Error::Shape("tomographic overlap bases differ".into()));
        }
        let count = self.settings(left.basis().modes())?.len();
        self.check_attempts(count, 2)?;
        let left = self.reconstruct(
            left,
            crate::photonic::runtime::shot_seed(seed, 0, 0x4c45_4654),
            limits,
        )?;
        let right = self.reconstruct(
            right,
            crate::photonic::runtime::shot_seed(seed, 1, 0x0052_4947_4854),
            limits,
        )?;
        let fidelity = left
            .state
            .uhlmann_fidelity(&right.state, self.max_modes + 1, 2e-5)?[0];
        let bound = (2.
            * ((2. * left.trace_distance_bound).sqrt() + (2. * right.trace_distance_bound).sqrt()))
        .min(1.);
        Ok(OpticalOverlapEstimate {
            squared_fidelity: fidelity,
            absolute_error_bound: bound,
            attempts: left.attempts + right.attempts,
            failure_probability: (left.failure_probability + right.failure_probability).min(1.),
            left,
            right,
        })
    }
    fn check_attempts(&self, settings: usize, states: usize) -> Result<()> {
        let required = settings
            .checked_mul(states)
            .and_then(|n| n.checked_mul(self.shots))
            .ok_or(Error::Overflow)?;
        if required > self.max_attempts {
            Err(Error::ResourceLimit {
                resource: "optical tomography attempts",
                required,
                limit: self.max_attempts,
            })
        } else {
            Ok(())
        }
    }
}
fn validate_input(state: &FockDensity) -> Result<()> {
    if state.entries().dims()[0] != 1
        || !matches!(
            state.basis().sector(),
            PhotonSector::Fixed(1) | PhotonSector::UpTo(1)
        )
    {
        return Err(Error::Unsupported(
            "passive tomography requires one single-photon/vacuum state".into(),
        ));
    }
    state.validate(2e-5)?;
    if let Ok(vacuum) = state.basis().index(&vec![0; state.basis().modes()]) {
        let re = crate::complex::read(state.entries().real())?;
        let im = crate::complex::read(state.entries().imag())?;
        let d = state.basis().size();
        for i in 0..d {
            if i != vacuum && (re[vacuum * d + i].abs() > 1e-7 || im[vacuum * d + i].abs() > 1e-7) {
                return Err(Error::Unsupported(
                    "vacuum/photon coherence requires an external phase reference".into(),
                ));
            }
        }
    }
    Ok(())
}
