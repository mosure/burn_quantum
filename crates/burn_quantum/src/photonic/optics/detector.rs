use super::{FockBasis, FockDensity, FockState, WavepacketLayout};
use crate::{Error, ResourceLimits, Result};
use burn::tensor::{Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// Independent finite-resolution detector with efficiency and a Bernoulli dark count.
/// Counts saturate at `resolution`. Resolution one is a threshold detector.
/// Dark counts mean at most one additional count per detection window, not Poisson noise.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(try_from = "DetectorData", into = "DetectorData")]
pub struct Detector {
    efficiency: f64,
    resolution: usize,
    dark_probability: f64,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DetectorData {
    efficiency: f64,
    resolution: usize,
    dark_probability: f64,
}
impl TryFrom<DetectorData> for Detector {
    type Error = Error;
    fn try_from(v: DetectorData) -> Result<Self> {
        Self::new(v.efficiency, v.resolution, v.dark_probability)
    }
}
impl From<Detector> for DetectorData {
    fn from(v: Detector) -> Self {
        Self {
            efficiency: v.efficiency,
            resolution: v.resolution,
            dark_probability: v.dark_probability,
        }
    }
}
impl Detector {
    /// Validate efficiency/dark probabilities and finite saturation resolution in 1..=64.
    pub fn new(efficiency: f64, resolution: usize, dark_probability: f64) -> Result<Self> {
        if [efficiency, dark_probability]
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.).contains(x))
            || resolution == 0
            || resolution > 64
        {
            return Err(Error::InvalidValue("invalid optical detector".into()));
        }
        Ok(Self {
            efficiency,
            resolution,
            dark_probability,
        })
    }
    /// Maximum reported count; this bin includes saturated events.
    pub fn resolution(&self) -> usize {
        self.resolution
    }
    /// Ideal number resolution, saturating only above the supplied count.
    pub fn ideal(resolution: usize) -> Result<Self> {
        Self::new(1., resolution, 0.)
    }
    /// On/off detector with the same explicit noise assumptions.
    pub fn threshold(efficiency: f64, dark_probability: f64) -> Result<Self> {
        Self::new(efficiency, 1, dark_probability)
    }
    /// Conditional detector response P(record | photons), including zero counts.
    pub fn response(&self, photons: usize) -> Result<Vec<f64>> {
        if photons > 64 {
            return Err(Error::InvalidValue(
                "detector reference accepts at most 64 photons".into(),
            ));
        }
        let mut out = vec![0.; self.resolution + 1];
        for count in 0..=photons {
            let p = super::state::binomial(photons, count)
                * self.efficiency.powi(count as i32)
                * (1. - self.efficiency).powi((photons - count) as i32);
            out[count.min(self.resolution)] += p * (1. - self.dark_probability);
            out[(count + 1).min(self.resolution)] += p * self.dark_probability;
        }
        Ok(out)
    }
}
/// Unconditioned detector records and their native tensor probabilities.
#[derive(Clone, Debug)]
pub struct DetectionDistribution {
    records: Vec<Vec<usize>>,
    probabilities: Tensor<2>,
}
impl DetectionDistribution {
    /// Per-mode observed counts in column order; saturated values remain explicit.
    pub fn records(&self) -> &[Vec<usize>] {
        &self.records
    }
    /// `[batch,records]`, retaining all zero-detection attempts and native gradients.
    pub fn probabilities(&self) -> Tensor<2> {
        self.probabilities.clone()
    }
}
impl FockState {
    /// Terminal readout with independent per-mode detector responses; no host readback.
    pub fn detect(
        &self,
        detectors: &[Detector],
        max_outcomes: usize,
        limits: ResourceLimits,
    ) -> Result<DetectionDistribution> {
        detect(
            self.basis(),
            self.probabilities(),
            detectors,
            max_outcomes,
            limits,
        )
    }
}
impl FockDensity {
    /// Terminal readout with independent per-mode detector responses; no postselection.
    pub fn detect(
        &self,
        detectors: &[Detector],
        max_outcomes: usize,
        limits: ResourceLimits,
    ) -> Result<DetectionDistribution> {
        detect(
            self.basis(),
            self.probabilities(),
            detectors,
            max_outcomes,
            limits,
        )
    }
}
fn detect(
    basis: &FockBasis,
    probabilities: Tensor<2>,
    detectors: &[Detector],
    max_outcomes: usize,
    limits: ResourceLimits,
) -> Result<DetectionDistribution> {
    detect_occupations(
        basis.occupations(),
        basis.modes(),
        probabilities,
        detectors,
        max_outcomes,
        limits,
    )
}
fn detect_occupations(
    occupations: &[Vec<usize>],
    modes: usize,
    probabilities: Tensor<2>,
    detectors: &[Detector],
    max_outcomes: usize,
    limits: ResourceLimits,
) -> Result<DetectionDistribution> {
    if detectors.len() != modes {
        return Err(Error::Shape("one detector is required per mode".into()));
    }
    let mut count = 1usize;
    for detector in detectors {
        count = count
            .checked_mul(detector.resolution + 1)
            .ok_or(Error::Overflow)?;
        if count > max_outcomes {
            return Err(Error::ResourceLimit {
                resource: "detector outcomes",
                required: count,
                limit: max_outcomes,
            });
        }
    }
    let dim = occupations.len();
    let [batch, _] = probabilities.dims();
    limits.check_state(batch, count)?;
    let size = dim.checked_mul(count).ok_or(Error::Overflow)?;
    if size > limits.max_matrix_elements {
        return Err(Error::ResourceLimit {
            resource: "detector response matrix",
            required: size,
            limit: limits.max_matrix_elements,
        });
    }
    let mut records = Vec::with_capacity(count);
    for index in 0..count {
        let mut value = index;
        let mut record = vec![0; detectors.len()];
        for (mode, detector) in detectors.iter().enumerate().rev() {
            record[mode] = value % (detector.resolution + 1);
            value /= detector.resolution + 1;
        }
        records.push(record);
    }
    let mut matrix = vec![0.; size];
    for (i, occupation) in occupations.iter().enumerate() {
        let responses: Vec<_> = detectors
            .iter()
            .zip(occupation)
            .map(|(d, &n)| d.response(n))
            .collect::<Result<_>>()?;
        for (j, record) in records.iter().enumerate() {
            matrix[i * count + j] = record
                .iter()
                .enumerate()
                .map(|(m, &r)| responses[m][r])
                .product();
        }
    }
    let response = Tensor::<2>::from_data(
        TensorData::new(matrix, [dim, count]),
        (&probabilities.device(), probabilities.dtype()),
    );
    Ok(DetectionDistribution {
        records,
        probabilities: crate::precision::matmul2(probabilities, response),
    })
}

impl FockState {
    /// Detect spatial counts without resolving orthogonal internal photon modes.
    pub fn detect_spatial(
        &self,
        layout: WavepacketLayout,
        detectors: &[Detector],
        max_outcomes: usize,
        limits: ResourceLimits,
    ) -> Result<DetectionDistribution> {
        detect_occupations(
            &layout.spatial_occupations(self.basis())?,
            layout.spatial_modes(),
            self.probabilities(),
            detectors,
            max_outcomes,
            limits,
        )
    }
}
impl FockDensity {
    /// Detect spatial counts after summing internal occupations, retaining all missed events.
    pub fn detect_spatial(
        &self,
        layout: WavepacketLayout,
        detectors: &[Detector],
        max_outcomes: usize,
        limits: ResourceLimits,
    ) -> Result<DetectionDistribution> {
        detect_occupations(
            &layout.spatial_occupations(self.basis())?,
            layout.spatial_modes(),
            self.probabilities(),
            detectors,
            max_outcomes,
            limits,
        )
    }
}
