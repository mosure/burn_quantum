use super::{MpsLimits, limit, tensor::*};
use crate::{ComplexTensor, Error, Result};
use burn::tensor::{DType, Device};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Approximation accounting, excluding finite-precision arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TruncationReport {
    /// Sum of discarded normalized Schmidt weights (not an infidelity bound).
    pub discarded_weight: f64,
    /// Conservative trace distance: sum sqrt(discarded weight), capped at one.
    /// Selective measurements amplify this bound by at most 2/p(outcome).
    pub trace_distance_bound: f64,
    /// Number of nonzero rank reductions.
    pub truncations: usize,
    /// Largest |norm²-1| observed before normalizing a center after a gate.
    pub max_norm_drift: f64,
}
/// Portable site tensor. Entries are row-major `[left, physical, right]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SiteData {
    /// Bond/local shape.
    pub shape: [usize; 3],
    /// Complex finite values.
    pub values: Vec<Complex64>,
}
impl SiteData {
    pub(crate) fn validate(&self, limits: MpsLimits) -> Result<()> {
        let [l, d, r] = self.shape;
        if l == 0 || d < 2 || r == 0 {
            return Err(Error::Shape("invalid MPS site shape".into()));
        }
        limit("MPS bond", l.max(r), limits.max_bond)?;
        let n = crate::error::checked_product(self.shape)?;
        limits.tensor(n)?;
        if self.values.len() != n
            || self
                .values
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(Error::InvalidValue("invalid MPS site values".into()));
        }
        Ok(())
    }
    pub(crate) fn tensor(
        &self,
        device: &Device,
        dtype: DType,
        limits: MpsLimits,
    ) -> Result<ComplexTensor<3>> {
        self.validate(limits)?;
        from(&self.values, self.shape, device, dtype)
    }
    pub(crate) fn read(site: &ComplexTensor<3>) -> Result<Self> {
        Ok(Self {
            shape: site.dims(),
            values: data(site)?,
        })
    }
}
/// Versioned full MPS checkpoint; restore rechecks canonical gauges and norm.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MpsCheckpoint {
    /// Currently 1.
    pub version: u32,
    /// Ordered sites, including entanglement across every cut.
    pub sites: Vec<SiteData>,
    /// Unique mixed-canonical center.
    pub center: usize,
    /// Approximation history (caller must authenticate persisted files).
    pub report: TruncationReport,
}
/// Normalized mixed-canonical open-boundary MPS, independent of Hilbert-size overflow.
#[derive(Clone, Debug)]
pub struct MpsState {
    pub(super) sites: Vec<ComplexTensor<3>>,
    pub(super) center: usize,
    pub(super) limits: MpsLimits,
    pub(super) report: TruncationReport,
}
impl MpsState {
    /// Product basis state; supports more than 64 qubits without a dense Register.
    pub fn basis(
        dimensions: &[usize],
        digits: &[usize],
        device: &Device,
        dtype: DType,
        limits: MpsLimits,
    ) -> Result<Self> {
        limits.validate()?;
        crate::backend::supported_dtype(dtype)?;
        limit("MPS sites", dimensions.len(), limits.max_sites)?;
        if dimensions.is_empty()
            || dimensions.len() != digits.len()
            || dimensions.iter().any(|d| *d < 2)
        {
            return Err(Error::InvalidDimensions(
                "MPS needs matching nonempty dimensions and digits".into(),
            ));
        }
        let count = dimensions
            .iter()
            .try_fold(0usize, |a, d| a.checked_add(*d).ok_or(Error::Overflow))?;
        limit("MPS persistent state", count, limits.max_state_elements)?;
        let mut sites = Vec::with_capacity(dimensions.len());
        for (&d, &digit) in dimensions.iter().zip(digits) {
            limits.tensor(d)?;
            if digit >= d {
                return Err(Error::Basis);
            }
            let mut values = vec![Complex64::new(0., 0.); d];
            values[digit] = 1.0.into();
            sites.push(from(&values, [1, d, 1], device, dtype)?);
        }
        Ok(Self {
            sites,
            center: 0,
            limits,
            report: TruncationReport::default(),
        })
    }
    /// Local dimensions; no exponentially large product is computed.
    pub fn dimensions(&self) -> Vec<usize> {
        self.sites.iter().map(|s| s.dims()[1]).collect()
    }
    /// Number of complex stored entries.
    pub fn storage_elements(&self) -> usize {
        self.sites
            .iter()
            .map(|s| s.dims().iter().product::<usize>())
            .sum()
    }
    /// Internal bond dimensions, ordered left to right.
    pub fn bonds(&self) -> Vec<usize> {
        self.sites
            .iter()
            .take(self.sites.len() - 1)
            .map(|s| s.dims()[2])
            .collect()
    }
    /// Approximation and numerical diagnostics.
    pub fn report(&self) -> TruncationReport {
        self.report
    }
    /// Native execution device (complex SVD remains host-side).
    pub fn device(&self) -> Device {
        self.sites[0].device()
    }
    /// Explicit complete checkpoint with device readback.
    pub fn checkpoint(&self) -> Result<MpsCheckpoint> {
        Ok(MpsCheckpoint {
            version: 1,
            sites: self
                .sites
                .iter()
                .map(SiteData::read)
                .collect::<Result<_>>()?,
            center: self.center,
            report: self.report,
        })
    }
    /// Restore after checking all size, gauge, finite-value and norm invariants.
    pub fn restore(
        checkpoint: &MpsCheckpoint,
        device: &Device,
        dtype: DType,
        limits: MpsLimits,
    ) -> Result<Self> {
        limits.validate()?;
        crate::backend::supported_dtype(dtype)?;
        limit("MPS sites", checkpoint.sites.len(), limits.max_sites)?;
        if checkpoint.version != 1
            || checkpoint.sites.is_empty()
            || checkpoint.center >= checkpoint.sites.len()
        {
            return Err(Error::InvalidValue(
                "invalid MPS checkpoint version/center".into(),
            ));
        }
        validate_report(checkpoint.report, limits)?;
        let mut count = 0usize;
        for (i, s) in checkpoint.sites.iter().enumerate() {
            s.validate(limits)?;
            count = count.checked_add(s.values.len()).ok_or(Error::Overflow)?;
            limit("MPS persistent state", count, limits.max_state_elements)?;
            if (i == 0 && s.shape[0] != 1)
                || (i + 1 == checkpoint.sites.len() && s.shape[2] != 1)
                || (i > 0 && checkpoint.sites[i - 1].shape[2] != s.shape[0])
            {
                return Err(Error::Shape("MPS checkpoint bond mismatch".into()));
            }
            validate_gauge(s, i.cmp(&checkpoint.center), 2e-5)?;
        }
        let state = Self {
            sites: checkpoint
                .sites
                .iter()
                .map(|s| s.tensor(device, dtype, limits))
                .collect::<Result<_>>()?,
            center: checkpoint.center,
            limits,
            report: checkpoint.report,
        };
        // Local gauge tolerances do not bound their accumulated chain norm error.
        // This streaming contraction uses only a bond-sized environment.
        validate_norm(state.expectation_product(&[])?)?;
        Ok(state)
    }
}
pub(super) fn accumulate(
    mut report: TruncationReport,
    lost: f64,
    limits: MpsLimits,
) -> Result<TruncationReport> {
    report.discarded_weight += lost;
    report.trace_distance_bound = (report.trace_distance_bound + lost.sqrt()).min(1.);
    if lost > 0. {
        report.truncations = report.truncations.checked_add(1).ok_or(Error::Overflow)?;
    }
    if report.trace_distance_bound > limits.max_truncation_distance + 1e-12 {
        return Err(Error::InvalidValue(format!(
            "MPS truncation distance {} exceeds budget {}",
            report.trace_distance_bound, limits.max_truncation_distance
        )));
    }
    Ok(report)
}
pub(super) fn validate_report(r: TruncationReport, limits: MpsLimits) -> Result<()> {
    if !r.discarded_weight.is_finite()
        || r.discarded_weight < 0.
        || !r.trace_distance_bound.is_finite()
        || !(0.0..=1.).contains(&r.trace_distance_bound)
        || !r.max_norm_drift.is_finite()
        || r.max_norm_drift < 0.
        || r.trace_distance_bound > limits.max_truncation_distance + 1e-12
    {
        return Err(Error::InvalidValue("invalid MPS error report".into()));
    }
    Ok(())
}
pub(crate) fn validate_gauge(
    site: &SiteData,
    position: std::cmp::Ordering,
    tolerance: f64,
) -> Result<()> {
    let [l, d, r] = site.shape;
    let v = &site.values;
    match position {
        std::cmp::Ordering::Equal => {
            let norm: f64 = v.iter().map(|z| z.norm_sqr()).sum();
            if (norm - 1.).abs() > tolerance {
                return Err(Error::InvalidValue("MPS center is not normalized".into()));
            }
        }
        std::cmp::Ordering::Less => {
            for a in 0..r {
                for b in 0..r {
                    let dot: Complex64 =
                        (0..l * d).map(|x| v[x * r + a].conj() * v[x * r + b]).sum();
                    if (dot
                        - if a == b {
                            Complex64::new(1., 0.)
                        } else {
                            Complex64::new(0., 0.)
                        })
                    .norm()
                        > tolerance
                    {
                        return Err(Error::InvalidValue(
                            "MPS left gauge is not isometric".into(),
                        ));
                    }
                }
            }
        }
        std::cmp::Ordering::Greater => {
            for a in 0..l {
                for b in 0..l {
                    let dot: Complex64 = (0..d * r)
                        .map(|x| v[a * d * r + x] * v[b * d * r + x].conj())
                        .sum();
                    if (dot
                        - if a == b {
                            Complex64::new(1., 0.)
                        } else {
                            Complex64::new(0., 0.)
                        })
                    .norm()
                        > tolerance
                    {
                        return Err(Error::InvalidValue(
                            "MPS right gauge is not isometric".into(),
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Check the complete chain, rather than inferring its norm from approximate gauges.
pub(crate) fn validate_norm(norm: Complex64) -> Result<()> {
    if !norm.re.is_finite() || !norm.im.is_finite() || (norm - Complex64::new(1., 0.)).norm() > 2e-5
    {
        return Err(Error::InvalidValue(
            "MPS global norm differs from one".into(),
        ));
    }
    Ok(())
}
