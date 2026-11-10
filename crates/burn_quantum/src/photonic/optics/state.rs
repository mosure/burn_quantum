use super::{FockBasis, Interferometer, ModeId, PhotonSector};
use crate::{ComplexTensor, DensityMatrix, Error, Matrix, Register, ResourceLimits, Result};
use burn::tensor::{DType, Device, Int, Tensor, TensorData};

/// Batched pure Fock amplitudes `[batch,basis]` on an explicit occupation basis.
#[derive(Clone, Debug)]
pub struct FockState {
    basis: FockBasis,
    amplitudes: ComplexTensor<2>,
}
impl FockState {
    /// Import amplitudes after metadata checks; call `validate` to check normalization.
    pub fn from_amplitudes(
        basis: FockBasis,
        amplitudes: ComplexTensor<2>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let [batch, dim] = amplitudes.dims();
        if dim != basis.size() {
            return Err(Error::Shape("Fock amplitude/basis mismatch".into()));
        }
        limits.check_state(batch, dim)?;
        Ok(Self { basis, amplitudes })
    }
    /// Prepare the same exact photon occupation in each batch entry.
    pub fn number(
        basis: FockBasis,
        occupation: &[usize],
        batch: usize,
        device: &Device,
        dtype: DType,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let dim = basis.size();
        let index = basis.index(occupation)?;
        let size = limits.check_state(batch, dim)?;
        crate::backend::supported_dtype(dtype)?;
        let mut data = vec![0.; size];
        for b in 0..batch {
            data[b * dim + index] = 1.;
        }
        let re = Tensor::from_data(TensorData::new(data, [batch, dim]), (device, dtype));
        Self::from_amplitudes(
            basis,
            ComplexTensor::new(re.clone(), re.zeros_like())?,
            limits,
        )
    }
    /// Immutable mode/occupation layout.
    pub fn basis(&self) -> &FockBasis {
        &self.basis
    }
    /// Split-complex amplitudes retaining the native autodiff graph.
    pub fn amplitudes(&self) -> &ComplexTensor<2> {
        &self.amplitudes
    }
    /// Exact unconditioned probabilities in occupation order.
    pub fn probabilities(&self) -> Tensor<2> {
        self.amplitudes.abs_squared()
    }
    /// Explicit host normalization and finiteness validation.
    pub fn validate(&self, tolerance: f64) -> Result<()> {
        crate::error::tolerance(tolerance)?;
        for row in crate::complex::read(self.probabilities())?.chunks_exact(self.basis.size()) {
            if row.iter().any(|x| !x.is_finite())
                || (row.iter().sum::<f64>() - 1.).abs() > tolerance
            {
                return Err(Error::InvalidValue("invalid Fock pure state".into()));
            }
        }
        Ok(())
    }
    /// Apply a passive interferometer exactly in the retained total sectors.
    pub fn apply(&self, interferometer: &Interferometer, limits: ResourceLimits) -> Result<Self> {
        let [batch, dim] = self.amplitudes.dims();
        limits.check_state(batch, dim)?;
        let op = interferometer
            .lift(&self.basis, limits)?
            .to_tensor(&self.amplitudes.device(), self.amplitudes.dtype())?;
        let re = self.amplitudes.real().reshape([batch, 1, dim]);
        let im = self.amplitudes.imag().reshape([batch, 1, dim]);
        let ur = op.real().transpose();
        let ui = op.imag().transpose();
        let real = crate::precision::matmul(re.clone(), ur.clone())
            - crate::precision::matmul(im.clone(), ui.clone());
        let imag = crate::precision::matmul(re, ui) + crate::precision::matmul(im, ur);
        Self::from_amplitudes(
            self.basis.clone(),
            ComplexTensor::new(real.reshape([batch, dim]), imag.reshape([batch, dim]))?,
            limits,
        )
    }
    /// Apply exp(i n phi) with a scalar or per-batch native angle tensor.
    /// Values remain on-device; callers supply finite angles.
    pub fn phase_shift(
        &self,
        mode: ModeId,
        angle: Tensor<1>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let [batch, dim] = self.amplitudes.dims();
        limits.check_state(batch, dim)?;
        let phase = phase_factors(&self.basis, mode, angle, &self.amplitudes.real(), batch)?;
        let re = self.amplitudes.real();
        let im = self.amplitudes.imag();
        Self::from_amplitudes(
            self.basis.clone(),
            ComplexTensor::new(
                re.clone() * phase.0.clone() - im.clone() * phase.1.clone(),
                re * phase.1 + im * phase.0,
            )?,
            limits,
        )
    }
    /// Form the exact density matrix, preserving native gradients.
    pub fn density(&self, limits: ResourceLimits) -> Result<FockDensity> {
        let [b, n] = self.amplitudes.dims();
        limits.check_state(b, n.checked_mul(n).ok_or(Error::Overflow)?)?;
        let re = self.amplitudes.real();
        let im = self.amplitudes.imag();
        let real = re.clone().reshape([b, n, 1]) * re.clone().reshape([b, 1, n])
            + im.clone().reshape([b, n, 1]) * im.clone().reshape([b, 1, n]);
        let imag = im.reshape([b, n, 1]) * re.clone().reshape([b, 1, n])
            - re.reshape([b, n, 1]) * self.amplitudes.imag().reshape([b, 1, n]);
        FockDensity::from_entries(self.basis.clone(), ComplexTensor::new(real, imag)?, limits)
    }
}

/// Exact mixed Fock state `[batch,basis,basis]`, including loss sectors.
#[derive(Clone, Debug)]
pub struct FockDensity {
    basis: FockBasis,
    entries: ComplexTensor<3>,
}
impl FockDensity {
    /// Metadata-only import; use `validate` for physical-state validation.
    pub fn from_entries(
        basis: FockBasis,
        entries: ComplexTensor<3>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let [b, n, m] = entries.dims();
        if n != basis.size() || m != n {
            return Err(Error::Shape("Fock density/basis mismatch".into()));
        }
        let size = n.checked_mul(n).ok_or(Error::Overflow)?;
        if size > i32::MAX as usize {
            return Err(Error::Unsupported("Fock density index range".into()));
        }
        limits.check_state(b, size)?;
        Ok(Self { basis, entries })
    }
    /// Immutable mode/occupation layout.
    pub fn basis(&self) -> &FockBasis {
        &self.basis
    }
    /// Split-complex matrix entries.
    pub fn entries(&self) -> &ComplexTensor<3> {
        &self.entries
    }
    /// Exact occupation probabilities, with no postselection.
    pub fn probabilities(&self) -> Tensor<2> {
        let [b, n, _] = self.entries.dims();
        let indices = Tensor::<1, Int>::from_data(
            TensorData::new((0..n).map(|i| (i * n + i) as i64).collect(), [n]),
            &self.entries.device(),
        );
        self.entries.real().reshape([b, n * n]).select(1, indices)
    }
    /// Trace per batch item.
    pub fn trace(&self) -> Tensor<1> {
        self.probabilities()
            .sum_dim(1)
            .reshape([self.entries.dims()[0]])
    }
    /// Check Hermiticity, unit trace and positivity on host.
    pub fn validate(&self, tolerance: f64) -> Result<()> {
        if self.basis.size() > 1 {
            DensityMatrix::from_entries(
                Register::new(vec![self.basis.size()])?,
                self.entries.clone(),
                ResourceLimits {
                    max_state_elements: self.entries.dims().iter().product(),
                    ..Default::default()
                },
            )?
            .validate(tolerance)
        } else {
            crate::error::tolerance(tolerance)?;
            if crate::complex::read(self.entries.real())?
                .iter()
                .any(|x| !x.is_finite() || (*x - 1.).abs() > tolerance)
                || crate::complex::read(self.entries.imag())?
                    .iter()
                    .any(|x| !x.is_finite() || x.abs() > tolerance)
            {
                Err(Error::InvalidValue("invalid vacuum density".into()))
            } else {
                Ok(())
            }
        }
    }
    /// Apply the physical passive interferometer without sector leakage.
    pub fn apply(&self, interferometer: &Interferometer, limits: ResourceLimits) -> Result<Self> {
        self.left_right(&interferometer.lift(&self.basis, limits)?, limits)
    }
    /// Native differentiable phase evolution exp(i(n_row-n_column) phi).
    pub fn phase_shift(
        &self,
        mode: ModeId,
        angle: Tensor<1>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let [b, n, _] = self.entries.dims();
        limits.check_state(b, n.checked_mul(n).ok_or(Error::Overflow)?)?;
        let (co, si) = phase_factors(&self.basis, mode, angle, &self.probabilities(), b)?;
        let re = co.clone().reshape([b, n, 1]) * co.clone().reshape([b, 1, n])
            + si.clone().reshape([b, n, 1]) * si.clone().reshape([b, 1, n]);
        let im = si.clone().reshape([b, n, 1]) * co.clone().reshape([b, 1, n])
            - co.reshape([b, n, 1]) * si.reshape([b, 1, n]);
        let real = self.entries.real() * re.clone() - self.entries.imag() * im.clone();
        let imag = self.entries.real() * im + self.entries.imag() * re;
        Self::from_entries(self.basis.clone(), ComplexTensor::new(real, imag)?, limits)
    }
    /// Independent pure loss on one mode, with intensity transmission in `[0,1]`.
    /// Requires every lower photon sector; fixed positive sectors reject actual loss.
    pub fn lose(&self, mode: ModeId, transmission: f64, limits: ResourceLimits) -> Result<Self> {
        self.basis.check_mode(mode)?;
        if !transmission.is_finite() || !(0.0..=1.).contains(&transmission) {
            return Err(Error::InvalidValue("invalid optical transmission".into()));
        }
        if transmission == 1. {
            return Ok(self.clone());
        }
        if matches!(self.basis.sector(),PhotonSector::Fixed(n) if n>0) {
            return Err(Error::Unsupported(
                "loss needs every lower total-photon sector".into(),
            ));
        }
        let [batch, d, _] = self.entries.dims();
        limits.check_state(batch, d.checked_mul(d).ok_or(Error::Overflow)?)?;
        limits.check_matrix(1, d)?;
        let mut real = self.entries.real().zeros_like();
        let mut imag = self.entries.imag().zeros_like();
        for lost in 0..=self.basis.max_photons() {
            let mut entries = vec![0.; d * d];
            for (input, occupation) in self.basis.occupations().iter().enumerate() {
                let n = occupation[mode.0];
                if n < lost {
                    continue;
                }
                let mut output = occupation.clone();
                output[mode.0] -= lost;
                let amplitude = (binomial(n, lost)
                    * transmission.powi((n - lost) as i32)
                    * (1. - transmission).powi(lost as i32))
                .sqrt();
                entries[self.basis.index(&output)? * d + input] = amplitude;
            }
            let branch = self.left_right(&Matrix::real(d, entries)?, limits)?;
            real = real + branch.entries.real();
            imag = imag + branch.entries.imag();
        }
        Self::from_entries(self.basis.clone(), ComplexTensor::new(real, imag)?, limits)
    }
    fn left_right(&self, matrix: &Matrix, limits: ResourceLimits) -> Result<Self> {
        let [b, n, _] = self.entries.dims();
        limits.check_state(b, n.checked_mul(n).ok_or(Error::Overflow)?)?;
        limits.check_matrix(1, n)?;
        let op = matrix.to_tensor(&self.entries.device(), self.entries.dtype())?;
        let ur = op.real();
        let ui = op.imag();
        let re = self.entries.real();
        let im = self.entries.imag();
        let tr = crate::precision::matmul(ur.clone(), re.clone())
            - crate::precision::matmul(ui.clone(), im.clone());
        let ti =
            crate::precision::matmul(ur.clone(), im) + crate::precision::matmul(ui.clone(), re);
        let real = crate::precision::matmul(tr.clone(), ur.clone().transpose())
            + crate::precision::matmul(ti.clone(), ui.clone().transpose());
        let imag = crate::precision::matmul(ti, ur.transpose())
            - crate::precision::matmul(tr, ui.transpose());
        Self::from_entries(self.basis.clone(), ComplexTensor::new(real, imag)?, limits)
    }
}
fn phase_factors<const D: usize>(
    basis: &FockBasis,
    mode: ModeId,
    angle: Tensor<1>,
    tensor: &Tensor<D>,
    batch: usize,
) -> Result<(Tensor<2>, Tensor<2>)> {
    basis.check_mode(mode)?;
    crate::complex::compatible(&angle, tensor)?;
    let [angles] = angle.dims();
    if angles != 1 && angles != batch {
        return Err(Error::Shape(
            "phase angle must have one or batch entries".into(),
        ));
    }
    let numbers: Vec<_> = basis
        .occupations()
        .iter()
        .map(|o| o[mode.0] as f64)
        .collect();
    let counts = Tensor::<2>::from_data(
        TensorData::new(numbers, [1, basis.size()]),
        (&angle.device(), angle.dtype()),
    );
    let phase = angle.reshape([angles, 1]).expand([batch, 1]) * counts;
    Ok((phase.clone().cos(), phase.sin()))
}
pub(crate) fn binomial(n: usize, k: usize) -> f64 {
    (1..=k.min(n - k)).fold(1., |x, i| x * (n + 1 - i) as f64 / i as f64)
}
