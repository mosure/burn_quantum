//! Batched pure-state simulation.
use crate::complex::{compatible, read};
use crate::error::tolerance;
use crate::{ComplexTensor, Error, Register, ResourceLimits, Result};
use burn::tensor::{DType, Device, Tensor, TensorData};

/// Batched pure states, stored as `[batch, product(local_dimensions)]`.
#[derive(Clone, Debug)]
pub struct StateVector {
    register: Register,
    amplitudes: ComplexTensor<2>,
}

pub(crate) fn compatible_states(left: &StateVector, right: &StateVector) -> Result<()> {
    if left.register() != right.register() {
        return Err(Error::Shape("state registers differ".into()));
    }
    let a = left.amplitudes();
    let b = right.amplitudes();
    compatible(&a.real(), &b.real())?;
    compatible(&a.real(), &b.imag())?;
    compatible(&a.imag(), &b.real())?;
    compatible(&a.imag(), &b.imag())
}

impl StateVector {
    /// Prepare a computational-basis state in every batch entry.
    pub fn basis(
        register: Register,
        digits: &[usize],
        batch: usize,
        device: &Device,
        dtype: DType,
        limits: ResourceLimits,
    ) -> Result<Self> {
        crate::backend::supported_dtype(dtype)?;
        let index = register.encode(digits)?;
        let count = limits.check_state(batch, register.size())?;
        let mut real = vec![0.0f64; count];
        for b in 0..batch {
            real[b * register.size() + index] = 1.;
        }
        let shape = [batch, register.size()];
        let amplitudes = ComplexTensor::new(
            Tensor::from_data(TensorData::new(real, shape), (device, dtype)),
            Tensor::zeros(shape, (device, dtype)),
        )?;
        Self::from_amplitudes(register, amplitudes, limits)
    }
    /// Prepare all zeros in the computational basis.
    pub fn zero(
        register: Register,
        batch: usize,
        device: &Device,
        dtype: DType,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let digits = vec![0; register.wires()];
        Self::basis(register, &digits, batch, device, dtype, limits)
    }
    /// Import amplitudes with metadata validation only, preserving gradients.
    /// Use [`Self::validate`] to check finite values and unit norm explicitly.
    pub fn from_amplitudes(
        register: Register,
        amplitudes: ComplexTensor<2>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let [batch, n] = amplitudes.dims();
        if n != register.size() {
            return Err(Error::Shape("amplitudes do not match register".into()));
        }
        limits.check_state(batch, n)?;
        Ok(Self {
            register,
            amplitudes,
        })
    }
    /// Register layout.
    pub fn register(&self) -> &Register {
        &self.register
    }
    /// Complex amplitudes. Cloning their components preserves autodiff.
    pub fn amplitudes(&self) -> &ComplexTensor<2> {
        &self.amplitudes
    }
    /// Batch size.
    pub fn batch_size(&self) -> usize {
        self.amplitudes.dims()[0]
    }
    /// Tensor device.
    pub fn device(&self) -> Device {
        self.amplitudes.device()
    }
    /// Exact computational-basis probabilities; differentiable.
    pub fn probabilities(&self) -> Tensor<2> {
        self.amplitudes.abs_squared()
    }
    /// Squared norm per batch entry.
    pub fn norm_squared(&self) -> Tensor<1> {
        self.probabilities().sum_dim(1).reshape([self.batch_size()])
    }
    /// Pairwise pure-state fidelities `|<self_b | other_b>|²`, retaining gradients.
    /// Registers, batch counts, dtypes and execution contexts must match. Inputs
    /// must be normalized; this metadata-only operation does not rescale them.
    pub fn fidelity(&self, other: &Self) -> Result<Tensor<1>> {
        compatible_states(self, other)?;
        if self.batch_size() != other.batch_size() {
            return Err(Error::Shape("fidelity batch counts differ".into()));
        }
        let a = self.amplitudes();
        let b = other.amplitudes();
        let re = (a.real() * b.real() + a.imag() * b.imag()).sum_dim(1);
        let im = (a.real() * b.imag() - a.imag() * b.real()).sum_dim(1);
        Ok((re.square() + im.square()).reshape([self.batch_size()]))
    }
    /// Explicit synchronous validation of finite normalized states.
    pub fn validate(&self, tol: f64) -> Result<()> {
        tolerance(tol)?;
        for norm in read(self.norm_squared())? {
            if !norm.is_finite() || (norm - 1.).abs() > tol {
                return Err(Error::InvalidValue(format!("state norm squared is {norm}")));
            }
        }
        Ok(())
    }
    /// Marginal probabilities for an ordered nonempty subset of wires.
    pub fn marginal(&self, wires: &[usize], limits: ResourceLimits) -> Result<Tensor<2>> {
        use burn::tensor::Int;
        let k = self.register.local_size(wires)?;
        let n = self.register.size();
        let b = self.batch_size();
        limits.check_state(b, n)?;
        limits.check_plan(n)?;
        let (forward, _) = self.register.permutation(wires)?;
        let indices = Tensor::<1, Int>::from_data(TensorData::new(forward, [n]), &self.device());
        Ok(self
            .probabilities()
            .select(1, indices)
            .reshape([b, n / k, k])
            .sum_dim(1)
            .reshape([b, k]))
    }
}
