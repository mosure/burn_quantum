//! Exact mixed states, partial traces, and completely positive trace-preserving channels.
use crate::complex::read;
use crate::error::{checked_product, tolerance};
use crate::{
    ComplexTensor, DenseSimulator, Error, Matrix, Observable, Register, ResourceLimits, Result,
    StateVector, Unitary,
};
use burn::tensor::{Int, Tensor, TensorData};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// A validated finite Kraus decomposition satisfying sum(K†K) = I.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "Vec<Matrix>", into = "Vec<Matrix>")]
pub struct KrausChannel {
    operators: Vec<Matrix>,
}
impl KrausChannel {
    /// Check finite, equal-size operators and trace preservation at 1e-10.
    pub fn new(operators: Vec<Matrix>) -> Result<Self> {
        let n = operators
            .first()
            .ok_or_else(|| Error::Shape("empty Kraus channel".into()))?
            .dimension();
        if operators.iter().any(|op| op.dimension() != n) {
            return Err(Error::Shape("Kraus dimensions differ".into()));
        }
        for a in 0..n {
            for b in 0..n {
                let mut sum = Complex64::new(0., 0.);
                for op in &operators {
                    for k in 0..n {
                        sum += op.entries()[k * n + a].conj() * op.entries()[k * n + b];
                    }
                }
                if !sum.re.is_finite()
                    || !sum.im.is_finite()
                    || (sum - if a == b { 1. } else { 0. }).norm() > 1e-10
                {
                    return Err(Error::InvalidValue(
                        "Kraus channel is not trace preserving".into(),
                    ));
                }
            }
        }
        Ok(Self { operators })
    }
    /// Local Kraus operators.
    pub fn operators(&self) -> &[Matrix] {
        &self.operators
    }
    /// Qubit amplitude damping with probability gamma.
    pub fn amplitude_damping(gamma: f64) -> Result<Self> {
        probability(gamma)?;
        Self::new(vec![
            Matrix::real(2, vec![1., 0., 0., (1. - gamma).sqrt()])?,
            Matrix::real(2, vec![0., gamma.sqrt(), 0., 0.])?,
        ])
    }
    /// Qubit phase-flip channel: (1-p) rho + p Z rho Z.
    pub fn phase_flip(p: f64) -> Result<Self> {
        probability(p)?;
        Self::new(vec![
            Matrix::real(2, vec![(1. - p).sqrt(), 0., 0., (1. - p).sqrt()])?,
            Matrix::real(2, vec![p.sqrt(), 0., 0., -p.sqrt()])?,
        ])
    }
    /// Qubit depolarizing channel: (1-p) rho + p I/2.
    pub fn depolarizing(p: f64) -> Result<Self> {
        probability(p)?;
        let gates = [
            Unitary::identity(2)?,
            Unitary::x(),
            Unitary::y(),
            Unitary::z(),
        ];
        let scales = [
            (1. - 3. * p / 4.).sqrt(),
            (p / 4.).sqrt(),
            (p / 4.).sqrt(),
            (p / 4.).sqrt(),
        ];
        let ops: Result<Vec<_>> = gates
            .iter()
            .zip(scales)
            .map(|(g, s)| Matrix::new(2, g.matrix().entries().iter().map(|x| *x * s).collect()))
            .collect();
        Self::new(ops?)
    }
}
fn probability(p: f64) -> Result<()> {
    if p.is_finite() && (0.0..=1.0).contains(&p) {
        Ok(())
    } else {
        Err(Error::InvalidValue(
            "channel probability outside [0,1]".into(),
        ))
    }
}
impl TryFrom<Vec<Matrix>> for KrausChannel {
    type Error = Error;
    fn try_from(v: Vec<Matrix>) -> Result<Self> {
        Self::new(v)
    }
}
impl From<KrausChannel> for Vec<Matrix> {
    fn from(v: KrausChannel) -> Self {
        v.operators
    }
}

/// Batched density matrices `[batch, row, column]` on a finite register.
#[derive(Clone, Debug)]
pub struct DensityMatrix {
    register: Register,
    entries: ComplexTensor<3>,
}
impl DensityMatrix {
    /// Metadata-only import. Call [`Self::validate`] for physical-state checks.
    pub fn from_entries(
        register: Register,
        entries: ComplexTensor<3>,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let [b, r, c] = entries.dims();
        if r != register.size() || c != r {
            return Err(Error::Shape(
                "density matrix shape differs from register".into(),
            ));
        }
        let n2 = checked_product([r, c])?;
        if n2 > i32::MAX as usize {
            return Err(Error::Unsupported(
                "density gather indices exceed signed 32-bit range".into(),
            ));
        }
        limits.check_state(b, n2)?;
        Ok(Self { register, entries })
    }
    /// Form |psi><psi| with no host readback, preserving gradients.
    pub fn from_pure(state: &StateVector, limits: ResourceLimits) -> Result<Self> {
        let b = state.batch_size();
        let n = state.register().size();
        limits.check_state(b, checked_product([n, n])?)?;
        let re = state.amplitudes().real().reshape([b, n, 1]);
        let im = state.amplitudes().imag().reshape([b, n, 1]);
        let out_re = crate::precision::matmul(re.clone(), re.clone().transpose())
            + crate::precision::matmul(im.clone(), im.clone().transpose());
        let out_im = crate::precision::matmul(im, re.clone().transpose())
            - crate::precision::matmul(re, state.amplitudes().imag().reshape([b, 1, n]));
        Self::from_entries(
            state.register().clone(),
            ComplexTensor::new(out_re, out_im)?,
            limits,
        )
    }
    /// Register layout.
    pub fn register(&self) -> &Register {
        &self.register
    }
    /// Split-complex matrix entries.
    pub fn entries(&self) -> &ComplexTensor<3> {
        &self.entries
    }
    /// Number of density matrices.
    pub fn batch_size(&self) -> usize {
        self.entries.dims()[0]
    }
    /// Exact computational-basis probabilities, extracted from the real diagonal.
    pub fn probabilities(&self) -> Tensor<2> {
        let n = self.register.size();
        let b = self.batch_size();
        let indices: Vec<i64> = (0..n).map(|i| (i * n + i) as i64).collect();
        let indices =
            Tensor::<1, Int>::from_data(TensorData::new(indices, [n]), &self.entries.device());
        self.entries.real().reshape([b, n * n]).select(1, indices)
    }
    /// Trace per batch entry.
    pub fn trace(&self) -> Tensor<1> {
        self.probabilities().sum_dim(1).reshape([self.batch_size()])
    }
    /// Purity Tr(rho²) for a Hermitian input.
    pub fn purity(&self) -> Tensor<1> {
        self.entries
            .abs_squared()
            .sum_dim(2)
            .sum_dim(1)
            .reshape([self.batch_size()])
    }
    /// Explicit host validation of finiteness, Hermiticity, unit trace, and
    /// positive semidefiniteness up to tolerance (regularized Cholesky, O(N³)).
    pub fn validate(&self, tol: f64) -> Result<()> {
        tolerance(tol)?;
        let re = read(self.entries.real())?;
        let im = read(self.entries.imag())?;
        let n = self.register.size();
        for batch in 0..self.batch_size() {
            let start = batch * n * n;
            let values: Vec<Complex64> = (0..n * n)
                .map(|i| Complex64::new(re[start + i], im[start + i]))
                .collect();
            Matrix::new(n, values.clone())?.check_hermitian(tol)?;
            let trace: Complex64 = (0..n).map(|i| values[i * n + i]).sum();
            if (trace - 1.).norm() > tol {
                return Err(Error::InvalidValue("density trace is not one".into()));
            }
            let mut lower = vec![Complex64::new(0., 0.); n * n];
            for i in 0..n {
                for j in 0..=i {
                    let residual = values[i * n + j]
                        - (0..j)
                            .map(|k| lower[i * n + k] * lower[j * n + k].conj())
                            .sum::<Complex64>();
                    if i == j {
                        let pivot = residual.re + tol;
                        if !pivot.is_finite() || pivot <= 0. {
                            return Err(Error::InvalidValue(
                                "density matrix is not positive semidefinite".into(),
                            ));
                        }
                        lower[i * n + j] = pivot.sqrt().into();
                    } else {
                        lower[i * n + j] = residual / lower[j * n + j].re;
                    }
                }
            }
        }
        Ok(())
    }
    pub(crate) fn left_right(
        &self,
        engine: &DenseSimulator,
        wires: &[usize],
        matrix: &Matrix,
    ) -> Result<Self> {
        let n = self.register.size();
        let b = self.batch_size();
        let n2 = checked_product([n, n])?;
        self.register.local_size(wires)?;
        engine.limits.check_state(b, n2)?;
        engine.limits.check_matrix(1, matrix.dimension())?;
        let doubled = Register::new(
            self.register
                .dimensions()
                .iter()
                .chain(self.register.dimensions())
                .copied()
                .collect(),
        )?;
        let flat = ComplexTensor::new(
            self.entries.real().reshape([b, n2]),
            self.entries.imag().reshape([b, n2]),
        )?;
        let op = matrix.to_tensor(&self.entries.device(), self.entries.dtype())?;
        let rows = engine.contract(&doubled, &flat, wires, &op)?;
        let columns: Vec<_> = wires.iter().map(|w| w + self.register.wires()).collect();
        let out = engine.contract(&doubled, &rows, &columns, &op.conj())?;
        Self::from_entries(
            self.register.clone(),
            ComplexTensor::new(out.real().reshape([b, n, n]), out.imag().reshape([b, n, n]))?,
            engine.limits,
        )
    }
    /// Unitary evolution U rho U† on local targets.
    pub fn apply(&self, engine: &DenseSimulator, wires: &[usize], gate: &Unitary) -> Result<Self> {
        self.left_right(engine, wires, gate.matrix())
    }
    /// Exact local quantum channel sum(K rho K†).
    pub fn channel(
        &self,
        engine: &DenseSimulator,
        wires: &[usize],
        channel: &KrausChannel,
    ) -> Result<Self> {
        let mut result = self.left_right(engine, wires, &channel.operators[0])?;
        for matrix in &channel.operators[1..] {
            let term = self.left_right(engine, wires, matrix)?;
            result.entries = ComplexTensor::new(
                result.entries.re + term.entries.re,
                result.entries.im + term.entries.im,
            )?;
        }
        Ok(result)
    }
    /// Local expectation Tr(rho O), retaining all tensor gradients.
    pub fn expectation(
        &self,
        engine: &DenseSimulator,
        observable: &Observable,
    ) -> Result<Tensor<1>> {
        let reduced = self.reduced(observable.wires(), engine.limits)?;
        let n = reduced.register.size();
        if observable.matrix().dimension() != n {
            return Err(Error::Shape("observable dimension mismatch".into()));
        }
        engine.limits.check_matrix(1, n)?;
        let op = observable
            .matrix()
            .to_tensor(&self.entries.device(), self.entries.dtype())?;
        let product = reduced.entries.real() * op.real().transpose()
            - reduced.entries.imag() * op.imag().transpose();
        Ok(product.sum_dim(2).sum_dim(1).reshape([self.batch_size()]))
    }
    /// Partial trace, retaining `wires` in the requested order.
    pub fn reduced(&self, wires: &[usize], limits: ResourceLimits) -> Result<Self> {
        let k = self.register.local_size(wires)?;
        let n = self.register.size();
        let b = self.batch_size();
        let env = n / k;
        let k2 = checked_product([k, k])?;
        limits.check_state(b, k2)?;
        let count = checked_product([k2, env])?;
        limits.check_plan(count)?;
        let kept = Register::new(
            wires
                .iter()
                .map(|&w| self.register.dimensions()[w])
                .collect(),
        )?;
        let (packed, _) = self.register.permutation(wires)?;
        let mut indices = Vec::with_capacity(count);
        for row in 0..k {
            for col in 0..k {
                for e in 0..env {
                    let original_row = packed[e * k + row] as usize;
                    let original_col = packed[e * k + col] as usize;
                    indices.push((original_row * n + original_col) as i64);
                }
            }
        }
        let indices =
            Tensor::<1, Int>::from_data(TensorData::new(indices, [count]), &self.entries.device());
        let re = self
            .entries
            .real()
            .reshape([b, n * n])
            .select(1, indices.clone())
            .reshape([b, k2, env])
            .sum_dim(2)
            .reshape([b, k, k]);
        let im = self
            .entries
            .imag()
            .reshape([b, n * n])
            .select(1, indices)
            .reshape([b, k2, env])
            .sum_dim(2)
            .reshape([b, k, k]);
        Self::from_entries(kept, ComplexTensor::new(re, im)?, limits)
    }
}

impl StateVector {
    /// Reduced density matrix without first allocating the full density matrix.
    pub fn reduced_density(
        &self,
        wires: &[usize],
        limits: ResourceLimits,
    ) -> Result<DensityMatrix> {
        let k = self.register().local_size(wires)?;
        let n = self.register().size();
        let b = self.batch_size();
        limits.check_state(b, checked_product([k, k])?)?;
        limits.check_state(b, n)?;
        limits.check_plan(n)?;
        let (forward, _) = self.register().permutation(wires)?;
        let indices = Tensor::<1, Int>::from_data(TensorData::new(forward, [n]), &self.device());
        let re = self
            .amplitudes()
            .real()
            .select(1, indices.clone())
            .reshape([b, n / k, k]);
        let im = self
            .amplitudes()
            .imag()
            .select(1, indices)
            .reshape([b, n / k, k]);
        let out_re = crate::precision::matmul(re.clone().transpose(), re.clone())
            + crate::precision::matmul(im.clone().transpose(), im.clone());
        let out_im = crate::precision::matmul(im.clone().transpose(), re.clone())
            - crate::precision::matmul(re.transpose(), im);
        let register = Register::new(
            wires
                .iter()
                .map(|&w| self.register().dimensions()[w])
                .collect(),
        )?;
        DensityMatrix::from_entries(register, ComplexTensor::new(out_re, out_im)?, limits)
    }
}
