//! Validated constant matrices and differentiable two-level rotations.
use crate::error::{checked_product, tolerance};
use crate::{ComplexTensor, Error, Result};
use burn::tensor::{DType, Device, Tensor, TensorData};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

/// Maximum entries allocated by built-in dense gate constructors.
pub const MAX_BUILTIN_MATRIX_ELEMENTS: usize = 65_536;

#[derive(Serialize, Deserialize)]
struct MatrixData {
    dimension: usize,
    entries: Vec<Complex64>,
}

/// Finite square row-major complex matrix. Not necessarily unitary or Hermitian.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "MatrixData", into = "MatrixData")]
pub struct Matrix {
    dimension: usize,
    entries: Vec<Complex64>,
}

impl Matrix {
    /// Construct a finite square matrix, with entries indexed `[output, input]`.
    pub fn new(dimension: usize, entries: Vec<Complex64>) -> Result<Self> {
        let length = checked_product([dimension, dimension])?;
        if dimension == 0 || entries.len() != length {
            return Err(Error::Shape("matrix must be nonempty and square".into()));
        }
        if entries
            .iter()
            .any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            return Err(Error::InvalidValue(
                "matrix contains nonfinite entries".into(),
            ));
        }
        Ok(Self { dimension, entries })
    }
    /// Construct from a real-valued row-major matrix.
    pub fn real(dimension: usize, entries: Vec<f64>) -> Result<Self> {
        Self::new(
            dimension,
            entries
                .into_iter()
                .map(|x| Complex64::new(x, 0.0))
                .collect(),
        )
    }
    /// Matrix dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Row-major entries.
    pub fn entries(&self) -> &[Complex64] {
        &self.entries
    }
    /// Hermitian adjoint.
    pub fn adjoint(&self) -> Self {
        let n = self.dimension;
        let entries = (0..n * n)
            .map(|i| self.entries[(i % n) * n + i / n].conj())
            .collect();
        Self {
            dimension: n,
            entries,
        }
    }
    /// Convert to split-complex tensors of shape `[1, output, input]`.
    pub fn to_tensor(&self, device: &Device, dtype: DType) -> Result<ComplexTensor<3>> {
        crate::backend::supported_dtype(dtype)?;
        ComplexTensor::new(
            Tensor::from_data(
                TensorData::new(
                    self.entries.iter().map(|x| x.re).collect(),
                    [1, self.dimension, self.dimension],
                ),
                (device, dtype),
            ),
            Tensor::from_data(
                TensorData::new(
                    self.entries.iter().map(|x| x.im).collect(),
                    [1, self.dimension, self.dimension],
                ),
                (device, dtype),
            ),
        )
    }
    pub(crate) fn check_unitary(&self, tol: f64) -> Result<()> {
        tolerance(tol)?;
        let n = self.dimension;
        for a in 0..n {
            for b in 0..n {
                let dot: Complex64 = (0..n)
                    .map(|k| self.entries[k * n + a].conj() * self.entries[k * n + b])
                    .sum();
                if !dot.re.is_finite()
                    || !dot.im.is_finite()
                    || (dot - if a == b { 1.0 } else { 0.0 }).norm() > tol
                {
                    return Err(Error::InvalidValue("matrix is not unitary".into()));
                }
            }
        }
        Ok(())
    }
    pub(crate) fn check_hermitian(&self, tol: f64) -> Result<()> {
        tolerance(tol)?;
        let n = self.dimension;
        for a in 0..n {
            for b in 0..n {
                if (self.entries[a * n + b] - self.entries[b * n + a].conj()).norm() > tol {
                    return Err(Error::InvalidValue("observable is not Hermitian".into()));
                }
            }
        }
        Ok(())
    }
}
impl TryFrom<MatrixData> for Matrix {
    type Error = Error;
    fn try_from(v: MatrixData) -> Result<Self> {
        Self::new(v.dimension, v.entries)
    }
}
impl From<Matrix> for MatrixData {
    fn from(v: Matrix) -> Self {
        Self {
            dimension: v.dimension,
            entries: v.entries,
        }
    }
}

/// Validated constant unitary. Deserialization revalidates at tolerance 1e-10.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Matrix", into = "Matrix")]
pub struct Unitary(Matrix);

impl Unitary {
    /// Validate a constant unitary with an absolute entrywise tolerance of 1e-10.
    pub fn new(matrix: Matrix) -> Result<Self> {
        matrix.check_unitary(1e-10)?;
        Ok(Self(matrix))
    }
    /// Underlying matrix.
    pub fn matrix(&self) -> &Matrix {
        &self.0
    }
    /// Inverse unitary.
    pub fn adjoint(&self) -> Self {
        Self(self.0.adjoint())
    }
    /// Identity on a local dimension.
    pub fn identity(dimension: usize) -> Result<Self> {
        let mut entries = blank(dimension)?;
        for i in 0..dimension {
            entries[i * dimension + i] = Complex64::new(1.0, 0.0);
        }
        Ok(Self(Matrix::new(dimension, entries)?))
    }
    /// Qubit Pauli X.
    pub fn x() -> Self {
        Self(Matrix::real(2, vec![0., 1., 1., 0.]).unwrap())
    }
    /// Qubit Pauli Y.
    pub fn y() -> Self {
        Self(
            Matrix::new(
                2,
                vec![
                    0.0.into(),
                    Complex64::new(0., -1.),
                    Complex64::new(0., 1.),
                    0.0.into(),
                ],
            )
            .unwrap(),
        )
    }
    /// Qubit Pauli Z.
    pub fn z() -> Self {
        Self(Matrix::real(2, vec![1., 0., 0., -1.]).unwrap())
    }
    /// Qubit Hadamard.
    pub fn h() -> Self {
        Self(
            Matrix::real(
                2,
                vec![FRAC_1_SQRT_2, FRAC_1_SQRT_2, FRAC_1_SQRT_2, -FRAC_1_SQRT_2],
            )
            .unwrap(),
        )
    }
    /// Qubit phase gate diag(1, exp(i angle)).
    pub fn phase(angle: f64) -> Result<Self> {
        if !angle.is_finite() {
            return Err(Error::InvalidValue("nonfinite phase".into()));
        }
        Ok(Self(Matrix::new(
            2,
            vec![
                1.0.into(),
                0.0.into(),
                0.0.into(),
                Complex64::from_polar(1.0, angle),
            ],
        )?))
    }
    /// S gate.
    pub fn s() -> Self {
        Self::phase(PI / 2.).unwrap()
    }
    /// T gate.
    pub fn t() -> Self {
        Self::phase(PI / 4.).unwrap()
    }
    /// Generalized shift: |j> maps to |j + amount mod d>.
    pub fn shift(dimension: usize, amount: usize) -> Result<Self> {
        let mut entries = blank(dimension)?;
        for j in 0..dimension {
            entries[((j + amount % dimension) % dimension) * dimension + j] = 1.0.into();
        }
        Ok(Self(Matrix::new(dimension, entries)?))
    }
    /// Generalized clock: |j> maps to exp(2 pi i j/d)|j>.
    pub fn clock(dimension: usize) -> Result<Self> {
        let mut entries = blank(dimension)?;
        for j in 0..dimension {
            entries[j * dimension + j] =
                Complex64::from_polar(1.0, TAU * j as f64 / dimension as f64);
        }
        Ok(Self(Matrix::new(dimension, entries)?))
    }
    /// Positive-sign discrete Fourier transform on a qudit.
    pub fn fourier(dimension: usize) -> Result<Self> {
        let mut entries = blank(dimension)?;
        for j in 0..dimension {
            for k in 0..dimension {
                entries[j * dimension + k] = Complex64::from_polar(
                    1.0 / (dimension as f64).sqrt(),
                    TAU * j as f64 * k as f64 / dimension as f64,
                );
            }
        }
        Ok(Self(Matrix::new(dimension, entries)?))
    }
    /// SUM on equal-dimensional qudits: |a,b> maps to |a,a+b mod d>.
    pub fn sum(dimension: usize) -> Result<Self> {
        if dimension < 2 {
            return Err(Error::InvalidDimensions("SUM dimension < 2".into()));
        }
        let n = checked_product([dimension, dimension])?;
        let mut entries = blank(n)?;
        for a in 0..dimension {
            for b in 0..dimension {
                entries[(a * dimension + (a + b) % dimension) * n + a * dimension + b] = 1.0.into();
            }
        }
        Ok(Self(Matrix::new(n, entries)?))
    }
    /// Controlled NOT, with the first target wire the control.
    pub fn cnot() -> Self {
        Self::sum(2).unwrap()
    }
    /// Controlled constant gate, activating on one specified control level.
    pub fn controlled(&self, control_dimension: usize, level: usize) -> Result<Self> {
        if control_dimension < 2 || level >= control_dimension {
            return Err(Error::Basis);
        }
        let k = self.0.dimension;
        let n = checked_product([control_dimension, k])?;
        let mut entries = blank(n)?;
        for c in 0..control_dimension {
            for a in 0..k {
                for b in 0..k {
                    entries[(c * k + a) * n + c * k + b] = if c == level {
                        self.0.entries[a * k + b]
                    } else {
                        (if a == b { 1.0 } else { 0.0 }).into()
                    };
                }
            }
        }
        Ok(Self(Matrix::new(n, entries)?))
    }
    /// Swap two equal-dimensional qudits.
    pub fn swap(dimension: usize) -> Result<Self> {
        if dimension < 2 {
            return Err(Error::InvalidDimensions("SWAP dimension < 2".into()));
        }
        let n = checked_product([dimension, dimension])?;
        let mut entries = blank(n)?;
        for a in 0..dimension {
            for b in 0..dimension {
                entries[(b * dimension + a) * n + a * dimension + b] = 1.0.into();
            }
        }
        Ok(Self(Matrix::new(n, entries)?))
    }
}
impl TryFrom<Matrix> for Unitary {
    type Error = Error;
    fn try_from(m: Matrix) -> Result<Self> {
        Self::new(m)
    }
}
impl From<Unitary> for Matrix {
    fn from(u: Unitary) -> Matrix {
        u.0
    }
}

fn blank(dimension: usize) -> Result<Vec<Complex64>> {
    if dimension < 2 {
        return Err(Error::InvalidDimensions("gate dimension < 2".into()));
    }
    let required = checked_product([dimension, dimension])?;
    if required > MAX_BUILTIN_MATRIX_ELEMENTS {
        return Err(Error::ResourceLimit {
            resource: "built-in gate",
            required,
            limit: MAX_BUILTIN_MATRIX_ELEMENTS,
        });
    }
    Ok(vec![Complex64::new(0., 0.); required])
}

/// Rotation axis in a selected two-level subspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    /// exp(-i theta X / 2).
    X,
    /// exp(-i theta Y / 2).
    Y,
    /// exp(-i theta Z / 2).
    Z,
}

/// Construct a batch of analytic SU(2) rotations embedded in a qudit.
///
/// `angle` has shape `[batch]`; no values leave the autodiff graph.
pub fn rotation(
    axis: Axis,
    dimension: usize,
    levels: [usize; 2],
    angle: Tensor<1>,
) -> Result<ComplexTensor<3>> {
    let [a, b] = levels;
    if a == b || a >= dimension || b >= dimension {
        return Err(Error::Basis);
    }
    let mut identity = blank(dimension)?;
    let mut cosine = identity.clone();
    let mut sine = identity.clone();
    for j in 0..dimension {
        if j == a || j == b {
            cosine[j * dimension + j] = 1.0.into();
        } else {
            identity[j * dimension + j] = 1.0.into();
        }
    }
    match axis {
        Axis::X => {
            sine[a * dimension + b] = Complex64::new(0., -1.);
            sine[b * dimension + a] = Complex64::new(0., -1.);
        }
        Axis::Y => {
            sine[a * dimension + b] = (-1.0).into();
            sine[b * dimension + a] = 1.0.into();
        }
        Axis::Z => {
            sine[a * dimension + a] = Complex64::new(0., -1.);
            sine[b * dimension + b] = Complex64::new(0., 1.);
        }
    }
    let device = angle.device();
    let dtype = angle.dtype();
    let [batch] = angle.dims();
    if batch == 0 {
        return Err(Error::Shape("empty angle batch".into()));
    }
    let id = Matrix::new(dimension, identity)?.to_tensor(&device, dtype)?;
    let co = Matrix::new(dimension, cosine)?.to_tensor(&device, dtype)?;
    let si = Matrix::new(dimension, sine)?.to_tensor(&device, dtype)?;
    let half = (angle / 2.).reshape([batch, 1, 1]);
    let c = half.clone().cos();
    let s = half.sin();
    ComplexTensor::new(
        id.re + co.re * c.clone() + si.re * s.clone(),
        id.im + co.im * c + si.im * s,
    )
}
