use super::{FockBasis, ModeId};
use crate::{Error, Matrix, ResourceLimits, Result, Unitary};
use num_complex::Complex64;
use std::collections::BTreeMap;

/// Passive mode unitary lifted to normalized bosonic occupations.
#[derive(Clone, Debug)]
pub struct Interferometer {
    matrix: Unitary,
}
impl Interferometer {
    /// Validate an optical mode matrix; 1..=256 modes are supported.
    pub fn new(matrix: Matrix) -> Result<Self> {
        if matrix.dimension() > 256 {
            return Err(Error::InvalidDimensions("at most 256 optical modes".into()));
        }
        Ok(Self {
            matrix: Unitary::new(matrix)?,
        })
    }
    /// The single-photon mode transformation, indexed `[output,input]`.
    pub fn matrix(&self) -> &Matrix {
        self.matrix.matrix()
    }
    /// Real Ry-convention beamsplitter, with block [[cos(theta/2),-sin(theta/2)],
    /// [sin(theta/2),cos(theta/2)]] on ordered modes `(a,b)`.
    /// This matches Perceval's Ry convention; theta=pi/2 is balanced.
    pub fn beam_splitter(modes: usize, a: ModeId, b: ModeId, theta: f64) -> Result<Self> {
        if modes == 0 || modes > 256 || a == b || a.0 >= modes || b.0 >= modes || !theta.is_finite()
        {
            return Err(Error::InvalidValue(
                "invalid beamsplitter modes or angle".into(),
            ));
        }
        let mut entries = vec![Complex64::new(0., 0.); modes * modes];
        for i in 0..modes {
            entries[i * modes + i] = 1.0.into();
        }
        let (s, c) = (theta / 2.).sin_cos();
        entries[a.0 * modes + a.0] = c.into();
        entries[b.0 * modes + b.0] = c.into();
        entries[a.0 * modes + b.0] = (-s).into();
        entries[b.0 * modes + a.0] = s.into();
        Self::new(Matrix::new(modes, entries)?)
    }
    /// Phase exp(i phi) on one mode, identity on the others.
    pub fn phase(modes: usize, mode: ModeId, phi: f64) -> Result<Self> {
        if modes == 0 || modes > 256 || mode.0 >= modes || !phi.is_finite() {
            return Err(Error::InvalidValue("invalid optical phase".into()));
        }
        let mut entries = vec![Complex64::new(0., 0.); modes * modes];
        for i in 0..modes {
            entries[i * modes + i] = if i == mode.0 {
                Complex64::from_polar(1., phi)
            } else {
                1.0.into()
            };
        }
        Self::new(Matrix::new(modes, entries)?)
    }
    /// Route each input mode to the specified output; must be a bijection.
    pub fn permutation(outputs: &[usize]) -> Result<Self> {
        let m = outputs.len();
        if m == 0 || m > 256 {
            return Err(Error::InvalidDimensions("invalid permutation size".into()));
        }
        let mut seen = vec![false; m];
        let mut entries = vec![Complex64::new(0., 0.); m * m];
        for (input, &output) in outputs.iter().enumerate() {
            if output >= m || seen[output] {
                return Err(Error::Basis);
            }
            seen[output] = true;
            entries[output * m + input] = 1.0.into();
        }
        Self::new(Matrix::new(m, entries)?)
    }
    /// Inverse physical mode transformation.
    pub fn adjoint(&self) -> Self {
        Self {
            matrix: self.matrix.adjoint(),
        }
    }
    /// Exact number-conserving lift, including factorial occupation normalization.
    /// Checks matrix storage and a conservative 2^24 scalar-work cap before expansion.
    pub fn lift(&self, basis: &FockBasis, limits: ResourceLimits) -> Result<Matrix> {
        let m = basis.modes();
        let d = basis.size();
        if self.matrix().dimension() != m {
            return Err(Error::Shape("interferometer/basis mode mismatch".into()));
        }
        limits.check_matrix(1, d)?;
        let work = d
            .checked_mul(d)
            .and_then(|n| n.checked_mul(m))
            .and_then(|n| n.checked_mul(basis.max_photons().max(1)))
            .ok_or(Error::Overflow)?;
        if work > 1 << 24 {
            return Err(Error::ResourceLimit {
                resource: "Fock lifting work",
                required: work,
                limit: 1 << 24,
            });
        }
        let mut entries = vec![Complex64::new(0., 0.); d * d];
        for (input, occupation) in basis.occupations().iter().enumerate() {
            let mut polynomial = BTreeMap::from([(vec![0; m], Complex64::new(1., 0.))]);
            for (source, &photons) in occupation.iter().enumerate() {
                for k in 1..=photons {
                    let mut next = BTreeMap::<Vec<usize>, Complex64>::new();
                    for (state, amplitude) in polynomial {
                        for output in 0..m {
                            let u = self.matrix().entries()[output * m + source];
                            if u.norm_sqr() == 0. {
                                continue;
                            }
                            let mut shifted = state.clone();
                            shifted[output] += 1;
                            let factor = (shifted[output] as f64 / k as f64).sqrt();
                            *next.entry(shifted).or_default() += amplitude * u * factor;
                        }
                    }
                    polynomial = next;
                }
            }
            for (occupation, amplitude) in polynomial {
                entries[basis.index(&occupation)? * d + input] = amplitude;
            }
        }
        Matrix::new(d, entries)
    }
}
