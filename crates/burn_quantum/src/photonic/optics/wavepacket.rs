//! Exact finite internal-mode representation of photon distinguishability.
use super::{FockBasis, FockState, Interferometer, ModeId};
use crate::{ComplexTensor, Error, Matrix, ResourceLimits, Result};
use burn::tensor::{DType, Device, Tensor, TensorData};
use num_complex::Complex64;
use std::collections::BTreeMap;

/// Spatial ports tensored with a finite orthonormal internal-mode basis.
/// Fine mode ordering is `spatial * internal_modes + internal`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavepacketLayout {
    spatial: usize,
    internal: usize,
}
impl WavepacketLayout {
    /// Create a layout with at most 256 fine optical modes.
    pub fn new(spatial: usize, internal: usize) -> Result<Self> {
        let total = spatial.checked_mul(internal).ok_or(Error::Overflow)?;
        if spatial == 0 || internal == 0 || total > 256 {
            return Err(Error::InvalidDimensions(
                "wavepacket layout requires 1..=256 fine modes".into(),
            ));
        }
        Ok(Self { spatial, internal })
    }
    /// Number of spatial ports, before resolving internal labels.
    pub fn spatial_modes(&self) -> usize {
        self.spatial
    }
    /// Number of orthogonal wavepacket basis functions per port.
    pub fn internal_modes(&self) -> usize {
        self.internal
    }
    /// Number of fine coherent optical modes.
    pub fn modes(&self) -> usize {
        self.spatial * self.internal
    }
    /// Resolve a spatial port/internal basis label.
    pub fn mode(&self, spatial: usize, internal: usize) -> Result<ModeId> {
        if spatial >= self.spatial || internal >= self.internal {
            return Err(Error::Basis);
        }
        Ok(ModeId(spatial * self.internal + internal))
    }
    /// Lift a spatial interferometer as U tensor identity on internal wavepackets.
    pub fn lift(&self, spatial: &Interferometer) -> Result<Interferometer> {
        if spatial.matrix().dimension() != self.spatial {
            return Err(Error::Shape(
                "spatial interferometer/layout mismatch".into(),
            ));
        }
        let n = self.modes();
        let mut values = vec![Complex64::new(0., 0.); n * n];
        for a in 0..self.spatial {
            for b in 0..self.spatial {
                for internal in 0..self.internal {
                    values[(a * self.internal + internal) * n + b * self.internal + internal] =
                        spatial.matrix().entries()[a * self.spatial + b];
                }
            }
        }
        Interferometer::new(Matrix::new(n, values)?)
    }
    /// Sum internal occupations into spatial counts. Detector labels are incoherently ignored.
    pub fn spatial_occupations(&self, basis: &FockBasis) -> Result<Vec<Vec<usize>>> {
        if basis.modes() != self.modes() {
            return Err(Error::Shape("wavepacket basis/layout mismatch".into()));
        }
        Ok(basis
            .occupations()
            .iter()
            .map(|o| {
                o.chunks_exact(self.internal)
                    .map(|x| x.iter().sum())
                    .collect()
            })
            .collect())
    }
}
/// One photon creation operator with a normalized internal wavepacket.
#[derive(Clone, Debug)]
pub struct PhotonPacket {
    spatial: usize,
    wavepacket: Vec<Complex64>,
}
impl PhotonPacket {
    /// Specify a spatial port and finite unit-norm internal amplitudes.
    pub fn new(spatial: usize, wavepacket: Vec<Complex64>) -> Result<Self> {
        if wavepacket.is_empty()
            || wavepacket.len() > 256
            || wavepacket
                .iter()
                .any(|a| !a.re.is_finite() || !a.im.is_finite())
            || (wavepacket.iter().map(|a| a.norm_sqr()).sum::<f64>() - 1.).abs() > 1e-10
        {
            return Err(Error::InvalidValue(
                "photon wavepacket must be finite and normalized".into(),
            ));
        }
        Ok(Self {
            spatial,
            wavepacket,
        })
    }
    /// Complex internal-state overlap; squared magnitude is two-photon HOM visibility.
    pub fn overlap(&self, other: &Self) -> Result<Complex64> {
        if self.wavepacket.len() != other.wavepacket.len() {
            return Err(Error::Shape("wavepacket dimensions differ".into()));
        }
        Ok(self
            .wavepacket
            .iter()
            .zip(&other.wavepacket)
            .map(|(a, b)| a.conj() * b)
            .sum())
    }
}
impl FockState {
    /// Prepare the normalized product of specified photon creation operators.
    /// Photons at the same port are bosonically symmetrized; source-generation yield
    /// is not inferred from this normalized state preparation.
    pub fn wavepackets(
        basis: FockBasis,
        layout: WavepacketLayout,
        packets: &[PhotonPacket],
        batch: usize,
        device: &Device,
        dtype: DType,
        limits: ResourceLimits,
    ) -> Result<Self> {
        let dim = basis.size();
        limits.check_state(batch, dim)?;
        let vector = packet_amplitudes(&basis, layout, packets)?;
        let re: Vec<f64> = (0..batch)
            .flat_map(|_| vector.iter().map(|z| z.re))
            .collect();
        let im: Vec<f64> = (0..batch)
            .flat_map(|_| vector.iter().map(|z| z.im))
            .collect();
        Self::from_amplitudes(
            basis,
            ComplexTensor::new(
                Tensor::from_data(TensorData::new(re, [batch, dim]), (device, dtype)),
                Tensor::from_data(TensorData::new(im, [batch, dim]), (device, dtype)),
            )?,
            limits,
        )
    }
}
pub(crate) fn packet_amplitudes(
    basis: &FockBasis,
    layout: WavepacketLayout,
    packets: &[PhotonPacket],
) -> Result<Vec<Complex64>> {
    if basis.modes() != layout.modes() || packets.len() > basis.max_photons() {
        return Err(Error::Shape(
            "photon packets do not fit the Fock basis".into(),
        ));
    }
    let work = basis
        .size()
        .checked_mul(layout.modes())
        .and_then(|n| n.checked_mul(packets.len().max(1)))
        .ok_or(Error::Overflow)?;
    if work > 1 << 24 {
        return Err(Error::ResourceLimit {
            resource: "wavepacket preparation work",
            required: work,
            limit: 1 << 24,
        });
    }
    let mut polynomial = BTreeMap::from([(vec![0; layout.modes()], Complex64::new(1., 0.))]);
    for packet in packets {
        if packet.spatial >= layout.spatial || packet.wavepacket.len() != layout.internal {
            return Err(Error::Shape("photon packet/layout mismatch".into()));
        }
        let mut next = BTreeMap::<Vec<usize>, Complex64>::new();
        for (occupation, value) in polynomial {
            for (internal, &amplitude) in packet.wavepacket.iter().enumerate() {
                if amplitude.norm_sqr() == 0. {
                    continue;
                }
                let mode = layout.mode(packet.spatial, internal)?.0;
                let mut out = occupation.clone();
                out[mode] += 1;
                let coefficient = (out[mode] as f64).sqrt();
                *next.entry(out).or_default() += value * amplitude * coefficient;
            }
        }
        polynomial = next;
    }
    let norm = polynomial
        .values()
        .map(|a| a.norm_sqr())
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 0. {
        return Err(Error::InvalidValue(
            "wavepacket preparation has invalid norm".into(),
        ));
    }
    let mut out = vec![Complex64::new(0., 0.); basis.size()];
    for (occupation, value) in polynomial {
        out[basis.index(&occupation)?] = value / norm;
    }
    Ok(out)
}
