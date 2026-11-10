use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Index of a coherent optical mode within one Fock basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModeId(pub usize);

/// Total-photon sectors retained exactly; no per-mode truncation is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhotonSector {
    /// Exactly the specified total; passive optics only.
    Fixed(usize),
    /// Every total from vacuum through the specified maximum, including loss outputs.
    UpTo(usize),
}

/// Immutable enumerated bosonic occupation basis, ordered by total then lexicographically.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FockBasis {
    modes: usize,
    sector: PhotonSector,
    occupations: Vec<Vec<usize>>,
    indices: BTreeMap<Vec<usize>, usize>,
}
impl FockBasis {
    /// Enumerate at most `max_states` basis states, checking combinatorics first.
    /// This reference supports 1..=256 modes and at most 64 photons.
    pub fn new(modes: usize, sector: PhotonSector, max_states: usize) -> Result<Self> {
        let photons = match sector {
            PhotonSector::Fixed(n) | PhotonSector::UpTo(n) => n,
        };
        if modes == 0 || modes > 256 || photons > 64 {
            return Err(Error::InvalidDimensions(
                "Fock basis needs 1..=256 modes and 0..=64 photons".into(),
            ));
        }
        let (n, k) = match sector {
            PhotonSector::Fixed(p) => (modes + p - 1, p),
            PhotonSector::UpTo(p) => (modes + p, p),
        };
        let size = bounded_binomial(n, k, max_states)?;
        let labels = size.checked_mul(modes).ok_or(Error::Overflow)?;
        if labels > 1 << 24 {
            return Err(Error::ResourceLimit {
                resource: "Fock occupation labels",
                required: labels,
                limit: 1 << 24,
            });
        }
        let mut occupations = Vec::with_capacity(size);
        let first = match sector {
            PhotonSector::Fixed(p) => p,
            PhotonSector::UpTo(_) => 0,
        };
        for total in first..=photons {
            enumerate(
                modes,
                total,
                &mut Vec::with_capacity(modes),
                &mut occupations,
            );
        }
        let indices = occupations
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, o)| (o, i))
            .collect();
        Ok(Self {
            modes,
            sector,
            occupations,
            indices,
        })
    }
    /// Number of coherent optical modes.
    pub fn modes(&self) -> usize {
        self.modes
    }
    /// Retained total-photon sectors.
    pub fn sector(&self) -> PhotonSector {
        self.sector
    }
    /// Exact Hilbert-space dimension, including one-dimensional vacuum sectors.
    pub fn size(&self) -> usize {
        self.occupations.len()
    }
    /// Largest total photon occupation represented.
    pub fn max_photons(&self) -> usize {
        match self.sector {
            PhotonSector::Fixed(n) | PhotonSector::UpTo(n) => n,
        }
    }
    /// Occupations in matrix/tensor basis order.
    pub fn occupations(&self) -> &[Vec<usize>] {
        &self.occupations
    }
    /// Locate a physical occupation; excluded sectors are rejected.
    pub fn index(&self, occupation: &[usize]) -> Result<usize> {
        self.indices.get(occupation).copied().ok_or(Error::Basis)
    }
    pub(crate) fn check_mode(&self, mode: ModeId) -> Result<()> {
        if mode.0 >= self.modes {
            Err(Error::Basis)
        } else {
            Ok(())
        }
    }
}
fn enumerate(modes: usize, total: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if modes == 1 {
        prefix.push(total);
        out.push(prefix.clone());
        prefix.pop();
    } else {
        for n in 0..=total {
            prefix.push(n);
            enumerate(modes - 1, total - n, prefix, out);
            prefix.pop();
        }
    }
}
pub(crate) fn bounded_binomial(n: usize, k: usize, limit: usize) -> Result<usize> {
    let mut count = 1u128;
    for i in 1..=k.min(n - k) {
        count = count * (n + 1 - i) as u128 / i as u128;
        if count > limit as u128 {
            return Err(Error::ResourceLimit {
                resource: "Fock basis states",
                required: usize::try_from(count).unwrap_or(usize::MAX),
                limit,
            });
        }
    }
    if count > limit as u128 {
        return Err(Error::ResourceLimit {
            resource: "Fock basis states",
            required: 1,
            limit,
        });
    }
    Ok(count as usize)
}
