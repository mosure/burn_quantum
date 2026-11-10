use super::*;
use crate::photonic::stabilizer::{Pauli, PauliProduct};
use crate::{Error, Result};
impl PlanarSurgery {
    /// Build checked local geometry and certify each protected CSS distance by graph search.
    pub fn new(distance: usize, basis: MergeBasis, limits: SurgeryLimits) -> Result<Self> {
        let code = PlanarCode::new(distance, limits.max_data_qubits / 2)?;
        let n = code.qubits();
        let d = code.distance();
        let mut split = code.x_checks(0)?;
        let xcount = split.len();
        split.extend(code.z_checks(0)?);
        let per_patch = split.len();
        split.extend(code.x_checks(n)?);
        split.extend(code.z_checks(n)?);
        let boundary: Vec<_> = (0..d - 1)
            .map(|i| {
                let a = match basis {
                    MergeBasis::ZZ => i,
                    MergeBasis::XX => xcount + i * d,
                };
                [a, a + per_patch]
            })
            .collect();
        let pauli = match basis {
            MergeBasis::ZZ => Pauli::Z,
            MergeBasis::XX => Pauli::X,
        };
        let mut merged: Vec<_> = (0..d)
            .map(|i| {
                let q = match basis {
                    MergeBasis::ZZ => i,
                    MergeBasis::XX => i * d,
                };
                PauliProduct::new(vec![(q, pauli), (n + q, pauli)])
            })
            .collect::<Result<_>>()?;
        for (index, check) in split.iter().enumerate() {
            if !boundary.iter().any(|pair| pair.contains(&index)) {
                merged.push(check.clone());
            }
        }
        for &[a, b] in &boundary {
            let (negative, product) = split[a].multiply_commuting(&split[b])?;
            if negative {
                return Err(Error::InvalidValue("negative merged generator".into()));
            }
            merged.push(product);
        }
        for (resource, required, limit) in [
            ("surgery data sites", 2 * n, limits.max_data_qubits),
            (
                "surgery checks",
                merged.len().max(split.len()),
                limits.max_checks,
            ),
            (
                "surgery measurements",
                d.checked_mul(merged.len() + split.len())
                    .ok_or(Error::Overflow)?,
                limits.max_measurements,
            ),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        if merged.len() != 2 * n - 1
            || !independent(&merged, 2 * n)
            || merged
                .iter()
                .enumerate()
                .any(|(i, p)| merged[..i].iter().any(|q| p.anticommutes(q)))
        {
            return Err(Error::InvalidValue("invalid merged CSS algebra".into()));
        }
        let mut coordinates = vec![];
        for patch in 0..2 {
            let point = |x: isize, y: isize| match basis {
                MergeBasis::ZZ => {
                    if patch == 0 {
                        [-x, y]
                    } else {
                        [x + 2, y]
                    }
                }
                MergeBasis::XX => {
                    if patch == 0 {
                        [x, -y]
                    } else {
                        [x, y + 2]
                    }
                }
            };
            for x in 0..d {
                for y in 0..d {
                    coordinates.push(point(2 * x as isize, 2 * y as isize));
                }
            }
            for x in 0..d - 1 {
                for y in 1..d {
                    coordinates.push(point(2 * x as isize + 1, 2 * y as isize - 1));
                }
            }
        }
        let mut plan = Self {
            code,
            basis,
            merged,
            split,
            boundary,
            coordinates,
            distance: SurgeryDistance::default(),
        };
        plan.distance = plan.certify_distance(limits.max_distance_work)?;
        if plan.distance.x_distance.min(plan.distance.z_distance) != d {
            return Err(Error::InvalidValue(
                "merged patch failed distance certificate".into(),
            ));
        }
        Ok(plan)
    }
}

// Bit-packed symplectic row elimination certifies independence, rather than
// inferring the number of logical degrees of freedom from a check count.
fn independent(checks: &[PauliProduct], sites: usize) -> bool {
    let words = (2 * sites).div_ceil(64);
    let mut pivots: Vec<Option<Vec<u64>>> = vec![None; 2 * sites];
    for check in checks {
        let mut row = vec![0u64; words];
        for &(site, pauli) in check.terms() {
            if pauli != Pauli::Z {
                row[site / 64] ^= 1 << (site % 64);
            }
            if pauli != Pauli::X {
                row[(sites + site) / 64] ^= 1 << ((sites + site) % 64);
            }
        }
        loop {
            let Some((word, value)) = row.iter().enumerate().find(|(_, value)| **value != 0) else {
                return false;
            };
            let pivot = 64 * word + value.trailing_zeros() as usize;
            if let Some(prior) = &pivots[pivot] {
                for (value, previous) in row.iter_mut().zip(prior) {
                    *value ^= previous;
                }
            } else {
                pivots[pivot] = Some(row);
                break;
            }
        }
    }
    true
}
