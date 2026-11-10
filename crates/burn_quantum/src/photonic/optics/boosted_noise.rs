//! Calibrated first-level Grice optics with explicit environment and internal modes.
use super::{Detector, FockBasis, OpticalBellInstrument, PhotonSector};
use crate::{Error, Result};
use num_complex::Complex64;
use std::collections::BTreeMap;

/// Complete bounded calibration for the supplied two-signal/two-ancilla Grice recipe.
#[derive(Clone, Debug)]
pub struct GriceNoiseConfig {
    /// Intensity transmission of each input port's two rails, before the analyzer.
    pub rail_transmissions: [[f64; 2]; 4],
    /// Coherent input phase offsets in radians, per port/rail.
    pub rail_phases: [[f64; 2]; 4],
    /// Normalized pure internal-mode kets for the four photons, in a shared basis.
    /// One to four labels suffice for an arbitrary pure four-photon Gram matrix.
    pub wavepackets: [Vec<Complex64>; 4],
    /// Independent finite-resolution, inefficient and dark-counting output detectors.
    pub detectors: [Detector; 8],
    /// Caller-supplied calibration identity, retained on the compiled instrument.
    pub calibration: String,
    /// Maximum distinct environment/output amplitude rows retained during compilation.
    pub max_terms: usize,
    /// Maximum counted scalar expansion, response and effect-accumulation operations.
    pub max_work: usize,
}
impl Default for GriceNoiseConfig {
    fn default() -> Self {
        Self {
            rail_transmissions: [[1.; 2]; 4],
            rail_phases: [[0.; 2]; 4],
            wavepackets: std::array::from_fn(|_| vec![1.0.into()]),
            detectors: [Detector::ideal(4).expect("valid resolution"); 8],
            calibration: "ideal-grice-explicit-environment-v1".into(),
            max_terms: 262_144,
            max_work: 1 << 28,
        }
    }
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Row {
    lost_rails: u8,
    occupation: Vec<u8>,
}
struct Budget {
    used: usize,
    maximum: usize,
}
impl Budget {
    fn use_work(&mut self, count: usize) -> Result<()> {
        self.used = self.used.checked_add(count).ok_or(Error::Overflow)?;
        if self.used > self.maximum {
            return Err(Error::ResourceLimit {
                resource: "calibrated Grice scalar work",
                required: self.used,
                limit: self.maximum,
            });
        }
        Ok(())
    }
}
impl OpticalBellInstrument {
    /// Derive every noisy POVM effect from the published first-level optical recipe.
    /// The supplied ancilla is Phi+; input attenuation has explicit lost-rail records,
    /// and unresolved internal modes are traced after interference. The ideal click
    /// classifier is kept fixed: dark counts/distinguishability can cause false heralds.
    pub fn grice_calibrated(config: GriceNoiseConfig) -> Result<Self> {
        validate(&config)?;
        let mut budget = Budget {
            used: 0,
            maximum: config.max_work,
        };
        let labels = ideal_labels(&mut budget, config.max_terms)?;
        let rows = amplitudes(&config, &mut budget)?;
        let internal = config.wavepackets[0].len();
        let mut responses = BTreeMap::<Vec<usize>, [f64; 6]>::new();
        let mut effects = vec![vec![Complex64::new(0., 0.); 81]; 6];
        for (row, amplitude) in rows {
            let occupation: Vec<_> = row
                .occupation
                .chunks_exact(internal)
                .map(|c| c.iter().map(|&n| n as usize).sum())
                .collect();
            if !responses.contains_key(&occupation) {
                responses.insert(
                    occupation.clone(),
                    record_weights(&occupation, &config.detectors, &labels, &mut budget)?,
                );
            }
            let weights = &responses[&occupation];
            budget.use_work(6 * 81)?;
            for (effect, &weight) in effects.iter_mut().zip(weights) {
                if weight == 0. {
                    continue;
                }
                for a in 0..9 {
                    for b in 0..9 {
                        effect[a * 9 + b] += amplitude[a].conj() * amplitude[b] * weight;
                    }
                }
            }
        }
        Self::from_effects(super::bell::grice_records(), effects, config.calibration, 2)
    }
}
fn validate(c: &GriceNoiseConfig) -> Result<()> {
    let internal = c.wavepackets[0].len();
    if !(1..=4).contains(&internal)
        || c.max_terms == 0
        || c.max_work == 0
        || c.calibration.is_empty()
        || c.calibration.len() > 4096
        || c.rail_transmissions
            .iter()
            .flatten()
            .any(|p| !p.is_finite() || !(0.0..=1.).contains(p))
        || c.rail_phases.iter().flatten().any(|p| !p.is_finite())
        || c.wavepackets.iter().any(|p| {
            p.len() != internal
                || p.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
                || (p.iter().map(|z| z.norm_sqr()).sum::<f64>() - 1.).abs() > 1e-10
        })
    {
        return Err(Error::InvalidValue(
            "invalid calibrated Grice optical model".into(),
        ));
    }
    Ok(())
}
fn amplitudes(c: &GriceNoiseConfig, budget: &mut Budget) -> Result<BTreeMap<Row, [Complex64; 9]>> {
    let unitary = OpticalBellInstrument::grice_interferometer()?;
    let internal = c.wavepackets[0].len();
    let mut rows = BTreeMap::<Row, [Complex64; 9]>::new();
    for a in 0..3 {
        for b in 0..3 {
            for ancillary in 0..2 {
                let ports = [a, b, ancillary, ancillary];
                let present: Vec<_> = (0..4).filter(|&p| ports[p] < 2).collect();
                for mask in 0..1usize << present.len() {
                    let mut coefficient = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.);
                    let mut lost_rails = 0u8;
                    let mut surviving = vec![];
                    for (bit, &port) in present.iter().enumerate() {
                        let rail = ports[port];
                        let eta = c.rail_transmissions[port][rail];
                        coefficient *= Complex64::from_polar(1., c.rail_phases[port][rail]);
                        if mask & (1 << bit) == 0 {
                            coefficient *= eta.sqrt();
                            surviving.push((port, rail));
                        } else {
                            coefficient *= (1. - eta).sqrt();
                            lost_rails |= 1 << (2 * port + rail);
                        }
                    }
                    if coefficient.norm_sqr() == 0. {
                        continue;
                    }
                    let mut polynomial = BTreeMap::from([(vec![0u8; 8 * internal], coefficient)]);
                    for (port, rail) in surviving {
                        let mut next = BTreeMap::<Vec<u8>, Complex64>::new();
                        for (occupation, value) in polynomial {
                            for output in 0..8 {
                                for label in 0..internal {
                                    let amplitude = unitary.matrix().entries()
                                        [output * 8 + 2 * port + rail]
                                        * c.wavepackets[port][label];
                                    if amplitude.norm_sqr() == 0. {
                                        continue;
                                    }
                                    budget.use_work(1)?;
                                    let mode = output * internal + label;
                                    let mut updated = occupation.clone();
                                    updated[mode] += 1;
                                    let factor = (updated[mode] as f64).sqrt();
                                    *next.entry(updated).or_default() += value * amplitude * factor;
                                    if next.len() > c.max_terms {
                                        return Err(Error::ResourceLimit {
                                            resource: "Grice polynomial terms",
                                            required: next.len(),
                                            limit: c.max_terms,
                                        });
                                    }
                                }
                            }
                        }
                        polynomial = next;
                    }
                    for (occupation, value) in polynomial {
                        if value.norm_sqr() == 0. {
                            continue;
                        }
                        let row = Row {
                            lost_rails,
                            occupation,
                        };
                        rows.entry(row).or_insert([Complex64::new(0., 0.); 9])[a * 3 + b] += value;
                        if rows.len() > c.max_terms {
                            return Err(Error::ResourceLimit {
                                resource: "Grice environment rows",
                                required: rows.len(),
                                limit: c.max_terms,
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(rows)
}
fn ideal_labels(budget: &mut Budget, max_terms: usize) -> Result<Vec<(Vec<usize>, usize)>> {
    let rows = amplitudes(
        &GriceNoiseConfig {
            max_terms,
            ..Default::default()
        },
        budget,
    )?;
    let mut labels: Vec<(Vec<usize>, usize)> = vec![];
    for (row, a) in rows {
        if row.occupation.iter().map(|&n| n as usize).sum::<usize>() != 4 {
            continue;
        }
        let bell = [a[1] + a[3], a[1] - a[3], a[0] + a[4], a[0] - a[4]];
        let support: Vec<_> = bell
            .iter()
            .enumerate()
            .filter(|(_, v)| v.norm_sqr() > 1e-12)
            .map(|(i, _)| i)
            .collect();
        let label = match support.as_slice() {
            [] => continue,
            [only] => *only,
            [2, 3] => 4,
            _ => {
                return Err(Error::InvalidValue(
                    "invalid ideal Grice classifier support".into(),
                ));
            }
        };
        labels.push((row.occupation.iter().map(|&n| n as usize).collect(), label));
    }
    // Ensure the pattern dictionary is in one well-defined complete four-photon sector.
    let basis = FockBasis::new(8, PhotonSector::Fixed(4), 330)?;
    for (occupation, _) in &labels {
        basis.index(occupation)?;
    }
    Ok(labels)
}
fn record_weights(
    occupation: &[usize],
    detectors: &[Detector; 8],
    labels: &[(Vec<usize>, usize)],
    budget: &mut Budget,
) -> Result<[f64; 6]> {
    let responses = detectors
        .iter()
        .zip(occupation)
        .map(|(d, &n)| d.response(n))
        .collect::<Result<Vec<_>>>()?;
    let mut weights = [0.; 6];
    budget.use_work(labels.len() * 8)?;
    for (record, label) in labels {
        weights[*label] += record
            .iter()
            .zip(&responses)
            .map(|(&n, p)| p.get(n).copied().unwrap_or(0.))
            .product::<f64>();
    }
    let accepted = weights[..5].iter().sum::<f64>();
    if accepted > 1. + 1e-10 {
        return Err(Error::InvalidValue(
            "Grice detector response exceeds one".into(),
        ));
    }
    weights[5] = (1. - accepted).max(0.);
    Ok(weights)
}
