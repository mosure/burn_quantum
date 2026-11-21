use crate::{Error, Result};
use std::f64::consts::{PI, TAU};

/// One generalized reflection pair. Each selected subspace acquires exp(+i phase).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectionPhases {
    /// Phase applied to the good subspace first, in radians.
    pub good: f64,
    /// Phase applied to the prepared source state second, in radians.
    pub source: f64,
}

#[derive(Clone, Copy, Debug)]
enum Rule {
    Standard,
    Chebyshev { attenuation: f64, failure: f64 },
}

/// Constant-size description of an ordered sequence of generalized Grover rounds.
#[derive(Clone, Copy, Debug)]
pub struct AmplificationSchedule {
    rounds: usize,
    rule: Rule,
}

impl AmplificationSchedule {
    /// Ordinary pi reflections, including zero rounds (preparation alone).
    pub fn standard(rounds: usize) -> Self {
        Self {
            rounds,
            rule: Rule::Standard,
        }
    }

    /// Choose the integer nearest the first success peak for a known initial mass in (0, 1].
    /// At a half-probability tie, preparation alone suffices.
    pub fn known_success(initial_success: f64) -> Result<Self> {
        positive_probability(initial_success)?;
        let rounds = if initial_success >= 0.5 {
            0.
        } else {
            (PI / (4. * initial_success.sqrt().asin())).floor()
        };
        Ok(Self::standard(round_count(rounds)?))
    }

    /// Yoder–Low–Chuang fixed-point schedule. For any initial good mass at least
    /// `minimum_success`, the **completed** schedule has failure probability at most
    /// `max_failure` in exact arithmetic. Both inputs use probability units.
    /// Intermediate rounds can dip. Phases use the smallest qualifying odd degree L.
    pub fn fixed_point(minimum_success: f64, max_failure: f64) -> Result<Self> {
        positive_probability(minimum_success)?;
        if !(0. < max_failure && max_failure < 1.) {
            return Err(Error::InvalidValue(
                "maximum failure probability must be in (0, 1)".into(),
            ));
        }
        let delta = max_failure.sqrt();
        // acosh(1/delta) and atanh(sqrt(minimum_success)), also stable near endpoints.
        let attenuation = ((1. - max_failure).sqrt() / delta).asinh();
        let already_sufficient = if max_failure >= 0.5 {
            minimum_success >= 1. - max_failure
        } else {
            1. - minimum_success <= max_failure
        };
        let rounds = if already_sufficient {
            0
        } else {
            let slope = (minimum_success.sqrt() / (1. - minimum_success).sqrt()).asinh();
            round_count(((attenuation / slope - 1.) * 0.5).ceil().max(0.))?
        };
        Ok(Self {
            rounds,
            rule: Rule::Chebyshev {
                attenuation,
                failure: max_failure,
            },
        })
    }

    /// Number of selective good-space phases. A compute/phase/uncompute oracle costs
    /// two bit-oracle executions per such phase.
    pub fn rounds(self) -> usize {
        self.rounds
    }

    /// Iterate phase pairs in physical execution order, without allocating a phase table.
    pub fn phases(self) -> impl ExactSizeIterator<Item = ReflectionPhases> {
        let degree = 2. * self.rounds as f64 + 1.;
        let width = match self.rule {
            Rule::Standard => 0.,
            Rule::Chebyshev { attenuation, .. } => (attenuation / degree).tanh(),
        };
        (0..self.rounds).map(move |i| match self.rule {
            Rule::Standard => ReflectionPhases {
                good: PI,
                source: PI,
            },
            Rule::Chebyshev { .. } => {
                // Eq. 11: the paper's source exponent is -alpha, its target exponent +beta.
                let alpha = |j: usize| 2. * 1_f64.atan2(width * (TAU * j as f64 / degree).tan());
                ReflectionPhases {
                    good: -alpha(self.rounds - i),
                    source: -alpha(i + 1),
                }
            }
        })
    }

    /// Analytic success for this complete schedule. Uses the rotation law or the
    /// Chebyshev success polynomial independently of circuit synthesis.
    pub fn predicted_success(self, initial_success: f64) -> Result<f64> {
        if !(0. ..=1.).contains(&initial_success) {
            return Err(Error::InvalidValue(
                "initial success probability must be in [0, 1]".into(),
            ));
        }
        if self.rounds == 0 || initial_success == 0. || initial_success == 1. {
            return Ok(initial_success);
        }
        let degree = 2. * self.rounds as f64 + 1.;
        Ok(match self.rule {
            Rule::Standard => (degree * initial_success.sqrt().asin()).sin().powi(2),
            Rule::Chebyshev {
                attenuation,
                failure,
            } => chebyshev_success(initial_success, degree, attenuation, failure),
        })
    }
}

// Evaluate 1 - delta² T_L(x)², x = sqrt(1-p) cosh(k/L), k = acosh(1/delta).
// Directly forming x rounds it to 1 for small promises, erasing the oscillations.
fn chebyshev_success(p: f64, degree: f64, k: f64, failure: f64) -> f64 {
    let root_width = (k / degree).tanh();
    let width = root_width * root_width;
    let bad = (1. - p).sqrt();
    if p >= width {
        let angle = (p - width).sqrt().atan2(bad);
        return (1. - failure) + failure * (degree * angle).sin().powi(2);
    }
    // Outside the passband, d = k - L acosh(x). Rationalizing the two
    // square-root differences retains relative accuracy even for p << width.
    let change = (p / (1. + bad) + p / (root_width + (width - p).sqrt())) / (1. + root_width);
    let d = -degree * (-change).ln_1p();
    // sinh(2k-d) sinh(d) / cosh(k)^2, using bounded exponentials.
    ((-2. * (2. * k - d)).exp_m1() * (-2. * d).exp_m1() / (1. + (-2. * k).exp()).powi(2))
        .clamp(0., 1.)
}

fn positive_probability(value: f64) -> Result<()> {
    if 0. < value && value <= 1. {
        Ok(())
    } else {
        Err(Error::InvalidValue(
            "initial success probability must be in (0, 1]".into(),
        ))
    }
}

fn round_count(value: f64) -> Result<usize> {
    // Phase indices and the odd polynomial degree must be representable exactly in f64.
    if value > ((1_u64 << 52) - 1) as f64 || value >= usize::MAX as f64 {
        return Err(Error::Overflow);
    }
    Ok(value as usize)
}
