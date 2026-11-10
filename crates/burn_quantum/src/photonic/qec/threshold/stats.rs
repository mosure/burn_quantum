use crate::{Error, Result};
use serde::{Deserialize, Serialize};
/// Two-sided Clopper–Pearson interval for a predeclared fixed binomial sample.
/// Exact refers to binomial coverage; numerical endpoints use bounded f64 beta
/// inversion. This is not an optional-stopping confidence sequence.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinomialInterval {
    /// Lower probability endpoint.
    pub lower: f64,
    /// Upper probability endpoint.
    pub upper: f64,
}
impl BinomialInterval {
    /// Invert exact binomial tails for 1..=10 million trials and alpha in [1e-12,1).
    pub fn clopper_pearson(events: usize, trials: usize, alpha: f64) -> Result<Self> {
        if trials == 0
            || trials > 10_000_000
            || events > trials
            || !alpha.is_finite()
            || !(1e-12..1.).contains(&alpha)
        {
            return Err(Error::InvalidValue(
                "binomial interval inputs: 1..=10000000 fixed trials, alpha in [1e-12,1)".into(),
            ));
        }
        let tail = alpha / 2.;
        let lower = if events == 0 {
            0.
        } else {
            beta_quantile(tail, events as f64, (trials - events + 1) as f64)?
        };
        let upper = if events == trials {
            1.
        } else {
            1. - beta_quantile(tail, (trials - events) as f64, (events + 1) as f64)?
        };
        // Outward numerical padding, far below the sampling uncertainty in the
        // supported trial range. The underlying interval is not a normal fit.
        Ok(Self {
            lower: (lower - 1e-11).max(0.),
            upper: (upper + 1e-11).min(1.),
        })
    }
    /// Bonferroni simultaneous coverage for all predeclared intervals, including
    /// every inspected noise/size/model cell, regardless of dependence between cells.
    pub fn simultaneous(
        events: usize,
        trials: usize,
        family_alpha: f64,
        planned_intervals: usize,
    ) -> Result<Self> {
        if planned_intervals == 0
            || !family_alpha.is_finite()
            || !(0.0..1.0).contains(&family_alpha)
        {
            return Err(Error::InvalidValue("interval family definition".into()));
        }
        Self::clopper_pearson(events, trials, family_alpha / planned_intervals as f64)
    }
}
/// Statement about the tested finite sizes only, not an asymptotic threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FiniteSizeTrend {
    /// Failure intervals decrease strictly at every adjacent tested size.
    Suppressed,
    /// Failure intervals increase strictly at every adjacent tested size.
    Increased,
    /// Overlap prevents the stipulated directional conclusion.
    Unresolved,
    /// Some adjacent sizes show resolved increases and others resolved decreases.
    NonMonotone,
}
/// Require disjoint simultaneous intervals at every adjacent supplied size.
/// Intervals must belong to the same model, noise level and fixed run protocol;
/// this function cannot infer those provenance constraints from probabilities.
pub fn compare_sizes(points: &[(usize, BinomialInterval)]) -> Result<FiniteSizeTrend> {
    if points.len() < 2
        || points.windows(2).any(|p| p[0].0 >= p[1].0)
        || points.iter().any(|(_, v)| {
            !v.lower.is_finite()
                || !v.upper.is_finite()
                || v.lower < 0.
                || v.upper > 1.
                || v.lower > v.upper
        })
    {
        return Err(Error::InvalidValue(
            "ordered finite-size interval points".into(),
        ));
    }
    let down = points
        .windows(2)
        .filter(|p| p[1].1.upper < p[0].1.lower)
        .count();
    let up = points
        .windows(2)
        .filter(|p| p[1].1.lower > p[0].1.upper)
        .count();
    Ok(if down == points.len() - 1 {
        FiniteSizeTrend::Suppressed
    } else if up == points.len() - 1 {
        FiniteSizeTrend::Increased
    } else if down > 0 && up > 0 {
        FiniteSizeTrend::NonMonotone
    } else {
        FiniteSizeTrend::Unresolved
    })
}
/// Finite-size suppression-to-growth bracket over a predeclared noise grid.
/// This deliberately performs no optimistic scaling fit or extrapolation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FiniteSizeBracket {
    /// Lower probability endpoint.
    pub lower: f64,
    /// Upper probability endpoint.
    pub upper: f64,
}
/// Tightest tested suppression-to-growth interval, or None when the grid does
/// not establish consistent endpoints. No claim outside the tested size family.
pub fn crossing_bracket(points: &[(f64, FiniteSizeTrend)]) -> Result<Option<FiniteSizeBracket>> {
    if points
        .iter()
        .any(|(p, _)| !p.is_finite() || !(0.0..=1.0).contains(p))
        || points.windows(2).any(|p| p[0].0 >= p[1].0)
    {
        return Err(Error::InvalidValue("ordered noise grid".into()));
    }
    if points
        .iter()
        .any(|(_, t)| *t == FiniteSizeTrend::NonMonotone)
    {
        return Ok(None);
    }
    let lower = points
        .iter()
        .filter(|(_, t)| *t == FiniteSizeTrend::Suppressed)
        .map(|(p, _)| *p)
        .next_back();
    let upper = points
        .iter()
        .find(|(_, t)| *t == FiniteSizeTrend::Increased)
        .map(|(p, _)| *p);
    Ok(match (lower, upper) {
        (Some(lower), Some(upper)) if lower < upper => Some(FiniteSizeBracket { lower, upper }),
        _ => None,
    })
}
// Integer-shape beta parameters suffice for exact binomial inversion. Stirling's
// expansion beyond 16 has an omitted term <2e-15 in log-gamma; smaller integer
// arguments use an independent finite log-factorial sum.
fn log_gamma(x: f64) -> f64 {
    if x <= 16. {
        return (1..x as usize).map(|k| (k as f64).ln()).sum();
    }
    let inv = 1. / x;
    let q = inv * inv;
    (x - 0.5) * x.ln() - x
        + 0.5 * (2. * std::f64::consts::PI).ln()
        + inv * (1. / 12. + q * (-1. / 360. + q * (1. / 1260. + q * (-1. / 1680. + q / 1188.))))
}
fn beta_fraction(a: f64, b: f64, x: f64) -> Result<f64> {
    const TINY: f64 = 1e-300;
    let safe = |v: f64| if v.abs() < TINY { TINY.copysign(v) } else { v };
    let mut c = 1.;
    let mut d = 1. / safe(1. - (a + b) * x / (a + 1.));
    let mut h = d;
    for m in 1..=4096 {
        let m = m as f64;
        let twice = 2. * m;
        let aa = m * (b - m) * x / ((a - 1. + twice) * (a + twice));
        d = 1. / safe(1. + aa * d);
        c = safe(1. + aa / c);
        h *= d * c;
        let aa = -(a + m) * (a + b + m) * x / ((a + twice) * (a + 1. + twice));
        d = 1. / safe(1. + aa * d);
        c = safe(1. + aa / c);
        let delta = d * c;
        h *= delta;
        if (delta - 1.).abs() < 2e-14 && h.is_finite() {
            return Ok(h);
        }
    }
    Err(Error::InvalidValue(
        "incomplete beta did not converge".into(),
    ))
}
fn beta_cdf(a: f64, b: f64, x: f64) -> Result<f64> {
    if x <= 0. {
        return Ok(0.);
    }
    if x >= 1. {
        return Ok(1.);
    }
    let scale =
        (log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * x.ln() + b * (-x).ln_1p()).exp();
    Ok(if x < (a + 1.) / (a + b + 2.) {
        scale * beta_fraction(a, b, x)? / a
    } else {
        1. - scale * beta_fraction(b, a, 1. - x)? / b
    })
}
fn beta_quantile(target: f64, a: f64, b: f64) -> Result<f64> {
    if a == 1. {
        return Ok(-((1. - target).ln() / b).exp_m1());
    }
    if b == 1. {
        return Ok((target.ln() / a).exp());
    }
    let mut low = 0.;
    let mut high = 1.;
    for _ in 0..64 {
        let middle = (low + high) / 2.;
        if beta_cdf(a, b, middle)? < target {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok((low + high) / 2.)
}
