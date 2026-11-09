//! Tensor-independent gradient/metric estimates from six measured scalars.
use super::validation::{finite, invalid, matrix_size, positive};
use crate::Result as QuantumResult;

/// One stochastic gradient and metric estimate. Matrices use row-major storage.
#[derive(Clone, Debug)]
pub struct QnSpsaSample {
    /// SPSA loss gradient estimate.
    pub gradient: Vec<f64>,
    /// Symmetric Fubini–Study metric estimate, which may be indefinite.
    pub metric: Vec<f64>,
    /// Mean of the two perturbed losses; not the loss at the center or candidate.
    pub loss_estimate: f64,
}

/// Combine one pair of ±1 perturbation directions and six measured scalars.
///
/// Loss order: L(theta+c*d1), L(theta-c*d1). Fidelity order, with the first
/// argument fixed at theta: F(theta,theta+c*d1), F(theta,theta-c*d1),
/// F(theta,theta+c*(d1+d2)), F(theta,theta+c*(-d1+d2)).
/// Returns g_hat = -deltaF * (d1*d2^T + d2*d1^T)/(8*c²), estimating QFI/4.
/// Fidelity roundoff within 1e-5 of `[0,1]` is accepted without clipping.
pub fn qn_spsa_sample(
    perturbation: f64,
    direction1: &[f64],
    direction2: &[f64],
    losses: [f64; 2],
    fidelities: [f64; 4],
    max_metric_elements: usize,
) -> QuantumResult<QnSpsaSample> {
    let size = matrix_size(direction1.len(), max_metric_elements)?;
    let denominator = 8.0 * perturbation * perturbation;
    if !positive(perturbation) || !positive(denominator) {
        return Err(invalid(
            "perturbation or its squared denominator is not representable",
        ));
    }
    if direction1.len() != direction2.len()
        || direction1
            .iter()
            .chain(direction2)
            .any(|x| *x != 1.0 && *x != -1.0)
    {
        return Err(invalid(
            "directions must be equal-length nonempty Rademacher vectors",
        ));
    }
    finite(&losses)?;
    finite(&fidelities)?;
    if fidelities
        .iter()
        .any(|&x| !(-1e-5..=1.0 + 1e-5).contains(&x))
    {
        return Err(invalid("fidelity must be a squared overlap in [0,1]"));
    }
    let slope = (losses[0] - losses[1]) / (2.0 * perturbation);
    let coefficient =
        -((fidelities[2] - fidelities[0]) - (fidelities[3] - fidelities[1])) / denominator;
    let gradient: Vec<_> = direction1.iter().map(|&x| slope * x).collect();
    let mut metric = Vec::with_capacity(size);
    for (&a, &b) in direction1.iter().zip(direction2) {
        for (&c, &d) in direction1.iter().zip(direction2) {
            metric.push(coefficient * (a * d + b * c));
        }
    }
    let loss_estimate = 0.5 * losses[0] + 0.5 * losses[1];
    finite(&gradient)?;
    finite(&metric)?;
    finite(&[loss_estimate])?;
    Ok(QnSpsaSample {
        gradient,
        metric,
        loss_estimate,
    })
}
