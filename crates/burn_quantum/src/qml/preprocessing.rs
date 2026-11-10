//! Host preprocessing fitted explicitly on a caller-selected training partition.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Per-feature population standardization. Constant training columns use scale one.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Standardizer {
    mean: Vec<f64>,
    scale: Vec<f64>,
}
impl Standardizer {
    /// Fit only the supplied rows; callers must keep validation/test rows separate.
    pub fn fit(rows: &[Vec<f64>]) -> Result<Self> {
        let width = rows.first().map(Vec::len).unwrap_or(0);
        if width == 0
            || rows
                .iter()
                .any(|r| r.len() != width || r.iter().any(|v| !v.is_finite()))
        {
            return Err(Error::Shape(
                "standardization needs finite rectangular training rows".into(),
            ));
        }
        let mean = (0..width)
            .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64)
            .collect::<Vec<_>>();
        let scale = (0..width)
            .map(|j| {
                let sigma = (rows.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>()
                    / rows.len() as f64)
                    .sqrt();
                if sigma < 1e-12 { 1. } else { sigma }
            })
            .collect::<Vec<_>>();
        if mean.iter().chain(&scale).any(|v| !v.is_finite()) {
            return Err(Error::InvalidValue(
                "nonfinite fitted standardization".into(),
            ));
        }
        Ok(Self { mean, scale })
    }
    /// Fitted feature means.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }
    /// Fitted feature scales.
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }
    /// Transform one row without refitting or clipping held-out observations.
    pub fn transform(&self, row: &[f64]) -> Result<Vec<f64>> {
        if row.len() != self.mean.len()
            || self.mean.is_empty()
            || self.scale.len() != self.mean.len()
            || row.iter().chain(&self.mean).any(|v| !v.is_finite())
            || self.scale.iter().any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err(Error::Shape(
                "invalid standardizer or input feature row".into(),
            ));
        }
        let result = row
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((&x, &m), &s)| (x - m) / s)
            .collect::<Vec<_>>();
        if result.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidValue(
                "nonfinite standardized features".into(),
            ));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn training_statistics_are_not_refit_on_heldout_rows() {
        let scaler = Standardizer::fit(&[vec![1., 5.], vec![3., 5.]]).unwrap();
        assert_eq!(scaler.mean(), &[2., 5.]);
        assert_eq!(scaler.scale(), &[1., 1.]);
        assert_eq!(scaler.transform(&[102., 8.]).unwrap(), vec![100., 3.]);
        assert_eq!(scaler.mean(), &[2., 5.]);
        assert!(scaler.transform(&[f64::NAN, 0.]).is_err());
        assert!(Standardizer::fit(&[]).is_err());
    }
}
