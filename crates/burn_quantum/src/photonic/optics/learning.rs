//! Explicit finite-shot phase probes and mixed-state geometry for photonic learning.
use super::{DetectionDistribution, Detector, FockDensity, FockState, Interferometer, ModeId};
use crate::{Error, ResourceLimits, Result};
use burn::tensor::Tensor;
use nalgebra::{DMatrix, linalg::SymmetricEigen};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Sampled mean of a bounded classical detector readout, including every attempt.
#[derive(Clone, Copy, Debug)]
pub struct ShotEstimate {
    /// Unbiased sample mean of the supplied outcome values.
    pub mean: f64,
    /// Unbiased per-shot sample variance.
    pub variance: f64,
    /// Estimated standard error of the mean; not a confidence interval.
    pub standard_error: f64,
    /// Physical attempted shots, including no-click records.
    pub attempts: usize,
}
/// Independent two-shift derivative estimate with explicit physical shot cost.
#[derive(Clone, Copy, Debug)]
pub struct PhaseGradient {
    /// Estimated derivative of the unconditional readout mean.
    pub gradient: f64,
    /// Standard error propagated from independent plus/minus experiments.
    pub standard_error: f64,
    /// Two physical probe budgets per batch item.
    pub attempts: usize,
}
/// One complete categorical finite-shot sample; zero-click outcomes remain explicit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShotHistogram {
    /// Counts in the original `DetectionDistribution::records()` order.
    pub counts: Vec<usize>,
    /// Physical attempted shots, including zero clicks.
    pub attempts: usize,
}
/// Finite-shot plug-in estimate of one classical readout Fisher diagonal.
/// Squared noisy derivatives and denominator clipping introduce finite-sample bias.
/// This is neither the quantum Fisher information nor the Bures metric.
#[derive(Clone, Copy, Debug)]
pub struct ClassicalFisherEstimate {
    /// Sum of estimated derivative squared divided by clipped outcome probability.
    pub value: f64,
    /// Three independent physical probe budgets per batch item.
    pub attempts: usize,
    /// Outcomes whose central empirical probability was below the chosen floor.
    pub regularized_outcomes: usize,
}
/// One trainable phase occurrence on a zero/one-photon state, followed by fixed optics.
/// This restriction makes the ±pi/2, coefficient-1/2 parameter-shift rule exact.
#[derive(Clone, Debug)]
pub struct SinglePhotonPhaseExperiment {
    input: FockState,
    mode: ModeId,
    after: Interferometer,
    detectors: Vec<Detector>,
    max_outcomes: usize,
    limits: ResourceLimits,
}
impl SinglePhotonPhaseExperiment {
    /// Fix preparation/readout while leaving exactly one optical phase parameter free.
    pub fn new(
        input: FockState,
        mode: ModeId,
        after: Interferometer,
        detectors: Vec<Detector>,
        max_outcomes: usize,
        limits: ResourceLimits,
    ) -> Result<Self> {
        input.basis().check_mode(mode)?;
        if input.basis().max_photons() > 1 {
            return Err(Error::Unsupported(
                "two-shift rule requires a zero/one-photon phase generator".into(),
            ));
        }
        if after.matrix().dimension() != input.basis().modes()
            || detectors.len() != input.basis().modes()
        {
            return Err(Error::Shape(
                "phase experiment mode/readout mismatch".into(),
            ));
        }
        Ok(Self {
            input,
            mode,
            after,
            detectors,
            max_outcomes,
            limits,
        })
    }
    /// Native differentiable probabilities; no sampling or conditional normalization.
    pub fn evaluate(&self, angles: Tensor<1>) -> Result<DetectionDistribution> {
        self.input
            .phase_shift(self.mode, angles, self.limits)?
            .apply(&self.after, self.limits)?
            .detect(&self.detectors, self.max_outcomes, self.limits)
    }
    /// Independent finite-shot parameter-shift probes of a declared detector readout.
    /// No gradient is propagated through sampling; the returned estimator is host data.
    pub fn gradient(
        &self,
        angles: Tensor<1>,
        readout: &[f64],
        shots_per_probe: usize,
        seed: u64,
        max_attempts: usize,
    ) -> Result<Vec<PhaseGradient>> {
        if shots_per_probe < 2 {
            return Err(Error::InvalidValue(
                "phase probes require at least two shots".into(),
            ));
        }
        let batch = self.input.amplitudes().dims()[0];
        let required = batch
            .checked_mul(shots_per_probe)
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::Overflow)?;
        if required > max_attempts {
            return Err(Error::ResourceLimit {
                resource: "phase-shift attempts",
                required,
                limit: max_attempts,
            });
        }
        let plus = self.evaluate(angles.clone() + std::f64::consts::FRAC_PI_2)?;
        let minus = self.evaluate(angles - std::f64::consts::FRAC_PI_2)?;
        let a = plus.sample_readout(
            readout,
            shots_per_probe,
            crate::photonic::runtime::shot_seed(seed, 0, 0x504c_5553),
            max_attempts,
        )?;
        let b = minus.sample_readout(
            readout,
            shots_per_probe,
            crate::photonic::runtime::shot_seed(seed, 0, 0x004d_494e_5553),
            max_attempts,
        )?;
        Ok(a.into_iter()
            .zip(b)
            .map(|(a, b)| PhaseGradient {
                gradient: (a.mean - b.mean) / 2.,
                standard_error: (a.standard_error.powi(2) + b.standard_error.powi(2)).sqrt() / 2.,
                attempts: a.attempts + b.attempts,
            })
            .collect())
    }
    /// Estimate a classical Fisher diagonal using three independent shot histograms.
    /// The phase derivative uses the exact single-photon shift rule. The returned
    /// nonlinear plug-in metric is biased at finite shots and near rare outcomes.
    pub fn classical_fisher(
        &self,
        angles: Tensor<1>,
        shots_per_probe: usize,
        probability_floor: f64,
        seed: u64,
        max_attempts: usize,
    ) -> Result<Vec<ClassicalFisherEstimate>> {
        if shots_per_probe < 2
            || !probability_floor.is_finite()
            || probability_floor <= 0.
            || probability_floor > 1.
            || !(1. / probability_floor).is_finite()
        {
            return Err(Error::InvalidValue(
                "invalid finite-shot Fisher probes".into(),
            ));
        }
        let required = self.input.amplitudes().dims()[0]
            .checked_mul(shots_per_probe)
            .and_then(|n| n.checked_mul(3))
            .ok_or(Error::Overflow)?;
        if required > max_attempts {
            return Err(Error::ResourceLimit {
                resource: "classical Fisher attempts",
                required,
                limit: max_attempts,
            });
        }
        let probes = [
            0.,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::FRAC_PI_2,
        ];
        let mut histograms = Vec::with_capacity(3);
        for (i, shift) in probes.into_iter().enumerate() {
            histograms.push(self.evaluate(angles.clone() + shift)?.sample_counts(
                shots_per_probe,
                crate::photonic::runtime::shot_seed(seed, i as u64, 0x4649_5348_4552),
                max_attempts,
            )?);
        }
        (0..histograms[0].len())
            .map(|b| {
                let mut value = 0.;
                let mut regularized_outcomes = 0;
                for i in 0..histograms[0][b].counts.len() {
                    let p = histograms[0][b].counts[i] as f64 / shots_per_probe as f64;
                    let derivative = (histograms[1][b].counts[i] as f64
                        - histograms[2][b].counts[i] as f64)
                        / (2. * shots_per_probe as f64);
                    regularized_outcomes += usize::from(p < probability_floor);
                    value += derivative.powi(2) / p.max(probability_floor);
                }
                if !value.is_finite() {
                    return Err(Error::InvalidValue(
                        "finite-shot Fisher statistic overflow".into(),
                    ));
                }
                Ok(ClassicalFisherEstimate {
                    value,
                    attempts: 3 * shots_per_probe,
                    regularized_outcomes,
                })
            })
            .collect::<Result<Vec<_>>>()
    }
}
impl DetectionDistribution {
    /// Draw complete categorical records with deterministic per-batch identities.
    /// Sampling is synchronous host work; native simulation remains on its selected device.
    pub fn sample_counts(
        &self,
        shots_per_batch: usize,
        seed: u64,
        max_attempts: usize,
    ) -> Result<Vec<ShotHistogram>> {
        let probabilities = self.probabilities();
        let [batch, outcomes] = probabilities.dims();
        if shots_per_batch == 0 {
            return Err(Error::InvalidValue("zero optical shot budget".into()));
        }
        let required = batch.checked_mul(shots_per_batch).ok_or(Error::Overflow)?;
        if required > max_attempts {
            return Err(Error::ResourceLimit {
                resource: "optical shot attempts",
                required,
                limit: max_attempts,
            });
        }
        let data = crate::complex::read(probabilities)?;
        data.chunks_exact(outcomes)
            .enumerate()
            .map(|(row, values)| {
                if values.iter().any(|x| !x.is_finite() || *x < -1e-6)
                    || (values.iter().sum::<f64>() - 1.).abs() > 1e-5
                {
                    return Err(Error::InvalidValue(
                        "detector probabilities are not normalized".into(),
                    ));
                }
                let mut total = 0.;
                let cdf: Vec<_> = values
                    .iter()
                    .map(|p| {
                        total += p.max(0.);
                        total
                    })
                    .collect();
                let mut rng = ChaCha8Rng::seed_from_u64(crate::photonic::runtime::shot_seed(
                    seed,
                    row as u64,
                    0x0052_4541_444f_5554,
                ));
                let mut counts = vec![0; outcomes];
                for _ in 0..shots_per_batch {
                    let draw = rng.random::<f64>() * total;
                    let index = cdf.partition_point(|&p| p <= draw).min(outcomes - 1);
                    counts[index] += 1;
                }
                Ok(ShotHistogram {
                    counts,
                    attempts: shots_per_batch,
                })
            })
            .collect()
    }
    /// Sample an unconditional detector readout, with at least two shots per batch.
    /// Values must be finite. Exact state probabilities are read synchronously once.
    pub fn sample_readout(
        &self,
        readout: &[f64],
        shots_per_batch: usize,
        seed: u64,
        max_attempts: usize,
    ) -> Result<Vec<ShotEstimate>> {
        let outcomes = self.probabilities().dims()[1];
        if shots_per_batch < 2
            || readout.len() != outcomes
            || readout.iter().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidValue(
                "invalid finite-shot readout/budget".into(),
            ));
        }
        let mut estimates = vec![];
        for histogram in self.sample_counts(shots_per_batch, seed, max_attempts)? {
            let mean = histogram
                .counts
                .iter()
                .zip(readout)
                .map(|(&n, &x)| n as f64 * x)
                .sum::<f64>()
                / shots_per_batch as f64;
            let variance = histogram
                .counts
                .iter()
                .zip(readout)
                .map(|(&n, &x)| n as f64 * (x - mean).powi(2))
                .sum::<f64>()
                / (shots_per_batch - 1) as f64;
            if !mean.is_finite() || !variance.is_finite() {
                return Err(Error::InvalidValue("readout statistics overflow".into()));
            }
            estimates.push(ShotEstimate {
                mean,
                variance,
                standard_error: (variance / shots_per_batch as f64).sqrt(),
                attempts: shots_per_batch,
            });
        }
        Ok(estimates)
    }
}
impl FockDensity {
    /// Native Hilbert-Schmidt feature kernel Tr(rho† sigma), not Uhlmann fidelity.
    /// For physical density matrices this real kernel is positive semidefinite.
    pub fn hilbert_schmidt_kernel(
        &self,
        other: &Self,
        limits: ResourceLimits,
    ) -> Result<Tensor<2>> {
        if self.basis() != other.basis() {
            return Err(Error::Shape("density kernel bases differ".into()));
        }
        crate::complex::compatible(&self.entries().real(), &other.entries().real())?;
        let [a, n, _] = self.entries().dims();
        let b = other.entries().dims()[0];
        limits.check_state(a, b)?;
        let re = crate::precision::matmul2(
            self.entries().real().reshape([a, n * n]),
            other.entries().real().reshape([b, n * n]).transpose(),
        );
        let im = crate::precision::matmul2(
            self.entries().imag().reshape([a, n * n]),
            other.entries().imag().reshape([b, n * n]).transpose(),
        );
        Ok(re + im)
    }
    /// Squared Uhlmann fidelity (Tr sqrt(sqrt(rho) sigma sqrt(rho)))² per paired batch.
    /// Explicit synchronous host-F64 spectral reference, without autodiff. Normalized
    /// physical densities are required; basis dimension is bounded by min(max_dimension,256).
    pub fn uhlmann_fidelity(
        &self,
        other: &Self,
        max_dimension: usize,
        tolerance: f64,
    ) -> Result<Vec<f64>> {
        if self.basis() != other.basis() || self.entries().dims() != other.entries().dims() {
            return Err(Error::Shape(
                "mixed-state fidelity requires paired equal bases/batches".into(),
            ));
        }
        let [batch, n, _] = self.entries().dims();
        let limit = max_dimension.min(256);
        if n > limit {
            return Err(Error::ResourceLimit {
                resource: "mixed-state fidelity dimension",
                required: n,
                limit,
            });
        }
        self.validate(tolerance)?;
        other.validate(tolerance)?;
        let ar = crate::complex::read(self.entries().real())?;
        let ai = crate::complex::read(self.entries().imag())?;
        let br = crate::complex::read(other.entries().real())?;
        let bi = crate::complex::read(other.entries().imag())?;
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let start = b * n * n;
            let a: Vec<_> = (0..n * n)
                .map(|i| Complex64::new(ar[start + i], ai[start + i]))
                .collect();
            let c: Vec<_> = (0..n * n)
                .map(|i| Complex64::new(br[start + i], bi[start + i]))
                .collect();
            let a = DMatrix::from_row_slice(n, n, &a);
            let sigma = DMatrix::from_row_slice(n, n, &c);
            let eigen = SymmetricEigen::new((&a + a.adjoint()) * Complex64::new(0.5, 0.));
            if eigen
                .eigenvalues
                .iter()
                .any(|v| !v.is_finite() || *v < -tolerance)
            {
                return Err(Error::InvalidValue(
                    "negative mixed-state fidelity input".into(),
                ));
            }
            let diagonal = DMatrix::from_diagonal(
                &eigen
                    .eigenvalues
                    .map(|x| Complex64::new(x.max(0.).sqrt(), 0.)),
            );
            let root = &eigen.eigenvectors * diagonal * eigen.eigenvectors.adjoint();
            let product = &root * sigma * &root;
            let middle =
                SymmetricEigen::new((&product + product.adjoint()) * Complex64::new(0.5, 0.));
            if middle
                .eigenvalues
                .iter()
                .any(|v| !v.is_finite() || *v < -tolerance)
            {
                return Err(Error::InvalidValue(
                    "invalid mixed-state fidelity product".into(),
                ));
            }
            let f = middle
                .eigenvalues
                .iter()
                .map(|v| v.max(0.).sqrt())
                .sum::<f64>()
                .powi(2);
            if !f.is_finite() || f > 1. + 4. * tolerance {
                return Err(Error::InvalidValue(
                    "mixed-state fidelity exceeds one".into(),
                ));
            }
            out.push(f.clamp(0., 1.));
        }
        Ok(out)
    }
}
/// Symmetric finite-difference Bures metric from squared Uhlmann fidelities to
/// theta±delta. In the delta->0 limit this is QFI/4, not classical outcome Fisher.
/// Finite delta and fidelity uncertainty create bias/error; this is not a full metric matrix.
pub fn bures_metric_from_fidelities(plus: f64, minus: f64, delta: f64) -> Result<f64> {
    let denominator = delta * delta;
    if [plus, minus]
        .iter()
        .any(|x| !x.is_finite() || !(0.0..=1.).contains(x))
        || !delta.is_finite()
        || delta <= 0.
        || !denominator.is_finite()
        || denominator <= 0.
    {
        return Err(Error::InvalidValue("invalid Bures probes".into()));
    }
    let metric =
        ((1. - plus) / (1. + plus.sqrt()) + (1. - minus) / (1. + minus.sqrt())) / denominator;
    if !metric.is_finite() {
        return Err(Error::InvalidValue(
            "Bures probe step is numerically unresolved".into(),
        ));
    }
    Ok(metric)
}
