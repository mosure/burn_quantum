//! Explicit nondifferentiable terminal measurement with local reproducible RNG.
use crate::complex::read;
use crate::error::checked_product;
use crate::{Error, Register, Result, StateVector};
use burn::tensor::Tensor;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Computational-basis outcomes, one flat mixed-radix index per shot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Samples {
    /// Outcomes grouped by batch entry. The original state is not collapsed.
    pub outcomes: Vec<Vec<usize>>,
    /// Seed used by the local ChaCha8 stream.
    pub seed: u64,
}
impl Samples {
    /// Draw terminal samples after validating the distribution on the host.
    /// `max_samples` limits batch * shots before any output allocation.
    pub fn from_probabilities(
        probabilities: Tensor<2>,
        shots: usize,
        seed: u64,
        max_samples: usize,
    ) -> Result<Self> {
        let [batch, n] = probabilities.dims();
        if batch == 0 || n == 0 {
            return Err(Error::Shape("empty sampling distribution".into()));
        }
        let required = checked_product([batch, shots])?;
        if required > max_samples {
            return Err(Error::ResourceLimit {
                resource: "samples",
                required,
                limit: max_samples,
            });
        }
        let data = read(probabilities)?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut outcomes = Vec::with_capacity(batch);
        for row in data.chunks_exact(n) {
            let total: f64 = row.iter().sum();
            if row.iter().any(|p| !p.is_finite() || *p < 0.)
                || !total.is_finite()
                || (total - 1.).abs() > 1e-5
            {
                return Err(Error::InvalidValue(
                    "sampling requires finite nonnegative normalized probabilities".into(),
                ));
            }
            let mut cdf = Vec::with_capacity(n);
            let mut acc = 0.;
            for p in row {
                acc += p;
                cdf.push(acc);
            }
            let mut samples = Vec::with_capacity(shots);
            for _ in 0..shots {
                let u = rng.random::<f64>() * total;
                let index = cdf.partition_point(|&p| p <= u).min(n - 1);
                samples.push(index);
            }
            outcomes.push(samples);
        }
        Ok(Self { outcomes, seed })
    }
    /// Sample a pure state without mutating or differentiating through measurement.
    pub fn from_state(
        state: &StateVector,
        shots: usize,
        seed: u64,
        max_samples: usize,
    ) -> Result<Self> {
        Self::from_probabilities(state.probabilities(), shots, seed, max_samples)
    }
    /// Decode flat outcomes to subsystem digits for one batch entry.
    pub fn digits(&self, batch: usize, register: &Register) -> Result<Vec<Vec<usize>>> {
        let samples = self.outcomes.get(batch).ok_or(Error::Basis)?;
        samples
            .iter()
            .map(|&index| register.decode(index))
            .collect()
    }
}
