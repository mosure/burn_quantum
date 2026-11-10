//! Native Burn tensor updates, callback orchestration, and checkpoint restoration.
use super::validation::{finite, invalid, positive};
use super::{QnSpsaConfig, QnSpsaState, QnSpsaStep, QnSpsaStepReport, metric, qn_spsa_sample};
use crate::{Error, Result as QuantumResult, complex::read};
use burn::tensor::{DType, Tensor, TensorData};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

impl QnSpsaConfig {
    /// Initialize a host optimizer for one flat Burn parameter tensor.
    pub fn init(&self, parameters: usize) -> QuantumResult<QnSpsa> {
        let size = self.validate(parameters)?;
        let mut metric = vec![0.0; size];
        for i in 0..parameters {
            metric[i * parameters + i] = self.initial_metric_scale;
        }
        Ok(QnSpsa {
            config: self.clone(),
            parameters,
            iteration: 0,
            metric,
        })
    }
}

/// Quantum natural SPSA with synchronous evaluator callbacks and a dense host F64 metric.
/// No backpropagation is required. The parameter update is not differentiable.
/// Failed steps leave optimizer state unchanged; callback side effects cannot be rolled back.
#[derive(Clone, Debug)]
pub struct QnSpsa {
    config: QnSpsaConfig,
    parameters: usize,
    iteration: u64,
    metric: Vec<f64>,
}
impl QnSpsa {
    /// Configuration used by this optimizer.
    pub fn config(&self) -> &QnSpsaConfig {
        &self.config
    }
    /// Completed attempts, including blocked updates.
    pub fn iteration(&self) -> u64 {
        self.iteration
    }
    /// Raw row-major running fidelity metric, without regularization.
    pub fn metric(&self) -> &[f64] {
        &self.metric
    }
    /// Export deterministic optimizer state. Parameters are checkpointed separately.
    pub fn state(&self) -> QnSpsaState {
        QnSpsaState {
            version: 1,
            config: self.config.clone(),
            parameters: self.parameters,
            iteration: self.iteration,
            metric: self.metric.clone(),
        }
    }
    /// Restore a checked checkpoint, including schedule and random-stream position.
    pub fn from_state(state: QnSpsaState) -> QuantumResult<Self> {
        if state.version != 1 {
            return Err(Error::Unsupported("QN-SPSA checkpoint schema".into()));
        }
        let size = state.config.validate(state.parameters)?;
        if state.metric.len() != size {
            return Err(Error::Shape("QN-SPSA checkpoint metric shape".into()));
        }
        finite(&state.metric)?;
        for i in 0..state.parameters {
            for j in 0..i {
                if state.metric[i * state.parameters + j] != state.metric[j * state.parameters + i]
                {
                    return Err(invalid("checkpoint metric must be symmetric"));
                }
            }
        }
        Ok(Self {
            config: state.config,
            parameters: state.parameters,
            iteration: state.iteration,
            metric: state.metric,
        })
    }
    /// Materialize |metric| + beta I for diagnostics. Uses a bounded host eigensolve.
    pub fn regularized_metric(&self) -> QuantumResult<Vec<f64>> {
        metric::regularized(&self.metric, self.parameters, self.config.regularization)
    }
    /// Attempt an update using ordinary Burn tensors and scalar loss/fidelity callbacks.
    /// Callbacks receive fresh, untracked probe tensors on the parameter device.
    /// The result is a new leaf preserving the input gradient-requirement flag.
    /// A rejected proposal still advances the schedule and learns from its metric samples.
    pub fn step<L, F>(
        &mut self,
        parameters: Tensor<1>,
        loss: &mut L,
        fidelity: &mut F,
    ) -> QuantumResult<QnSpsaStep>
    where
        L: FnMut(Tensor<1>) -> QuantumResult<f64>,
        F: FnMut(Tensor<1>, Tensor<1>) -> QuantumResult<f64>,
    {
        if parameters.dims() != [self.parameters] {
            return Err(Error::Shape("QN-SPSA parameter shape changed".into()));
        }
        crate::backend::supported_dtype(parameters.dtype())?;
        let next = self.iteration.checked_add(1).ok_or(Error::Overflow)?;
        let eta = self.config.learning_rate
            / (next as f64 + self.config.stability_constant).powf(self.config.learning_rate_decay);
        let eps = self.config.perturbation / (next as f64).powf(self.config.perturbation_decay);
        if !positive(eta) || !positive(eps) || !positive(8.0 * eps * eps) {
            return Err(invalid("scheduled rate or perturbation underflow/overflow"));
        }
        let center = read(parameters.clone())?;
        finite(&center)?;
        let dtype = parameters.dtype();
        let cast = |x: f64| {
            if dtype == DType::F32 {
                (x as f32) as f64
            } else {
                x
            }
        };
        for &x in &center {
            if !cast(x + eps).is_finite()
                || !cast(x - eps).is_finite()
                || cast(x + eps) == x
                || cast(x - eps) == x
            {
                return Err(invalid(
                    "perturbation is not resolvable in the parameter dtype",
                ));
            }
        }
        let device = parameters.device();
        let tensor = |values: Vec<f64>| -> QuantumResult<Tensor<1>> {
            finite(&values)?;
            if values.iter().any(|&x| !cast(x).is_finite()) {
                return Err(invalid("probe/update overflows the tensor dtype"));
            }
            Ok(Tensor::from_data(
                TensorData::new(values, [self.parameters]),
                (&device, dtype),
            ))
        };
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        rng.set_stream(self.iteration);
        let mut gradient = vec![0.0; self.parameters];
        let mut metric_sample = vec![0.0; self.metric.len()];
        let mut loss_estimate = 0.0;
        let weight = 1.0 / self.config.resamplings as f64;
        for _ in 0..self.config.resamplings {
            let d1: Vec<f64> = (0..self.parameters)
                .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
                .collect();
            let d2: Vec<f64> = (0..self.parameters)
                .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
                .collect();
            let probe = |s1: f64, s2: f64| {
                tensor(
                    center
                        .iter()
                        .zip(&d1)
                        .zip(&d2)
                        .map(|((&x, &a), &b)| x + eps * (s1 * a + s2 * b))
                        .collect(),
                )
            };
            let plus = probe(1.0, 0.0)?;
            let minus = probe(-1.0, 0.0)?;
            let losses = [loss(plus.clone())?, loss(minus.clone())?];
            finite(&losses)?;
            let anchor = tensor(center.clone())?;
            let fidelities = [
                fidelity(anchor.clone(), plus)?,
                fidelity(anchor.clone(), minus)?,
                fidelity(anchor.clone(), probe(1.0, 1.0)?)?,
                fidelity(anchor, probe(-1.0, 1.0)?)?,
            ];
            let sample = qn_spsa_sample(
                eps,
                &d1,
                &d2,
                losses,
                fidelities,
                self.config.max_metric_elements,
            )?;
            for (average, value) in gradient.iter_mut().zip(sample.gradient) {
                *average += weight * value;
            }
            for (average, value) in metric_sample.iter_mut().zip(sample.metric) {
                *average += weight * value;
            }
            loss_estimate += weight * sample.loss_estimate;
        }
        let weight = 1.0 / (next as f64 + 1.0);
        let metric: Vec<_> = self
            .metric
            .iter()
            .zip(metric_sample)
            .map(|(&old, new)| (1.0 - weight) * old + weight * new)
            .collect();
        finite(&metric)?;
        finite(&gradient)?;
        let direction = if next > self.config.metric_delay {
            metric::solve(
                &metric,
                self.parameters,
                self.config.regularization,
                gradient,
            )?
        } else {
            gradient
        };
        finite(&direction)?;
        let proposal: Vec<_> = center
            .iter()
            .zip(&direction)
            .map(|(&x, &d)| x - eta * d)
            .collect();
        let proposed_step_norm = direction
            .iter()
            .fold(0.0f64, |norm, &x| norm.hypot(eta * x));
        finite(&[proposed_step_norm, loss_estimate])?;
        let candidate = tensor(proposal)?;
        let (accepted, loss_before, loss_proposed) =
            if let Some(tolerance) = self.config.blocking_tolerance {
                let before = loss(tensor(center)?)?;
                let after = loss(candidate.clone())?;
                finite(&[before, after])?;
                (
                    after <= before || after - before <= tolerance,
                    Some(before),
                    Some(after),
                )
            } else {
                (true, None, None)
            };
        let result = if accepted {
            candidate.set_require_grad(parameters.is_require_grad())
        } else {
            parameters
        };
        self.metric = metric;
        self.iteration = next;
        Ok(QnSpsaStep {
            parameters: result,
            report: QnSpsaStepReport {
                iteration: next,
                learning_rate: eta,
                perturbation: eps,
                accepted,
                loss_estimate,
                loss_before,
                loss_proposed,
                proposed_step_norm,
                loss_evaluations: 2 * self.config.resamplings
                    + 2 * usize::from(self.config.blocking_tolerance.is_some()),
                fidelity_evaluations: 4 * self.config.resamplings,
            },
        })
    }
}
