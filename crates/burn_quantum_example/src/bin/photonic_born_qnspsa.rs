//! Lossy mixed-state optical Born fitting with QN-SPSA and a Bures metric evaluator.
//! The squared Uhlmann evaluator is an exact host spectral reference, not a hardware
//! overlap measurement. Finite-shot training/validation data are independently sampled.
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{
    ComplexTensor, Result, optim::QnSpsaConfig, photonic::optics::*,
    qml::distribution_cross_entropy,
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
const ETA: f64 = 0.8;
const RADIUS: f64 = 0.6;
const TARGET: f64 = 1.1;
const STEPS: usize = 70;
fn state(theta: Tensor<1>, device: &Device) -> Result<FockDensity> {
    let limits = Default::default();
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)?;
    // Declared incoherent input rail mixture; no data or no-click conditioning.
    let re = Tensor::from_data(
        TensorData::new(
            vec![
                0.,
                0.,
                0.,
                0.,
                (1. - RADIUS) / 2.,
                0.,
                0.,
                0.,
                (1. + RADIUS) / 2.,
            ],
            [1, 3, 3],
        ),
        (device, DType::F32),
    );
    FockDensity::from_entries(
        FockBasis::new(2, PhotonSector::UpTo(1), 3)?,
        ComplexTensor::new(re.clone(), re.zeros_like())?,
        limits,
    )?
    .apply(&bs, limits)?
    .phase_shift(ModeId(0), theta, limits)?
    .apply(&bs.adjoint(), limits)?
    .lose(ModeId(0), ETA, limits)?
    .lose(ModeId(1), ETA, limits)
}
fn readout(theta: Tensor<1>, device: &Device) -> Result<DetectionDistribution> {
    state(theta, device)?.detect(&[Detector::ideal(1)?; 2], 4, Default::default())
}
fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let target = readout(Tensor::from_data([TARGET as f32], &device), &device)?;
    let analytic: Vec<_> = target
        .records()
        .iter()
        .map(|r| match r.as_slice() {
            [0, 0] => 1. - ETA,
            [1, 0] => ETA * (1. + RADIUS * TARGET.cos()) / 2.,
            [0, 1] => ETA * (1. - RADIUS * TARGET.cos()) / 2.,
            _ => 0.,
        })
        .collect();
    let oracle = values(target.probabilities())?
        .iter()
        .zip(&analytic)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0f64, f64::max);
    ensure(
        oracle < 3e-6,
        "Born target disagrees with analytic noisy optical law",
    )?;
    let training = target.sample_counts(8192, 41, 8192)?.remove(0);
    let validation = target.sample_counts(16384, 97, 16384)?.remove(0);
    let train_probabilities: Vec<_> = training.counts.iter().map(|&n| n as f32 / 8192.).collect();
    let validation_probabilities: Vec<_> = validation
        .counts
        .iter()
        .map(|&n| n as f64 / 16384.)
        .collect();
    let train = Tensor::from_data(TensorData::new(train_probabilities, [1, 4]), &device);
    let baseline: Vec<_> = target
        .records()
        .iter()
        .map(|r| match r.as_slice() {
            [0, 0] => 1. - ETA,
            [1, 0] | [0, 1] => ETA / 2.,
            _ => 0.,
        })
        .collect();
    let baseline_tvd = baseline
        .iter()
        .zip(&analytic)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 2.;
    let mut worst_tvd = 0f64;
    let mut worst_heldout = 0f64;
    let mut worst_metric = 0f64;
    let mut loss_calls = 0;
    let mut fidelity_calls = 0;
    for (seed, initial) in [(7, 0.4), (11, 2.), (19, -0.5)] {
        let mut parameters = Tensor::from_data([initial as f32], &device);
        let mut loss = |p: Tensor<1>| -> Result<f64> {
            Ok(distribution_cross_entropy(
                readout(p, &device)?.probabilities(),
                train.clone(),
                1e-7,
            )?
            .into_scalar::<f64>())
        };
        let mut fidelity = |a: Tensor<1>, b: Tensor<1>| -> Result<f64> {
            Ok(state(a, &device)?.uhlmann_fidelity(&state(b, &device)?, 3, 3e-6)?[0])
        };
        let mut optimizer = QnSpsaConfig::new()
            .with_seed(seed)
            .with_learning_rate(0.2)
            .with_learning_rate_decay(0.)
            .with_perturbation(0.12)
            .with_perturbation_decay(0.)
            .with_initial_metric_scale(ETA * RADIUS.powi(2) / 4.)
            .with_regularization(0.02)
            .with_blocking_tolerance(Some(0.))
            .init(1)?;
        for _ in 0..STEPS {
            let update = optimizer.step(parameters, &mut loss, &mut fidelity)?;
            loss_calls += update.report.loss_evaluations;
            fidelity_calls += update.report.fidelity_evaluations;
            parameters = update.parameters;
        }
        let probabilities = values(readout(parameters, &device)?.probabilities())?;
        worst_tvd = worst_tvd.max(
            probabilities
                .iter()
                .zip(&analytic)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / 2.,
        );
        worst_heldout = worst_heldout.max(
            probabilities
                .iter()
                .zip(&validation_probabilities)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / 2.,
        );
        // Fixed-radius loss mixture has Bures metric eta*r²/4. Finite differences add O(c²).
        worst_metric =
            worst_metric.max((optimizer.state().metric[0] - ETA * RADIUS.powi(2) / 4.).abs());
    }
    ensure(
        worst_tvd < 0.02 && worst_tvd < baseline_tvd / 4. && worst_heldout < 0.025,
        "mixed optical Born fitting missed held-out distribution",
    )?;
    ensure(
        worst_metric < 0.002,
        "squared-Uhlmann QN-SPSA metric missed analytic Bures metric",
    )?;
    ensure(
        loss_calls == 3 * STEPS * 4 && fidelity_calls == 3 * STEPS * 4,
        "QN-SPSA evaluator accounting changed",
    )?;
    Report::new(
        "photonic_born_qnspsa",
        7,
        STEPS,
        &[
            ("seed_count", 3.),
            ("worst_true_tvd", worst_tvd),
            ("worst_heldout_tvd", worst_heldout),
            ("uniform_photon_baseline_tvd", baseline_tvd),
            ("max_bures_metric_error", worst_metric),
            ("analytic_bures_metric", ETA * RADIUS.powi(2) / 4.),
            ("training_shots", 8192.),
            ("heldout_shots", 16384.),
            ("exact_simulator_loss_calls", loss_calls as f64),
            ("host_spectral_fidelity_calls", fidelity_calls as f64),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn mixed_optical_born_model_uses_bures_natural_updates_and_heldout_samples() {
        super::run().unwrap();
    }
}
