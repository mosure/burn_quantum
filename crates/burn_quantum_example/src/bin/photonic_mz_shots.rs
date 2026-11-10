//! Finite-shot phase calibration: independent noisy residual and parameter-shift probes.
//! Every attempted photon, including detector misses, is charged to the shot budget.
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::photonic::{optics::*, runtime::shot_seed};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
const SHOTS: usize = 512;
const STEPS: usize = 80;
const TRAIN: usize = 8;
const TRUE_OFFSET: f64 = 0.65;
const EFFICIENCY: f64 = 0.8;
fn analytic(x: f64, offset: f64) -> f64 {
    EFFICIENCY * ((x + offset) / 2.).cos().powi(2)
}
fn experiment(batch: usize, device: &Device) -> ExampleResult<SinglePhotonPhaseExperiment> {
    let limits = Default::default();
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)?;
    let input = FockState::number(
        FockBasis::new(2, PhotonSector::UpTo(1), 3)?,
        &[1, 0],
        batch,
        device,
        DType::F32,
        limits,
    )?
    .apply(&bs, limits)?;
    Ok(SinglePhotonPhaseExperiment::new(
        input,
        ModeId(0),
        bs.adjoint(),
        vec![Detector::threshold(EFFICIENCY, 0.)?; 2],
        4,
        limits,
    )?)
}
fn angles(points: &[f64], offset: f64, device: &Device) -> Tensor<1> {
    Tensor::from_data(
        TensorData::new(
            points.iter().map(|&x| (x + offset) as f32).collect(),
            [points.len()],
        ),
        device,
    )
}
fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let training: Vec<_> = (0..TRAIN)
        .map(|i| -1.5 + 3. * i as f64 / (TRAIN - 1) as f64)
        .collect();
    let testing: Vec<_> = (0..17)
        .map(|i| -1.5 + 3. * (i as f64 + 0.5) / 17.)
        .collect();
    ensure(
        training
            .iter()
            .all(|a| testing.iter().all(|b| (a - b).abs() > 1e-10)),
        "phase splits overlap",
    )?;
    let training_experiment = experiment(TRAIN, &device)?;
    let testing_experiment = experiment(testing.len(), &device)?;
    let distribution = training_experiment.evaluate(angles(&training, 0., &device))?;
    let index = distribution
        .records()
        .iter()
        .position(|r| r == &[1, 0])
        .unwrap();
    let readout: Vec<_> = distribution
        .records()
        .iter()
        .map(|r| f64::from(r == &[1, 0]))
        .collect();
    let baseline = testing
        .iter()
        .map(|&x| (analytic(x, 0.) - analytic(x, TRUE_OFFSET)).powi(2))
        .sum::<f64>()
        / testing.len() as f64;
    let mut worst_mse = 0f64;
    let mut worst_offset = 0f64;
    let mut attempts = 0usize;
    for (seed, initial) in [(7, -0.4), (11, 0.1), (19, 1.2)] {
        let mut offset = initial;
        for step in 0..STEPS {
            let theta = angles(&training, offset, &device);
            // Independent streams make residual * derivative an unbiased MSE-gradient estimator.
            let means = training_experiment
                .evaluate(theta.clone())?
                .sample_readout(
                    &readout,
                    SHOTS,
                    shot_seed(seed, step as u64, 1),
                    TRAIN * SHOTS,
                )?;
            let derivatives = training_experiment.gradient(
                theta,
                &readout,
                SHOTS,
                shot_seed(seed, step as u64, 2),
                2 * TRAIN * SHOTS,
            )?;
            let gradient = training
                .iter()
                .zip(&means)
                .zip(&derivatives)
                .map(|((&x, m), g)| 2. * (m.mean - analytic(x, TRUE_OFFSET)) * g.gradient)
                .sum::<f64>()
                / TRAIN as f64;
            offset -= 1.2 * gradient;
            attempts += means.iter().map(|m| m.attempts).sum::<usize>()
                + derivatives.iter().map(|g| g.attempts).sum::<usize>();
        }
        // Held-out model quality is evaluated exactly, independently of the training shot stream.
        let prediction = testing_experiment.evaluate(angles(&testing, offset, &device))?;
        let predictions = values(prediction.probabilities().slice_dim(1, index..index + 1))?;
        let mse = predictions
            .iter()
            .zip(&testing)
            .map(|(&p, &x)| (p - analytic(x, TRUE_OFFSET)).powi(2))
            .sum::<f64>()
            / testing.len() as f64;
        worst_mse = worst_mse.max(mse);
        worst_offset = worst_offset.max((offset - TRUE_OFFSET).abs());
    }
    ensure(
        attempts == 3 * STEPS * TRAIN * SHOTS * 3,
        "shot accounting omitted a probe",
    )?;
    ensure(
        worst_mse < 1e-4 && worst_mse < baseline / 30. && worst_offset < 0.05,
        "finite-shot calibration missed held-out targets",
    )?;
    Report::new(
        "photonic_mz_shots",
        7,
        STEPS,
        &[
            ("seed_count", 3.),
            ("worst_test_mse", worst_mse),
            ("uncalibrated_baseline_mse", baseline),
            ("worst_offset_error", worst_offset),
            ("total_physical_attempts", attempts as f64),
            ("shots_per_probe", SHOTS as f64),
            ("detector_efficiency", EFFICIENCY),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn finite_shot_training_generalizes_with_complete_attempt_accounting() {
        super::run().unwrap();
    }
}
