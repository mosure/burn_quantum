//! Lossy mixed-state optical kernel classification with a matched classical control.
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::photonic::optics::*;
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
const ETA: f64 = 0.6;
fn features(points: &[f64], device: &Device) -> ExampleResult<FockDensity> {
    let limits = Default::default();
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)?;
    let phases = Tensor::from_data(
        TensorData::new(points.iter().map(|&x| x as f32).collect(), [points.len()]),
        device,
    );
    Ok(FockState::number(
        FockBasis::new(2, PhotonSector::UpTo(1), 3)?,
        &[1, 0],
        points.len(),
        device,
        DType::F32,
        limits,
    )?
    .apply(&bs, limits)?
    .phase_shift(ModeId(0), phases, limits)?
    .density(limits)?
    .lose(ModeId(0), ETA, limits)?
    .lose(ModeId(1), ETA, limits)?)
}
fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let training: Vec<_> = (0..20)
        .map(|i| std::f64::consts::TAU * (i as f64 + 0.2) / 20.)
        .collect();
    let testing: Vec<_> = (0..65)
        .map(|i| std::f64::consts::TAU * (i as f64 + 0.37) / 65.)
        .collect();
    ensure(
        training
            .iter()
            .all(|a| testing.iter().all(|b| (a - b).abs() > 1e-10)),
        "kernel data splits overlap",
    )?;
    let labels: Vec<_> = training
        .iter()
        .map(|x| if x.cos() > 0. { 1. } else { -1. })
        .collect();
    let positive = labels.iter().filter(|&&y| y > 0.).count();
    let negative = labels.len() - positive;
    let kernel = values(
        features(&testing, &device)?
            .hilbert_schmidt_kernel(&features(&training, &device)?, Default::default())?,
    )?;
    let mut errors = 0;
    let mut disagreement = 0;
    let mut max_oracle = 0f64;
    for (i, &x) in testing.iter().enumerate() {
        let mut score = 0.;
        let mut control = 0.;
        for (j, (&z, &label)) in training.iter().zip(&labels).enumerate() {
            // Explicit finite feature map: vacuum, constant photon term, cos(x), sin(x).
            let analytic = (1. - ETA).powi(2) + ETA.powi(2) * ((x - z) / 2.).cos().powi(2);
            let weight = label
                / if label > 0. {
                    positive as f64
                } else {
                    negative as f64
                };
            score += kernel[i * training.len() + j] * weight;
            control += analytic * weight;
            max_oracle = max_oracle.max((kernel[i * training.len() + j] - analytic).abs());
        }
        errors += usize::from((score > 0.) != (x.cos() > 0.));
        disagreement += usize::from((score > 0.) != (control > 0.));
    }
    let accuracy = 1. - errors as f64 / testing.len() as f64;
    let majority = testing
        .iter()
        .filter(|x| x.cos() > 0.)
        .count()
        .max(testing.iter().filter(|x| x.cos() <= 0.).count()) as f64
        / testing.len() as f64;
    ensure(
        accuracy > 0.95 && accuracy > majority + 0.4,
        "optical kernel classifier failed held-out labels",
    )?;
    ensure(
        disagreement == 0 && max_oracle < 3e-6,
        "native mixed-state kernel disagrees with matched classical features",
    )?;
    Report::new(
        "photonic_kernel",
        0,
        1,
        &[
            ("test_accuracy", accuracy),
            ("majority_baseline", majority),
            ("matched_control_disagreements", disagreement as f64),
            ("max_kernel_oracle_error", max_oracle),
            ("physical_transmission", ETA),
            ("state_purity", (1. - ETA).powi(2) + ETA.powi(2)),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn lossy_kernel_classification_matches_control_on_heldout_inputs() {
        super::run().unwrap();
    }
}
