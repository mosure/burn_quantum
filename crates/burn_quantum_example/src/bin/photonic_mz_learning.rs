//! Native Burn calibration of a lossy Mach-Zehnder interferometer on disjoint phases.
//! Three seeded initializations; target counts follow an independent analytic law.
use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{ResourceLimits, photonic::optics::*};
use burn_quantum_example::{
    ExampleResult, Report, ensure, execution_device, initial_angles, values,
};
const STEPS: usize = 120;
const TRUE_OFFSET: f64 = 0.65;
const EFFICIENCY: f64 = 0.8;
#[derive(Module, Debug)]
struct PhaseModel {
    offset: Param<Tensor<1>>,
}
fn analytic(x: f64, offset: f64) -> f64 {
    EFFICIENCY * ((x + offset) / 2.).cos().powi(2)
}
fn dataset(count: usize, midpoint: bool, device: &Device) -> (Tensor<1>, Tensor<2>, Vec<f64>) {
    let xs: Vec<_> = (0..count)
        .map(|i| {
            -1.5 + 3.
                * if midpoint {
                    (i as f64 + 0.5) / count as f64
                } else {
                    i as f64 / (count - 1) as f64
                }
        })
        .collect();
    let ys: Vec<_> = xs
        .iter()
        .map(|&x| analytic(x, TRUE_OFFSET) as f32)
        .collect();
    (
        Tensor::from_data(
            TensorData::new(xs.iter().map(|&x| x as f32).collect(), [count]),
            device,
        ),
        Tensor::from_data(TensorData::new(ys, [count, 1]), device),
        xs,
    )
}
fn evaluate(model: &PhaseModel, x: Tensor<1>, device: &Device) -> ExampleResult<Tensor<2>> {
    let limits = ResourceLimits::default();
    let batch = x.dims()[0];
    let basis = FockBasis::new(2, PhotonSector::UpTo(1), 3)?;
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)?;
    let state = FockState::number(basis, &[1, 0], batch, device, DType::F32, limits)?
        .apply(&bs, limits)?
        .phase_shift(ModeId(0), x + model.offset.val(), limits)?
        .apply(&bs.adjoint(), limits)?;
    // Inefficient detectors retain missed photons as all-zero records; no conditioning.
    let readout = state.detect(&[Detector::threshold(EFFICIENCY, 0.)?; 2], 4, limits)?;
    let index = readout.records().iter().position(|r| r == &[1, 0]).unwrap();
    Ok(readout.probabilities().slice_dim(1, index..index + 1))
}
fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let (train_x, train_y, train_points) = dataset(12, false, &device);
    let (test_x, test_y, test_points) = dataset(17, true, &device);
    ensure(
        train_points
            .iter()
            .all(|a| test_points.iter().all(|b| (a - b).abs() > 1e-10)),
        "train/test phases overlap",
    )?;
    let baseline = test_points
        .iter()
        .map(|&x| (analytic(x, 0.) - analytic(x, TRUE_OFFSET)).powi(2))
        .sum::<f64>()
        / 17.;
    let mut worst_mse = 0f64;
    let mut worst_offset = 0f64;
    let mut worst_oracle = 0f64;
    for seed in [7, 11, 19] {
        let mut model = PhaseModel {
            offset: Param::from_tensor(Tensor::from_data(
                [initial_angles(1, seed)[0] as f32],
                &device,
            )),
        };
        let mut optimizer = AdamConfig::new().init();
        for _ in 0..STEPS {
            let loss = (evaluate(&model, train_x.clone(), &device)? - train_y.clone())
                .square()
                .mean();
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(0.06, model, gradients);
        }
        let prediction = evaluate(&model, test_x.clone(), &device)?;
        let mse = (prediction.clone() - test_y.clone())
            .square()
            .mean()
            .into_scalar::<f64>();
        let offset = values(model.offset.val())?[0];
        let oracle_error = values(prediction)?
            .iter()
            .zip(&test_points)
            .map(|(&actual, &x)| (actual - analytic(x, offset)).abs())
            .fold(0f64, f64::max);
        worst_mse = worst_mse.max(mse);
        worst_offset = worst_offset.max((offset - TRUE_OFFSET).abs());
        worst_oracle = worst_oracle.max(oracle_error);
    }
    ensure(
        worst_mse < 1e-5 && worst_mse < baseline / 100.,
        "optical learning missed held-out count probabilities",
    )?;
    ensure(
        worst_offset < 0.02 && worst_oracle < 2e-5,
        "phase calibration disagrees with the analytic optical law",
    )?;
    Report::new(
        "photonic_mz_learning",
        7,
        STEPS,
        &[
            ("initialization_count", 3.),
            ("worst_test_mse", worst_mse),
            ("uncalibrated_baseline_mse", baseline),
            ("worst_offset_error", worst_offset),
            ("max_analytic_error", worst_oracle),
            ("detector_efficiency", EFFICIENCY),
            ("photons_per_attempt", 1.),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn learns_physical_count_probabilities_on_disjoint_inputs() {
        super::run().unwrap();
    }
}
