//! Two-photon Hong-Ou-Mandel interference with physical loss and imperfect detection.
use burn::tensor::DType;
use burn_quantum::{ResourceLimits, photonic::optics::*};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let limits = ResourceLimits::default();
    let basis = FockBasis::new(2, PhotonSector::UpTo(2), 6)?;
    let splitter =
        Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)?;
    let state = FockState::number(basis.clone(), &[1, 1], 1, &device, DType::F32, limits)?
        .apply(&splitter, limits)?;
    state.validate(2e-5)?;
    let probabilities = values(state.probabilities())?;
    let coincidence = probabilities[basis.index(&[1, 1])?];
    let bunching = probabilities[basis.index(&[2, 0])?] + probabilities[basis.index(&[0, 2])?];
    let lossy =
        state
            .density(limits)?
            .lose(ModeId(0), 0.8, limits)?
            .lose(ModeId(1), 0.8, limits)?;
    let readout = lossy.detect(&[Detector::new(0.9, 2, 0.)?; 2], 9, limits)?;
    let detected = values(readout.probabilities())?;
    let no_click = detected[readout.records().iter().position(|r| r == &[0, 0]).unwrap()];
    let two_clicks = readout
        .records()
        .iter()
        .zip(&detected)
        .filter(|(r, _)| r.iter().sum::<usize>() == 2)
        .map(|(_, p)| p)
        .sum::<f64>();
    ensure(
        coincidence.abs() < 2e-5 && (bunching - 1.).abs() < 2e-5,
        "HOM interference failed",
    )?;
    ensure(
        (no_click - 0.28f64.powi(2)).abs() < 2e-5 && (two_clicks - 0.72f64.powi(2)).abs() < 2e-5,
        "loss/detection accounting failed",
    )?;
    Report::new(
        "photonic_hom",
        0,
        0,
        &[
            ("coincidence_probability", coincidence),
            ("bunching_probability", bunching),
            ("no_click_probability", no_click),
            ("two_detected_probability", two_clicks),
            ("density_trace", values(lossy.trace())?[0]),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn physical_bunching_and_all_loss_events_are_preserved() {
        super::run().unwrap();
    }
}
