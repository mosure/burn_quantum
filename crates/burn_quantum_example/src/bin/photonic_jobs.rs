//! Asynchronous local optical jobs with immutable calibration and raw missed-photon records.
use burn::tensor::{DType, Tensor};
use burn_quantum::{
    Error,
    photonic::{
        optics::*,
        runtime::{JobIdentity, jobs::*},
    },
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};
fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let limits = Default::default();
    let angles = [0.4f64, 1.1];
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)?;
    let input = FockState::number(
        FockBasis::new(2, PhotonSector::UpTo(1), 3)?,
        &[1, 0],
        2,
        &device,
        DType::F32,
        limits,
    )?
    .apply(&bs, limits)?;
    let experiment = SinglePhotonPhaseExperiment::new(
        input,
        ModeId(0),
        bs.adjoint(),
        vec![Detector::threshold(0.8, 0.)?; 2],
        4,
        limits,
    )?;
    let distribution = experiment.evaluate(Tensor::from_data([0.4f32, 1.1], &device))?;
    let probabilities = values(distribution.probabilities())?;
    let records = distribution.records().to_vec();
    let count = records.len();
    let left = records.iter().position(|v| v == &[1, 0]).unwrap();
    for (i, angle) in angles.iter().enumerate() {
        ensure(
            (probabilities[i * count + left] - 0.8 * (angle / 2.).cos().powi(2)).abs() < 2e-6,
            "native optical calibration oracle",
        )?;
    }
    let capability = JobCapabilities {
        executor: "local-fock-sampler-v1".into(),
        programs: vec!["mz-2-mode-phase-v1".into()],
        compiler: "passive-optics-v1".into(),
        physical_model: "single-photon-mz-eta0.8-v1".into(),
        calibration: "detector-eta0.8-dark0-v1".into(),
        calibration_expires_at: 200,
        readouts: vec!["left_click".into()],
        precision_bits: vec![32],
        max_attempts: 4096,
        max_parameters: 1,
    };
    let mut previous = None;
    let mut worst_error = 0.0f64;
    let mut photons = 0;
    let mut missed = 0;
    for workers in [1, 3] {
        let records = records.clone();
        let probabilities = probabilities.clone();
        let service = LocalSimulationJobs::new(
            capability.clone(),
            JobServiceLimits {
                workers,
                ..Default::default()
            },
            move |request, _, seed, _| {
                let row = angles
                    .iter()
                    .position(|x| request.bindings == vec![*x])
                    .ok_or_else(|| {
                        Error::InvalidValue("uncached immutable phase binding".into())
                    })?;
                let p = &probabilities[row * count..(row + 1) * count];
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let mut draw = rng.random::<f64>() * p.iter().sum::<f64>();
                let mut selected = count - 1;
                for (i, mass) in p.iter().enumerate() {
                    if draw < *mass {
                        selected = i;
                        break;
                    }
                    draw -= mass;
                }
                Ok(RawSimulationShot {
                    raw: records[selected].iter().map(|v| *v as u64).collect(),
                    accepted: records[selected].iter().any(|v| *v != 0),
                    values: vec![f64::from(selected == left)],
                    photons: 1,
                    latency_ticks: 3,
                })
            },
        )?;
        for (i, &angle) in angles.iter().enumerate() {
            service.submit(
                JobId(i as u64),
                SimulationRequest {
                    version: 1,
                    identity: JobIdentity {
                        program: "mz-2-mode-phase-v1".into(),
                        calibration: capability.calibration.clone(),
                        executor: capability.executor.clone(),
                        seed: 311 + i as u64,
                        attempts: 4096,
                    },
                    compiler: capability.compiler.clone(),
                    physical_model: capability.physical_model.clone(),
                    bindings: vec![angle],
                    readouts: vec!["left_click".into()],
                    precision_bits: 32,
                    error_budget: 0.,
                    max_result_bytes: 2 << 20,
                    timeout_ms: 10_000,
                },
                100,
            )?;
        }
        let deadline = Instant::now() + Duration::from_secs(15);
        for id in 0..2 {
            loop {
                let status = service.status(JobId(id))?;
                if status == JobStatus::Completed {
                    break;
                }
                ensure(
                    matches!(status, JobStatus::Running | JobStatus::Queued),
                    "local optical job failed",
                )?;
                ensure(Instant::now() < deadline, "local optical job deadline")?;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let mut all = vec![];
        for (i, &angle) in angles.iter().enumerate() {
            let result = service.result(JobId(i as u64))?;
            ensure(
                result.execution_kind == "local_simulator" && result.shots.len() == 4096,
                "complete raw local provenance",
            )?;
            let mean = result.shots.iter().map(|(_, s)| s.values[0]).sum::<f64>() / 4096.;
            worst_error = worst_error.max((mean - 0.8 * (angle / 2.).cos().powi(2)).abs());
            if workers == 1 {
                photons += result.shots.iter().map(|(_, s)| s.photons).sum::<u64>();
                missed += result.shots.iter().filter(|(_, s)| !s.accepted).count();
            }
            all.push(result.shots);
        }
        if let Some(before) = previous {
            ensure(
                before == all,
                "raw attempt replay independent of worker count",
            )?;
        }
        previous = Some(all);
    }
    ensure(
        worst_error < 0.03,
        "finite-shot optical readout within conservative sampling tolerance",
    )?;
    ensure(
        photons == 8192 && missed > 0,
        "missed photons count against physical budget",
    )?;
    Report::new(
        "photonic_jobs",
        311,
        0,
        &[
            ("physical_preparations_per_layout", photons as f64),
            ("missed_photon_attempts", missed as f64),
            ("worst_probability_error", worst_error),
            ("worker_layouts", 2.),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn asynchronous_photonic_jobs() {
        super::run().unwrap();
    }
}
