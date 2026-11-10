//! Seeded finite-shot classification/kernel/Born benchmark with held-out calibration.
//! Classical controls use the identical measured records and density feature map.
//! Photon/time/energy totals use explicit illustrative assumptions, not hardware telemetry.
use burn::tensor::{DType, Device, Tensor};
use burn_quantum::{
    ComplexTensor,
    photonic::{optics::*, runtime::shot_seed},
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use serde::Serialize;
const ETA: f64 = 0.72;
const VISIBILITY: f64 = 0.78;
const OFFSET: f64 = 0.37;
const TOMOGRAPHY_SHOTS: usize = 256;
const CALIBRATION_SHOTS: usize = 2048;
const BORN_SHOTS: usize = 1024;
#[derive(Debug, Serialize)]
struct SeedRecord {
    seed: u64,
    classification_accuracy: f64,
    kernel_accuracy: f64,
    majority_baseline: f64,
    matched_classical_disagreements: usize,
    max_kernel_oracle_error: f64,
    born_true_tvd: f64,
    born_heldout_tvd: f64,
    uncalibrated_born_tvd: f64,
    fitted_survival: f64,
    fitted_visibility: f64,
    fitted_phase: f64,
    calibration_allocation: ShotAllocation,
    calibration_learning_curve: Vec<CalibrationLearningPoint>,
    kernel_learning_curve: Vec<KernelLearningPoint>,
    cost: OpticalResourceCost,
}
/// Actual pilot/full calibration estimates, evaluated against identical held-out records.
#[derive(Debug, Serialize)]
struct CalibrationLearningPoint {
    completed_calibration_attempts: usize,
    classification_accuracy: f64,
    born_true_tvd: f64,
    born_heldout_tvd: f64,
    classification_cost: OpticalResourceCost,
    born_cost: OpticalResourceCost,
}
/// Nested balanced training subsets; each point charges its own train/test preparations.
#[derive(Debug, Serialize)]
struct KernelLearningPoint {
    training_examples: usize,
    heldout_examples: usize,
    kernel_accuracy: f64,
    matched_classical_accuracy: f64,
    matched_classical_disagreements: usize,
    cost: OpticalResourceCost,
}
fn state(phase: f64, device: &Device) -> ExampleResult<FockDensity> {
    let limits = Default::default();
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)?;
    let rho = FockState::number(
        FockBasis::new(2, PhotonSector::UpTo(1), 3)?,
        &[1, 0],
        1,
        device,
        DType::F32,
        limits,
    )?
    .apply(&bs, limits)?
    .phase_shift(ModeId(0), Tensor::from_data([phase as f32], device), limits)?
    .density(limits)?;
    let spread = VISIBILITY.acos();
    let noise = CalibratedNoiseEnsemble::new(
        [-spread, spread]
            .into_iter()
            .map(|noise| OpticalNoiseBranch {
                probability: 0.5,
                transmissions: vec![ETA; 2],
                phases: vec![OFFSET + noise, 0.],
            })
            .collect(),
        "bench-phase-loss-v1".into(),
    )?;
    Ok(noise.apply(&rho, limits)?)
}
fn counts(phase: f64, shots: usize, seed: u64, device: &Device) -> ExampleResult<[usize; 3]> {
    let bs = Interferometer::beam_splitter(2, ModeId(0), ModeId(1), std::f64::consts::FRAC_PI_2)?;
    let distribution = state(phase, device)?
        .apply(&bs.adjoint(), Default::default())?
        .detect(&[Detector::ideal(1)?; 2], 4, Default::default())?;
    let sample = distribution.sample_counts(shots, seed, shots)?.remove(0);
    Ok([[0, 0], [1, 0], [0, 1]]
        .map(|r| sample.counts[distribution.records().iter().position(|o| o == &r).unwrap()]))
}
fn analytic(phase: f64) -> [f64; 3] {
    [
        1. - ETA,
        ETA * (1. + VISIBILITY * (phase + OFFSET).cos()) / 2.,
        ETA * (1. - VISIBILITY * (phase + OFFSET).cos()) / 2.,
    ]
}
fn tvd(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.
}
fn batch(states: &[FockDensity]) -> ExampleResult<FockDensity> {
    Ok(FockDensity::from_entries(
        states[0].basis().clone(),
        ComplexTensor::new(
            Tensor::cat(states.iter().map(|s| s.entries().real()).collect(), 0),
            Tensor::cat(states.iter().map(|s| s.entries().imag()).collect(), 0),
        )?,
        Default::default(),
    )?)
}
fn score(kernel: &[f64], labels: &[bool], norms: [f64; 2]) -> f64 {
    let mut means = [0.; 2];
    let mut totals = [0; 2];
    for (&k, &label) in kernel.iter().zip(labels) {
        let c = usize::from(label);
        means[c] += k;
        totals[c] += 1;
    }
    2. * (means[1] / totals[1] as f64 - means[0] / totals[0] as f64) - norms[1] + norms[0]
}
fn run() -> ExampleResult<(Report, Vec<SeedRecord>, OpticalCostModel)> {
    let device = execution_device()?;
    let train: Vec<_> = (0..12)
        .map(|i| std::f64::consts::TAU * (i as f64 + 0.2) / 12.)
        .collect();
    let test: Vec<_> = (0..25)
        .map(|i| std::f64::consts::TAU * (i as f64 + 0.37) / 25.)
        .collect();
    ensure(
        train
            .iter()
            .all(|a| test.iter().all(|b| (a - b).abs() > 1e-10)),
        "benchmark train/test overlap",
    )?;
    let labels: Vec<_> = train.iter().map(|x| x.cos() > 0.).collect();
    let protocol = OpticalTomography::new(TOMOGRAPHY_SHOTS, 2, 3 * TOMOGRAPHY_SHOTS, 0.01)?;
    let model = OpticalCostModel {
        mean_photons_per_attempt: 1.,
        seconds_per_attempt: 1e-6,
        joules_per_attempt: 1e-12,
        parallel_lanes: 4,
    };
    let mut runs = vec![];
    for seed in [7, 11, 19, 23, 31] {
        const PILOT: usize = 256;
        let mut probes = vec![];
        for i in 0..4 {
            let phase = i as f64 * std::f64::consts::FRAC_PI_2;
            probes.push(CalibrationProbe {
                phase,
                counts: counts(phase, PILOT, shot_seed(seed, i, 1), &device)?,
            });
        }
        let pilot_fit = MachZehnderCalibration::fit(&probes, 0.1)?;
        let items: Vec<_> = probes
            .iter()
            .map(|p| {
                let mean = (p.counts[1] as f64 - p.counts[2] as f64) / PILOT as f64;
                let variance = ((p.counts[1] + p.counts[2]) as f64 / PILOT as f64 - mean * mean)
                    * PILOT as f64
                    / (PILOT - 1) as f64;
                ShotAllocationItem {
                    completed: PILOT,
                    variance,
                    coefficient: 1.,
                }
            })
            .collect();
        let allocation = ShotAllocationPolicy {
            minimum_per_item: PILOT,
            maximum_per_item: 2 * CALIBRATION_SHOTS,
            round_budget: 4 * (CALIBRATION_SHOTS - PILOT),
            total_budget: 4 * CALIBRATION_SHOTS,
            variance_floor: 1e-3,
        }
        .allocate(&items)?;
        for (i, probe) in probes.iter_mut().enumerate() {
            let extra = counts(
                probe.phase,
                allocation.additional[i],
                shot_seed(seed, i as u64, 11),
                &device,
            )?;
            for (total, n) in probe.counts.iter_mut().zip(extra) {
                *total += n;
            }
        }
        let fit = MachZehnderCalibration::fit(&probes, 0.1)?;
        let mut attempts = fit.attempts;
        let mut reconstructed = vec![];
        for (i, &phase) in train.iter().chain(&test).enumerate() {
            let result = protocol.reconstruct(
                &state(phase, &device)?,
                shot_seed(seed, i as u64, 2),
                Default::default(),
            )?;
            attempts += result.attempts;
            reconstructed.push(result.state);
        }
        let train_states = batch(&reconstructed[..train.len()])?;
        let test_states = batch(&reconstructed[train.len()..])?;
        let kernel =
            values(test_states.hilbert_schmidt_kernel(&train_states, Default::default())?)?;
        let gram = values(train_states.hilbert_schmidt_kernel(&train_states, Default::default())?)?;
        let mut norms = [0.; 2];
        let class_counts = [
            labels.iter().filter(|&&x| !x).count(),
            labels.iter().filter(|&&x| x).count(),
        ];
        for a in 0..train.len() {
            for b in 0..train.len() {
                if labels[a] == labels[b] {
                    let c = usize::from(labels[a]);
                    norms[c] +=
                        gram[a * train.len() + b] / (class_counts[c] * class_counts[c]) as f64;
                }
            }
        }
        let mut vectors = vec![];
        for rho in &reconstructed {
            vectors.push((values(rho.entries().real())?, values(rho.entries().imag())?));
        }
        let mut direct_correct = 0;
        let mut kernel_correct = 0;
        let mut disagreements = 0;
        let mut max_error = 0f64;
        let mut classical_rows = vec![];
        for (i, &phase) in test.iter().enumerate() {
            let (re, im) = &vectors[train.len() + i];
            let coherence = re[7] * fit.phase_offset.cos() + im[7] * fit.phase_offset.sin();
            direct_correct += usize::from((coherence > 0.) == (phase.cos() > 0.));
            let control: Vec<_> = vectors[..train.len()]
                .iter()
                .map(|(r, s)| {
                    re.iter().zip(r).map(|(a, b)| a * b).sum::<f64>()
                        + im.iter().zip(s).map(|(a, b)| a * b).sum::<f64>()
                })
                .collect();
            let row = &kernel[i * train.len()..(i + 1) * train.len()];
            for (a, b) in row.iter().zip(&control) {
                max_error = max_error.max((a - b).abs());
            }
            let prediction = score(row, &labels, norms) > 0.;
            let classical = score(&control, &labels, norms) > 0.;
            kernel_correct += usize::from(prediction == (phase.cos() > 0.));
            disagreements += usize::from(prediction != classical);
            classical_rows.push(control);
        }
        let mut born_true = 0.;
        let mut born_heldout = 0.;
        let mut uncalibrated = 0.;
        let mut born_evaluation = vec![];
        for (i, phase) in [0.21, 1.17, 2.29, 4.88].into_iter().enumerate() {
            let observed = counts(phase, BORN_SHOTS, shot_seed(seed, i as u64, 3), &device)?;
            attempts += BORN_SHOTS;
            let empirical = observed.map(|n| n as f64 / BORN_SHOTS as f64);
            born_evaluation.push((phase, empirical));
            let prediction = fit.probabilities(phase)?;
            born_true += tvd(&prediction, &analytic(phase)) / 4.;
            born_heldout += tvd(&prediction, &empirical) / 4.;
            uncalibrated += tvd(
                &[0., (phase / 2.).cos().powi(2), (phase / 2.).sin().powi(2)],
                &analytic(phase),
            ) / 4.;
        }
        let mut calibration_learning_curve = vec![];
        for estimate in [&pilot_fit, &fit] {
            let correct = test
                .iter()
                .enumerate()
                .filter(|(i, phase)| {
                    let (re, im) = &vectors[train.len() + i];
                    let coherence =
                        re[7] * estimate.phase_offset.cos() + im[7] * estimate.phase_offset.sin();
                    (coherence > 0.) == (phase.cos() > 0.)
                })
                .count();
            let mut true_tvd = 0.;
            let mut heldout_tvd = 0.;
            for &(phase, empirical) in &born_evaluation {
                let prediction = estimate.probabilities(phase)?;
                true_tvd += tvd(&prediction, &analytic(phase)) / born_evaluation.len() as f64;
                heldout_tvd += tvd(&prediction, &empirical) / born_evaluation.len() as f64;
            }
            calibration_learning_curve.push(CalibrationLearningPoint {
                completed_calibration_attempts: estimate.attempts,
                classification_accuracy: correct as f64 / test.len() as f64,
                born_true_tvd: true_tvd,
                born_heldout_tvd: heldout_tvd,
                classification_cost: model
                    .cost(estimate.attempts + test.len() * 3 * TOMOGRAPHY_SHOTS)?,
                born_cost: model.cost(estimate.attempts + born_evaluation.len() * BORN_SHOTS)?,
            });
        }
        // Nested balanced training subsets reuse the measured record pool.
        // Their costs are per-budget experiments, not extra queries charged again.
        let training_order: Vec<_> = (0..6).flat_map(|i| [i, i + 6]).collect();
        let mut kernel_learning_curve = vec![];
        for training_examples in [4, 8, 12] {
            let selected = &training_order[..training_examples];
            let selected_labels: Vec<_> = selected.iter().map(|&i| labels[i]).collect();
            let totals = [
                selected_labels.iter().filter(|&&v| !v).count(),
                selected_labels.iter().filter(|&&v| v).count(),
            ];
            ensure(
                totals.iter().all(|&n| n > 0),
                "learning-curve training subset lacks a class",
            )?;
            let mut native_norms = [0.; 2];
            let mut classical_norms = [0.; 2];
            for &a in selected {
                for &b in selected {
                    if labels[a] == labels[b] {
                        let class = usize::from(labels[a]);
                        let denominator = (totals[class] * totals[class]) as f64;
                        native_norms[class] += gram[a * train.len() + b] / denominator;
                        let (ar, ai) = &vectors[a];
                        let (br, bi) = &vectors[b];
                        classical_norms[class] +=
                            (ar.iter().zip(br).map(|(a, b)| a * b).sum::<f64>()
                                + ai.iter().zip(bi).map(|(a, b)| a * b).sum::<f64>())
                                / denominator;
                    }
                }
            }
            let mut native_correct = 0;
            let mut classical_correct = 0;
            let mut disagreements = 0;
            for (i, &phase) in test.iter().enumerate() {
                let native_row: Vec<_> = selected
                    .iter()
                    .map(|&j| kernel[i * train.len() + j])
                    .collect();
                let classical_row: Vec<_> =
                    selected.iter().map(|&j| classical_rows[i][j]).collect();
                let native = score(&native_row, &selected_labels, native_norms) > 0.;
                let classical = score(&classical_row, &selected_labels, classical_norms) > 0.;
                native_correct += usize::from(native == (phase.cos() > 0.));
                classical_correct += usize::from(classical == (phase.cos() > 0.));
                disagreements += usize::from(native != classical);
            }
            ensure(
                disagreements == 0,
                "learning curve differs from matched-cost classical control",
            )?;
            kernel_learning_curve.push(KernelLearningPoint {
                training_examples,
                heldout_examples: test.len(),
                kernel_accuracy: native_correct as f64 / test.len() as f64,
                matched_classical_accuracy: classical_correct as f64 / test.len() as f64,
                matched_classical_disagreements: disagreements,
                cost: model.cost((training_examples + test.len()) * 3 * TOMOGRAPHY_SHOTS)?,
            });
        }
        let classification_accuracy = direct_correct as f64 / test.len() as f64;
        let kernel_accuracy = kernel_correct as f64 / test.len() as f64;
        let positives = test.iter().filter(|x| x.cos() > 0.).count();
        let majority = positives.max(test.len() - positives) as f64 / test.len() as f64;
        ensure(
            classification_accuracy >= 0.88 && kernel_accuracy >= 0.84,
            "finite-shot classifier missed held-out labels",
        )?;
        ensure(
            disagreements == 0 && max_error < 3e-6,
            "native density kernel disagrees with identical-data classical features",
        )?;
        ensure(
            born_true < 0.035 && born_heldout < 0.05 && born_true < uncalibrated / 5.,
            "calibrated Born distribution failed independent holdout",
        )?;
        ensure(
            (fit.phase_offset - OFFSET).abs() < 0.12 && (fit.visibility - VISIBILITY).abs() < 0.07,
            "calibration parameters missed identifiable reference",
        )?;
        runs.push(SeedRecord {
            seed,
            classification_accuracy,
            kernel_accuracy,
            majority_baseline: majority,
            matched_classical_disagreements: disagreements,
            max_kernel_oracle_error: max_error,
            born_true_tvd: born_true,
            born_heldout_tvd: born_heldout,
            uncalibrated_born_tvd: uncalibrated,
            fitted_survival: fit.survival,
            fitted_visibility: fit.visibility,
            fitted_phase: fit.phase_offset,
            calibration_allocation: allocation,
            calibration_learning_curve,
            kernel_learning_curve,
            cost: model.cost(attempts)?,
        });
    }
    let report = Report::new(
        "photonic_benchmarks",
        7,
        0,
        &[
            ("seed_count", runs.len() as f64),
            ("workloads", 3.),
            (
                "worst_classification_accuracy",
                runs.iter()
                    .map(|r| r.classification_accuracy)
                    .fold(1f64, f64::min),
            ),
            (
                "worst_kernel_accuracy",
                runs.iter().map(|r| r.kernel_accuracy).fold(1f64, f64::min),
            ),
            (
                "worst_born_true_tvd",
                runs.iter().map(|r| r.born_true_tvd).fold(0f64, f64::max),
            ),
            (
                "total_physical_attempts",
                runs.iter().map(|r| r.cost.attempts as f64).sum(),
            ),
            ("matched_control_additional_physical_attempts", 0.),
        ],
    )?;
    Ok((report, runs, model))
}
fn main() -> ExampleResult<()> {
    let (report, runs, model) = run()?;
    let mut json = serde_json::to_value(report)?;
    json["runs"] = serde_json::to_value(runs)?;
    json["illustrative_cost_model"] = serde_json::to_value(model)?;
    json["matched_control"] = serde_json::json!(
        "identical measured records, scalar density-feature inner product; zero extra physical queries; no quantum advantage claim"
    );
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn calibrated_photonic_workloads_pass_seeded_matched_data_controls() {
        let (_, runs, _) = super::run().unwrap();
        for run in runs {
            assert_eq!(
                run.calibration_learning_curve
                    .iter()
                    .map(|p| p.completed_calibration_attempts)
                    .collect::<Vec<_>>(),
                vec![1024, 8192]
            );
            assert_eq!(
                run.kernel_learning_curve
                    .iter()
                    .map(|p| p.training_examples)
                    .collect::<Vec<_>>(),
                vec![4, 8, 12]
            );
            assert_eq!(
                run.kernel_learning_curve
                    .iter()
                    .map(|p| p.cost.attempts)
                    .collect::<Vec<_>>(),
                vec![22272, 25344, 28416]
            );
            let final_calibration = run.calibration_learning_curve.last().unwrap();
            assert_eq!(
                final_calibration.classification_accuracy,
                run.classification_accuracy
            );
            assert_eq!(final_calibration.born_true_tvd, run.born_true_tvd);
            assert_eq!(final_calibration.classification_cost.attempts, 27392);
            assert_eq!(final_calibration.born_cost.attempts, 12288);
            assert_eq!(
                run.kernel_learning_curve.last().unwrap().kernel_accuracy,
                run.kernel_accuracy
            );
            assert_eq!(run.cost.attempts, 40704);
        }
    }
}
