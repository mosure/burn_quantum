use super::*;
use burn_quantum::algorithms::amplification::{AmplificationSchedule, OddParityExperiment};

const RATE: f64 = 0.04;
const LOWER_BOUND: f64 = 0.08;
const FAILURE: f64 = 0.01;

fn configure(config: DemoConfig) -> Result<(OddParityExperiment, AmplificationSchedule)> {
    let experiment = OddParityExperiment::new(config.size, RATE)?;
    let fixed = AmplificationSchedule::fixed_point(LOWER_BOUND, FAILURE)?;
    let schedule = match config.variant {
        0 => fixed,
        1 => AmplificationSchedule::known_success(experiment.initial_success())?,
        _ => AmplificationSchedule::standard(fixed.rounds()),
    };
    Ok((experiment, schedule))
}

pub(super) fn preview(config: DemoConfig, limits: TraceLimits) -> Result<Preview> {
    let (experiment, schedule) = configure(config)?;
    let amp = experiment.amplifier();
    catalog::preview_register(
        amp.oracle().register(),
        amp.operation_count(schedule.rounds())?,
        limits,
    )
}

pub(super) async fn build(
    config: DemoConfig,
    device: &Device,
    limits: TraceLimits,
) -> Result<VisualizationDocument> {
    let kind = DemoKind::AmplitudeAmplification;
    let (experiment, schedule) = configure(config)?;
    let rounds = schedule.rounds();
    let algorithm = experiment.labeled_circuit(schedule, limits.max_frames.saturating_sub(1))?;
    let trace = simulate(
        &algorithm.circuit,
        &[],
        &[],
        &algorithm.labels,
        device,
        limits,
    )
    .await?;
    let mut round_success = Vec::with_capacity(rounds + 1);
    for round in 0..=rounds {
        let step = experiment.amplifier().operation_count(round)?;
        round_success.push(experiment.readout(&trace.frames[step].probabilities)?.0);
    }
    let final_mass = round_success[rounds];
    let clean = experiment
        .readout(&trace.frames.last().unwrap().probabilities)?
        .1;
    let reference = schedule.predicted_success(experiment.initial_success())?;
    let (title, interpretation) = match config.variant {
        0 => (
            "Fixed-point amplification",
            "An 8% lower bound on initial odd-parity mass guarantees 99% final success.",
        ),
        1 => (
            "Known-mass Grover search",
            "The initial odd-parity mass sets the stopping round nearest the first probability peak.",
        ),
        _ => (
            "Equal-budget Grover search",
            "Five ordinary Grover rounds show how success can fall and recover as the state rotates.",
        ),
    };
    let study = StudyReport {
        title: title.into(),
        summary: format!("Flag odd fault counts among {} data bits with a 4% event rate per bit. {interpretation}", config.size),
        metrics: vec![
            StudyMetric::new("Final odd-parity mass", 100. * final_mass, "%"),
            StudyMetric::new("Before amplification", 100. * round_success[0], "%"),
            StudyMetric::new("Analytic final mass", 100. * reference, "%"),
            StudyMetric::new("Bit-oracle executions", (2 * rounds) as f64, "calls"),
            StudyMetric::new("Clean flag at output", 100. * clean, "%"),
        ],
        notes: vec![
            config.variant_label(kind),
            format!("The event model is a coherent preparation with one Ry rotation per data bit. The last wire, q{}, starts and ends in zero. Odd data strings are verified by counting set bits.", config.size),
            "Each plotted point sums the measured probabilities of all odd data strings after preparation or a complete round. Circuit view also shows the intermediate compute, phase and uncompute gates.".into(),
            "The initial odd-event probability is (1 - 0.92^n)/2. Standard rounds follow the Grover rotation law. Fixed-point phases use the Yoder–Low–Chuang Chebyshev polynomial. The guarantee applies at the end of the complete schedule.".into(),
            "Each good-space phase calls the reversible parity oracle forward and backward. The CLI also runs a seeded exponential search with verified terminal samples. Dense simulation stores 2^(n+1) amplitudes.".into(),
        ],
        amplification: Some(AmplificationSummary { round_success }),
        classification: None,
        sat: None,
        photonic: None,
    };
    let document = VisualizationDocument {
        schema_version: 2,
        title: kind.title().into(),
        description: format!("{} {interpretation}", kind.description()),
        circuit: algorithm.circuit,
        parameters: vec![],
        inputs: vec![],
        operation_labels: algorithm.labels,
        trace,
        fabric: None,
        optimization: vec![],
        protocol: None,
        study: Some(study),
    };
    document.validate_with(limits)?;
    Ok(document)
}
