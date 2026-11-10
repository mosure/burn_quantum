//! Real-data task adapter; training and preprocessing live in the reusable core crate.
use super::*;
use burn_quantum::qml::benchmarks::wine::{self, WineConfig};

fn configuration(config: DemoConfig) -> WineConfig {
    WineConfig {
        qubits: config.size,
        depth: config.depth,
        ..Default::default()
    }
}
pub(super) fn preview(config: DemoConfig, limits: TraceLimits) -> Result<Preview> {
    let config = configuration(config);
    training::preview(&wine::ansatz(&config)?.0, config.steps + 1, limits)
}
pub(super) async fn build(
    config: DemoConfig,
    device: &Device,
    limits: TraceLimits,
) -> Result<VisualizationDocument> {
    preview(config, limits)?;
    let run =
        wine::run_with_progress(&configuration(config), device, training::yield_to_ui).await?;
    let parameters = run.history.last().unwrap().parameters.clone();
    let validation_loss = run.history.iter().map(|p| p.validation_loss).collect();
    let iterates = run
        .history
        .iter()
        .enumerate()
        .map(|(i, p)| training::Iterate {
            parameters: p.parameters.clone(),
            loss: p.train_loss,
            evaluations: 2 * (i + 1),
        })
        .collect();
    let (trace, optimization) =
        training::trace(&run.circuit, &run.probe_inputs, iterates, device, limits).await?;
    let kind = DemoKind::WineClassifier;
    let study=StudyReport {
        title:"Wine cultivar classification".into(),
        summary:"Predict one of three cultivars from 13 real chemical measurements. All inputs are reuploaded into an entangling quantum circuit; a jointly trained classical head reads every qubit's Z expectation.".into(),
        metrics:vec![
            StudyMetric::new("Held-out QML accuracy",100.*run.test.accuracy,"%"),
            StudyMetric::new("Linear baseline accuracy",100.*run.linear_baseline.accuracy,"%"),
            StudyMetric::new("Majority baseline accuracy",100.*run.majority_accuracy,"%"),
            StudyMetric::new("Held-out cross entropy",run.test.cross_entropy,"nats"),
            StudyMetric::new("Quantum parameters",run.circuit.num_parameters() as f64,"angles"),
            StudyMetric::new("Total learned parameters",run.learned_parameters as f64,"scalars"),
            StudyMetric::new("Batched quantum forwards",run.quantum_forward_passes as f64,"passes"),
            StudyMetric::new("State dimension / row",run.circuit.register().size() as f64,"amplitudes"),
        ],
        notes:vec![
            wine::DATASET_CITATION.into(),
            format!("Seed {}. Disjoint train/validation/test partitions: {}/{}/{}. Standardization is fitted only on training rows; each angle is 0.8 atan(z).",run.config.seed,run.split.train.len(),run.split.validation.len(),run.split.test.len()),
            format!("Lowest validation loss selects update {}. Held-out test labels do not select parameters or ridge regularization. The linear baseline uses all 13 standardized features.",run.selected_iteration),
            format!("State and circuit views show source row {} (zero-based) throughout training. Basis probabilities describe this quantum encoding, not the three class probabilities. The final timeline frame is the last iterate, not necessarily the selected checkpoint.",run.probe_id),
            "Exact simulator expectations and autodiff, Adam, 64 updates; no finite-shot noise. Training cost grows exponentially with qubits and linearly with dataset rows. The classical baseline is an essential comparison; this is no claim of quantum advantage.".into(),
        ],
        classification:Some(ClassificationSummary {classes:vec!["Cultivar 1".into(),"Cultivar 2".into(),"Cultivar 3".into()],confusion:run.test.confusion.iter().map(|r|r.to_vec()).collect(),split_sizes:[run.split.train.len(),run.split.validation.len(),run.split.test.len()],selected_iteration:run.selected_iteration,validation_loss}),
        sat:None,photonic:None,
    };
    let document = VisualizationDocument {
        schema_version: 1,
        title: kind.title().into(),
        description: kind.description().into(),
        circuit: run.circuit,
        parameters,
        inputs: run.probe_inputs,
        operation_labels: run.labels,
        trace,
        fabric: None,
        optimization,
        protocol: None,
        study: Some(study),
    };
    document.validate_with(limits)?;
    Ok(document)
}
