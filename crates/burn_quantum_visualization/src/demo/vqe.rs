//! Bounded quantum many-body optimization with independently computed references.
use super::*;
use burn_quantum::qml::benchmarks::ising::{self, IsingConfig};
fn configuration(config: DemoConfig) -> IsingConfig {
    IsingConfig {
        qubits: config.size,
        depth: config.depth,
        ..Default::default()
    }
}
pub(super) fn preview(config: DemoConfig, limits: TraceLimits) -> Result<Preview> {
    let config = configuration(config);
    training::preview(&ising::ansatz(&config)?.0, config.steps + 1, limits)
}
pub(super) async fn build(
    config: DemoConfig,
    device: &Device,
    limits: TraceLimits,
) -> Result<VisualizationDocument> {
    preview(config, limits)?;
    let run =
        ising::run_with_progress(&configuration(config), device, training::yield_to_ui).await?;
    let last = run.history.last().unwrap();
    let parameters = last.parameters.clone();
    let energy = last.energy;
    let initial_energy = run.history[0].energy;
    let (trace, optimization) = training::trace(
        &run.circuit,
        &[],
        run.history
            .into_iter()
            .enumerate()
            .map(|(i, p)| training::Iterate {
                parameters: p.parameters,
                loss: p.energy,
                evaluations: i + 1,
            })
            .collect(),
        device,
        limits,
    )
    .await?;
    let kind = DemoKind::Vqe;
    let study=StudyReport {title:"Ground-state preparation".into(),summary:"Learn an entangled approximation to the ground state of an open transverse-field Ising chain: H = -Σ ZZ - Σ X, with J = h = 1.".into(),
        metrics:vec![StudyMetric::new("Variational energy",energy,"J"),StudyMetric::new("Exact ground energy",run.exact_energy,"J"),StudyMetric::new("Energy error / spin",(energy-run.exact_energy)/config.size as f64,"J"),StudyMetric::new("Uniform product reference",run.product_energy,"J"),StudyMetric::new("Initial energy",initial_energy,"J"),StudyMetric::new("Learned parameters",parameters.len() as f64,"angles"),StudyMetric::new("Local Hamiltonian terms",(2*config.size-1) as f64,"terms"),StudyMetric::new("State dimension",run.circuit.register().size() as f64,"amplitudes")],
        notes:vec!["80 Adam updates with exact Burn autodiff and local X / ZZ measurements. The training objective never constructs a global Hamiltonian.".into(),"Independent CPU diagonalization supplies the exact reference only for this bounded 4–8-spin benchmark. The uniform product-state reference minimizes -(n-1) cos²θ - n sinθ; it is not the best unrestricted classical algorithm.".into(),"Depth limits expressibility; larger circuits need not achieve lower error after the same training budget. Dense state simulation and reference diagonalization are exponential; no computational advantage is claimed.".into()],classification:None,sat:None,photonic:None};
    let document = VisualizationDocument {
        schema_version: 1,
        title: kind.title().into(),
        description: kind.description().into(),
        circuit: run.circuit,
        parameters,
        inputs: vec![],
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
