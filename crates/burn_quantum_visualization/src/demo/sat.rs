//! Exact three-literal Boolean constraints, a reversible oracle and checked readout.
use super::*;
use burn_quantum::sat::benchmarks::ThreeSatBenchmark;

fn benchmark(config: DemoConfig, limits: TraceLimits) -> Result<ThreeSatBenchmark> {
    // Keep every selectable case within the browser's retained trajectory and
    // export budgets. The core/CLI also support six variables without a timeline.
    if !(3..=5).contains(&config.size) || config.variant > 1 {
        return Err(burn_quantum::Error::InvalidValue(
            "3SAT gallery needs 3–5 variables and SAT/UNSAT variant 0/1".into(),
        ));
    }
    ThreeSatBenchmark::new(config.size, config.variant == 1, limits.resources())
}

fn preview_benchmark(benchmark: &ThreeSatBenchmark, limits: TraceLimits) -> Result<Preview> {
    let circuit = benchmark.circuit();
    let steps = circuit.operations().len();
    limits.check(circuit.register(), steps + 1)?;
    let dimension = circuit.register().size();
    let local: usize = circuit
        .register()
        .dimensions()
        .iter()
        .map(|d| 16 * d * d + 8 * d + 32)
        .sum();
    Ok(Preview {
        wires: circuit.register().wires(),
        dimension,
        steps,
        snapshot_bytes: (24 * dimension + local) * (steps + 1),
    })
}

pub(super) fn preview(config: DemoConfig, limits: TraceLimits) -> Result<Preview> {
    preview_benchmark(&benchmark(config, limits)?, limits)
}

pub(super) async fn build(
    config: DemoConfig,
    device: &Device,
    limits: TraceLimits,
) -> Result<VisualizationDocument> {
    let benchmark = benchmark(config, limits)?;
    let info = preview_benchmark(&benchmark, limits)?;
    let trace = simulate(
        benchmark.circuit(),
        &[],
        &[],
        benchmark.labels(),
        device,
        limits,
    )
    .await?;
    let readout = benchmark.readout(&trace.frames.last().unwrap().probabilities)?;
    let verdict = if readout.reference_proves_unsatisfiable {
        "No assignment satisfies all clauses. The exhaustive classical reference proves this tiny fixture UNSAT; quantum search failure alone would be inconclusive."
    } else {
        "The final distribution favors configurations satisfying every clause. The highest-probability assignment is independently checked against the original CNF."
    };
    let study = StudyReport {
        title: "3SAT · Boolean configuration constraints".into(),
        summary: format!(
            "{} variables, {} exact-three-literal clauses. Uniform preparation → compute every clause → mark all-clause success → uncompute → diffuse, once. {verdict}",
            readout.variables, readout.clauses
        ),
        metrics: vec![
            StudyMetric::new("Satisfying probability", 100. * readout.success_probability, "%"),
            StudyMetric::new("Uniform assignment baseline", 100. * readout.uniform_success_probability, "%"),
            StudyMetric::new("Analytic one-round reference", 100. * readout.analytic_success_probability, "%"),
            StudyMetric::new("All work qubits clean", 100. * readout.clean_ancilla_probability, "%"),
            StudyMetric::new("Exact reference solutions", readout.satisfying_assignments.len() as f64, "assignments"),
            StudyMetric::new("Boolean variables", readout.variables as f64, "bits"),
            StudyMetric::new("Clause work qubits", readout.clauses as f64, "qubits"),
            StudyMetric::new("Total simulated qubits", readout.total_qubits as f64, "qubits"),
            StudyMetric::new("Dense state size", readout.basis_states as f64, "amplitudes"),
            StudyMetric::new("Circuit operations", readout.circuit_operations as f64, "operations"),
            StudyMetric::new("Oracle operations", readout.oracle_operations as f64, "operations"),
            StudyMetric::new("Oracle calls", readout.oracle_queries as f64, "calls"),
            StudyMetric::new("Retained numeric snapshots", info.snapshot_bytes as f64 / 1_048_576., "MiB"),
        ],
        notes: vec![
            "A Boolean configuration enables/disables each x_i. Each clause permits at least one of its three signed literals; negative literals mean disabled. Every displayed clause has three distinct variables. These are illustrative constraints, not industrial benchmark instances.".into(),
            format!("q0…q{} encode x1…x{} in big-endian order; q{}…q{} are one clean work qubit per clause. Timeline frames show the actual reversible oracle, including temporary clause values and their uncomputation.", readout.variables - 1, readout.variables, readout.variables, readout.total_qubits - 1),
            "The oracle compiles directly from clauses. Exhaustive enumeration of 2^n assignments is a separately identified classical reference and never constructs the oracle or chooses the fixed one-round budget.".into(),
            "The peak assignment is inspected from the exact simulator distribution, not a sampled measurement. Satisfying probability marginalizes over the work qubits; their cleanup is measured separately. A missing quantum witness never establishes UNSAT.".into(),
            "The one-round reference is sin²(3 asin(sqrt(M/2^n))), with M from the independent classical count. These bounded fixtures illustrate amplification and verification; no quantum speedup or practical SAT scaling is claimed.".into(),
            "Dense simulation stores 2^(variables + clauses) complex amplitudes per frame. Structured controlled gates avoid dense oracle matrices, but do not remove exponential state storage. Gallery: 3–5 variables; CLI: 3–6 variables; custom CNF uses the core GroverSat solver.".into(),
        ],
        classification: None,
        sat: Some(SatSummary {
            variables: readout.variables,
            clauses: benchmark.formula().clauses().to_vec(),
            witness: readout.verified_witness,
            satisfying_assignments: readout.satisfying_assignments.len(),
            success_probability: readout.success_probability,
            uniform_probability: readout.uniform_success_probability,
        }),
        photonic: None,
    };
    let document = VisualizationDocument {
        schema_version: 2,
        title: "3SAT · constraint solving".into(),
        description: format!(
            "A genuine exact-3SAT circuit computes, phase-marks and uncomputes {} Boolean clauses, then performs one Grover diffusion. {} variables + {} clause work qubits = {} simulated qubits. Success {:.2}% versus uniform {:.2}%. {verdict}",
            readout.clauses,
            readout.variables,
            readout.clauses,
            readout.total_qubits,
            100. * readout.success_probability,
            100. * readout.uniform_success_probability
        ),
        circuit: benchmark.circuit().clone(),
        parameters: vec![],
        inputs: vec![],
        operation_labels: benchmark.labels().to_vec(),
        trace,
        fabric: None,
        optimization: vec![],
        protocol: None,
        study: Some(study),
    };
    document.validate_with(limits)?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_sat_and_unsat_traces_are_exportable_and_budget_checked() {
        for variant in [0, 1] {
            let config = DemoConfig {
                size: 5,
                depth: 2,
                variant,
            };
            let document =
                pollster::block_on(build(config, &Device::flex(), TraceLimits::default())).unwrap();
            let study = document.study.as_ref().unwrap();
            let sat = study.sat.as_ref().unwrap();
            assert_eq!(sat.variables, 5);
            assert_eq!(sat.witness.is_none(), variant == 1);
            assert!((sat.success_probability - if variant == 0 { 1. } else { 0. }).abs() < 5e-5);
            assert_eq!(
                document.trace.frames.len(),
                document.circuit.operations().len() + 1
            );
            // Maximum-size SAT and UNSAT trajectories exceed the budget only
            // when indented. Their unchanged compact documents must round-trip.
            assert!(
                serde_json::to_string_pretty(&document).unwrap().len()
                    > crate::io::MAX_DOCUMENT_BYTES
            );
            let json = crate::io::document_json(&document).unwrap();
            assert!(json.len() < crate::io::MAX_DOCUMENT_BYTES);
            assert!(!json.contains('\n'));
            let imported = crate::io::parse_document(&json).unwrap();
            assert_eq!(imported.trace.frames.len(), document.trace.frames.len());
            for (before, after) in document.trace.frames.iter().zip(&imported.trace.frames) {
                assert_eq!(before.step, after.step);
                assert_eq!(before.label, after.label);
                assert_eq!(before.probabilities.len(), after.probabilities.len());
                assert!(
                    before
                        .probabilities
                        .iter()
                        .zip(&after.probabilities)
                        .all(|(a, b)| (a - b).abs() < 1e-14)
                );
                // JSON's decimal parser may differ by an f64 ULP. This bound is
                // many orders below the F32 simulator's numerical tolerance.
                assert!(
                    before
                        .amplitudes
                        .as_ref()
                        .unwrap()
                        .iter()
                        .zip(after.amplitudes.as_ref().unwrap())
                        .all(|(a, b)| (a.re - b.re).abs() < 1e-14 && (a.im - b.im).abs() < 1e-14)
                );
            }
            eprintln!(
                "3SAT maximum-size variant {variant}: {} exported bytes",
                json.len()
            );
            let limits = TraceLimits {
                max_trace_elements: 100,
                ..Default::default()
            };
            assert!(preview(config, limits).is_err());
        }
    }
}
