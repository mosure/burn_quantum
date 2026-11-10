//! Bounded document interchange and atomic import validation.
use burn::tensor::Device;
use burn_quantum_visualization::{
    demo::{self, DemoKind},
    io,
    simulation::TraceLimits,
};

#[test]
fn imported_trace_retains_scientific_state_and_invalid_import_does_not_replace_pending() {
    let document = pollster::block_on(demo::build(
        DemoKind::Bell,
        &Device::flex(),
        TraceLimits::default(),
    ))
    .unwrap();
    let json = io::document_json(&document).unwrap();
    let decoded = io::parse_document(&json).unwrap();
    assert_eq!(decoded.trace, document.trace);
    io::queue_document(decoded).unwrap();
    let mut invalid = document.clone();
    invalid.trace.frames.last_mut().unwrap().probabilities[0] = -0.1;
    assert!(io::queue_document(invalid).is_err());
    assert_eq!(io::take_pending_document().unwrap().trace, document.trace);
    assert!(io::take_pending_document().is_none());
    assert!(io::parse_document(&" ".repeat(io::MAX_DOCUMENT_BYTES + 1)).is_err());
    assert!(io::parse_document("{}").is_err());
}

#[test]
fn dense_gate_and_aggregate_work_budgets_run_before_core_unitary_validation() {
    let document = pollster::block_on(demo::build(
        DemoKind::Bell,
        &Device::flex(),
        TraceLimits::default(),
    ))
    .unwrap();
    let original = serde_json::to_value(document).unwrap();
    let mut value = original.clone();
    // Individually oversized matrix: rejection is the import preflight budget,
    // before either core matrix-shape or cubic unitarity validation can execute.
    value["circuit"]["operations"][0]["Unitary"]["gate"]["dimension"] = serde_json::json!(257);
    let error = io::parse_document(&value.to_string())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("preflight array or dimension exceeds"),
        "{error}"
    );

    // Both dimensions/entry counts fit individually, but their sum of d^3
    // exceeds the work budget. Deliberately nonunitary values also prove that
    // checked Unitary deserialization has not run ahead of the preflight.
    let matrix = serde_json::json!({
        "dimension": 256,
        "entries": vec![[0.0f64, 0.0]; 65_536],
    });
    let gate = serde_json::json!({"Unitary": {"wires": [0], "gate": matrix}});
    let mut value = original.clone();
    value["circuit"]["operations"] = serde_json::json!([gate.clone(), gate]);
    let text = value.to_string();
    assert!(text.len() < io::MAX_DOCUMENT_BYTES);
    let error = io::parse_document(&text).unwrap_err().to_string();
    assert!(
        error.contains("aggregate unitary validation work"),
        "{error}"
    );

    let mut value = original.clone();
    value["circuit"]["operations"][0]["Unitary"]["gate"]["dimension"] = serde_json::json!(u64::MAX);
    let error = io::parse_document(&value.to_string())
        .unwrap_err()
        .to_string();
    assert!(error.contains("preflight"), "{error}");

    // A bounded byte input can still contain an excessive number of operations.
    let mut value = original.clone();
    value["circuit"]["operations"] =
        serde_json::json!(vec![original["circuit"]["operations"][0].clone(); 257]);
    value["operation_labels"] = serde_json::json!(vec!["H"; 257]);
    let error = io::parse_document(&value.to_string())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("preflight array or dimension exceeds"),
        "{error}"
    );
}
