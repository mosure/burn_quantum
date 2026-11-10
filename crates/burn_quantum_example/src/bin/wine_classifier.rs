//! Real UCI Wine classification: thirteen chemical measurements, three cultivars.
//!
//! `wine_classifier [qubits=4] [layers=2] [updates=64] [seed=731]`.
//! Training, preprocessing, exact simulator gradients, validation selection and
//! held-out evaluation are the same reusable core pipeline used by the explorer.
//! This is a small public benchmark, not evidence of quantum advantage.
use burn_quantum::qml::benchmarks::wine::{self, WineConfig, WineRun};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device};

fn validate(result: &WineRun) -> ExampleResult<()> {
    ensure(
        result.predictions.len() == result.split.test.len()
            && result.test.confusion.iter().flatten().sum::<usize>() == result.split.test.len(),
        "held-out accounting is incomplete",
    )?;
    ensure(
        result.predictions.iter().all(|p| {
            p.probabilities
                .iter()
                .all(|x| x.is_finite() && *x >= 0. && *x <= 1.)
                && (p.probabilities.iter().sum::<f64>() - 1.).abs() < 1e-6
        }),
        "invalid class probabilities",
    )?;
    ensure(
        result.history.len() == result.config.steps + 1
            && result.history[result.selected_iteration].validation_loss
                <= result.history[0].validation_loss,
        "checkpoint must be chosen using validation loss",
    )?;
    Ok(())
}

fn main() -> ExampleResult<()> {
    let mut config = WineConfig::default();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure(
        args.len() <= 4,
        "usage: wine_classifier [qubits] [layers] [updates] [seed]",
    )?;
    if let Some(value) = args.first() {
        config.qubits = value.parse()?;
    }
    if let Some(value) = args.get(1) {
        config.depth = value.parse()?;
    }
    if let Some(value) = args.get(2) {
        config.steps = value.parse()?;
    }
    if let Some(value) = args.get(3) {
        config.seed = value.parse()?;
    }
    config.validate()?;
    let start = std::time::Instant::now();
    let result = pollster::block_on(wine::run(&config, &execution_device()?))?;
    validate(&result)?;
    let selected = &result.history[result.selected_iteration];
    let report = Report::new(
        "wine_classifier",
        config.seed,
        config.steps,
        &[
            ("qubits", config.qubits as f64),
            ("layers", config.depth as f64),
            ("initial_train_cross_entropy", result.history[0].train_loss),
            ("selected_train_cross_entropy", selected.train_loss),
            ("validation_accuracy", selected.validation_accuracy),
            ("test_accuracy", result.test.accuracy),
            ("ridge_test_accuracy", result.linear_baseline.accuracy),
            ("majority_test_accuracy", result.majority_accuracy),
            ("elapsed_seconds", start.elapsed().as_secs_f64()),
        ],
    )?;
    let mut output = serde_json::to_value(report)?;
    output.as_object_mut().unwrap().extend(
        serde_json::json!({
            "dataset":wine::DATASET_CITATION,
            "dataset_sha256":wine::DATASET_SHA256,
            "config":config,
            "model":result.model(),
            "split":result.split,
            "selected_iteration":result.selected_iteration,
            "held_out":result.test,
            "linear_ridge_baseline":result.linear_baseline,
            "ridge_regularization":result.linear_regularization,
            "learned_parameters":result.learned_parameters,
            "quantum_forward_passes":result.quantum_forward_passes,
            "history":result.history,
            "predictions":result.predictions,
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_real_held_out_cultivars_with_a_fair_classical_baseline() {
        let result = pollster::block_on(wine::run(
            &WineConfig::default(),
            &execution_device().unwrap(),
        ))
        .unwrap();
        validate(&result).unwrap();
        let selected = &result.history[result.selected_iteration];
        assert!(selected.train_loss < result.history[0].train_loss * 0.75);
        assert!(
            result.test.accuracy >= 0.70,
            "held-out accuracy {}",
            result.test.accuracy
        );
        assert!(result.test.accuracy > result.majority_accuracy + 0.15);
        assert!(result.linear_baseline.accuracy >= 0.85);
        // Persist the entire selected model, then evaluate raw held-out features
        // through the same public API available to new measurement consumers.
        let serialized = serde_json::to_string(&result.model()).unwrap();
        let restored: wine::WineModel = serde_json::from_str(&serialized).unwrap();
        let data = wine::dataset().unwrap();
        let rows = result
            .split
            .test
            .iter()
            .map(|&id| data[id].features.clone())
            .collect::<Vec<_>>();
        let probabilities =
            pollster::block_on(restored.predict(&rows, &execution_device().unwrap())).unwrap();
        for (actual, expected) in probabilities.iter().zip(&result.predictions) {
            for (a, b) in actual.iter().zip(expected.probabilities) {
                assert!((a - b).abs() < 2e-5);
            }
        }
        assert_eq!(
            result.model().parameters,
            result.history[result.selected_iteration].parameters
        );
        // JSON's parser may differ by one f64 ULP; the model's actual tensor
        // precision is F32, which must survive persistence exactly.
        for (actual, expected) in restored.parameters.iter().zip(&result.parameters) {
            assert_eq!(*actual as f32, *expected as f32);
        }
        for invalid_rows in [
            vec![],
            vec![vec![0.; 12]],
            vec![vec![f64::NAN; 13]],
            vec![vec![0.; 13]; 4097],
        ] {
            assert!(
                pollster::block_on(restored.predict(&invalid_rows, &execution_device().unwrap()))
                    .is_err()
            );
        }
        let mut invalid = restored.clone();
        invalid.head_weights.clear();
        assert!(invalid.validate().is_err());
        invalid = restored.clone();
        invalid.parameters[0] = f64::INFINITY;
        assert!(invalid.validate().is_err());
        invalid = restored.clone();
        invalid.parameters[0] = f64::MAX;
        assert!(invalid.validate().is_err());
        let mut invalid_json = serde_json::to_value(&restored).unwrap();
        invalid_json["scaler"]["scale"][0] = serde_json::json!(0.);
        let invalid: wine::WineModel = serde_json::from_value(invalid_json).unwrap();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn training_is_reproducible_and_updates_quantum_angles() {
        let config = WineConfig {
            steps: 2,
            ..Default::default()
        };
        let device = execution_device().unwrap();
        let a = pollster::block_on(wine::run(&config, &device)).unwrap();
        let mut updates = Vec::new();
        let b = pollster::block_on(wine::run_with_progress(&config, &device, |iteration| {
            updates.push(iteration);
            std::future::ready(())
        }))
        .unwrap();
        assert_eq!(updates, vec![0, 1, 2]);
        for (a, b) in a.history.iter().zip(&b.history) {
            assert!((a.train_loss - b.train_loss).abs() < 1e-5);
            for (a, b) in a.parameters.iter().zip(&b.parameters) {
                assert!((a - b).abs() < 1e-5);
            }
        }
        assert!(
            a.history[0]
                .parameters
                .iter()
                .zip(&a.history[2].parameters)
                .any(|(a, b)| (a - b).abs() > 1e-4)
        );
    }
}
