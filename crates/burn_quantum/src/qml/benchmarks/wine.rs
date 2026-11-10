//! Real chemical-analysis classification with an entangling, data-reuploading quantum model.
//!
//! UCI Wine: 178 observations, 13 chemical constituents, three cultivars. All
//! preprocessing is fitted on training rows. A fixed validation partition selects
//! checkpoints and ridge regularization; the held-out test set is read only after
//! those choices. The hybrid model trains quantum angles and a classical linear
//! readout together. Exact simulator gradients and expectations are used, not QPU
//! shots. Dense simulation remains exponential in qubits.
use crate::{
    Angle, Circuit, Error, Observable, QuantumLayer, QuantumLayerConfig, Register, Result,
    Rotation, gate::Axis, qml::preprocessing::Standardizer,
};
use burn::{
    module::{Module, Param},
    optim::{AdamConfig, GradientsParams},
    tensor::{DType, Device, Tensor, TensorData, activation::log_softmax},
};
use nalgebra::DMatrix;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Citation and license for the vendored data, separate from the Rust code license.
pub const DATASET_CITATION: &str =
    "Aeberhard & Forina (1992), UCI Wine, doi:10.24432/C5PC7J; CC BY 4.0";
/// SHA-256 of the unmodified UCI `wine.data` fixture.
pub const DATASET_SHA256: &str = "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659";
/// Feature names in the original dataset order.
pub const FEATURE_NAMES: [&str; 13] = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315",
    "Proline",
];

/// Bounded portable training configuration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct WineConfig {
    /// Dense quantum register size, 4–8.
    pub qubits: usize,
    /// Reuploading/entangling layers, 1–3.
    pub depth: usize,
    /// Adam updates, 1–128.
    pub steps: usize,
    /// Local seed for stratified splitting and explicit initialization.
    pub seed: u64,
}
impl Default for WineConfig {
    fn default() -> Self {
        Self {
            qubits: 4,
            depth: 2,
            steps: 64,
            seed: 731,
        }
    }
}
impl WineConfig {
    /// Reject unsupported complexity before allocating any state.
    pub fn validate(&self) -> Result<()> {
        if !(4..=8).contains(&self.qubits)
            || !(1..=3).contains(&self.depth)
            || !(1..=128).contains(&self.steps)
        {
            return Err(Error::InvalidValue(
                "Wine benchmark supports 4–8 qubits, 1–3 layers and 1–128 updates".into(),
            ));
        }
        Ok(())
    }
}

/// One original observation. `id` is the zero-based row in `wine.data`.
#[derive(Clone, Debug, Serialize)]
pub struct Observation {
    /// Original row identity, retained across partitions.
    pub id: usize,
    /// Thirteen measured chemical features.
    pub features: Vec<f64>,
    /// Zero-based cultivar class.
    pub class: usize,
}
/// Load the bundled public dataset; no network or generated labels are involved.
pub fn dataset() -> Result<Vec<Observation>> {
    include_str!("data/wine.data")
        .lines()
        .enumerate()
        .map(|(id, line)| {
            let values = line
                .split(',')
                .map(str::parse::<f64>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| Error::InvalidValue("invalid bundled Wine row".into()))?;
            if values.len() != 14
                || !(1. ..=3.).contains(&values[0])
                || values[0].fract() != 0.
                || values.iter().any(|v| !v.is_finite())
            {
                return Err(Error::InvalidValue(
                    "invalid bundled Wine observation".into(),
                ));
            }
            Ok(Observation {
                id,
                class: values[0] as usize - 1,
                features: values[1..].to_vec(),
            })
        })
        .collect()
}
/// Original row IDs in disjoint, deterministic, class-stratified partitions.
#[derive(Clone, Debug, Serialize)]
pub struct Split {
    /// Approximately sixty percent, used for fitting all model parameters.
    pub train: Vec<usize>,
    /// Approximately twenty percent, used only for checkpoint/hyperparameter selection.
    pub validation: Vec<usize>,
    /// Remaining observations, evaluated once after selection.
    pub test: Vec<usize>,
}
/// Deterministic stratification; shuffling uses a local RNG, never backend-global state.
pub fn split(seed: u64) -> Result<Split> {
    let data = dataset()?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut result = Split {
        train: vec![],
        validation: vec![],
        test: vec![],
    };
    for class in 0..3 {
        let mut ids = data
            .iter()
            .filter(|r| r.class == class)
            .map(|r| r.id)
            .collect::<Vec<_>>();
        ids.shuffle(&mut rng);
        let train = ids.len() * 3 / 5;
        let validation = ids.len() / 5;
        result.train.extend_from_slice(&ids[..train]);
        result
            .validation
            .extend_from_slice(&ids[train..train + validation]);
        result.test.extend_from_slice(&ids[train + validation..]);
    }
    // Interleave classes during batched execution; preserve deterministic row IDs.
    result.train.shuffle(&mut rng);
    result.validation.shuffle(&mut rng);
    result.test.shuffle(&mut rng);
    Ok(result)
}

/// One measured training iterate. Parameters contain the circuit angles only.
#[derive(Clone, Debug, Serialize)]
pub struct WineStep {
    /// Completed Adam updates, starting at zero.
    pub iteration: usize,
    /// Actual train cross entropy at this iterate.
    pub train_loss: f64,
    /// Validation cross entropy, used to choose the final checkpoint.
    pub validation_loss: f64,
    /// Training classification accuracy.
    pub train_accuracy: f64,
    /// Validation classification accuracy.
    pub validation_accuracy: f64,
    /// Circuit angle values; no fabricated fidelity metric is attached to Adam.
    pub parameters: Vec<f64>,
}
/// A single held-out prediction, including its original source row ID.
#[derive(Clone, Debug, Serialize)]
pub struct Prediction {
    /// Dataset source row.
    pub id: usize,
    /// Ground truth cultivar.
    pub expected: usize,
    /// Maximum-probability cultivar.
    pub predicted: usize,
    /// Three normalized class probabilities from the learned classical readout.
    pub probabilities: [f64; 3],
}
/// Independent held-out classification summary; rows=true, columns=predicted.
#[derive(Clone, Debug, Serialize)]
pub struct ClassificationMetrics {
    /// Fraction correctly classified.
    pub accuracy: f64,
    /// Mean categorical cross entropy.
    pub cross_entropy: f64,
    /// Ground-truth rows and prediction columns.
    pub confusion: [[usize; 3]; 3],
}
/// Auditable output of one complete experiment.
#[derive(Clone, Debug)]
pub struct WineRun {
    /// Exact configuration used.
    pub config: WineConfig,
    /// Public circuit structure, reusable with ordinary Burn input/parameter tensors.
    pub circuit: Circuit,
    /// One label per circuit operation.
    pub labels: Vec<String>,
    /// Training/validation/test source row IDs.
    pub split: Split,
    /// Fitted on training observations only.
    pub scaler: Standardizer,
    /// Every actual iterate, including initialization.
    pub history: Vec<WineStep>,
    /// Lowest-validation-loss iterate, chosen without consulting the test set.
    pub selected_iteration: usize,
    /// Selected checkpoint's quantum angles.
    pub parameters: Vec<f64>,
    /// Selected checkpoint's classical head weights, row-major `[qubits, 3]`.
    pub head_weights: Vec<f64>,
    /// Selected checkpoint's three classical biases.
    pub head_bias: Vec<f64>,
    /// One complete test-set evaluation after checkpoint selection.
    pub predictions: Vec<Prediction>,
    /// Held-out QML metrics.
    pub test: ClassificationMetrics,
    /// Held-out linear-ridge metrics using all thirteen standardized inputs.
    pub linear_baseline: ClassificationMetrics,
    /// Ridge regularization chosen only on validation loss.
    pub linear_regularization: f64,
    /// Test accuracy of the most common training class.
    pub majority_accuracy: f64,
    /// Test row used to illustrate quantum states, chosen before model evaluation.
    pub probe_id: usize,
    /// All thirteen bounded angle inputs for that row.
    pub probe_inputs: Vec<f64>,
    /// Actual number of batched quantum forward passes, including final test inference.
    pub quantum_forward_passes: usize,
    /// Total quantum+classical learned scalar parameters.
    pub learned_parameters: usize,
}

/// Portable selected checkpoint for inference on new chemical measurements.
///
/// Restore with serde JSON, then call [`Self::predict`] with raw features in
/// [`FEATURE_NAMES`] order. Shape, scalar precision and resource limits are
/// checked before tensor allocation. The scaler remains fitted on training data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WineModel {
    /// Checkpoint format version; currently one.
    pub schema_version: u32,
    /// Circuit architecture and training provenance.
    pub config: WineConfig,
    /// Fitted training-set statistics.
    pub scaler: Standardizer,
    /// Validation-selected circuit angles.
    pub parameters: Vec<f64>,
    /// Row-major classical linear head of shape `[qubits, 3]`.
    pub head_weights: Vec<f64>,
    /// Three class biases.
    pub head_bias: Vec<f64>,
}
impl WineRun {
    /// Export the validation-selected model, independent of the last iterate.
    pub fn model(&self) -> WineModel {
        WineModel {
            schema_version: 1,
            config: self.config,
            scaler: self.scaler.clone(),
            parameters: self.parameters.clone(),
            head_weights: self.head_weights.clone(),
            head_bias: self.head_bias.clone(),
        }
    }
}
impl WineModel {
    /// Validate checkpoint metadata and finite F32-representable model values.
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.schema_version != 1
            || self.parameters.len() != 2 * self.config.qubits * self.config.depth
            || self.head_weights.len() != 3 * self.config.qubits
            || self.head_bias.len() != 3
            || self.scaler.mean().len() != 13
            || self
                .parameters
                .iter()
                .chain(&self.head_weights)
                .chain(&self.head_bias)
                .any(|x| !x.is_finite() || !(*x as f32).is_finite())
        {
            return Err(Error::InvalidValue("invalid Wine model checkpoint".into()));
        }
        // Standardizer checks its own serialized shape, finiteness and scale.
        self.scaler.transform(&[0.; 13])?;
        Ok(())
    }

    /// Predict three cultivar probabilities for 1–4096 new raw feature rows.
    ///
    /// Exact inference uses at most 1,048,576 complex state elements. These are
    /// preflight limits, not a cap on total allocator or backend temporary memory.
    pub async fn predict(&self, rows: &[Vec<f64>], device: &Device) -> Result<Vec<[f64; 3]>> {
        self.validate()?;
        if rows.is_empty() || rows.len() > 4096 {
            return Err(Error::Shape(
                "Wine inference accepts 1–4096 observations per batch".into(),
            ));
        }
        crate::ResourceLimits {
            max_state_elements: 1 << 20,
            ..Default::default()
        }
        .check_state(rows.len(), 1 << self.config.qubits)?;
        let standardized = rows
            .iter()
            .map(|r| self.scaler.transform(r))
            .collect::<Result<Vec<_>>>()?;
        let device = device.clone().without_autodiff();
        let (circuit, _) = ansatz(&self.config)?;
        let model = Classifier {
            quantum: QuantumLayerConfig::new(
                circuit,
                (0..self.config.qubits).map(Observable::z).collect(),
                self.parameters.clone(),
            )
            .init(&device)?,
            weights: Param::from_data(
                TensorData::new(
                    self.head_weights
                        .iter()
                        .map(|x| *x as f32)
                        .collect::<Vec<_>>(),
                    [self.config.qubits, 3],
                ),
                &device,
            ),
            bias: Param::from_data(
                TensorData::new(
                    self.head_bias.iter().map(|x| *x as f32).collect::<Vec<_>>(),
                    [3],
                ),
                &device,
            ),
        };
        probabilities(&read(model.logits(tensor(&encode(&standardized), &device))?).await?)
    }
}

/// Construct the circuit and labels without allocating simulation state.
/// All thirteen features occur in every layer, including at larger/smaller register sizes.
pub fn ansatz(config: &WineConfig) -> Result<(Circuit, Vec<String>)> {
    config.validate()?;
    let n = config.qubits;
    let mut circuit = Circuit::new(Register::qubits(n)?, 2 * n * config.depth, 13);
    let mut labels = Vec::new();
    for wire in 0..n {
        circuit.unitary(vec![wire], crate::Unitary::h())?;
        labels.push(format!("Prepare |+> q{wire}"));
    }
    for layer in 0..config.depth {
        for (feature, name) in FEATURE_NAMES.iter().enumerate() {
            let wire = feature % n;
            let axis = if (feature / n + layer).is_multiple_of(2) {
                Axis::Y
            } else {
                Axis::Z
            };
            circuit.rotate(wire, Rotation::qubit(axis, Angle::input(feature)))?;
            labels.push(format!("Encode {} · layer {}", name, layer + 1));
            if feature % n == n - 1 {
                circuit.push(crate::Operation::ControlledX {
                    controls: vec![(wire, 1)],
                    target: (wire + 1) % n,
                })?;
                labels.push(format!("Feature mixing q{wire} → q{}", (wire + 1) % n));
            }
        }
        for wire in 0..n {
            // RY follows RZ: a terminal RZ would commute with the Z readout and
            // CNOT network, leaving half the last layer's angles unobservable.
            for (offset, axis) in [(0, Axis::Z), (1, Axis::Y)] {
                let index = 2 * (layer * n + wire) + offset;
                circuit.rotate(wire, Rotation::qubit(axis, Angle::parameter(index)))?;
                labels.push(format!("Train R{axis:?} q{wire} · layer {}", layer + 1));
            }
        }
        for wire in 0..n {
            circuit.push(crate::Operation::ControlledX {
                controls: vec![(wire, 1)],
                target: (wire + 1) % n,
            })?;
            labels.push(format!("Entangle q{wire} → q{}", (wire + 1) % n));
        }
    }
    Ok((circuit, labels))
}

#[derive(Module, Debug)]
struct Classifier {
    quantum: QuantumLayer,
    weights: Param<Tensor<2>>,
    bias: Param<Tensor<1>>,
}
impl Classifier {
    fn logits(&self, input: Tensor<2>) -> Result<Tensor<2>> {
        let batch = input.dims()[0];
        Ok(self
            .quantum
            .forward(Some(input), batch)?
            .matmul(self.weights.val())
            + self.bias.val().reshape([1, 3]))
    }
}
async fn read<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f64>> {
    tensor
        .into_data_async()
        .await
        .map_err(|e| Error::Readback(format!("{e:?}")))?
        .convert::<f64>()
        .try_to_vec::<f64>()
        .map_err(|e| Error::Readback(format!("{e:?}")))
}
fn tensor(rows: &[Vec<f64>], device: &Device) -> Tensor<2> {
    Tensor::from_data(
        TensorData::new(
            rows.iter().flatten().copied().collect::<Vec<_>>(),
            [rows.len(), 13],
        ),
        (device, DType::F32),
    )
}
fn probabilities(logits: &[f64]) -> Result<Vec<[f64; 3]>> {
    if !logits.len().is_multiple_of(3) || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidValue(
            "nonfinite classification logits".into(),
        ));
    }
    Ok(logits
        .chunks_exact(3)
        .map(|x| {
            let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let p = [(x[0] - max).exp(), (x[1] - max).exp(), (x[2] - max).exp()];
            let sum = p.iter().sum::<f64>();
            p.map(|x| x / sum)
        })
        .collect())
}
fn metrics(
    probabilities: &[[f64; 3]],
    rows: &[usize],
    data: &[Observation],
) -> ClassificationMetrics {
    let mut result = ClassificationMetrics {
        accuracy: 0.,
        cross_entropy: 0.,
        confusion: [[0; 3]; 3],
    };
    for (p, &id) in probabilities.iter().zip(rows) {
        let predicted = argmax(p);
        let label = data[id].class;
        result.confusion[label][predicted] += 1;
        result.accuracy += f64::from(predicted == label);
        result.cross_entropy -= p[label].max(1e-12).ln();
    }
    result.accuracy /= rows.len() as f64;
    result.cross_entropy /= rows.len() as f64;
    result
}
fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}
fn standardized(
    rows: &[usize],
    data: &[Observation],
    scaler: &Standardizer,
) -> Result<Vec<Vec<f64>>> {
    rows.iter()
        .map(|&id| scaler.transform(&data[id].features))
        .collect()
}
fn encode(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|r| r.iter().map(|z| 0.8 * z.atan()).collect())
        .collect()
}

/// Train the hybrid classifier and evaluate the selected checkpoint once on held-out data.
/// All tensor reads are async, so the same function is usable in a WebGPU/WASM application.
pub async fn run(config: &WineConfig, device: &Device) -> Result<WineRun> {
    run_with_progress(config, device, |_| std::future::ready(())).await
}

/// Train with an async notification after each recorded iterate's tensor work.
///
/// The callback receives the recorded update index (zero through `steps`) and
/// can yield to an application event loop. It does not receive or mutate model
/// state. Native callers may use [`run`] to avoid any scheduling overhead.
pub async fn run_with_progress<F, Fut>(
    config: &WineConfig,
    device: &Device,
    mut on_update: F,
) -> Result<WineRun>
where
    F: FnMut(usize) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let (circuit, labels) = ansatz(config)?;
    let data = dataset()?;
    let split = split(config.seed)?;
    let scaler = Standardizer::fit(
        &split
            .train
            .iter()
            .map(|&id| data[id].features.clone())
            .collect::<Vec<_>>(),
    )?;
    let train_raw = standardized(&split.train, &data, &scaler)?;
    let val_raw = standardized(&split.validation, &data, &scaler)?;
    let train_rows = encode(&train_raw);
    let val_rows = encode(&val_raw);
    let training_device = device.clone().autodiff();
    let inference_device = device.clone().without_autodiff();
    let train_x = tensor(&train_rows, &training_device);
    let val_x = tensor(&val_rows, &inference_device);
    let target = split
        .train
        .iter()
        .flat_map(|&id| {
            let label = data[id].class;
            (0..3).map(move |class| f32::from(label == class))
        })
        .collect::<Vec<_>>();
    let targets = Tensor::<2>::from_data(
        TensorData::new(target, [split.train.len(), 3]),
        &training_device,
    );
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed ^ 0x57494e45);
    let initial = (0..circuit.num_parameters())
        .map(|_| rng.random_range(-0.3..0.3))
        .collect::<Vec<_>>();
    let mut model = Classifier {
        quantum: QuantumLayerConfig::new(
            circuit.clone(),
            (0..config.qubits).map(Observable::z).collect(),
            initial,
        )
        .init(&training_device)?,
        weights: Param::from_data(
            TensorData::new(
                (0..config.qubits * 3)
                    .map(|_| rng.random_range(-0.15f32..0.15))
                    .collect::<Vec<_>>(),
                [config.qubits, 3],
            ),
            &training_device,
        ),
        bias: Param::from_data(TensorData::new(vec![0f32; 3], [3]), &training_device),
    };
    let mut optimizer = AdamConfig::new().init();
    let mut history = Vec::with_capacity(config.steps + 1);
    let mut selected = model.valid();
    let mut selected_iteration = 0;
    let mut best_loss = f64::INFINITY;
    for iteration in 0..=config.steps {
        let logits = model.logits(train_x.clone())?;
        let loss = -(log_softmax(logits.clone(), 1) * targets.clone())
            .sum_dim(1)
            .mean();
        let train_metrics = metrics(&probabilities(&read(logits).await?)?, &split.train, &data);
        let valid = model.valid();
        let validation_metrics = metrics(
            &probabilities(&read(valid.logits(val_x.clone())?).await?)?,
            &split.validation,
            &data,
        );
        if validation_metrics.cross_entropy < best_loss {
            best_loss = validation_metrics.cross_entropy;
            selected = valid;
            selected_iteration = iteration;
        }
        history.push(WineStep {
            iteration,
            train_loss: train_metrics.cross_entropy,
            validation_loss: validation_metrics.cross_entropy,
            train_accuracy: train_metrics.accuracy,
            validation_accuracy: validation_metrics.accuracy,
            parameters: read(model.quantum.parameters.val()).await?,
        });
        if iteration < config.steps {
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(0.04, model, gradients);
        }
        on_update(iteration).await;
    }
    // Test features/labels enter model evaluation only after all selection decisions.
    let test_raw = standardized(&split.test, &data, &scaler)?;
    let test_rows = encode(&test_raw);
    let predicted =
        probabilities(&read(selected.logits(tensor(&test_rows, &inference_device))?).await?)?;
    let test = metrics(&predicted, &split.test, &data);
    let predictions = predicted
        .into_iter()
        .zip(&split.test)
        .map(|(probabilities, &id)| Prediction {
            id,
            expected: data[id].class,
            predicted: argmax(&probabilities),
            probabilities,
        })
        .collect();
    let (linear_baseline, linear_regularization) =
        ridge_baseline(&train_raw, &val_raw, &test_raw, &split, &data)?;
    let mut counts = [0usize; 3];
    for &id in &split.train {
        counts[data[id].class] += 1;
    }
    let majority = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .unwrap()
        .0;
    let majority_accuracy = split
        .test
        .iter()
        .filter(|&&id| data[id].class == majority)
        .count() as f64
        / split.test.len() as f64;
    let probe_id = split.test[0];
    Ok(WineRun {
        config: *config,
        circuit,
        labels,
        split,
        scaler,
        history,
        selected_iteration,
        parameters: read(selected.quantum.parameters.val()).await?,
        head_weights: read(selected.weights.val()).await?,
        head_bias: read(selected.bias.val()).await?,
        predictions,
        test,
        linear_baseline,
        linear_regularization,
        majority_accuracy,
        probe_id,
        probe_inputs: test_rows[0].clone(),
        quantum_forward_passes: 2 * (config.steps + 1) + 1,
        learned_parameters: 2 * config.qubits * config.depth + 3 * config.qubits + 3,
    })
}

fn ridge_baseline(
    train: &[Vec<f64>],
    validation: &[Vec<f64>],
    test: &[Vec<f64>],
    split: &Split,
    data: &[Observation],
) -> Result<(ClassificationMetrics, f64)> {
    let design = |rows: &[Vec<f64>]| {
        DMatrix::from_fn(rows.len(), 14, |i, j| if j == 13 { 1. } else { rows[i][j] })
    };
    let x = design(train);
    let v = design(validation);
    let t = design(test);
    let y = DMatrix::from_fn(train.len(), 3, |i, j| {
        f64::from(data[split.train[i]].class == j)
    });
    let gram = x.transpose() * &x;
    let rhs = x.transpose() * y;
    let mut best = f64::INFINITY;
    let mut chosen = None;
    for lambda in [0.001, 0.01, 0.1, 1., 10.] {
        let mut regularized = gram.clone();
        for j in 0..13 {
            regularized[(j, j)] += lambda;
        }
        let weights = regularized
            .cholesky()
            .ok_or_else(|| Error::InvalidValue("ridge solve is not positive definite".into()))?
            .solve(&rhs);
        let logits = &v * &weights;
        let values = (0..logits.nrows())
            .flat_map(|i| {
                (0..3).map({
                    let logits = &logits;
                    move |j| 4. * logits[(i, j)]
                })
            })
            .collect::<Vec<_>>();
        let score = metrics(&probabilities(&values)?, &split.validation, data).cross_entropy;
        if score < best {
            best = score;
            chosen = Some((weights, lambda));
        }
    }
    let (weights, lambda) = chosen.unwrap();
    let logits = t * weights;
    let values = (0..logits.nrows())
        .flat_map(|i| {
            (0..3).map({
                let logits = &logits;
                move |j| 4. * logits[(i, j)]
            })
        })
        .collect::<Vec<_>>();
    Ok((metrics(&probabilities(&values)?, &split.test, data), lambda))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_dataset_and_stratified_partitions_are_complete_disjoint_and_reproducible() {
        let data = dataset().unwrap();
        assert_eq!(data.len(), 178);
        assert_eq!(
            (0..3)
                .map(|c| data.iter().filter(|r| r.class == c).count())
                .collect::<Vec<_>>(),
            [59, 71, 48]
        );
        let a = split(731).unwrap();
        let b = split(731).unwrap();
        assert_eq!(a.train, b.train);
        let mut ids = a
            .train
            .iter()
            .chain(&a.validation)
            .chain(&a.test)
            .copied()
            .collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, (0..178).collect::<Vec<_>>());
        for partition in [&a.train, &a.validation, &a.test] {
            for class in 0..3 {
                assert!(partition.iter().any(|&id| data[id].class == class));
            }
        }
    }
    #[test]
    fn every_measured_feature_is_present_at_all_supported_complexities() {
        for qubits in [4, 6, 8] {
            for depth in [1, 2, 3] {
                let (circuit, labels) = ansatz(&WineConfig {
                    qubits,
                    depth,
                    ..Default::default()
                })
                .unwrap();
                assert_eq!(circuit.num_parameters(), 2 * qubits * depth);
                assert_eq!(labels.len(), circuit.operations().len());
                let mut counts = [0; 13];
                for operation in circuit.operations() {
                    if let crate::Operation::Rotate { rotation, .. } = operation
                        && let Angle::Input { index, .. } = rotation.angle
                    {
                        counts[index] += 1;
                    }
                }
                assert_eq!(counts, [depth; 13]);
            }
        }
        assert!(
            ansatz(&WineConfig {
                qubits: 100,
                ..Default::default()
            })
            .is_err()
        );
    }
}
