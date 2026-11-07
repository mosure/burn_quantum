//! Quantum circuit Born machine: fit an explicit correlated three-bit distribution.
//! Exact probability cross-entropy trains the circuit; seeded shots evaluate it.
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Tensor, TensorData};
use burn_quantum::{
    Circuit, QuantumLayerConfig, Register, Samples,
    qml::{distribution_cross_entropy, ry_cnot_layer},
};
use burn_quantum_example::{
    ExampleResult, Report, ensure, execution_device, initial_angles, values,
};

const SEED: u64 = 19;
const STEPS: usize = 250;
const TARGET: [f64; 8] = [0.05, 0.05, 0.1, 0.3, 0.3, 0.1, 0.05, 0.05];

fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let mut circuit = Circuit::new(Register::qubits(3)?, 12, 0);
    for layer in 0..4 {
        ry_cnot_layer(&mut circuit, &[0, 1, 2], 3 * layer)?;
    }
    let mut model =
        QuantumLayerConfig::new(circuit, vec![], initial_angles(12, SEED)).init(&device)?;
    let target = Tensor::<2>::from_data(TensorData::new(TARGET.to_vec(), [1, 8]), &device);
    let entropy = -TARGET.iter().map(|p| p * p.ln()).sum::<f64>();
    let initial =
        distribution_cross_entropy(model.state(None, 1)?.probabilities(), target.clone(), 1e-7)?
            .into_scalar::<f64>()
            - entropy;
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..STEPS {
        let loss = distribution_cross_entropy(
            model.state(None, 1)?.probabilities(),
            target.clone(),
            1e-7,
        )?;
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.05, model, gradients);
    }
    let state = model.state(None, 1)?;
    state.validate(1e-5)?;
    let kl = distribution_cross_entropy(state.probabilities(), target, 1e-7)?.into_scalar::<f64>()
        - entropy;
    let learned = values(state.probabilities())?;
    let tv = learned
        .iter()
        .zip(TARGET)
        .map(|(p, q)| (p - q).abs())
        .sum::<f64>()
        / 2.;
    let samples = Samples::from_state(&state, 4096, SEED, 4096)?;
    let mut counts = [0usize; 8];
    for &i in &samples.outcomes[0] {
        counts[i] += 1;
    }
    let shot_tv = counts
        .iter()
        .zip(TARGET)
        .map(|(n, p)| (*n as f64 / 4096. - p).abs())
        .sum::<f64>()
        / 2.;
    ensure(
        (-1e-5..0.005).contains(&kl) && tv < 0.03,
        "Born machine did not fit its target distribution",
    )?;
    ensure(
        shot_tv < 0.06,
        "Born-machine sampled distribution failed its statistical tolerance",
    )?;
    Report::new(
        "born_machine",
        SEED,
        STEPS,
        &[
            ("initial_kl_divergence", initial),
            ("kl_divergence", kl),
            ("total_variation", tv),
            ("sample_total_variation", shot_tv),
            ("shots", 4096.),
        ],
    )
}

fn main() -> ExampleResult<()> {
    run()?.print()
}

#[cfg(test)]
mod tests {
    #[test]
    fn learns_and_samples_target_distribution() {
        super::run().unwrap();
    }
}
