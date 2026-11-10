//! Depth-one QAOA for MaxCut on a triangle, checked against exact enumeration.
//! C = sum_edges (I-ZuZv)/2; U_C(gamma) = exp(-i gamma C),
//! U_B(beta) = exp(-i beta sum X). Maximize the expected cut by minimizing its negative.
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Tensor, TensorData};
use burn_quantum::{QuantumLayerConfig, Samples, algorithms::qaoa_maxcut};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

const SEED: u64 = 11;
const STEPS: usize = 140;
const EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (0, 2)];

fn cut(index: usize) -> f64 {
    EDGES
        .iter()
        .filter(|&&(a, b)| ((index >> (2 - a)) & 1) != ((index >> (2 - b)) & 1))
        .count() as f64
}

fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let circuit = qaoa_maxcut(3, &EDGES, &[(0.4, 0.2)])?.circuit;
    let mut model = QuantumLayerConfig::new(circuit, vec![], vec![0.4, 0.2]).init(&device)?;
    let costs: Vec<f64> = (0..8).map(cut).collect();
    let optimum = costs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let cost = Tensor::<2>::from_data(TensorData::new(costs, [1, 8]), &device);
    let initial = (model.state(None, 1)?.probabilities() * cost.clone())
        .sum()
        .into_scalar::<f64>();
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..STEPS {
        let loss = -(model.state(None, 1)?.probabilities() * cost.clone()).sum();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.04, model, gradients);
    }
    let state = model.state(None, 1)?;
    state.validate(1e-5)?;
    let expected = (state.probabilities() * cost).sum().into_scalar::<f64>();
    let optimal_mass: f64 = values(state.probabilities())?
        .iter()
        .enumerate()
        .filter(|(i, _)| cut(*i) == optimum)
        .map(|(_, p)| p)
        .sum();
    let samples = Samples::from_state(&state, 512, SEED, 512)?;
    let best = samples.outcomes[0]
        .iter()
        .map(|&i| cut(i))
        .fold(0., f64::max);
    ensure(
        (optimum - 2.).abs() < 1e-12 && expected > 1.995 && expected <= optimum + 1e-5,
        "QAOA failed the enumerated triangle objective",
    )?;
    ensure(
        best == optimum && optimal_mass > 0.995,
        "QAOA failed to sample optimal cuts",
    )?;
    Report::new(
        "qaoa_maxcut",
        SEED,
        STEPS,
        &[
            ("initial_expected_cut", initial),
            ("expected_cut", expected),
            ("exact_optimal_cut", optimum),
            ("optimal_probability", optimal_mass),
            ("best_sampled_cut", best),
        ],
    )
}

fn main() -> ExampleResult<()> {
    run()?.print()
}

#[cfg(test)]
mod tests {
    #[test]
    fn recovers_enumerated_maxcut() {
        super::run().unwrap();
    }
}
