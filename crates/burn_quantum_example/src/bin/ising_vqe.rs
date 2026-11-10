//! Multi-qubit Ising energy minimization; optional arguments are qubits and depth.
use burn_quantum::qml::benchmarks::ising::{self, IsingConfig};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device};
fn run(config: IsingConfig) -> ExampleResult<Report> {
    let result = pollster::block_on(ising::run(&config, &execution_device()?))?;
    let initial = result.history[0].energy;
    let energy = result.history.last().unwrap().energy;
    ensure(
        energy >= result.exact_energy - 2e-4,
        "energy violates the independent variational bound",
    )?;
    ensure(
        energy < initial - 0.2,
        "VQE failed to improve the initial energy",
    )?;
    Report::new(
        "ising_vqe",
        0,
        config.steps,
        &[
            ("qubits", config.qubits as f64),
            ("depth", config.depth as f64),
            ("initial_energy", initial),
            ("energy", energy),
            ("exact_ground_energy", result.exact_energy),
            (
                "error_per_spin",
                (energy - result.exact_energy) / config.qubits as f64,
            ),
            ("uniform_product_energy", result.product_energy),
        ],
    )
}
fn main() -> ExampleResult<()> {
    let mut config = IsingConfig::default();
    let mut args = std::env::args().skip(1);
    if let Some(n) = args.next() {
        config.qubits = n.parse()?;
    }
    if let Some(n) = args.next() {
        config.depth = n.parse()?;
    }
    if args.next().is_some() {
        return Err("usage: ising_vqe [qubits 4..8] [depth 1..3]".into());
    }
    run(config)?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn optimizes_multiqubit_ising_energy() {
        super::run(Default::default()).unwrap();
    }
}
