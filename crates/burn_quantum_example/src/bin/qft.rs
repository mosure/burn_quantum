//! Positive-sign QFT of |5 mod 2^n>; optional CLI qubit count 2..=12.
use burn_quantum::algorithms::qft;
use burn_quantum_example::*;
fn run(n: usize) -> ExampleResult<()> {
    let input = 5 % (1 << n);
    let state = algorithm_state(&qft(n, input, false)?)?;
    let re = values(state.amplitudes().real())?;
    let im = values(state.amplitudes().imag())?;
    let size = 1usize << n;
    let error = (0..size)
        .map(|y| {
            let phase = std::f64::consts::TAU * (input * y) as f64 / size as f64;
            (re[y] - phase.cos() / (size as f64).sqrt())
                .abs()
                .max((im[y] - phase.sin() / (size as f64).sqrt()).abs())
        })
        .fold(0., f64::max);
    ensure(error < 5e-5, "DFT oracle mismatch")?;
    Report::new(
        "qft",
        0,
        0,
        &[("qubits", n as f64), ("maximum_dft_error", error)],
    )?
    .print()
}

fn main() -> ExampleResult<()> {
    run(algorithm_size(5, 2, 12)?)
}
#[cfg(test)]
mod tests {
    #[test]
    fn canonical_numerical_acceptance() {
        super::run(5).unwrap();
    }
}
