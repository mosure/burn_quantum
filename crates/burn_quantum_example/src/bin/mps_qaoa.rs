//! Non-Clifford QAOA on a configurable 4–256-qubit ring (default 80), checked against its independent p=1 law.
//! MPS contractions use the selected Burn device; complex SVD is explicit host work.
use burn::tensor::DType;
use burn_quantum::{
    Matrix, Unitary,
    tensor_network::{MpsLimits, MpsState},
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device};
use std::f64::consts::PI;
fn trial(n: usize, beta: f64, gamma: f64) -> ExampleResult<(f64, usize, f64)> {
    let device = execution_device()?;
    let limits = MpsLimits {
        max_bond: 16,
        max_truncation_distance: 1e-4,
        ..Default::default()
    };
    let mut state = MpsState::basis(&vec![2; n], &vec![0; n], &device, DType::F32, limits)?;
    for i in 0..n {
        state.apply(&[i], &Unitary::h())?;
    }
    let mut cost = vec![0.0.into(); 16];
    for i in 0..4 {
        let angle = if i == 0 || i == 3 {
            gamma / 2.
        } else {
            -gamma / 2.
        };
        cost[i * 4 + i] = Unitary::phase(angle)?.matrix().entries()[3];
    }
    let cost = Unitary::new(Matrix::new(4, cost)?)?;
    for i in 0..n {
        state.apply(&[i, (i + 1) % n], &cost)?;
    }
    let mut mixer = vec![0.0.into(); 4];
    mixer[0] = beta.cos().into();
    mixer[3] = beta.cos().into();
    mixer[1] = Unitary::s().matrix().entries()[3] * -beta.sin();
    mixer[2] = Unitary::s().matrix().entries()[3] * -beta.sin();
    let mixer = Unitary::new(Matrix::new(2, mixer)?)?;
    for i in 0..n {
        state.apply(&[i], &mixer)?;
    }
    // Translation-invariant ring: check separated interior and wrap-around terms.
    let mut cut = 0.;
    for (a, b) in [(0, 1), (n / 2, n / 2 + 1), (n - 1, 0)] {
        let zz = state
            .expectation_product(&[
                (a, Unitary::z().matrix().clone()),
                (b, Unitary::z().matrix().clone()),
            ])?
            .re;
        cut += (1. - zz) / 6.;
    }
    let analytic = 0.5 + 0.25 * (4. * beta).sin() * (2. * gamma).sin();
    ensure(
        (cut - analytic).abs() < 2e-5,
        "QAOA MPS versus analytic triangle-free ring law",
    )?;
    ensure(
        state.report().trace_distance_bound < 1e-4,
        "QAOA obeyed declared accumulated truncation bound",
    )?;
    Ok((cut, state.storage_elements(), (cut - analytic).abs()))
}
fn run(qubits: usize) -> ExampleResult<Report> {
    ensure(
        (4..=256).contains(&qubits),
        "mps_qaoa supports 4–256 qubits",
    )?;
    // A bounded parameter search is a transparent derivative-free variational loop;
    // no autodiff derivative through the host SVD is implied.
    let mut best = 0.5;
    let mut worst_error = 0.0f64;
    for beta in [PI / 16., PI / 8., 3. * PI / 16.] {
        for gamma in [PI / 8., PI / 4., 3. * PI / 8.] {
            let (cut, _, error) = trial(8, beta, gamma)?;
            best = f64::max(best, cut);
            worst_error = worst_error.max(error);
        }
    }
    let (large, elements, error) = trial(qubits, PI / 8., PI / 4.)?;
    worst_error = worst_error.max(error);
    ensure(
        (best - 0.75).abs() < 2e-5 && (large - 0.75).abs() < 2e-5,
        "QAOA beats random 0.5 cut fraction at both sizes",
    )?;
    ensure(
        elements < qubits * 2 * 16 * 16,
        "MPS stored bounded entanglement instead of an exponential state",
    )?;
    Report::new(
        "mps_qaoa",
        0,
        9,
        &[
            ("qubits", qubits as f64),
            ("dense_amplitude_count_log2", qubits as f64),
            ("cut_fraction", large),
            ("random_cut_fraction", 0.5),
            ("complex_state_elements", elements as f64),
            ("maximum_analytic_error", worst_error),
        ],
    )
}
fn main() -> ExampleResult<()> {
    let mut args = std::env::args().skip(1);
    let qubits = args.next().map(|s| s.parse()).transpose()?.unwrap_or(80);
    ensure(args.next().is_none(), "usage: mps_qaoa [qubits 4..256]")?;
    run(qubits)?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn qaoa_nonclifford_scaling() {
        super::run(80).unwrap();
    }
}
