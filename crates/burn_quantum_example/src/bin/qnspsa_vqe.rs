//! Quantum natural SPSA for the two-qubit Heisenberg singlet, without backpropagation.
//! The correlated RY/RZ ansatz has a state-dependent Fubini–Study metric.
use burn::tensor::Tensor;
use burn_quantum::{
    Angle, Circuit, DenseSimulator, Matrix, Observable, QuantumLayerConfig, Register, Rotation,
    Unitary, gate::Axis, optim::QnSpsaConfig,
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

const SEED: u64 = 17;
const STEPS: usize = 80;
const RESAMPLINGS: usize = 2;

fn run() -> ExampleResult<Report> {
    // This optimizer asks for fresh evaluations; an autodiff context is unnecessary.
    let device = execution_device()?;
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(Register::qubits(2)?, 2, 0);
    circuit.unitary(vec![1], Unitary::x())?;
    circuit.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))?;
    circuit.unitary(vec![0, 1], Unitary::cnot())?;
    circuit.rotate(0, Rotation::qubit(Axis::Z, Angle::parameter(1)))?;
    let h = Observable::new(
        vec![0, 1],
        Matrix::real(
            4,
            vec![
                1., 0., 0., 0., 0., -1., 2., 0., 0., 2., -1., 0., 0., 0., 0., 1.,
            ],
        )?,
    )?;
    let mut model =
        QuantumLayerConfig::new(circuit.clone(), vec![h.clone()], vec![1.0, 2.0]).init(&device)?;
    let state = |p: Tensor<1>| circuit.run(&engine, Some(p), None, 1, &device);
    let mut loss = |p: Tensor<1>| -> burn_quantum::Result<f64> {
        Ok(h.expectation(&engine, &state(p)?)?.into_scalar::<f64>())
    };
    let mut fidelity = |a: Tensor<1>, b: Tensor<1>| -> burn_quantum::Result<f64> {
        Ok(state(a)?.fidelity(&state(b)?)?.into_scalar::<f64>())
    };
    let initial = loss(model.parameters.val())?;
    let mut optimizer = QnSpsaConfig::new()
        .with_seed(SEED)
        .with_learning_rate(0.08)
        .with_learning_rate_decay(0.2)
        .with_perturbation(0.12)
        .with_regularization(0.02)
        .with_resamplings(RESAMPLINGS)
        .with_blocking_tolerance(Some(0.0))
        .init(2)?;
    let mut loss_calls = 0;
    let mut fidelity_calls = 0;
    for _ in 0..STEPS {
        let update = optimizer.step(model.parameters.val(), &mut loss, &mut fidelity)?;
        loss_calls += update.report.loss_evaluations;
        fidelity_calls += update.report.fidelity_evaluations;
        // Param::map retains Burn's parameter identity and model-record layout.
        model.parameters = model.parameters.map(|_| update.parameters);
    }
    let final_state = model.state(None, 1)?;
    final_state.validate(1e-5)?;
    let energy = loss(model.parameters.val())?;
    let re = values(final_state.amplitudes().real())?;
    let im = values(final_state.amplitudes().imag())?;
    let singlet_fidelity = ((re[1] - re[2]).powi(2) + (im[1] - im[2]).powi(2)) / 2.;
    ensure(
        initial > -2.0 && (-3.00001..-2.99).contains(&energy),
        "QN-SPSA missed the analytic Heisenberg ground energy",
    )?;
    ensure(singlet_fidelity > 0.997, "QN-SPSA missed the singlet state")?;
    ensure(
        loss_calls == STEPS * (2 * RESAMPLINGS + 2) && fidelity_calls == STEPS * 4 * RESAMPLINGS,
        "QN-SPSA evaluator-call budget changed",
    )?;
    Report::new(
        "qnspsa_vqe",
        SEED,
        STEPS,
        &[
            ("initial_energy", initial),
            ("energy", energy),
            ("analytic_ground_energy", -3.),
            ("singlet_fidelity", singlet_fidelity),
            ("loss_evaluations", loss_calls as f64),
            ("fidelity_evaluations", fidelity_calls as f64),
        ],
    )
}

fn main() -> ExampleResult<()> {
    run()?.print()
}

#[cfg(test)]
mod tests {
    #[test]
    fn natural_spsa_reaches_the_singlet_without_backpropagation() {
        super::run().unwrap();
    }
}
