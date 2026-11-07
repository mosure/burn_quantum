//! Variational quantum eigensolver for H = XX + YY + ZZ on two qubits.
//! The singlet is an independent analytic reference with ground energy -3.
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::Tensor;
use burn_quantum::{
    Circuit, ComplexTensor, Matrix, Observable, QuantumLayerConfig, Register, ResourceLimits,
    StateVector, qml::ry_cnot_layer,
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, initial_angles};

const SEED: u64 = 7;
const STEPS: usize = 160;

fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let mut circuit = Circuit::new(Register::qubits(2)?, 4, 0);
    ry_cnot_layer(&mut circuit, &[0, 1], 0)?;
    ry_cnot_layer(&mut circuit, &[0, 1], 2)?;
    // Explicit two-site Hamiltonian in |00>, |01>, |10>, |11> order.
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
        QuantumLayerConfig::new(circuit, vec![h], initial_angles(4, SEED)).init(&device)?;
    let initial_energy = model.forward(None, 1)?.into_scalar::<f64>();
    let mut optimizer = AdamConfig::new().init();
    for _ in 0..STEPS {
        let energy = model.forward(None, 1)?.mean();
        let gradients = GradientsParams::from_grads(energy.backward(), &model);
        model = optimizer.step(0.07, model, gradients);
    }
    let final_energy = model.forward(None, 1)?.into_scalar::<f64>();
    let a = std::f32::consts::FRAC_1_SQRT_2;
    let target = StateVector::from_amplitudes(
        Register::qubits(2)?,
        ComplexTensor::new(
            Tensor::from_floats([[0., a, -a, 0.]], &device),
            Tensor::zeros([1, 4], &device),
        )?,
        ResourceLimits::default(),
    )?;
    let state = model.state(None, 1)?;
    state.validate(1e-5)?;
    let fidelity = state.fidelity(&target)?.into_scalar::<f64>();
    ensure(
        (-3.00001..-2.995).contains(&final_energy),
        "VQE missed the analytic ground energy",
    )?;
    ensure(fidelity > 0.998, "VQE missed the singlet state")?;
    Report::new(
        "vqe",
        SEED,
        STEPS,
        &[
            ("initial_energy", initial_energy),
            ("energy", final_energy),
            ("exact_ground_energy", -3.),
            ("ground_state_fidelity", fidelity),
        ],
    )
}

fn main() -> ExampleResult<()> {
    run()?.print()
}

#[cfg(test)]
mod tests {
    #[test]
    fn reaches_heisenberg_ground_state() {
        super::run().unwrap();
    }
}
