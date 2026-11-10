//! Actual projective teleportation, checking every branch; optional transfers 1..=2.
use burn::tensor::DType;
use burn_quantum::{
    DenseSimulator, StateVector, algorithms::teleportation, protocol::BranchExecution,
};
use burn_quantum_example::*;
fn run(transfers: usize) -> ExampleResult<()> {
    pollster::block_on(async {
        let a = teleportation(transfers, 1.1, 0.7)?;
        let device = execution_device()?;
        let engine = DenseSimulator::default();
        let mut maximum: f64 = 0.;
        for branch in 0..1usize << (2 * transfers) {
            let initial = StateVector::zero(
                a.circuit.register().clone(),
                1,
                &device,
                DType::F32,
                engine.limits,
            )?;
            let outcomes = (0..2 * transfers)
                .map(|b| branch >> (2 * transfers - 1 - b) & 1)
                .collect();
            let mut run = BranchExecution::new(
                &a.circuit,
                a.protocol.as_ref().unwrap(),
                &engine,
                initial,
                outcomes,
                None,
                None,
            )?;
            while run.advance().await? {}
            ensure(
                (run.probability() - 1. / (1usize << (2 * transfers)) as f64).abs() < 5e-5,
                "branch probability mismatch",
            )?;
            for j in 0..transfers {
                let rho = run.state().reduced_density(&[3 * j + 2], engine.limits)?;
                let re = values(rho.entries().real())?;
                let im = values(rho.entries().imag())?;
                for (x, y) in [2. * re[1], -2. * im[1], re[0] - re[3]].into_iter().zip([
                    1.1f64.sin() * 0.7f64.cos(),
                    1.1f64.sin() * 0.7f64.sin(),
                    1.1f64.cos(),
                ]) {
                    maximum = maximum.max((x - y).abs());
                }
            }
        }
        ensure(maximum < 5e-5, "teleportation failed")?;
        Report::new(
            "teleportation",
            0,
            0,
            &[
                ("transfers", transfers as f64),
                ("maximum_bloch_error", maximum),
            ],
        )?
        .print()
    })
}

fn main() -> ExampleResult<()> {
    run(algorithm_size(1, 1, 2)?)
}
#[cfg(test)]
mod tests {
    #[test]
    fn canonical_numerical_acceptance() {
        super::run(2).unwrap();
    }
}

#[cfg(test)]
#[test]
fn fixed_branch_normalization_preserves_autodiff() {
    use burn::tensor::Tensor;
    use burn_quantum::{
        Angle, Circuit, Register, Rotation,
        gate::Axis,
        protocol::{Instruction, Protocol},
    };
    pollster::block_on(async {
        let device = execution_device().unwrap().autodiff();
        let engine = DenseSimulator::default();
        let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
        circuit
            .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
            .unwrap();
        let protocol = Protocol {
            instructions: vec![
                Instruction::Gate {
                    operation: 0,
                    conditions: vec![],
                },
                Instruction::Measure { wire: 0, bit: 0 },
            ],
        };
        for outcome in [0, 1] {
            let theta = Tensor::<1>::from_floats([0.73], &device).require_grad();
            let initial = StateVector::zero(
                circuit.register().clone(),
                1,
                &device,
                DType::F32,
                engine.limits,
            )
            .unwrap();
            let mut branch = BranchExecution::new(
                &circuit,
                &protocol,
                &engine,
                initial,
                vec![outcome],
                Some(theta.clone()),
                None,
            )
            .unwrap();
            while branch.advance().await.unwrap() {}
            let gradient = branch.state().norm_squared().sum().backward();
            let derivative = values(theta.grad(&gradient).unwrap()).unwrap()[0];
            assert!(
                derivative.abs() < 5e-5,
                "normalization derivative must vanish, got {derivative}"
            );
        }
    });
}
