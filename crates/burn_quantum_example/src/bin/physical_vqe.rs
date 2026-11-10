//! VQE through routed Choi resources and destructive fusions, with explicit abort cost.
//! This is an ideal-resource photonic protocol, not a fault-tolerant hardware claim.
use burn::optim::{GradientsParams, SgdConfig};
use burn::tensor::{DType, Tensor};
use burn_quantum::{
    gate::Axis,
    photonic::{compiler::*, *},
    *,
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
const STEPS: usize = 24;
fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let engine = DenseSimulator::default();
    let eta = RailTransmission::uniform(0.98)?;
    let ideal = RailTransmission::IDEAL;
    let fabric = Fabric::new(
        vec![
            FabricNode {
                kind: NodeKind::Source,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Source,
                storage: ideal,
            },
            FabricNode {
                kind: NodeKind::Fusion,
                storage: ideal,
            },
        ],
        vec![
            FabricLink {
                from: NodeId(0),
                to: NodeId(2),
                ticks: 1,
                transmission: eta,
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: eta,
            },
        ],
    )?;
    let mut circuit = Circuit::new(Register::qubits(2)?, 1, 0);
    circuit.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))?;
    circuit.unitary(vec![0, 1], Unitary::cnot())?;
    let physical = PhysicalCircuit::compile(
        &circuit,
        &fabric,
        PhysicalPlacement {
            inputs: vec![NodeId(0); 2],
            resource: NodeId(1),
            analyzer: NodeId(2),
            feedback_ticks: 1,
        },
        CompileOptions::default(),
    )?;
    let mut model = QuantumLayerConfig::new(circuit, vec![], vec![0.2]).init(&device)?;
    let initial = StateVector::basis(
        Register::qubits(2)?,
        &[0, 1],
        1,
        &device,
        DType::F32,
        engine.limits,
    )?;
    let mut entries = vec![0.; 81];
    for (index, value) in [(0, 1.), (1, -1.), (3, -1.), (4, 1.)] {
        entries[index * 9 + index] = value;
    }
    entries[12] = 2.;
    entries[28] = 2.;
    let h = Observable::new(vec![0, 1], Matrix::real(9, entries)?)?;
    let evaluate =
        |parameters: Tensor<1>| physical.execute(&engine, initial.clone(), Some(parameters), None);
    let conditional_energy = |run: &PhysicalRun| -> burn_quantum::Result<Tensor<1>> {
        Ok(run.state().expectation(&engine, &h)? / run.herald_probability())
    };
    let initial_energy = values(conditional_energy(&evaluate(model.parameters.val())?)?)?[0];
    let mut optimizer = SgdConfig::new().init();
    for _ in 0..STEPS {
        let result = evaluate(model.parameters.val())?;
        let energy = conditional_energy(&result)?.mean();
        let grads = GradientsParams::from_grads(energy.backward(), &model);
        model = optimizer.step(0.2, model, grads);
    }
    let result = evaluate(model.parameters.val())?;
    let energy = values(conditional_energy(&result)?)?[0];
    let herald = values(result.herald_probability())?[0];
    let expected_herald = (0.5 * 0.98f64.powi(2)).powi(3);
    let photons = values(result.expected_resource_photons())?[0];
    ensure(
        initial_energy > -1.0 && (-3.00005..-2.995).contains(&energy),
        "physical VQE missed singlet ground energy",
    )?;
    ensure(
        (herald - expected_herald).abs() < 2e-5,
        "physical herald probability omitted loss/fusions",
    )?;
    ensure(
        (photons - (2. + 4. * 0.5 * 0.98f64.powi(2))).abs() < 3e-5,
        "resource accounting ignored early termination",
    )?;
    Report::new(
        "physical_vqe",
        0,
        STEPS,
        &[
            ("initial_energy", initial_energy),
            ("conditional_energy", energy),
            ("exact_ground_energy", -3.),
            ("herald_probability", herald),
            ("analytic_herald_probability", expected_herald),
            ("abort_probability", values(result.abort_probability())?[0]),
            (
                "logical_survival_probability",
                values(result.logical_survival_probability()?)?[0],
            ),
            ("expected_resource_photons", photons),
            ("expected_fusions", values(result.expected_fusions())?[0]),
            ("finish_tick", physical.finish_tick() as f64),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn routed_physical_vqe_reaches_singlet_and_accounts_for_aborts() {
        super::run().unwrap();
    }
}
