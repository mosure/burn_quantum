//! MaxCut QAOA through finite buffers, heralded factories and calibrated optical POVMs.
use burn::optim::{GradientsParams, SgdConfig};
use burn::tensor::{DType, Tensor};
use burn_quantum::{
    gate::Axis,
    photonic::{compiler::*, fabric::*, factories::*, optics::*, *},
    *,
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
fn run() -> ExampleResult<Report> {
    let device = execution_device()?.autodiff();
    let engine = DenseSimulator::default();
    let nodes = vec![
        FabricNode {
            kind: NodeKind::Source,
            storage: RailTransmission::IDEAL,
        },
        FabricNode {
            kind: NodeKind::Source,
            storage: RailTransmission::IDEAL,
        },
        FabricNode {
            kind: NodeKind::Fusion,
            storage: RailTransmission::IDEAL,
        },
    ];
    let eta = 0.995;
    let fabric = Fabric::new(
        nodes,
        (0..2)
            .map(|from| FabricLink {
                from: NodeId(from),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::uniform(eta).unwrap(),
            })
            .collect(),
    )?;
    let mut circuit = Circuit::new(Register::qubits(2)?, 2, 0);
    for wire in 0..2 {
        circuit.unitary(vec![wire], Unitary::h())?;
    }
    circuit.unitary(vec![0, 1], Unitary::cnot())?;
    circuit.rotate(
        1,
        Rotation::qubit(
            Axis::Z,
            Angle::Parameter {
                index: 0,
                scale: -1.,
                offset: 0.,
            },
        ),
    )?;
    circuit.unitary(vec![0, 1], Unitary::cnot())?;
    for wire in 0..2 {
        circuit.rotate(
            wire,
            Rotation::qubit(
                Axis::X,
                Angle::Parameter {
                    index: 1,
                    scale: 2.,
                    offset: 0.,
                },
            ),
        )?;
    }
    let source = |photons| {
        ResourceFactoryModel::new(
            "bounded-emitter-p0.9-v1".into(),
            ResourceFactory::new(0.9, photons, 1, 2)?,
            None,
        )
    };
    let calibration = PhysicalCalibration::new(
        source(2)?,
        source(4)?,
        OpticalBellInstrument::standard(
            1.,
            [Detector::ideal(2)?; 4],
            "ideal-pnr-standard-v1".into(),
        )?,
        100_000,
    )?;
    let placement = PhysicalPlacement {
        inputs: vec![NodeId(0); 2],
        resource: NodeId(1),
        analyzer: NodeId(2),
        feedback_ticks: 1,
    };
    let initial_plan = PhysicalCircuit::compile_calibrated(
        &circuit,
        &fabric,
        placement.clone(),
        CompileOptions::default(),
        calibration.clone(),
        None,
    )?;
    let ticks: Vec<_> = initial_plan
        .gates()
        .iter()
        .flat_map(|g| {
            g.routes()
                .chunks_exact(2)
                .map(|pair| pair[0].arrival.max(pair[1].arrival))
        })
        .collect();
    let constraints = FabricConstraints {
        buffers: vec![2, 6, 2],
        configure_ticks: vec![0; 3],
        max_search_states: 10_000,
        max_reservations: 1000,
    };
    let plan = PhysicalCircuit::compile_calibrated(
        &circuit,
        &fabric,
        placement,
        CompileOptions::default(),
        calibration,
        Some((constraints, &ticks)),
    )?;
    let mut cut = vec![0.; 81];
    cut[10] = 1.;
    cut[3 * 9 + 3] = 1.;
    let cut = Observable::new(vec![0, 1], Matrix::real(9, cut)?)?;
    let initial = StateVector::zero(Register::qubits(2)?, 1, &device, DType::F32, engine.limits)?;
    let mut model = QuantumLayerConfig::new(circuit, vec![], vec![0.5, 0.2]).init(&device)?;
    let evaluate =
        |parameters: Tensor<1>| plan.execute(&engine, initial.clone(), Some(parameters), None);
    let objective = |run: &PhysicalRun| -> burn_quantum::Result<Tensor<1>> {
        Ok(run.state().expectation(&engine, &cut)? / run.herald_probability())
    };
    let before = values(objective(&evaluate(model.parameters.val())?)?)?[0];
    let mut optimizer = SgdConfig::new().init();
    for _ in 0..32 {
        let loss = -objective(&evaluate(model.parameters.val())?)?.mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.15, model, gradients);
        // Keep only one optimization step in flight on asynchronous accelerators.
        // This dispatch boundary preserves native gradients and all 32 updates.
        device.sync()?;
    }
    let result = evaluate(model.parameters.val())?;
    let energy = values(objective(&result)?)?[0];
    let parameters = values(model.parameters.val())?;
    let oracle = 0.5 + 0.5 * (4. * parameters[1]).sin() * parameters[0].sin();
    let herald = values(result.herald_probability())?[0];
    let oracle_herald = 0.99f64.powi(7) * (0.5 * eta * eta).powi(9);
    ensure(
        energy > 0.995 && energy > before,
        "physical QAOA failed to optimize the MaxCut edge",
    )?;
    ensure(
        (energy - oracle).abs() < 5e-5,
        "physical QAOA disagrees with analytic conditional objective",
    )?;
    ensure(
        (herald - oracle_herald).abs() < 2e-6,
        "source yield or routed fusion loss omitted",
    )?;
    Report::new(
        "physical_qaoa",
        0,
        32,
        &[
            ("initial_cut", before),
            ("conditional_cut", energy),
            ("analytic_cut", oracle),
            ("herald_probability", herald),
            ("analytic_herald_probability", oracle_herald),
            (
                "source_abort_probability",
                values(result.source_abort_probability())?[0],
            ),
            ("abort_probability", values(result.abort_probability())?[0]),
            (
                "expected_generation_photons",
                values(result.expected_resource_photons())?[0],
            ),
            ("expected_fusions", values(result.expected_fusions())?[0]),
            ("finish_tick", plan.finish_tick() as f64),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn calibrated_finite_qaoa_matches_analytic_maxcut_and_cost() {
        super::run().unwrap();
    }
}
