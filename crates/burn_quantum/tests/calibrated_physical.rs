//! Complete noisy teleportation with real source yield/cost and finite packet capacities.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::DType;
use burn_quantum::{
    photonic::{compiler::*, fabric::*, factories::*, optics::*, *},
    *,
};
use common::{close, values};
fn fabric() -> Fabric {
    Fabric::new(
        vec![
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
        ],
        vec![
            FabricLink {
                from: NodeId(0),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::IDEAL,
            },
            FabricLink {
                from: NodeId(1),
                to: NodeId(2),
                ticks: 1,
                transmission: RailTransmission::IDEAL,
            },
        ],
    )
    .unwrap()
}
fn placement(n: usize) -> PhysicalPlacement {
    PhysicalPlacement {
        inputs: vec![NodeId(0); n],
        resource: NodeId(1),
        analyzer: NodeId(2),
        feedback_ticks: 1,
    }
}
fn source(p: f64, noise: Option<KrausChannel>) -> ResourceFactoryModel {
    ResourceFactoryModel::new(
        "test-calibrated-emitter".into(),
        ResourceFactory::new(p, 6, 3, 2).unwrap(),
        noise,
    )
    .unwrap()
}
fn model(
    instrument: OpticalBellInstrument,
    p: f64,
    noise: Option<KrausChannel>,
) -> PhysicalCalibration {
    PhysicalCalibration::new(source(p, noise), source(1., None), instrument, 100_000).unwrap()
}
fn standard() -> OpticalBellInstrument {
    OpticalBellInstrument::standard(
        1.,
        [Detector::ideal(2).unwrap(); 4],
        "ideal-standard".into(),
    )
    .unwrap()
}
fn bell() -> StateVector {
    let e = DenseSimulator::default();
    let dev = runtime::execution().unwrap().device;
    let mut c = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    c.unitary(vec![0], Unitary::h()).unwrap();
    c.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    c.run_from(
        &e,
        StateVector::zero(c.register().clone(), 1, &dev, DType::F32, e.limits).unwrap(),
        None,
        None,
    )
    .unwrap()
}
fn gate() -> Circuit {
    let mut c = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    c.unitary(vec![0], Unitary::h()).unwrap();
    c
}
#[test]
fn standard_effects_match_existing_complete_channel_and_source_exhaustion_cost() {
    let c = gate();
    let f = fabric();
    let e = DenseSimulator::default();
    let ideal = PhysicalCircuit::compile(&c, &f, placement(2), CompileOptions::default())
        .unwrap()
        .execute(&e, bell(), None, None)
        .unwrap();
    let plan = PhysicalCircuit::compile_calibrated(
        &c,
        &f,
        placement(2),
        CompileOptions::default(),
        model(standard(), 0.4, None),
        None,
    )
    .unwrap();
    let r = plan.execute(&e, bell(), None, None).unwrap();
    close(
        &values(r.state().entries().real()),
        &values(ideal.state().entries().real() * 0.64),
        3e-6,
    );
    close(&values(r.herald_probability()), &[0.32], 2e-6);
    close(&values(r.abort_probability()), &[0.68], 2e-6);
    close(&values(r.source_abort_probability()), &[0.36], 2e-6);
    close(&values(r.expected_resource_photons()), &[9.6], 2e-6);
    close(&values(r.expected_fusions()), &[0.64], 2e-6);
    assert_eq!(r.optical_outcomes().len(), 5);
    assert_eq!(plan.resource_photons(), 12);
    assert!(plan.finish_tick() >= 9);
}
#[test]
fn arbitrary_emitted_resource_noise_changes_the_conditional_channel() {
    let mut z = vec![0.; 81];
    for i in 0..9 {
        z[i * 9 + i] = if i % 3 == 1 { -1. } else { 1. };
    }
    let noise = KrausChannel::new(vec![
        Matrix::real(
            9,
            (0..81)
                .map(|i| if i / 9 == i % 9 { 0.7f64.sqrt() } else { 0. })
                .collect(),
        )
        .unwrap(),
        Matrix::real(9, z.into_iter().map(|v| v * 0.3f64.sqrt()).collect()).unwrap(),
    ])
    .unwrap();
    let e = DenseSimulator::default();
    let c = gate();
    let f = fabric();
    let ideal = PhysicalCircuit::compile(&c, &f, placement(2), CompileOptions::default())
        .unwrap()
        .execute(&e, bell(), None, None)
        .unwrap();
    let flipped = ideal
        .state()
        .apply(&e, &[0], &DualRail::unitary(&Unitary::z()).unwrap())
        .unwrap();
    let expected = ideal.state().entries().real() * 0.7 + flipped.entries().real() * 0.3;
    let plan = PhysicalCircuit::compile_calibrated(
        &c,
        &f,
        placement(2),
        CompileOptions::default(),
        model(standard(), 1., Some(noise)),
        None,
    )
    .unwrap();
    let r = plan.execute(&e, bell(), None, None).unwrap();
    close(&values(r.state().entries().real()), &values(expected), 3e-6);
    close(
        &values(r.herald_probability() + r.abort_probability()),
        &[1.],
        3e-6,
    );
}
#[test]
fn grice_boost_is_an_actual_instrument_with_ancillary_photon_cost() {
    let e = DenseSimulator::default();
    let c = gate();
    let f = fabric();
    let ideal = PhysicalCircuit::compile(&c, &f, placement(2), CompileOptions::default())
        .unwrap()
        .execute(&e, bell(), None, None)
        .unwrap();
    let plan = PhysicalCircuit::compile_calibrated(
        &c,
        &f,
        placement(2),
        CompileOptions::default(),
        model(
            OpticalBellInstrument::grice(GriceConfig::default()).unwrap(),
            1.,
            None,
        ),
        None,
    )
    .unwrap();
    let r = plan.execute(&e, bell(), None, None).unwrap();
    close(
        &values(r.state().entries().real()),
        &values(ideal.state().entries().real() * 1.5),
        4e-6,
    );
    close(&values(r.herald_probability()), &[0.75], 3e-6);
    close(&values(r.expected_analyzer_ancillas()), &[2.], 2e-6);
    assert_eq!(r.optical_outcomes().len(), 6);
}
#[test]
fn finite_compiler_accounts_for_emitted_output_storage_and_rejects_overcommit() {
    let c = gate();
    let f = fabric();
    let constraints = |resource| FabricConstraints {
        buffers: vec![2, resource, 2],
        configure_ticks: vec![0, 1, 0],
        max_search_states: 1000,
        max_reservations: 100,
    };
    let options = CompileOptions::default();
    assert!(
        PhysicalCircuit::compile_finite(&c, &f, placement(2), options, constraints(1), &[4])
            .is_err()
    );
    let plan = PhysicalCircuit::compile_finite(&c, &f, placement(2), options, constraints(2), &[4])
        .unwrap();
    let e = DenseSimulator::default();
    let r = plan.execute(&e, bell(), None, None).unwrap();
    close(&values(r.herald_probability()), &[0.5], 2e-6);
    assert_eq!(plan.finish_tick(), 6);
    assert!(plan.constraints().is_some());
    assert!(
        PhysicalCircuit::compile_calibrated(
            &c,
            &f,
            placement(2),
            options,
            model(standard(), 1., None),
            Some((constraints(2), &[4]))
        )
        .is_err()
    );
}
#[test]
fn matched_wavelength_compilation_reaches_the_same_complete_channel() {
    let e = DenseSimulator::default();
    let c = gate();
    let f = fabric();
    let constraints = FabricConstraints {
        buffers: vec![2, 2, 2],
        configure_ticks: vec![0; 3],
        max_search_states: 1000,
        max_reservations: 100,
    };
    let rails = PhysicalRailOptions {
        fusion_ticks: vec![9],
        constraints: WavelengthConstraints {
            wavelengths: vec!["zero-rail-frequency".into(), "one-rail-frequency".into()],
            channels: vec![constraints; 2],
            max_skew_ticks: 0,
            policy: RoutePolicy::MaximumSurvival,
        },
        wavelengths: [0, 1],
    };
    let plan = PhysicalCircuit::compile_rails(
        &c,
        &f,
        placement(2),
        CompileOptions::default(),
        model(standard(), 1., None),
        rails,
    )
    .unwrap();
    let r = plan.execute(&e, bell(), None, None).unwrap();
    close(&values(r.herald_probability()), &[0.5], 3e-6);
    assert_eq!(plan.gates()[0].rail_routes().len(), 2);
    assert!(plan.rail_options().is_some());
}
#[test]
fn two_qubit_grice_gate_corrects_every_declared_bell_frame_on_an_entangled_reference() {
    let e = DenseSimulator::default();
    let d = runtime::execution().unwrap().device;
    let mut prep = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
    for i in 0..2 {
        prep.unitary(vec![i], Unitary::h()).unwrap();
        prep.unitary(vec![i, i + 2], Unitary::cnot()).unwrap();
    }
    let input = prep
        .run_from(
            &e,
            StateVector::zero(prep.register().clone(), 1, &d, DType::F32, e.limits).unwrap(),
            None,
            None,
        )
        .unwrap();
    let mut c = Circuit::new(Register::qubits(4).unwrap(), 0, 0);
    c.unitary(vec![1, 0], Unitary::cnot()).unwrap();
    let ideal = PhysicalCircuit::compile(&c, &fabric(), placement(4), CompileOptions::default())
        .unwrap()
        .execute(&e, input.clone(), None, None)
        .unwrap();
    let plan = PhysicalCircuit::compile_calibrated(
        &c,
        &fabric(),
        placement(4),
        CompileOptions::default(),
        model(
            OpticalBellInstrument::grice(GriceConfig::default()).unwrap(),
            1.,
            None,
        ),
        None,
    )
    .unwrap();
    let r = plan.execute(&e, input, None, None).unwrap();
    close(
        &values(r.state().entries().real()),
        &values(ideal.state().entries().real() * 2.25),
        6e-6,
    );
    close(
        &values(r.state().entries().imag()),
        &values(ideal.state().entries().imag() * 2.25),
        6e-6,
    );
    close(&values(r.herald_probability()), &[0.5625], 4e-6);
    assert_eq!(r.optical_outcomes().len(), 36);
}
#[test]
fn detector_false_heralds_do_not_manufacture_unknown_input_fidelity() {
    let e = DenseSimulator::default();
    let mut topology = fabric();
    let mut links = topology.links().to_vec();
    links[0].transmission = RailTransmission::uniform(0.).unwrap();
    topology = Fabric::new(topology.nodes().to_vec(), links).unwrap();
    let analyzer = OpticalBellInstrument::standard(
        1.,
        [Detector::new(1., 2, 0.2).unwrap(); 4],
        "dark-counts-not-valid-teleportation".into(),
    )
    .unwrap();
    let plan = PhysicalCircuit::compile_calibrated(
        &gate(),
        &topology,
        placement(2),
        CompileOptions::default(),
        model(analyzer, 1., None),
        None,
    )
    .unwrap();
    let r = plan.execute(&e, bell(), None, None).unwrap();
    let p = values(r.herald_probability())[0];
    assert!(p > 0.01);
    let rho = values(r.state().entries().real());
    let target = [(0, 0.5), (1, 0.5), (3, 0.5), (4, -0.5)];
    let fidelity = target
        .iter()
        .flat_map(|&(a, x)| target.iter().map(move |&(b, y)| (a, b, x * y)))
        .map(|(a, b, v)| v * rho[b * 9 + a])
        .sum::<f64>()
        / p;
    assert!((fidelity - 0.25).abs() < 3e-5);
    close(
        &values(r.herald_probability() + r.abort_probability()),
        &[1.],
        4e-6,
    );
}
#[test]
fn calibrated_resource_parameters_keep_exact_native_conditional_gradients() {
    use burn::tensor::Tensor;
    use burn_quantum::gate::Axis;
    let e = DenseSimulator::default();
    let d = runtime::execution().unwrap().device.autodiff();
    let mut c = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    c.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let p = PhysicalCircuit::compile_calibrated(
        &c,
        &fabric(),
        placement(1),
        CompileOptions::default(),
        model(standard(), 0.4, None),
        None,
    )
    .unwrap();
    let theta = Tensor::from_data([0.7f32], &d).require_grad();
    let r = p
        .execute(
            &e,
            StateVector::zero(c.register().clone(), 1, &d, DType::F32, e.limits).unwrap(),
            Some(theta.clone()),
            None,
        )
        .unwrap();
    let z = r.state().expectation(&e, &DualRail::z(0)).unwrap() / r.herald_probability();
    close(&values(z.clone()), &[0.7f64.cos()], 3e-5);
    let gradient = z.sum().backward();
    close(
        &values(theta.grad(&gradient).unwrap()),
        &[-0.7f64.sin()],
        5e-5,
    );
    close(&values(r.expected_ticks()), &[0.36 * 6. + 0.64 * 9.], 3e-5);
}

// A fresh graph is built for each execution; readouts keep every declared effect
// and derivatives through both the unknown input and the teleported rotation.
fn parameterized_case(plan: &PhysicalCircuit, include_records: bool) -> (Vec<f64>, Vec<f64>) {
    use burn::tensor::Tensor;
    use burn_quantum::gate::Axis;
    let engine = DenseSimulator::default();
    let device = runtime::execution().unwrap().device.autodiff();
    let parameters = Tensor::from_data([0.7f32, 0.43], &device).require_grad();
    let mut prep = Circuit::new(Register::qubits(2).unwrap(), 2, 0);
    prep.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(1)))
        .unwrap();
    prep.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    prep.rotate(0, Rotation::qubit(Axis::Z, Angle::Constant(0.29)))
        .unwrap();
    let input = prep
        .run_from(
            &engine,
            StateVector::zero(
                prep.register().clone(),
                1,
                &device,
                DType::F32,
                engine.limits,
            )
            .unwrap(),
            Some(parameters.clone()),
            None,
        )
        .unwrap();
    let result = plan
        .execute(&engine, input, Some(parameters.clone()), None)
        .unwrap();
    let mut samples = values(result.state().entries().real());
    samples.extend(values(result.state().entries().imag()));
    samples.extend(values(result.herald_probability()));
    samples.extend(values(result.abort_probability()));
    let mut objective = result
        .state()
        .expectation(&engine, &DualRail::z(0))
        .unwrap()
        + result.herald_probability() * 0.37;
    if include_records {
        for (index, record) in result.optical_outcomes().iter().enumerate() {
            samples.extend(values(record.probability.clone()));
            objective = objective + record.probability.clone() * ((index + 1) as f64 / 7.);
        }
        samples.extend(values(result.expected_resource_photons()));
        samples.extend(values(result.expected_ticks()));
    }
    let gradient = objective.sum().backward();
    let derivative = values(parameters.grad(&gradient).unwrap());
    device.sync().unwrap();
    (samples, derivative)
}
fn symbolic_gate() -> Circuit {
    use burn_quantum::gate::Axis;
    let mut c = Circuit::new(Register::qubits(2).unwrap(), 2, 0);
    c.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    c
}
#[test]
fn rejected_choi_marginal_shortcut_matches_generic_noisy_channel_and_input_gradients() {
    let c = symbolic_gate();
    let mut links = fabric().links().to_vec();
    links[0].transmission = RailTransmission::new(0.71, 0.89).unwrap();
    links[1].transmission = RailTransmission::new(0.83, 0.61).unwrap();
    let f = Fabric::new(fabric().nodes().to_vec(), links).unwrap();
    for analyzer in [
        OpticalBellInstrument::standard(
            0.87,
            [Detector::new(0.93, 2, 0.04).unwrap(); 4],
            "false-herald-regression".into(),
        )
        .unwrap(),
        OpticalBellInstrument::grice(GriceConfig::default()).unwrap(),
    ] {
        let make = |noise| {
            PhysicalCircuit::compile_calibrated(
                &c,
                &f,
                placement(2),
                CompileOptions::default(),
                model(analyzer.clone(), 0.4, noise),
                None,
            )
            .unwrap()
        };
        let fast = make(None);
        // An explicit identity CPTP map forces the complete resource/Kraus path.
        let generic = make(Some(
            KrausChannel::new(vec![Unitary::identity(9).unwrap().matrix().clone()]).unwrap(),
        ));
        let (a, da) = parameterized_case(&fast, true);
        let (b, db) = parameterized_case(&generic, true);
        close(&a, &b, 8e-6);
        close(&da, &db, 7e-5);
        assert!(da[0].abs() > 1e-3 && da[1].abs() > 1e-3);
    }
}
#[test]
fn composed_correction_matches_separate_density_evolutions_and_gradients() {
    let c = symbolic_gate();
    let f = fabric();
    let separate =
        PhysicalCircuit::compile(&c, &f, placement(2), CompileOptions::default()).unwrap();
    let composed = PhysicalCircuit::compile_calibrated(
        &c,
        &f,
        placement(2),
        CompileOptions::default(),
        model(standard(), 1., None),
        None,
    )
    .unwrap();
    let (a, da) = parameterized_case(&composed, false);
    let (b, db) = parameterized_case(&separate, false);
    close(&a, &b, 7e-6);
    close(&da, &db, 7e-5);
}
