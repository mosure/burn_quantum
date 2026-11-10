//! Scientific readout and import-budget regression tests, independent of the renderer.
use burn::tensor::Device;
use burn_quantum::{
    Circuit, DenseSimulator, DensityMatrix, KrausChannel, QuantumBackend, Register, Unitary,
};
use burn_quantum_visualization::{
    demo::{self, DemoKind},
    model::aggregate_probabilities,
    simulation::{self, TraceLimits},
};

fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 3e-5, "{a} != {b}");
}
#[test]
fn bloch_y_signs_and_entangled_reduced_states_are_physical() {
    pollster::block_on(async {
        let device = Device::flex();
        let limits = TraceLimits::default();
        let mut c = Circuit::new(Register::qubits(1).unwrap(), 0, 0);
        c.unitary(vec![0], Unitary::h()).unwrap();
        c.unitary(vec![0], Unitary::s()).unwrap();
        c.unitary(vec![0], Unitary::z()).unwrap();
        let labels = vec!["H".into(), "S".into(), "Z".into()];
        let trace = simulation::simulate(&c, &[], &[], &labels, &device, limits)
            .await
            .unwrap();
        for (frame, expected) in
            trace
                .frames
                .iter()
                .zip([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        {
            for (a, b) in frame.subsystems[0].bloch.unwrap().into_iter().zip(expected) {
                near(a, b);
            }
            near(frame.subsystems[0].purity, 1.);
        }
        for kind in [DemoKind::Bell, DemoKind::Ghz] {
            let doc = demo::build(kind, &device, limits).await.unwrap();
            let final_state = doc.trace.frames.last().unwrap();
            near(final_state.probabilities[0], 0.5);
            near(*final_state.probabilities.last().unwrap(), 0.5);
            for sub in &final_state.subsystems {
                for a in sub.bloch.unwrap() {
                    near(a, 0.);
                }
                near(sub.purity, 0.5);
            }
            for (step, frame) in doc.trace.frames.iter().enumerate() {
                assert_eq!(frame.step, step);
                near(frame.norm, 1.);
            }
            assert_eq!(doc.trace.frames.len(), doc.circuit.operations().len() + 1);
        }
    });
}

#[test]
fn mixed_state_snapshots_do_not_invent_pure_amplitudes() {
    pollster::block_on(async {
        let device = Device::flex();
        let limits = TraceLimits::default();
        let engine = DenseSimulator::new(limits.resources());
        let mut c = Circuit::new(Register::qubits(1).unwrap(), 0, 0);
        c.unitary(vec![0], Unitary::x()).unwrap();
        let state = engine.execute(&c, None, None, 1, &device).unwrap();
        let density = DensityMatrix::from_pure(&state, engine.limits)
            .unwrap()
            .channel(
                &engine,
                &[0],
                &KrausChannel::amplitude_damping(0.25).unwrap(),
            )
            .unwrap();
        let frame = simulation::snapshot_density(&density, 0, "Damped".into(), limits)
            .await
            .unwrap();
        assert!(frame.amplitudes.is_none());
        near(frame.probabilities[0], 0.25);
        near(frame.probabilities[1], 0.75);
        near(frame.subsystems[0].bloch.unwrap()[2], -0.5);
        near(frame.subsystems[0].purity, 0.625);
    });
}

#[test]
fn qutrit_interference_and_probability_aggregation_keep_all_mass() {
    pollster::block_on(async {
        let device = Device::flex();
        let limits = TraceLimits::default();
        let doc = demo::build(DemoKind::Qutrit, &device, limits)
            .await
            .unwrap();
        let state = doc.trace.frames.last().unwrap();
        assert!(state.subsystems[0].bloch.is_none());
        for (&a, b) in state.probabilities.iter().zip([1. / 9., 4. / 9., 4. / 9.]) {
            near(a, b);
        }
        let bins = aggregate_probabilities(&state.probabilities, 2).unwrap();
        assert_eq!(bins.len(), 2);
        near(bins.iter().map(|b| b.probability).sum(), 1.);
        assert_eq!((bins[0].start, bins[0].end, bins[1].end), (0, 2, 3));
        let interference = demo::build(DemoKind::Interference, &device, limits)
            .await
            .unwrap();
        near(
            interference.trace.frames.last().unwrap().probabilities[1],
            1.,
        );
        assert!(aggregate_probabilities(&[1.], 0).is_err());
        assert!(aggregate_probabilities(&[f64::NAN], 2).is_err());
    });
}

#[test]
fn optimization_frames_are_actual_iterates_and_costs_are_not_perturbed_losses() {
    pollster::block_on(async {
        let doc = demo::build(
            DemoKind::Optimization,
            &Device::flex(),
            TraceLimits::default(),
        )
        .await
        .unwrap();
        assert_eq!(doc.optimization.len(), 25);
        assert_eq!(doc.trace.frames.len(), 25);
        for (i, (p, frame)) in doc.optimization.iter().zip(&doc.trace.frames).enumerate() {
            near(p.loss, p.parameters[0].cos());
            near(p.loss, frame.probabilities[0] - frame.probabilities[1]);
            assert_eq!(p.loss_evaluations, 1 + 3 * i);
            assert_eq!(p.fidelity_evaluations, 4 * i);
            assert_eq!(p.probabilities, frame.probabilities);
        }
        assert!(doc.optimization.last().unwrap().loss < -0.98);
        assert!(doc.optimization.last().unwrap().loss < doc.optimization[0].loss);
        doc.validate().unwrap();
    });
}

#[test]
fn actual_photonic_schedule_and_untrusted_document_checks_are_bounded() {
    pollster::block_on(async {
        let device = Device::flex();
        let limits = TraceLimits::default();
        let doc = demo::build(DemoKind::PhotonicRouting, &device, limits)
            .await
            .unwrap();
        let fabric = doc.fabric.as_ref().unwrap();
        assert_eq!(
            (
                fabric.resource_count,
                fabric.photon_count,
                fabric.routes.len(),
                fabric.fusions.len()
            ),
            (2, 4, 2, 1)
        );
        assert_eq!(fabric.outputs, vec![0, 3]);
        assert!(fabric.routes.iter().all(|r| r.transmission[0] < 1.));
        let mut bad = doc.clone();
        bad.fabric.as_mut().unwrap().routes[0].hops[0].arrival += 1;
        assert!(bad.validate().is_err());
        let mut bad = doc.clone();
        bad.fabric.as_mut().unwrap().routes[0].transmission = [1.0; 2];
        assert!(bad.validate().is_err());
        let mut bad = doc.clone();
        bad.fabric.as_mut().unwrap().nodes[0].position[0] = f32::MAX;
        assert!(bad.validate().is_err());
        let mut bad = doc.clone();
        bad.trace.frames[0].probabilities[0] = f64::NAN;
        assert!(bad.validate().is_err());
        let mut bad = doc.clone();
        bad.trace.frames[0].subsystems[0].density[0].re = 0.;
        assert!(bad.validate().is_err());
        let mut bad = doc.clone();
        bad.trace.frames[0].amplitudes.as_mut().unwrap()[0].im = 0.1;
        assert!(bad.validate().is_err());
        let text = serde_json::to_string(&doc).unwrap();
        let restored: burn_quantum_visualization::model::VisualizationDocument =
            serde_json::from_str(&text).unwrap();
        restored.validate().unwrap();
        for limits in [
            TraceLimits {
                max_frames: 1,
                ..limits
            },
            TraceLimits {
                max_dimension: 2,
                ..limits
            },
            TraceLimits {
                max_trace_elements: 1,
                ..limits
            },
            TraceLimits {
                max_matrix_elements: 1,
                ..limits
            },
        ] {
            assert!(demo::build(DemoKind::Bell, &device, limits).await.is_err());
        }
    });
}

#[test]
fn twelve_qubit_trace_and_bounded_plots_preserve_scientific_mass() {
    pollster::block_on(async {
        let device = Device::flex();
        let mut circuit = Circuit::new(Register::qubits(12).unwrap(), 0, 0);
        let mut labels = vec![];
        for wire in 0..12 {
            circuit.unitary(vec![wire], Unitary::h()).unwrap();
            labels.push(format!("H q{wire}"));
        }
        let trace =
            simulation::simulate(&circuit, &[], &[], &labels, &device, TraceLimits::default())
                .await
                .unwrap();
        assert_eq!(trace.frames.len(), 13);
        let output = trace.frames.last().unwrap();
        assert_eq!(output.probabilities.len(), 4096);
        for &probability in &output.probabilities {
            near(probability, 1. / 4096.);
        }
        let bins = aggregate_probabilities(&output.probabilities, 64).unwrap();
        assert_eq!(bins.len(), 64);
        near(bins.iter().map(|b| b.probability).sum(), 1.);
        for state in &output.subsystems {
            near(state.bloch.unwrap()[0], 1.);
            near(state.purity, 1.);
        }
    });
}

#[test]
fn heterogeneous_endianness_input_bindings_and_nonpositive_imports_are_checked() {
    pollster::block_on(async {
        use burn_quantum::{Angle, Rotation, gate::Axis};
        let device = Device::flex();
        let limits = TraceLimits::default();
        let mut circuit = Circuit::new(Register::new(vec![2, 3]).unwrap(), 0, 0);
        circuit.unitary(vec![0], Unitary::x()).unwrap();
        circuit
            .unitary(vec![1], Unitary::shift(3, 2).unwrap())
            .unwrap();
        let trace = simulation::simulate(
            &circuit,
            &[],
            &[],
            &["X".into(), "Shift2".into()],
            &device,
            limits,
        )
        .await
        .unwrap();
        let frame = trace.frames.last().unwrap();
        near(frame.probabilities[5], 1.);
        near(frame.subsystems[0].bloch.unwrap()[2], -1.);
        near(frame.subsystems[1].probabilities[2], 1.);
        let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 0, 1);
        circuit
            .rotate(0, Rotation::qubit(Axis::Y, Angle::input(0)))
            .unwrap();
        let trace = simulation::simulate(
            &circuit,
            &[],
            &[std::f64::consts::FRAC_PI_2],
            &["Ry(input)".into()],
            &device,
            limits,
        )
        .await
        .unwrap();
        near(trace.frames[1].subsystems[0].bloch.unwrap()[0], 1.);
        assert!(
            simulation::simulate(&circuit, &[], &[], &["Ry(input)".into()], &device, limits)
                .await
                .is_err()
        );
        let mut doc = demo::build(DemoKind::Qutrit, &device, limits)
            .await
            .unwrap();
        let frame = doc.trace.frames.last_mut().unwrap();
        frame.amplitudes = None;
        frame.probabilities.fill(1. / 3.);
        frame.norm = 1.;
        let sub = &mut frame.subsystems[0];
        sub.probabilities.fill(1. / 3.);
        sub.density.fill(Default::default());
        for i in 0..3 {
            sub.density[3 * i + i].re = 1. / 3.;
        }
        sub.density[1].re = 0.4;
        sub.density[3].re = 0.4;
        sub.purity = 1. / 3. + 0.32;
        // Trace one and Hermitian with valid marginals, but a negative eigenvalue.
        assert!(doc.validate().is_err());
        assert!(aggregate_probabilities(&[f64::MAX, f64::MAX], 1).is_err());
    });
}
