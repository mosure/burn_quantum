//! Task outcomes and bounded import validation, independent of graphics rendering.
use burn::tensor::Device;
use burn_quantum_visualization::{
    demo::{self, DemoConfig, DemoKind},
    io,
    simulation::TraceLimits,
};

#[test]
fn wine_is_multiqubit_real_data_with_heldout_evaluation_and_checkpoint_selection() {
    pollster::block_on(async {
        let document = demo::build(
            DemoKind::WineClassifier,
            &Device::flex(),
            TraceLimits::default(),
        )
        .await
        .unwrap();
        assert_eq!(document.circuit.register().wires(), 4);
        assert_eq!(document.inputs.len(), 13);
        assert_eq!(document.parameters.len(), 16);
        let study = document.study.as_ref().unwrap();
        let c = study.classification.as_ref().unwrap();
        assert_eq!(c.split_sizes, [105, 34, 39]);
        let best = c
            .validation_loss
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(c.selected_iteration, best);
        let correct: usize = c.confusion.iter().enumerate().map(|(i, r)| r[i]).sum();
        assert!(
            correct >= 30,
            "real held-out classification should exceed its 15/39 majority baseline"
        );
        assert!(document.optimization.last().unwrap().loss < document.optimization[0].loss * 0.6);
        assert!(document.optimization.iter().all(|p| p.metric.is_empty()));
        io::parse_document(&io::document_json(&document).unwrap()).unwrap();
        let mut bad = document.clone();
        bad.study
            .as_mut()
            .unwrap()
            .classification
            .as_mut()
            .unwrap()
            .confusion[0][0] = usize::MAX;
        assert!(bad.validate().is_err());
        let mut bad = document.clone();
        bad.study
            .as_mut()
            .unwrap()
            .classification
            .as_mut()
            .unwrap()
            .selected_iteration = 65;
        assert!(bad.validate().is_err());
        let mut bad = document;
        bad.study.as_mut().unwrap().metrics[0].value = f64::NAN;
        assert!(bad.validate().is_err());
    });
}
#[test]
fn ising_vqe_improves_energy_and_respects_independent_ground_bound() {
    pollster::block_on(async {
        for (size, depth) in [(4, 2), (6, 3)] {
            let document = demo::build_configured(
                DemoKind::Vqe,
                DemoConfig {
                    size,
                    depth,
                    variant: 0,
                },
                &Device::flex(),
                TraceLimits::default(),
            )
            .await
            .unwrap();
            let metrics = &document.study.as_ref().unwrap().metrics;
            let metric = |name: &str| metrics.iter().find(|m| m.label == name).unwrap().value;
            let energy = metric("Variational energy");
            assert!(energy >= metric("Exact ground energy") - 2e-4);
            assert!(energy < metric("Initial energy") - 0.2);
            assert!(metric("Energy error / spin") < 0.03);
            assert!((energy - document.optimization.last().unwrap().loss).abs() < 1e-7);
            document.validate().unwrap();
        }
    });
}
#[test]
fn task_reports_reject_false_witnesses_and_impossible_yield_counts() {
    pollster::block_on(async {
        let mut sat = demo::build(DemoKind::ThreeSat, &Device::flex(), TraceLimits::default())
            .await
            .unwrap();
        sat.study
            .as_mut()
            .unwrap()
            .sat
            .as_mut()
            .unwrap()
            .satisfying_assignments = 0;
        assert!(sat.validate().is_err());
        let mut unsat = demo::build_configured(
            DemoKind::ThreeSat,
            DemoConfig {
                size: 3,
                depth: 2,
                variant: 1,
            },
            &Device::flex(),
            TraceLimits::default(),
        )
        .await
        .unwrap();
        unsat
            .study
            .as_mut()
            .unwrap()
            .sat
            .as_mut()
            .unwrap()
            .success_probability = 1.;
        assert!(unsat.validate().is_err());
        let mut photonic = demo::build(
            DemoKind::PhotonicRouting,
            &Device::flex(),
            TraceLimits::default(),
        )
        .await
        .unwrap();
        photonic
            .study
            .as_mut()
            .unwrap()
            .photonic
            .as_mut()
            .unwrap()
            .delivered = usize::MAX;
        assert!(photonic.validate().is_err());
    });
}
