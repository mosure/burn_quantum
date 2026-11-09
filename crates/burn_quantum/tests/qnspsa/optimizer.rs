//! Step accounting, state transitions, reproducibility, and noisy objectives.
use super::*;

#[test]
fn update_and_running_average_match_an_independent_scalar_recurrence() {
    let cfg = config().with_resamplings(3).with_regularization(0.02);
    let mut optimizer = cfg.init(1).unwrap();
    let mut theta = 0.7f64;
    let mut expected_metric = 1.;
    let mut p = tensor(&[theta], &Device::flex(), DType::F64);
    let point_metric = cfg.perturbation.sin().powi(2) / (4. * cfg.perturbation.powi(2));
    for k in 1..=4 {
        expected_metric = (k as f64 * expected_metric + point_metric) / (k as f64 + 1.);
        theta += cfg.learning_rate * theta.sin() * cfg.perturbation.sin()
            / cfg.perturbation
            / (expected_metric + cfg.regularization);
        let result = optimizer
            .step(p, &mut scalar_loss, &mut scalar_fidelity)
            .unwrap();
        close(&values(result.parameters.clone()), &[theta], 1e-11);
        close(optimizer.metric(), &[expected_metric], 1e-11);
        assert_eq!(result.report.loss_evaluations, 6);
        assert_eq!(result.report.fidelity_evaluations, 12);
        assert_eq!(result.report.iteration, k);
        assert_eq!(result.parameters.dtype(), DType::F64);
        assert!(result.report.accepted && result.report.loss_proposed.is_none());
        p = result.parameters;
    }
}

#[test]
fn evaluation_cost_does_not_grow_with_parameter_count() {
    for n in [1, 4, 9] {
        let losses = Cell::new(0);
        let fidelities = Cell::new(0);
        let mut loss = |p: Tensor<1>| {
            losses.set(losses.get() + 1);
            Ok(values(p).iter().map(|x| x * x / 2.).sum())
        };
        let mut fidelity = |a: Tensor<1>, b: Tensor<1>| {
            fidelities.set(fidelities.get() + 1);
            Ok(values(a)
                .iter()
                .zip(values(b))
                .map(|(x, y)| ((x - y) / 2.).cos().powi(2))
                .product())
        };
        let mut optimizer = config().with_resamplings(3).init(n).unwrap();
        let report = optimizer
            .step(
                tensor(&vec![0.3; n], &Device::flex(), DType::F64),
                &mut loss,
                &mut fidelity,
            )
            .unwrap()
            .report;
        assert_eq!((losses.get(), fidelities.get()), (6, 12));
        assert_eq!(
            (report.loss_evaluations, report.fidelity_evaluations),
            (6, 12)
        );
    }
}

#[test]
fn blocking_rejects_loss_increase_but_learns_metric_and_advances_schedule() {
    let mut optimizer = config()
        .with_learning_rate(100.)
        .with_blocking_tolerance(Some(0.))
        .init(1)
        .unwrap();
    let old = tensor(&[0.7], &Device::flex(), DType::F64);
    let mut quadratic = |p: Tensor<1>| Ok(values(p)[0].powi(2));
    let step = optimizer
        .step(old.clone(), &mut quadratic, &mut scalar_fidelity)
        .unwrap();
    assert!(!step.report.accepted);
    assert!(step.report.loss_proposed.unwrap() > step.report.loss_before.unwrap());
    close(&values(step.parameters), &values(old), 0.);
    assert_eq!(optimizer.iteration(), 1);
    assert!((optimizer.metric()[0] - 1.).abs() > 0.1);
    assert_eq!(
        (
            step.report.loss_evaluations,
            step.report.fidelity_evaluations
        ),
        (4, 4)
    );
}

#[test]
fn delay_and_power_schedules_use_completed_attempt_indices() {
    let cfg = config()
        .with_metric_delay(2)
        .with_learning_rate_decay(0.5)
        .with_perturbation_decay(0.1)
        .with_stability_constant(3.);
    let mut optimizer = cfg.init(1).unwrap();
    let mut p = tensor(&[0.7], &Device::flex(), DType::F64);
    let mut theta = 0.7f64;
    for k in 1..=3 {
        let rate = 0.1 / (k as f64 + 3.).sqrt();
        let eps = 0.05 / (k as f64).powf(0.1);
        let step = optimizer
            .step(p, &mut scalar_loss, &mut scalar_fidelity)
            .unwrap();
        let denominator = if k <= 2 {
            1.
        } else {
            optimizer.metric()[0] + cfg.regularization
        };
        theta += rate * theta.sin() * eps.sin() / eps / denominator;
        close(&values(step.parameters.clone()), &[theta], 1e-11);
        close(
            &[step.report.learning_rate, step.report.perturbation],
            &[rate, eps],
            1e-14,
        );
        p = step.parameters;
    }
}

#[test]
fn checkpoint_restores_metric_schedule_and_perturbation_stream() {
    let mut optimizer = config().with_resamplings(3).init(2).unwrap();
    let mut p = tensor(&[0.8, -0.2], &Device::flex(), DType::F64);
    let mut loss = |p: Tensor<1>| Ok(values(p).iter().map(|x| x.cos()).sum());
    let mut fidelity = |a: Tensor<1>, b: Tensor<1>| {
        Ok(values(a)
            .iter()
            .zip(values(b))
            .map(|(x, y)| ((x - y) / 2.).cos().powi(2))
            .product())
    };
    for _ in 0..3 {
        p = optimizer
            .step(p, &mut loss, &mut fidelity)
            .unwrap()
            .parameters;
    }
    let serialized = serde_json::to_string(&optimizer.state()).unwrap();
    let mut restored = QnSpsa::from_state(serde_json::from_str(&serialized).unwrap()).unwrap();
    let mut q = p.clone();
    for _ in 0..5 {
        p = optimizer
            .step(p, &mut loss, &mut fidelity)
            .unwrap()
            .parameters;
        q = restored
            .step(q, &mut loss, &mut fidelity)
            .unwrap()
            .parameters;
        close(&values(p.clone()), &values(q.clone()), 1e-12);
        close(optimizer.metric(), restored.metric(), 1e-12);
        assert_eq!(optimizer.iteration(), restored.iteration());
    }
}

#[test]
fn invalid_values_and_failed_callbacks_cannot_mutate_optimizer_state() {
    for cfg in [
        config().with_learning_rate(f64::NAN),
        config().with_perturbation(0.),
        config().with_regularization(-1.),
        config().with_resamplings(0),
        config().with_learning_rate_decay(-1.),
        config().with_blocking_tolerance(Some(f64::INFINITY)),
    ] {
        assert!(cfg.init(1).is_err());
    }
    assert!(config().init(0).is_err());
    assert!(matches!(
        config().with_max_metric_elements(3).init(2),
        Err(Error::ResourceLimit { .. })
    ));
    assert!(config().init(usize::MAX).is_err());
    let base = config().init(2).unwrap().state();
    for mode in 0..4 {
        let mut bad = base.clone();
        match mode {
            0 => bad.version = 2,
            1 => bad.metric.pop().map(|_| ()).unwrap(),
            2 => bad.metric[0] = f64::NAN,
            _ => bad.metric[1] = 0.1,
        }
        assert!(QnSpsa::from_state(bad).is_err());
    }
    let mut optimizer = config().init(1).unwrap();
    let before = serde_json::to_string(&optimizer.state()).unwrap();
    let p = tensor(&[0.5], &Device::flex(), DType::F64);
    assert!(
        optimizer
            .step(
                p.clone(),
                &mut |_| Err(Error::Readback("deliberate failure".into())),
                &mut scalar_fidelity
            )
            .is_err()
    );
    assert!(
        optimizer
            .step(p.clone(), &mut |_| Ok(f64::NAN), &mut scalar_fidelity)
            .is_err()
    );
    assert!(
        optimizer
            .step(p.clone(), &mut scalar_loss, &mut |_, _| Ok(1.1))
            .is_err()
    );
    assert_eq!(before, serde_json::to_string(&optimizer.state()).unwrap());
    let mut tiny = config().with_perturbation(1e-12).init(1).unwrap();
    let calls = Cell::new(0);
    assert!(
        tiny.step(
            tensor(&[0.5], &Device::flex(), DType::F32),
            &mut |_| {
                calls.set(calls.get() + 1);
                Ok(0.)
            },
            &mut scalar_fidelity
        )
        .is_err()
    );
    assert_eq!(calls.get(), 0);
    assert!(qn_spsa_sample(0.1, &[0.], &[1.], [0., 1.], [1.; 4], 1).is_err());
    assert!(qn_spsa_sample(1e-300, &[1.], &[1.], [0., 1.], [1.; 4], 1).is_err());
    assert!(qn_spsa_sample(0.1, &[1.], &[], [0., 1.], [1.; 4], 1).is_err());
}

#[cfg(feature = "autodiff")]
#[test]
fn burn_parameter_identity_and_leaf_gradient_flag_survive_update() {
    let device = selected().autodiff();
    let mut c = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    c.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let mut layer = QuantumLayerConfig::new(c, vec![Observable::z(0)], vec![0.7])
        .init(&device)
        .unwrap();
    let id = layer.parameters.id;
    let original = layer.parameters.val();
    let mut optimizer = config().init(1).unwrap();
    let step = optimizer
        .step(original.clone(), &mut scalar_loss, &mut scalar_fidelity)
        .unwrap();
    layer.parameters = layer.parameters.map(|_| step.parameters);
    assert_eq!(id, layer.parameters.id);
    assert_eq!(layer.parameters.val().device(), device);
    assert!(layer.parameters.val().is_require_grad());
    let grads = layer.parameters.val().sum().backward();
    close(
        &values(layer.parameters.val().grad(&grads).unwrap()),
        &[1.],
        1e-6,
    );
    assert!(original.grad(&grads).is_none());
}

#[test]
fn seeded_finite_shot_callbacks_improve_the_underlying_exact_loss() {
    let mut rng = ChaCha8Rng::seed_from_u64(88);
    let mut fid_rng = ChaCha8Rng::seed_from_u64(99);
    let estimate = |probability: f64, rng: &mut ChaCha8Rng| {
        (0..512)
            .filter(|_| rng.random::<f64>() < probability)
            .count() as f64
            / 512.
    };
    let mut loss = |p: Tensor<1>| {
        let angle = values(p)[0];
        Ok(2. * estimate((1. + angle.cos()) / 2., &mut rng) - 1.)
    };
    let mut fidelity = |a: Tensor<1>, b: Tensor<1>| {
        Ok(estimate(
            ((values(a)[0] - values(b)[0]) / 2.).cos().powi(2),
            &mut fid_rng,
        ))
    };
    let mut optimizer = config()
        .with_learning_rate(0.1)
        .with_perturbation(0.2)
        .with_regularization(0.05)
        .with_initial_metric_scale(0.25)
        .with_resamplings(4)
        .with_blocking_tolerance(Some(0.05))
        .init(1)
        .unwrap();
    let mut theta = tensor(&[0.7], &Device::flex(), DType::F64);
    for _ in 0..40 {
        theta = optimizer
            .step(theta, &mut loss, &mut fidelity)
            .unwrap()
            .parameters;
    }
    assert!(values(theta)[0].cos() < -0.98);
}
