//! Independent scalar estimators and closed-form metric solves.
use super::*;

#[test]
fn single_rotation_has_quarter_metric_not_qfi_or_loss_hessian() {
    let theta = 0.7f64;
    let c = 0.1f64;
    for a in [-1., 1.] {
        for b in [-1., 1.] {
            let f = |shift: f64| (shift / 2.).cos().powi(2);
            let sample = qn_spsa_sample(
                c,
                &[a],
                &[b],
                [(theta + c * a).cos(), (theta - c * a).cos()],
                [f(c * a), f(-c * a), f(c * (a + b)), f(c * (-a + b))],
                1,
            )
            .unwrap();
            close(&sample.gradient, &[-theta.sin() * c.sin() / c], 1e-12);
            close(&sample.metric, &[c.sin().powi(2) / (4. * c * c)], 1e-12);
        }
    }
}

#[test]
fn exhaustive_directions_recover_rank_one_off_diagonal_geometry() {
    let theta = 0.4f64;
    let c = 0.001;
    let mut metric = vec![0.; 4];
    let mut gradient = vec![0.; 2];
    let shift = |v: &[f64]| v[0] + 2. * v[1];
    for a in directions(2) {
        for b in directions(2) {
            let x = shift(&a) * c;
            let y = shift(&b) * c;
            let f = |z: f64| (z / 2.).cos().powi(2);
            let s = qn_spsa_sample(
                c,
                &a,
                &b,
                [(theta + x).cos(), (theta - x).cos()],
                [f(x), f(-x), f(x + y), f(-x + y)],
                4,
            )
            .unwrap();
            for (avg, x) in metric.iter_mut().zip(s.metric) {
                *avg += x / 16.;
            }
            for (avg, x) in gradient.iter_mut().zip(s.gradient) {
                *avg += x / 16.;
            }
        }
    }
    close(&metric, &[0.25, 0.5, 0.5, 1.0], 4e-6);
    close(&gradient, &[-theta.sin(), -2. * theta.sin()], 2e-6);
}

#[test]
fn spectral_regularization_handles_indefinite_and_null_directions() {
    let mut saved = config().with_regularization(0.1).init(3).unwrap().state();
    saved.metric = vec![0., 2., 0., 2., 0., 0., 0., 0., 0.];
    let optimizer = QnSpsa::from_state(saved).unwrap();
    close(
        &optimizer.regularized_metric().unwrap(),
        &[2.1, 0., 0., 0., 2.1, 0., 0., 0., 0.1],
        1e-12,
    );
    let mut saved = config().with_regularization(0.1).init(2).unwrap().state();
    saved.metric = vec![1., 2., 2., 1.];
    close(
        &QnSpsa::from_state(saved)
            .unwrap()
            .regularized_metric()
            .unwrap(),
        &[2.1, 1., 1., 2.1],
        1e-12,
    );
}

#[test]
fn coupled_update_matches_closed_form_two_by_two_natural_solve() {
    let cfg = config().with_regularization(0.07);
    let mut saved = cfg.init(2).unwrap().state();
    saved.metric = vec![1., 2., 2., 1.];
    let mut optimizer = QnSpsa::from_state(saved).unwrap();
    let center = [0.3, -0.5];
    let mut probes = Vec::new();
    let mut loss = |p: Tensor<1>| {
        let p = values(p);
        let value = p[0] - 0.6 * p[1];
        probes.push(p);
        Ok(value)
    };
    let mut fidelity = |a: Tensor<1>, b: Tensor<1>| {
        let a = values(a);
        let b = values(b);
        Ok(((a[0] + 2. * a[1] - b[0] - 2. * b[1]) / 2.).cos().powi(2))
    };
    let step = optimizer
        .step(
            tensor(&center, &Device::flex(), DType::F64),
            &mut loss,
            &mut fidelity,
        )
        .unwrap();
    let direction: Vec<_> = probes[0]
        .iter()
        .zip(&probes[1])
        .map(|(a, b)| (a - b) / (2. * cfg.perturbation))
        .collect();
    let slope = direction[0] - 0.6 * direction[1];
    let gradient = [slope * direction[0], slope * direction[1]];
    let raw = optimizer.metric();
    // Independent 2x2 spectral interpolation for |M|; no eigensolver in the reference.
    let mid = (raw[0] + raw[3]) / 2.;
    let spread = ((raw[0] - raw[3]) / 2.).hypot(raw[1]);
    let high = mid + spread;
    let low = mid - spread;
    let alpha = (high.abs() - low.abs()) / (high - low);
    let beta = low.abs() - alpha * low + cfg.regularization;
    let a = alpha * raw[0] + beta;
    let b = alpha * raw[1];
    let d = alpha * raw[3] + beta;
    let determinant = a * d - b * b;
    let natural = [
        (d * gradient[0] - b * gradient[1]) / determinant,
        (a * gradient[1] - b * gradient[0]) / determinant,
    ];
    close(
        &values(step.parameters),
        &[
            center[0] - cfg.learning_rate * natural[0],
            center[1] - cfg.learning_rate * natural[1],
        ],
        1e-12,
    );
}
