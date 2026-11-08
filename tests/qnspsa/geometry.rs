//! Independent complex tangent geometry and selected-device numerical parity.
use super::*;

struct Problem {
    circuit: Circuit,
    diagonal: Vec<f64>,
}
impl Problem {
    fn state(&self, p: Tensor<1>) -> Result<StateVector> {
        let device = p.device();
        self.circuit
            .run(&DenseSimulator::default(), Some(p), None, 1, &device)
    }
    fn loss(&self, p: Tensor<1>) -> Result<f64> {
        Ok(values(self.state(p)?.probabilities())
            .iter()
            .zip(&self.diagonal)
            .map(|(p, h)| p * h)
            .sum())
    }
    fn fidelity(&self, a: Tensor<1>, b: Tensor<1>) -> Result<f64> {
        Ok(self
            .state(a)?
            .fidelity(&self.state(b)?)?
            .into_scalar::<f64>())
    }
    fn average_sample(&self, theta: &[f64], c: f64, device: &Device, dtype: DType) -> QnSpsaSample {
        let n = theta.len();
        let mut result = QnSpsaSample {
            gradient: vec![0.; n],
            metric: vec![0.; n * n],
            loss_estimate: 0.,
        };
        let count = (1usize << (2 * n)) as f64;
        for a in directions(n) {
            for b in directions(n) {
                let probe = |u: f64, v: f64| {
                    tensor(
                        &theta
                            .iter()
                            .zip(&a)
                            .zip(&b)
                            .map(|((&x, &a), &b)| x + c * (u * a + v * b))
                            .collect::<Vec<_>>(),
                        device,
                        dtype,
                    )
                };
                let f = |p| self.fidelity(tensor(theta, device, dtype), p).unwrap();
                let s = qn_spsa_sample(
                    c,
                    &a,
                    &b,
                    [
                        self.loss(probe(1., 0.)).unwrap(),
                        self.loss(probe(-1., 0.)).unwrap(),
                    ],
                    [
                        f(probe(1., 0.)),
                        f(probe(-1., 0.)),
                        f(probe(1., 1.)),
                        f(probe(-1., 1.)),
                    ],
                    n * n,
                )
                .unwrap();
                for (avg, x) in result.gradient.iter_mut().zip(s.gradient) {
                    *avg += x / count;
                }
                for (avg, x) in result.metric.iter_mut().zip(s.metric) {
                    *avg += x / count;
                }
                result.loss_estimate += s.loss_estimate / count;
            }
        }
        result
    }
}
fn problem(d: usize) -> Problem {
    let mut circuit = Circuit::new(Register::new(vec![d]).unwrap(), 2, 0);
    if d == 2 {
        circuit
            .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
            .unwrap();
        circuit
            .rotate(0, Rotation::qubit(Axis::Z, Angle::parameter(1)))
            .unwrap();
    } else {
        circuit
            .unitary(vec![0], Unitary::fourier(d).unwrap())
            .unwrap();
        circuit
            .rotate(
                0,
                Rotation {
                    axis: Axis::X,
                    levels: [0, 1],
                    angle: Angle::parameter(0),
                },
            )
            .unwrap();
        circuit
            .rotate(
                0,
                Rotation {
                    axis: Axis::Y,
                    levels: [1, 2],
                    angle: Angle::parameter(1),
                },
            )
            .unwrap();
    }
    Problem {
        circuit,
        diagonal: (0..d)
            .map(|i| 1. - 2. * i as f64 / (d - 1) as f64)
            .collect(),
    }
}
fn tangent_reference(problem: &Problem, theta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let device = Device::flex();
    let state = |x: &[f64]| complex_values(&problem.state(tensor(x, &device, DType::F64)).unwrap());
    let psi = state(theta);
    let derivatives: Vec<Vec<Complex64>> = (0..theta.len())
        .map(|i| {
            let mut plus = theta.to_vec();
            let mut minus = theta.to_vec();
            plus[i] += 1e-5;
            minus[i] -= 1e-5;
            state(&plus)
                .iter()
                .zip(state(&minus))
                .map(|(p, m)| (p - m) / 2e-5)
                .collect()
        })
        .collect();
    let dot = |a: &[Complex64], b: &[Complex64]| {
        a.iter()
            .zip(b)
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>()
    };
    let mut metric = Vec::new();
    for a in &derivatives {
        for b in &derivatives {
            metric.push((dot(a, b) - dot(a, &psi) * dot(&psi, b)).re);
        }
    }
    let gradient = derivatives
        .iter()
        .map(|v| {
            2. * v
                .iter()
                .zip(&psi)
                .zip(&problem.diagonal)
                .map(|((d, p), h)| (p.conj() * d).re * h)
                .sum::<f64>()
        })
        .collect();
    (metric, gradient)
}

#[test]
fn qubit_and_qudit_estimators_match_independent_complex_tangent_geometry() {
    for d in [2, 3] {
        let p = problem(d);
        let theta = [0.6, -0.4];
        let (metric, gradient) = tangent_reference(&p, &theta);
        let estimate = p.average_sample(&theta, 0.002, &Device::flex(), DType::F64);
        close(&estimate.metric, &metric, 8e-6);
        close(&estimate.gradient, &gradient, 8e-6);
        if d == 2 {
            close(
                &metric,
                &[0.25, 0., 0., 0.25 * theta[0].sin().powi(2)],
                1e-10,
            );
        }
    }
}

#[test]
fn selected_backend_fidelity_estimates_match_cpu_and_physical_geometry() {
    for d in [2, 3] {
        let p = problem(d);
        let theta = [0.6, -0.4];
        let reference = p.average_sample(&theta, 0.08, &Device::flex(), DType::F32);
        let actual = p.average_sample(&theta, 0.08, &selected(), DType::F32);
        close(&actual.metric, &reference.metric, 2e-4);
        close(&actual.gradient, &reference.gradient, 5e-5);
        let (metric, gradient) = tangent_reference(&p, &theta);
        close(&actual.metric, &metric, 0.004);
        close(&actual.gradient, &gradient, 0.004);
    }
}

#[test]
fn global_phase_has_zero_metric_and_regularized_updates_stay_finite() {
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    circuit
        .rotate(0, Rotation::qubit(Axis::Z, Angle::parameter(0)))
        .unwrap();
    let p = Problem {
        circuit,
        diagonal: vec![1., -1.],
    };
    let sample = p.average_sample(&[0.7], 0.1, &selected(), DType::F32);
    close(&sample.metric, &[0.], 5e-5);
    let mut optimizer = config().with_initial_metric_scale(0.).init(1).unwrap();
    let step = optimizer
        .step(
            tensor(&[0.7], &selected(), DType::F32),
            &mut |x| p.loss(x),
            &mut |x, y| p.fidelity(x, y),
        )
        .unwrap();
    assert!(values(step.parameters).iter().all(|x| x.is_finite()));
    assert!(optimizer.regularized_metric().unwrap()[0] >= optimizer.config().regularization);
}

#[test]
fn selected_backend_natural_updates_converge_and_match_cpu_trajectory() {
    let mut circuit = Circuit::new(Register::qubits(1).unwrap(), 1, 0);
    circuit
        .rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))
        .unwrap();
    let p = Problem {
        circuit,
        diagonal: vec![1., -1.],
    };
    let train = |device: Device| {
        let mut optimizer = config()
            .with_learning_rate(0.15)
            .with_perturbation(0.1)
            .with_initial_metric_scale(0.25)
            .with_blocking_tolerance(Some(0.))
            .init(1)
            .unwrap();
        let mut theta = tensor(&[0.7], &device, DType::F32);
        let mut curve = Vec::new();
        for _ in 0..25 {
            theta = optimizer
                .step(theta, &mut |x| p.loss(x), &mut |x, y| p.fidelity(x, y))
                .unwrap()
                .parameters;
            curve.push(p.loss(theta.clone()).unwrap());
        }
        assert!(*curve.last().unwrap() < -0.9999);
        curve
    };
    close(&train(selected()), &train(Device::flex()), 2e-4);
}
