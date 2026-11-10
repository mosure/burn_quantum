use super::*;

#[test]
fn complete_instrument_matches_independent_scalar_partial_inner_products() {
    let dev = device();
    let data: Vec<_> = normalized(81, 11)
        .into_iter()
        .chain(normalized(81, 29))
        .collect();
    let rho = density(data.clone(), 4, 2, &dev);
    let branches = StandardFusion::branches(&DenseSimulator::default(), &rho, [2, 0]).unwrap();
    let mut total = vec![0.; 2];
    for (branch_index, branch) in branches.iter().enumerate() {
        let mut expected = vec![];
        let mut expected_weights = vec![];
        for batch in 0..2 {
            let mut matrix = vec![Complex64::new(0., 0.); 81];
            // Independent bras on ordered sites [2,0]; outputs are sites [1,3].
            let bras: Vec<Vec<(usize, usize, f64)>> = match branch_index {
                0 => vec![vec![
                    (0, 1, std::f64::consts::FRAC_1_SQRT_2),
                    (1, 0, std::f64::consts::FRAC_1_SQRT_2),
                ]],
                1 => vec![vec![
                    (0, 1, std::f64::consts::FRAC_1_SQRT_2),
                    (1, 0, -std::f64::consts::FRAC_1_SQRT_2),
                ]],
                2 => vec![vec![(0, 0, 1.)]],
                3 => vec![vec![(1, 1, 1.)]],
                _ => vec![
                    vec![(0, 2, 1.)],
                    vec![(1, 2, 1.)],
                    vec![(2, 0, 1.)],
                    vec![(2, 1, 1.)],
                    vec![(2, 2, 1.)],
                ],
            };
            for bra in bras {
                let mut vector = [Complex64::new(0., 0.); 9];
                for left in 0..3 {
                    for right in 0..3 {
                        for &(a, b, coefficient) in &bra {
                            vector[left * 3 + right] += data
                                [batch * 81 + ((b * 3 + left) * 3 + a) * 3 + right]
                                * coefficient;
                        }
                    }
                }
                for i in 0..9 {
                    for j in 0..9 {
                        matrix[i * 9 + j] += vector[i] * vector[j].conj();
                    }
                }
            }
            let weight = (0..9).map(|i| matrix[i * 9 + i].re).sum::<f64>();
            total[batch] += weight;
            expected_weights.push(weight);
            expected.extend(matrix);
        }
        close(&values(branch.probability()), &expected_weights, 3e-6);
        let actual = branch.unnormalized_state().unwrap();
        let complex: Vec<_> = values(actual.entries().real())
            .into_iter()
            .zip(values(actual.entries().imag()))
            .map(|(re, im)| Complex64::new(re, im))
            .collect();
        complex_close(&complex, &expected, 3e-6);
        branch
            .conditional_state(1e-8)
            .unwrap()
            .unwrap()
            .validate(2e-5)
            .unwrap();
    }
    close(&total, &[1., 1.], 1e-12);
}
#[test]
fn bell_success_is_state_dependent_and_failures_retain_only_known_parities() {
    let dev = device();
    let mut psi = vec![Complex64::new(0., 0.); 9];
    psi[1] = std::f64::consts::FRAC_1_SQRT_2.into();
    psi[3] = std::f64::consts::FRAC_1_SQRT_2.into();
    let branches = StandardFusion::branches(
        &DenseSimulator::default(),
        &density(psi, 2, 1, &dev),
        [0, 1],
    )
    .unwrap();
    close(
        &branches
            .iter()
            .map(|b| values(b.probability())[0])
            .collect::<Vec<_>>(),
        &[1., 0., 0., 0., 0.],
        2e-6,
    );
    assert!(branches[0].conditional_state(1e-8).unwrap().is_none());
    assert!(branches[1].conditional_state(1e-8).is_err());
    let parities = branches[2].outcomes()[0].parities();
    assert_eq!(parities.xx, None);
    assert_eq!(parities.zz, Some(Parity::Even));
    assert_eq!(
        FusionOutcome::Loss.parities(),
        FusionParities { xx: None, zz: None }
    );
    assert!(
        StandardFusion::branches(
            &DenseSimulator::default(),
            &density(vec![1.0.into(), 0.0.into(), 0.0.into()], 1, 1, &dev),
            [0, 0]
        )
        .is_err()
    );
}
#[test]
fn fusion_branch_weights_and_conditional_states_retain_native_gradients() {
    let dev = device().autodiff();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(DualRail::register(3).unwrap(), 1, 0);
    circuit
        .rotate(
            0,
            Rotation::qubit(burn_quantum::gate::Axis::Y, Angle::parameter(0)),
        )
        .unwrap();
    let theta = Tensor::from_data([0.7f32], &dev).require_grad();
    let state = circuit
        .run(&engine, Some(theta.clone()), None, 1, &dev)
        .unwrap();
    let rho = DensityMatrix::from_pure(&state, engine.limits).unwrap();
    let branches = StandardFusion::branches(&engine, &rho, [0, 1]).unwrap();
    let success = branches[0].probability() + branches[1].probability();
    close(
        &values(success.clone()),
        &[(0.7f64 / 2.).sin().powi(2)],
        2e-6,
    );
    let grads = success.sum().backward();
    close(
        &values(theta.grad(&grads).unwrap()),
        &[0.5 * 0.7f64.sin()],
        2e-5,
    );
    let normalized = branches[0].conditional_state(1e-8).unwrap().unwrap();
    close(&values(normalized.probabilities()), &[1., 0., 0.], 2e-6);
}

#[test]
fn conditional_objectives_differentiate_the_success_denominator() {
    let dev = device().autodiff();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(DualRail::register(3).unwrap(), 1, 0);
    circuit
        .rotate(
            0,
            Rotation::qubit(burn_quantum::gate::Axis::Y, Angle::parameter(0)),
        )
        .unwrap();
    circuit
        .rotate(
            1,
            Rotation::qubit(burn_quantum::gate::Axis::Y, Angle::Constant(1.1)),
        )
        .unwrap();
    // Ideal entangled input preparation, not an asserted deterministic optical CNOT.
    circuit
        .unitary(
            vec![0, 2],
            DualRail::unitary(&Unitary::x())
                .unwrap()
                .controlled(3, 1)
                .unwrap(),
        )
        .unwrap();
    let theta = Tensor::from_data([0.7f32], &dev).require_grad();
    let state = circuit
        .run(&engine, Some(theta.clone()), None, 1, &dev)
        .unwrap();
    let rho = DensityMatrix::from_pure(&state, engine.limits).unwrap();
    let branches = StandardFusion::branches(&engine, &rho, [0, 1]).unwrap();
    let conditional = branches[0].conditional_state(1e-8).unwrap().unwrap();
    let z = conditional.expectation(&engine, &DualRail::z(0)).unwrap();
    // The surviving vector is proportional to c_theta s_gamma |0> + s_theta c_gamma |1>.
    let a = (0.7f64 / 2.).cos().powi(2) * (1.1f64 / 2.).sin().powi(2);
    let b = (0.7f64 / 2.).sin().powi(2) * (1.1f64 / 2.).cos().powi(2);
    close(&values(branches[0].probability()), &[(a + b) / 2.], 2e-6);
    close(&values(z.clone()), &[(a - b) / (a + b)], 2e-6);
    let derivative =
        -0.7f64.sin() * (1.1f64 / 2.).sin().powi(2) * (1.1f64 / 2.).cos().powi(2) / (a + b).powi(2);
    let gradients = z.sum().backward();
    close(
        &values(theta.grad(&gradients).unwrap()),
        &[derivative],
        3e-5,
    );
}

#[test]
fn branch_weighted_learning_retains_success_and_zero_success_policy() {
    use burn_quantum::photonic::learning::heralded_readout;
    let dev = device().autodiff();
    let engine = DenseSimulator::default();
    let mut circuit = Circuit::new(DualRail::register(3).unwrap(), 1, 0);
    circuit
        .rotate(
            0,
            Rotation::qubit(burn_quantum::gate::Axis::Y, Angle::parameter(0)),
        )
        .unwrap();
    let theta = Tensor::from_data([0.7f32], &dev).require_grad();
    let state = circuit
        .run(&engine, Some(theta.clone()), None, 1, &dev)
        .unwrap();
    let rho = DensityMatrix::from_pure(&state, engine.limits).unwrap();
    let branches = StandardFusion::branches(&engine, &rho, [0, 1]).unwrap();
    let readout = heralded_readout(&branches, &engine, &DualRail::z(0), |r| {
        matches!(r[0], FusionOutcome::Success { .. })
    })
    .unwrap();
    close(
        &values(readout.probability()),
        &[(0.7f64 / 2.).sin().powi(2)],
        2e-6,
    );
    close(&values(readout.conditional(1e-8).unwrap()), &[1.], 2e-6);
    let gradients = readout.unconditional().sum().backward();
    close(
        &values(theta.grad(&gradients).unwrap()),
        &[0.5 * 0.7f64.sin()],
        3e-5,
    );
    let rejected = heralded_readout(&branches, &engine, &DualRail::z(0), |_| false).unwrap();
    close(&values(rejected.probability()), &[0.], 1e-6);
    assert!(rejected.conditional(1e-8).is_err());
}
