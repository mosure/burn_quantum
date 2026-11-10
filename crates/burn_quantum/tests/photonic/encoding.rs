use super::*;
use burn_quantum::gate::Axis;

#[test]
fn asymmetric_loss_retains_coherences_and_vacuum_without_postselection() {
    let dev = device();
    let engine = DenseSimulator::default();
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let rho = density(
        vec![s.into(), Complex64::new(0., s), 0.0.into()],
        1,
        1,
        &dev,
    );
    let eta = RailTransmission::new(0.8, 0.2).unwrap();
    let out = rho.channel(&engine, &[0], &eta.channel().unwrap()).unwrap();
    close(&values(out.probabilities()), &[0.4, 0.1, 0.5], 2e-6);
    close(
        &values(out.entries().imag()),
        &[0., -0.2, 0., 0.2, 0., 0., 0., 0., 0.],
        2e-6,
    );
    close(&values(out.trace()), &[1.], 2e-6);
    out.validate(1e-5).unwrap();
    let sequential = out.channel(&engine, &[0], &eta.channel().unwrap()).unwrap();
    let composed = rho
        .channel(&engine, &[0], &eta.then(eta).channel().unwrap())
        .unwrap();
    close(
        &values(sequential.entries().real()),
        &values(composed.entries().real()),
        2e-6,
    );
    close(
        &values(sequential.entries().imag()),
        &values(composed.entries().imag()),
        2e-6,
    );
    let vacuum = density(vec![0.0.into(), 0.0.into(), 1.0.into()], 1, 1, &dev);
    close(
        &values(
            vacuum
                .channel(
                    &engine,
                    &[0],
                    &RailTransmission::uniform(0.).unwrap().channel().unwrap(),
                )
                .unwrap()
                .probabilities(),
        ),
        &[0., 0., 1.],
        1e-6,
    );
    assert!(RailTransmission::new(f64::NAN, 1.).is_err());
    assert!(RailTransmission::uniform(-0.01).is_err());
    assert!(serde_json::from_str::<RailTransmission>("[1.1,0.5]").is_err());
    assert_eq!(
        RailTransmission::uniform(0.).unwrap().over_ticks(0),
        RailTransmission::IDEAL
    );
    close(
        &RailTransmission::uniform(0.5)
            .unwrap()
            .over_ticks(10)
            .values(),
        &[1. / 1024.; 2],
        1e-12,
    );
}
#[test]
fn local_frontend_preserves_bindings_gradients_and_rejects_entangling_operations() {
    let dev = device().autodiff();
    let mut logical = Circuit::new(Register::qubits(1).unwrap(), 1, 1);
    logical
        .rotate(0, Rotation::qubit(Axis::Y, Angle::input(0)))
        .unwrap();
    logical
        .rotate(
            0,
            Rotation::qubit(
                Axis::Y,
                Angle::Parameter {
                    index: 0,
                    scale: 2.,
                    offset: 0.1,
                },
            ),
        )
        .unwrap();
    let encoded = DualRail::encode_circuit(&logical).unwrap();
    assert_eq!(encoded.num_parameters(), 1);
    assert_eq!(encoded.num_inputs(), 1);
    let theta = Tensor::from_data([0.2f32], &dev).require_grad();
    let inputs = Tensor::from_data([[0.3f32], [0.5]], &dev).require_grad();
    let engine = DenseSimulator::default();
    let state = encoded
        .run(&engine, Some(theta.clone()), Some(inputs.clone()), 2, &dev)
        .unwrap();
    let rho = DensityMatrix::from_pure(&state, engine.limits)
        .unwrap()
        .channel(
            &engine,
            &[0],
            &RailTransmission::uniform(0.7).unwrap().channel().unwrap(),
        )
        .unwrap();
    let z = rho.expectation(&engine, &DualRail::z(0)).unwrap();
    close(
        &values(z.clone()),
        &[0.7 * 0.8f64.cos(), 0.7 * 1.0f64.cos()],
        2e-5,
    );
    let grads = z.sum().backward();
    close(
        &values(theta.grad(&grads).unwrap()),
        &[-1.4 * (0.8f64.sin() + 1.0f64.sin())],
        5e-5,
    );
    close(
        &values(inputs.grad(&grads).unwrap()),
        &[-0.7 * 0.8f64.sin(), -0.7 * 1.0f64.sin()],
        5e-5,
    );
    let mut bad = Circuit::new(Register::qubits(2).unwrap(), 0, 0);
    bad.unitary(vec![0, 1], Unitary::cnot()).unwrap();
    assert!(matches!(
        DualRail::encode_circuit(&bad),
        Err(Error::Unsupported(_))
    ));
    assert!(DualRail::register(usize::MAX).is_err());
    assert!(DualRail::register(0).is_err());
}
#[test]
fn resource_graphs_match_scalar_graph_amplitudes_and_roundtrip() {
    let resource = ResourceState::six_ring();
    let dev = device();
    let engine = DenseSimulator::default();
    let state = resource
        .circuit()
        .unwrap()
        .run(&engine, None, None, 1, &dev)
        .unwrap();
    let re = values(state.amplitudes().real());
    let im = values(state.amplitudes().imag());
    for (i, (&a, &b)) in re.iter().zip(&im).enumerate() {
        let digits = state.register().decode(i).unwrap();
        let expected = if digits.contains(&2) {
            0.
        } else {
            let odd = resource
                .edges()
                .iter()
                .filter(|&&(x, y)| digits[x] == 1 && digits[y] == 1)
                .count()
                % 2
                == 1;
            if odd { -0.125 } else { 0.125 }
        };
        assert!((a - expected).abs() < 2e-6 && b.abs() < 2e-6);
    }
    let text = serde_json::to_string(&resource).unwrap();
    assert_eq!(
        serde_json::from_str::<ResourceState>(&text).unwrap(),
        resource
    );
    assert!(ResourceState::graph(2, vec![(0, 1), (1, 0)]).is_err());
    assert!(ResourceState::graph(2, vec![(0, 0)]).is_err());
    assert!(ResourceState::graph(usize::MAX, vec![]).is_err());
}
