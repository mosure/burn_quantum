//! Exact occurrence-wise parameter shifts for supported analytic rotations.
use crate::{Angle, Circuit, DenseSimulator, Error, Observable, Operation, Result};
use burn::tensor::{Device, Tensor};
use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

/// Compute an expectation Jacobian of shape `[batch, num_parameters]`.
///
/// Each gate occurrence is shifted separately before accumulating chain-rule
/// contributions for shared parameters. Qubits use two terms; embedded qudit
/// rotations use four terms for the generator's frequencies 1/2 and 1.
/// This is an analytic simulator reference, not a hardware shot estimator.
pub fn parameter_shift(
    engine: &DenseSimulator,
    circuit: &Circuit,
    parameters: Tensor<1>,
    inputs: Option<Tensor<2>>,
    batch: usize,
    device: &Device,
    observable: &Observable,
) -> Result<Tensor<2>> {
    if circuit.num_parameters() == 0 {
        return Err(Error::Shape(
            "Jacobian requires at least one parameter".into(),
        ));
    }
    let state = circuit.run(
        engine,
        Some(parameters.clone()),
        inputs.clone(),
        batch,
        device,
    )?;
    let base = observable.expectation(engine, &state)? * 0.;
    let mut columns = vec![base; circuit.num_parameters()];
    for (occurrence, op) in circuit.operations().iter().enumerate() {
        let Operation::Rotate { wire, rotation } = op else {
            continue;
        };
        let Angle::Parameter { index, scale, .. } = rotation.angle else {
            continue;
        };
        // f'(x) = 1/2 [f(x+pi/2)-f(x-pi/2)]
        //       + (1-sqrt(2))/4 [f(x+pi)-f(x-pi)] for d > 2.
        let mut recipe = vec![(FRAC_PI_2, 0.5)];
        if circuit.register().dimensions()[*wire] > 2 {
            recipe.push((PI, (1. - SQRT_2) / 4.));
        }
        for (shift, coefficient) in recipe {
            let plus = circuit.run_shifted(
                engine,
                Some(parameters.clone()),
                inputs.clone(),
                batch,
                device,
                Some((occurrence, shift)),
            )?;
            let minus = circuit.run_shifted(
                engine,
                Some(parameters.clone()),
                inputs.clone(),
                batch,
                device,
                Some((occurrence, -shift)),
            )?;
            let difference =
                observable.expectation(engine, &plus)? - observable.expectation(engine, &minus)?;
            columns[index] = columns[index].clone() + difference * (scale * coefficient);
        }
    }
    Ok(Tensor::cat(
        columns.into_iter().map(|c| c.reshape([batch, 1])).collect(),
        1,
    ))
}
