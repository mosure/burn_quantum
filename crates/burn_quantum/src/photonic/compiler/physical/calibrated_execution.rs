//! General complete Bell POVMs contracted with conditional noisy resource states.
use super::execution::{adjoint, apply_tensor, embed, encode_state, mm};
use super::*;
use crate::photonic::{DualRail, Parity, optics::BellRecord};
use crate::{ComplexTensor, DenseSimulator, DensityMatrix, Matrix, Register, StateVector, Unitary};
use burn::tensor::Tensor;
use nalgebra::{DMatrix, linalg::SymmetricEigen};
use num_complex::Complex64 as C;
/// Marginal declared detector record conditional on reaching a physical gate.
/// A `Resolved` declaration under noisy calibration is not a fidelity certificate.
#[derive(Clone, Debug)]
pub struct CalibratedOutcome {
    /// Synthesized gate index.
    pub gate: usize,
    /// Complete declared records in fusion/input order.
    pub records: Vec<BellRecord>,
    /// Absolute probability including previous success and this resource's generation yield.
    pub probability: Tensor<1>,
}
struct EffectRows {
    record: BellRecord,
    rows: Vec<Vec<C>>,
}
impl PhysicalCircuit {
    pub(super) fn execute_calibrated(
        &self,
        engine: &DenseSimulator,
        input: StateVector,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
    ) -> Result<PhysicalRun> {
        let model = self.calibration.as_ref().unwrap();
        if input.register() != self.source.register() {
            return Err(Error::Shape(
                "physical circuit and input register differ".into(),
            ));
        }
        // Reuse the native encoder/binding preflight without physical operations.
        let mut initial = self.clone();
        initial.gates.clear();
        initial.finish = 0;
        initial.calibration = None;
        initial.final_storage.fill(RailTransmission::IDEAL);
        let mut result = initial.execute(engine, input, parameters.clone(), inputs.clone())?;
        let device = result.state.entries().device();
        let dtype = result.state.entries().dtype();
        let batch = result.state.batch_size();
        let effects: Vec<_> = model
            .analyzer()
            .effects()
            .iter()
            .map(|effect| {
                let eigen =
                    SymmetricEigen::new(DMatrix::from_row_slice(9, 9, effect.matrix().entries()));
                let rows = (0..9)
                    .filter(|&i| eigen.eigenvalues[i] > 0.)
                    .map(|i| {
                        (0..9)
                            .map(|j| {
                                eigen.eigenvectors[(j, i)].conj() * eigen.eigenvalues[i].sqrt()
                            })
                            .collect()
                    })
                    .collect();
                EffectRows {
                    record: effect.record(),
                    rows,
                }
            })
            .collect();
        let ranks = effects.iter().map(|e| e.rows.len()).sum::<usize>();
        for gate in &self.gates {
            let k = gate.wires.len();
            let source = model.resource(k)?;
            let noise = source.noise().map_or(1, |n| n.operators().len());
            let loss_terms = gate
                .resource_measured
                .iter()
                .filter(|&&t| t != RailTransmission::IDEAL)
                .count();
            let terms = noise
                .checked_mul(3usize.pow(loss_terms as u32))
                .and_then(|v| v.checked_mul(ranks.pow(k as u32)))
                .ok_or(Error::Overflow)?;
            if terms > model.max_terms() {
                return Err(Error::ResourceLimit {
                    resource: "calibrated contraction Kraus terms",
                    required: terms,
                    limit: model.max_terms(),
                });
            }
            let n = 3usize.pow(k as u32);
            engine.limits.check_state(batch, n * n)?;
            engine.limits.check_matrix(batch, n)?;
            engine.limits.check_plan(
                result
                    .state
                    .register()
                    .size()
                    .checked_mul(result.state.register().size())
                    .ok_or(Error::Overflow)?,
            )?;
            if source.noise().is_some() {
                engine.limits.check_matrix(batch, n * n)?;
            }
        }
        for (index, gate) in self.gates.iter().enumerate() {
            let k = gate.wires.len();
            let d = 1usize << k;
            let n = 3usize.pow(k as u32);
            let source = model.resource(k)?;
            let reached = result.state.trace();
            let yield_probability = source.factory().yield_probability();
            result.photons = result.photons + reached.clone() * source.factory().expected_photons();
            let exhausted = reached * (1. - yield_probability);
            result.source_abort = result.source_abort + exhausted.clone();
            result.ticks = result.ticks + exhausted.clone() * gate.start as f64;
            result.abort = result.abort + exhausted;
            result.state = scale(&result.state, yield_probability, engine)?;
            result.fusions = result.fusions + result.state.trace() * k as f64;
            result.ancillas = result.ancillas
                + result.state.trace() * (k * model.analyzer().supplied_ancillas()) as f64;
            for (&wire, &loss) in gate.wires.iter().zip(&gate.input) {
                if loss != RailTransmission::IDEAL {
                    result.state = result.state.channel(engine, &[wire], &loss.channel()?)?;
                }
            }
            let reduced = result.state.reduced(&gate.wires, engine.limits)?;
            let mut preparation = Circuit::new(
                Register::qubits(2 * k)?,
                self.source.num_parameters(),
                self.source.num_inputs(),
            );
            for i in 0..k {
                preparation.unitary(vec![i], Unitary::h())?;
                preparation.unitary(vec![i, k + i], Unitary::cnot())?;
            }
            match &gate.operation {
                Operation::Unitary { gate, .. } => {
                    preparation.unitary((k..2 * k).collect(), gate.clone())?;
                }
                Operation::Rotate { rotation, .. } => {
                    preparation.rotate(k, rotation.clone())?;
                }
                _ => unreachable!(),
            }
            let zero = StateVector::zero(
                preparation.register().clone(),
                batch,
                &device,
                dtype,
                engine.limits,
            )?;
            let choi = preparation.run_from(engine, zero, parameters.clone(), inputs.clone())?;
            let u = ComplexTensor::new(
                choi.amplitudes().real().reshape([batch, d, d]).transpose() * (d as f64).sqrt(),
                choi.amplitudes().imag().reshape([batch, d, d]).transpose() * (d as f64).sqrt(),
            )?;
            let encoded = encode_state(engine, choi)?;
            let mut ensemble = if let Some(noise) = source.noise() {
                noise
                    .operators()
                    .iter()
                    .map(|op| {
                        engine.apply_tensor(
                            &encoded,
                            &(0..2 * k).collect::<Vec<_>>(),
                            &op.to_tensor(&device, dtype)?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![encoded]
            };
            let measured_layout = DualRail::register(k)?;
            let code_support = (0..n)
                .map(|m| Ok(!measured_layout.decode(m)?.contains(&2)))
                .collect::<Result<Vec<_>>>()?;
            let mut supports =
                vec![source.noise().is_none().then_some(code_support); ensemble.len()];
            for (wire, &loss) in gate.resource_measured.iter().enumerate() {
                if loss != RailTransmission::IDEAL {
                    let mut next = vec![];
                    let mut next_supports = vec![];
                    for (state, support) in ensemble.iter().zip(&supports) {
                        for (operator, op) in loss.channel()?.operators().iter().enumerate() {
                            next_supports.push(if let Some(support) = support {
                                if operator == 0 {
                                    Some(support.clone())
                                } else {
                                    let mut mask = vec![false; n];
                                    for (m, &possible) in support.iter().enumerate() {
                                        let mut digits = measured_layout.decode(m)?;
                                        if possible && digits[wire] == operator - 1 {
                                            digits[wire] = 2;
                                            mask[measured_layout.encode(&digits)?] = true;
                                        }
                                    }
                                    Some(mask)
                                }
                            } else {
                                None
                            });
                            next.push(engine.apply_tensor(
                                state,
                                &[wire],
                                &op.to_tensor(&device, dtype)?,
                            )?);
                        }
                    }
                    ensemble = next;
                    supports = next_supports;
                }
            }
            let resources: Vec<_> = ensemble
                .into_iter()
                .map(|s| {
                    ComplexTensor::new(
                        s.amplitudes().real().reshape([batch, n, n]).transpose(),
                        s.amplitudes().imag().reshape([batch, n, n]).transpose(),
                    )
                })
                .collect::<Result<_>>()?;
            let resources: Vec<_> = resources.into_iter().zip(supports).collect();
            let mut successful: Option<DensityMatrix> = None;
            for code in 0..effects.len().pow(k as u32) {
                let mut v = code;
                let mut selected = vec![0; k];
                for i in (0..k).rev() {
                    selected[i] = v % effects.len();
                    v /= effects.len();
                }
                let records: Vec<_> = selected.iter().map(|&i| effects[i].record).collect();
                let resolved = records
                    .iter()
                    .all(|r| matches!(r, BellRecord::Resolved { .. }));
                if !resolved && source.noise().is_none() {
                    // The measured marginal of a unitary Choi resource is I/2^k,
                    // regardless of its parameter-bound output unitary. Contract
                    // it with the complete detector effect on the host; only the
                    // input density carries gradients in this rejected branch.
                    let effect =
                        rejected_effect(gate, model, &selected)?.to_tensor(&device, dtype)?;
                    let probability = trace_effect(&reduced, &effect);
                    result.abort = result.abort + probability.clone();
                    result.ticks = result.ticks + probability.clone() * gate.finish as f64;
                    result.optical_records.push(CalibratedOutcome {
                        gate: index,
                        records,
                        probability,
                    });
                    continue;
                }
                let correction = if resolved {
                    Some(correct(gate, &u, &records)?)
                } else {
                    None
                };
                let mut probability = Tensor::zeros([batch], (&device, dtype));
                let count = selected
                    .iter()
                    .map(|&i| effects[i].rows.len())
                    .product::<usize>();
                for row_code in 0..count {
                    let mut v = row_code;
                    let mut rows = vec![0; k];
                    for i in (0..k).rev() {
                        rows[i] = v % effects[selected[i]].rows.len();
                        v /= effects[selected[i]].rows.len();
                    }
                    let layout = DualRail::register(k)?;
                    let mut entries = vec![C::new(0., 0.); n * n];
                    for m in 0..n {
                        let md = layout.decode(m)?;
                        for i in 0..n {
                            let id = layout.decode(i)?;
                            entries[m * n + i] = (0..k)
                                .map(|j| effects[selected[j]].rows[rows[j]][id[j] * 3 + md[j]])
                                .product();
                        }
                    }
                    let active: Vec<_> = entries
                        .chunks(n)
                        .map(|row| row.iter().any(|v| v.norm_sqr() != 0.))
                        .collect();
                    let contraction = Matrix::new(n, entries)?.to_tensor(&device, dtype)?;
                    for (resource, support) in &resources {
                        if support
                            .as_ref()
                            .is_some_and(|mask| !mask.iter().zip(&active).any(|(&a, &b)| a && b))
                        {
                            continue;
                        }
                        let op = mm(resource, &contraction)?;
                        if let Some(correction) = &correction {
                            let branch = if gate
                                .resource_outputs
                                .iter()
                                .all(|&loss| loss == RailTransmission::IDEAL)
                            {
                                // (C K) rho (C K)^dagger is exactly the same channel;
                                // avoid two complete density evolutions when there
                                // is no intervening output-storage channel.
                                apply_tensor(
                                    &result.state,
                                    engine,
                                    &gate.wires,
                                    &mm(correction, &op)?,
                                )?
                            } else {
                                let mut branch =
                                    apply_tensor(&result.state, engine, &gate.wires, &op)?;
                                for (&wire, &loss) in gate.wires.iter().zip(&gate.resource_outputs)
                                {
                                    if loss != RailTransmission::IDEAL {
                                        branch =
                                            branch.channel(engine, &[wire], &loss.channel()?)?;
                                    }
                                }
                                apply_tensor(&branch, engine, &gate.wires, correction)?
                            };
                            probability = probability + branch.trace();
                            successful = Some(add(successful, branch, engine)?);
                        } else {
                            let effect = mm(&adjoint(&op)?, &op)?;
                            probability = probability + trace_effect(&reduced, &effect);
                        }
                    }
                }
                if !resolved {
                    result.abort = result.abort + probability.clone();
                    result.ticks = result.ticks + probability.clone() * gate.finish as f64;
                }
                result.optical_records.push(CalibratedOutcome {
                    gate: index,
                    records,
                    probability,
                });
            }
            result.state = match successful {
                Some(state) => state,
                None => scale(&result.state, 0., engine)?,
            };
        }
        for (wire, &loss) in self.final_storage.iter().enumerate() {
            if loss != RailTransmission::IDEAL {
                result.state = result.state.channel(engine, &[wire], &loss.channel()?)?;
            }
        }
        result.ticks = result.ticks + result.state.trace() * self.finish as f64;
        Ok(result)
    }
}
fn trace_effect(state: &DensityMatrix, effect: &ComplexTensor<3>) -> Tensor<1> {
    let product = state.entries().real() * effect.real().transpose()
        - state.entries().imag() * effect.imag().transpose();
    product.sum_dim(2).sum_dim(1).reshape([state.batch_size()])
}
/// Partial detector-effect contraction for an ideal emitted Choi state. Loss on
/// each measured resource half maps its I/2 code marginal to a diagonal qutrit
/// marginal. Output unitaries cancel under the trace; input loss remains in rho.
fn rejected_effect(
    gate: &PhysicalGate,
    model: &PhysicalCalibration,
    selected: &[usize],
) -> Result<Matrix> {
    let local: Vec<Vec<C>> = selected
        .iter()
        .zip(&gate.resource_measured)
        .map(|(&record, loss)| {
            let [a, b] = loss.values();
            let weights = [a / 2., b / 2., 1. - (a + b) / 2.];
            let effect = model.analyzer().effects()[record].matrix().entries();
            (0..9)
                .map(|i| {
                    let row = i / 3;
                    let col = i % 3;
                    (0..3)
                        .map(|m| effect[(row * 3 + m) * 9 + col * 3 + m] * weights[m])
                        .sum()
                })
                .collect()
        })
        .collect();
    let register = DualRail::register(selected.len())?;
    let n = register.size();
    let mut entries = Vec::with_capacity(n * n);
    for row in 0..n {
        let a = register.decode(row)?;
        for col in 0..n {
            let b = register.decode(col)?;
            entries.push(
                (0..selected.len())
                    .map(|i| local[i][a[i] * 3 + b[i]])
                    .product(),
            );
        }
    }
    Matrix::new(n, entries)
}
fn scale(state: &DensityMatrix, weight: f64, engine: &DenseSimulator) -> Result<DensityMatrix> {
    DensityMatrix::from_entries(
        state.register().clone(),
        ComplexTensor::new(
            state.entries().real() * weight,
            state.entries().imag() * weight,
        )?,
        engine.limits,
    )
}
fn add(
    previous: Option<DensityMatrix>,
    term: DensityMatrix,
    engine: &DenseSimulator,
) -> Result<DensityMatrix> {
    if let Some(p) = previous {
        DensityMatrix::from_entries(
            term.register().clone(),
            ComplexTensor::new(
                p.entries().real() + term.entries().real(),
                p.entries().imag() + term.entries().imag(),
            )?,
            engine.limits,
        )
    } else {
        Ok(term)
    }
}
fn pauli(x: bool, z: bool) -> Result<Unitary> {
    Ok(match (x, z) {
        (true, true) => Unitary::y(),
        (true, false) => Unitary::x(),
        (false, true) => Unitary::z(),
        _ => Unitary::identity(2)?,
    })
}
fn correct(
    gate: &PhysicalGate,
    u: &ComplexTensor<3>,
    records: &[BellRecord],
) -> Result<ComplexTensor<3>> {
    let mut x = vec![];
    let mut z = vec![];
    for record in records {
        let BellRecord::Resolved { xx, zz } = record else {
            unreachable!()
        };
        x.push(*zz == Parity::Odd);
        z.push(*xx == Parity::Odd);
    }
    let device = u.device();
    let dtype = u.dtype();
    if gate.wires.len() == 1 {
        let p = pauli(x[0], z[0])?.matrix().to_tensor(&device, dtype)?;
        return embed(&mm(&mm(u, &p)?, &adjoint(u)?)?, 1, true);
    }
    let Operation::Unitary { gate, .. } = &gate.operation else {
        unreachable!()
    };
    if gate == &Unitary::cnot() {
        x[1] ^= x[0];
        z[0] ^= z[1];
    } else if gate == &Unitary::swap(2)? {
        x.swap(0, 1);
        z.swap(0, 1);
    } else {
        z[0] ^= x[1];
        z[1] ^= x[0];
    }
    let a = DualRail::unitary(&pauli(x[0], z[0])?)?;
    let b = DualRail::unitary(&pauli(x[1], z[1])?)?;
    Matrix::new(
        9,
        (0..81)
            .map(|i| {
                a.matrix().entries()[(i / 9 / 3) * 3 + (i % 9 / 3)]
                    * b.matrix().entries()[(i / 9 % 3) * 3 + (i % 3)]
            })
            .collect(),
    )?
    .to_tensor(&device, dtype)
}
