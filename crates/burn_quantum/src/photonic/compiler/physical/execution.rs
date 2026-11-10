use super::*;
use crate::photonic::{DualRail, FusionOutcome, Parity};
use crate::{
    ComplexTensor, DenseSimulator, DensityMatrix, Matrix, Observable, Register, StateVector,
    Unitary,
};
use burn::tensor::{Int, Tensor, TensorData};

/// Marginal physical detector record at a gate reached after all earlier successes.
#[derive(Clone, Debug)]
pub struct PhysicalOutcome {
    /// Index into the compiled physical gate list.
    pub gate: usize,
    /// Joint standard-fusion observations in local input order.
    pub outcomes: Vec<FusionOutcome>,
    /// Absolute probability including the probability of reaching this gate.
    pub probability: Tensor<1>,
}
/// Exact abort-on-observed-failure result, marginalizing corrected success histories.
///
/// This retains marginal detector statistics rather than a full joint transcript.
/// Corrections are applied before histories are merged; unequal loss can leave a
/// mixture. The remaining state includes output vacuum and its trace is the
/// probability of all successful heralds, not the useful logical-output probability.
#[derive(Clone, Debug)]
pub struct PhysicalRun {
    pub(super) state: DensityMatrix,
    pub(super) records: Vec<PhysicalOutcome>,
    pub(super) abort: Tensor<1>,
    pub(super) photons: Tensor<1>,
    pub(super) fusions: Tensor<1>,
    pub(super) optical_records: Vec<super::calibrated_execution::CalibratedOutcome>,
    pub(super) source_abort: Tensor<1>,
    pub(super) ancillas: Tensor<1>,
    pub(super) ticks: Tensor<1>,
}
impl PhysicalRun {
    /// Subnormalized all-heralds-success state in original logical wire order.
    pub fn state(&self) -> &DensityMatrix {
        &self.state
    }
    /// Marginal joint detector records per gate, including failure and loss.
    pub fn outcomes(&self) -> &[PhysicalOutcome] {
        &self.records
    }
    /// Absolute probability that every fusion reported success.
    pub fn herald_probability(&self) -> Tensor<1> {
        self.state.trace()
    }
    /// Probability that execution terminated after an observed failure/loss.
    pub fn abort_probability(&self) -> Tensor<1> {
        self.abort.clone()
    }
    /// Expected newly supplied resource photons per attempt, accounting for aborts.
    pub fn expected_resource_photons(&self) -> Tensor<1> {
        self.photons.clone()
    }
    /// Expected attempted fusions, accounting for gate-level aborts.
    pub fn expected_fusions(&self) -> Tensor<1> {
        self.fusions.clone()
    }
    /// Expected duration until success or observed abort, including source windows/feedback.
    pub fn expected_ticks(&self) -> Tensor<1> {
        self.ticks.clone()
    }
    /// Calibrated declared optical records, including false heralds and rejected outcomes.
    pub fn optical_outcomes(&self) -> &[super::calibrated_execution::CalibratedOutcome] {
        &self.optical_records
    }
    /// Probability execution aborted because a bounded source failed to supply a resource.
    pub fn source_abort_probability(&self) -> Tensor<1> {
        self.source_abort.clone()
    }
    /// Expected supplied analyzer ancillas, separate from gate-resource generation photons.
    pub fn expected_analyzer_ancillas(&self) -> Tensor<1> {
        self.ancillas.clone()
    }
    /// Probability of successful heralds and one surviving photon at every output.
    pub fn logical_survival_probability(&self) -> Result<Tensor<1>> {
        let register = self.state.register();
        let indices: Vec<i64> = (0..register.size())
            .filter(|&i| !register.decode(i).unwrap().contains(&2))
            .map(|i| i as i64)
            .collect();
        let count = indices.len();
        let idx = Tensor::<1, Int>::from_data(
            TensorData::new(indices, [count]),
            &self.state.entries().device(),
        );
        Ok(self
            .state
            .probabilities()
            .select(1, idx)
            .sum_dim(1)
            .reshape([self.state.batch_size()]))
    }
}
impl PhysicalCircuit {
    /// Execute the exact physical channel on injected pure qubit inputs.
    ///
    /// Each gate prepares its parameter-bound ideal Choi resource. Successful
    /// records are corrected then marginalized; all other records terminate the
    /// attempt after that gate. Output loss is latent until a later detector or
    /// final readout: it never triggers unphysical loss-aware adaptive control.
    pub fn execute(
        &self,
        engine: &DenseSimulator,
        input: StateVector,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
    ) -> Result<PhysicalRun> {
        if self.calibration.is_some() {
            return self.execute_calibrated(engine, input, parameters, inputs);
        }
        if input.register() != self.source.register() {
            return Err(Error::Shape(
                "physical circuit and input register differ".into(),
            ));
        }
        let validator = Circuit::new(
            input.register().clone(),
            self.source.num_parameters(),
            self.source.num_inputs(),
        );
        let input = validator.run_from(engine, input, parameters.clone(), inputs.clone())?;
        let register = DualRail::register(input.register().wires())?;
        let n = register.size();
        engine
            .limits
            .check_state(input.batch_size(), n.checked_mul(n).ok_or(Error::Overflow)?)?;
        engine.limits.check_plan(n)?;
        if !self.gates.is_empty() {
            engine
                .limits
                .check_plan(n.checked_mul(n).ok_or(Error::Overflow)?)?;
            for gate in &self.gates {
                engine
                    .limits
                    .check_matrix(input.batch_size(), 3usize.pow(gate.wires.len() as u32))?;
            }
        }
        let encoded = encode_state(engine, input)?;
        let device = encoded.device();
        let dtype = encoded.amplitudes().dtype();
        let batch = encoded.batch_size();
        let mut state = DensityMatrix::from_pure(&encoded, engine.limits)?;
        let mut records = vec![];
        let mut abort = Tensor::zeros([batch], (&device, dtype));
        let mut photons = abort.clone();
        let mut fusions = abort.clone();
        let mut ticks = abort.clone();
        for (index, gate) in self.gates.iter().enumerate() {
            let k = gate.wires.len();
            let d = 1usize << k;
            photons = photons + state.trace() * (2 * k) as f64;
            fusions = fusions + state.trace() * k as f64;
            let mut resource = Circuit::new(
                Register::qubits(2 * k)?,
                self.source.num_parameters(),
                self.source.num_inputs(),
            );
            for i in 0..k {
                resource.unitary(vec![i], Unitary::h())?;
                resource.unitary(vec![i, k + i], Unitary::cnot())?;
            }
            match &gate.operation {
                Operation::Unitary { gate, .. } => {
                    resource.unitary((k..2 * k).collect(), gate.clone())?;
                }
                Operation::Rotate { rotation, .. } => {
                    resource.rotate(k, rotation.clone())?;
                }
                _ => unreachable!(),
            }
            let zero = StateVector::zero(
                resource.register().clone(),
                batch,
                &device,
                dtype,
                engine.limits,
            )?;
            let choi = resource.run_from(engine, zero, parameters.clone(), inputs.clone())?;
            let u = ComplexTensor::new(
                choi.amplitudes().real().reshape([batch, d, d]).transpose() * (d as f64).sqrt(),
                choi.amplitudes().imag().reshape([batch, d, d]).transpose() * (d as f64).sqrt(),
            )?;
            let embedded = embed(&u, k, false)?;
            let local = 3usize.pow(k as u32);
            let mut accumulated: Option<DensityMatrix> = None;
            for code in 0..5usize.pow(k as u32) {
                let outcomes = decode_outcomes(code, k);
                let success = outcomes
                    .iter()
                    .all(|x| matches!(x, FusionOutcome::Success { .. }));
                let probability;
                if success {
                    let right = right_factor(gate, &outcomes)?.to_tensor(&device, dtype)?;
                    let operator = mm(&embedded, &right)?;
                    let mut result = apply_tensor(&state, engine, &gate.wires, &operator)?;
                    for (&wire, &loss) in gate.wires.iter().zip(&gate.resource_outputs) {
                        if loss != RailTransmission::IDEAL {
                            result = result.channel(engine, &[wire], &loss.channel()?)?;
                        }
                    }
                    let correction = if k == 1 {
                        let p =
                            if matches!(outcomes[0], FusionOutcome::Success { xx: Parity::Even }) {
                                Unitary::x()
                            } else {
                                Unitary::y()
                            };
                        let pt = p.matrix().to_tensor(&device, dtype)?;
                        let r = mm(&mm(&u, &pt)?, &adjoint(&u)?)?;
                        embed(&r, 1, true)?
                    } else {
                        two_qubit_correction(gate, &outcomes)?
                            .matrix()
                            .to_tensor(&device, dtype)?
                    };
                    result = apply_tensor(&result, engine, &gate.wires, &correction)?;
                    probability = result.trace();
                    accumulated = Some(if let Some(previous) = accumulated {
                        DensityMatrix::from_entries(
                            state.register().clone(),
                            ComplexTensor::new(
                                previous.entries().real() + result.entries().real(),
                                previous.entries().imag() + result.entries().imag(),
                            )?,
                            engine.limits,
                        )?
                    } else {
                        result
                    });
                } else {
                    let mut diagonal = vec![0.; local * local];
                    let layout = DualRail::register(k)?;
                    for i in 0..local {
                        let digits = layout.decode(i)?;
                        let mut effect = 1.;
                        for j in 0..k {
                            effect *= effect_value(
                                digits[j],
                                outcomes[j],
                                gate.input[j],
                                gate.resource_measured[j],
                            );
                        }
                        diagonal[i * local + i] = effect;
                    }
                    probability = state.expectation(
                        engine,
                        &Observable::new(gate.wires.clone(), Matrix::real(local, diagonal)?)?,
                    )?;
                    abort = abort + probability.clone();
                    ticks = ticks + probability.clone() * gate.finish as f64;
                }
                records.push(PhysicalOutcome {
                    gate: index,
                    outcomes,
                    probability,
                });
            }
            state = accumulated.expect("every gate has successful instrument operators");
        }
        for (wire, &loss) in self.final_storage.iter().enumerate() {
            if loss != RailTransmission::IDEAL {
                state = state.channel(engine, &[wire], &loss.channel()?)?;
            }
        }
        ticks = ticks + state.trace() * self.finish as f64;
        Ok(PhysicalRun {
            state,
            records,
            abort,
            photons,
            fusions,
            optical_records: vec![],
            source_abort: Tensor::zeros([batch], (&device, dtype)),
            ancillas: Tensor::zeros([batch], (&device, dtype)),
            ticks,
        })
    }
}
fn decode_outcomes(mut code: usize, k: usize) -> Vec<FusionOutcome> {
    let mut out = vec![FusionOutcome::Loss; k];
    for i in (0..k).rev() {
        out[i] = match code % 5 {
            0 => FusionOutcome::Success { xx: Parity::Even },
            1 => FusionOutcome::Success { xx: Parity::Odd },
            2 => FusionOutcome::Failure { bit: false },
            3 => FusionOutcome::Failure { bit: true },
            _ => FusionOutcome::Loss,
        };
        code /= 5;
    }
    out
}
fn effect_value(
    digit: usize,
    outcome: FusionOutcome,
    a: RailTransmission,
    b: RailTransmission,
) -> f64 {
    if digit == 2 {
        return f64::from(matches!(outcome, FusionOutcome::Loss));
    }
    let a = a.values()[digit];
    let b = b.values();
    match outcome {
        FusionOutcome::Success { .. } => a * b[1 - digit] / 4.,
        FusionOutcome::Failure { bit } => {
            if usize::from(bit) == digit {
                a * b[digit] / 2.
            } else {
                0.
            }
        }
        FusionOutcome::Loss => 1. - a * (b[0] + b[1]) / 2.,
    }
}
fn right_factor(gate: &PhysicalGate, outcomes: &[FusionOutcome]) -> Result<Matrix> {
    let k = gate.wires.len();
    let layout = DualRail::register(k)?;
    let n = layout.size();
    let mut values = vec![0.; n * n];
    for col in 0..n {
        let digits = layout.decode(col)?;
        if digits.contains(&2) {
            continue;
        }
        let row = layout.encode(&digits.iter().map(|x| 1 - x).collect::<Vec<_>>())?;
        let mut value = 1. / (1usize << k) as f64;
        for j in 0..k {
            value *= (gate.input[j].values()[digits[j]]
                * gate.resource_measured[j].values()[1 - digits[j]])
                .sqrt();
            if digits[j] == 1 && matches!(outcomes[j], FusionOutcome::Success { xx: Parity::Odd }) {
                value = -value;
            }
        }
        values[row * n + col] = value;
    }
    Matrix::real(n, values)
}
fn two_qubit_correction(gate: &PhysicalGate, outcomes: &[FusionOutcome]) -> Result<Unitary> {
    let Operation::Unitary { gate, .. } = &gate.operation else {
        unreachable!()
    };
    let mut x = [true, true];
    let mut z = outcomes
        .iter()
        .map(|o| matches!(o, FusionOutcome::Success { xx: Parity::Odd }))
        .collect::<Vec<_>>();
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
    let mut gates = vec![];
    for i in 0..2 {
        gates.push(DualRail::unitary(&match (x[i], z[i]) {
            (true, true) => Unitary::y(),
            (true, false) => Unitary::x(),
            (false, true) => Unitary::z(),
            _ => Unitary::identity(2)?,
        })?);
    }
    let mut data = vec![];
    for row in 0..9 {
        for col in 0..9 {
            data.push(
                gates[0].matrix().entries()[(row / 3) * 3 + col / 3]
                    * gates[1].matrix().entries()[(row % 3) * 3 + col % 3],
            );
        }
    }
    Unitary::new(Matrix::new(9, data)?)
}
pub(super) fn embed(
    op: &ComplexTensor<3>,
    k: usize,
    vacuum_identity: bool,
) -> Result<ComplexTensor<3>> {
    let n = 3usize.pow(k as u32);
    let d = 1usize << k;
    let batch = op.dims()[0];
    let layout = DualRail::register(k)?;
    let qubits = Register::qubits(k)?;
    let mut indices = vec![];
    let mut masks = vec![];
    let mut identity = vec![];
    for row in 0..n {
        for col in 0..n {
            let r = layout.decode(row)?;
            let c = layout.decode(col)?;
            let code = !r.contains(&2) && !c.contains(&2);
            indices.push(if code {
                (qubits.encode(&r)? * d + qubits.encode(&c)?) as i64
            } else {
                0
            });
            masks.push(if code { 1. } else { 0. });
            identity.push(if vacuum_identity && !code && row == col {
                1.
            } else {
                0.
            });
        }
    }
    let dev = op.device();
    let dtype = op.dtype();
    let idx = Tensor::<1, Int>::from_data(TensorData::new(indices, [n * n]), &dev);
    let mask = Tensor::<2>::from_data(TensorData::new(masks, [1, n * n]), (&dev, dtype));
    let id = Tensor::<2>::from_data(TensorData::new(identity, [1, n * n]), (&dev, dtype));
    ComplexTensor::new(
        (op.real().reshape([batch, d * d]).select(1, idx.clone()) * mask.clone() + id)
            .reshape([batch, n, n]),
        (op.imag().reshape([batch, d * d]).select(1, idx) * mask).reshape([batch, n, n]),
    )
}
pub(super) fn adjoint(op: &ComplexTensor<3>) -> Result<ComplexTensor<3>> {
    ComplexTensor::new(op.real().transpose(), -op.imag().transpose())
}
pub(super) fn mm(a: &ComplexTensor<3>, b: &ComplexTensor<3>) -> Result<ComplexTensor<3>> {
    ComplexTensor::new(
        crate::precision::matmul(a.real(), b.real()) - crate::precision::matmul(a.imag(), b.imag()),
        crate::precision::matmul(a.real(), b.imag()) + crate::precision::matmul(a.imag(), b.real()),
    )
}
pub(super) fn apply_tensor(
    state: &DensityMatrix,
    engine: &DenseSimulator,
    wires: &[usize],
    op: &ComplexTensor<3>,
) -> Result<DensityMatrix> {
    let n = state.register().size();
    let b = state.batch_size();
    let dims = state.register().dimensions();
    let wide = StateVector::from_amplitudes(
        Register::new(dims.iter().chain(dims).copied().collect())?,
        ComplexTensor::new(
            state.entries().real().reshape([b, n * n]),
            state.entries().imag().reshape([b, n * n]),
        )?,
        engine.limits,
    )?;
    let wide = engine.apply_tensor(&wide, wires, op)?;
    let columns: Vec<_> = wires.iter().map(|w| w + dims.len()).collect();
    let wide = engine.apply_tensor(&wide, &columns, &op.conj())?;
    DensityMatrix::from_entries(
        state.register().clone(),
        ComplexTensor::new(
            wide.amplitudes().real().reshape([b, n, n]),
            wide.amplitudes().imag().reshape([b, n, n]),
        )?,
        engine.limits,
    )
}

pub(in crate::photonic::compiler) fn encode_state(
    engine: &DenseSimulator,
    input: StateVector,
) -> Result<StateVector> {
    let register = DualRail::register(input.register().wires())?;
    let n = register.size();
    engine.limits.check_state(input.batch_size(), n)?;
    engine.limits.check_plan(n)?;
    let mut indices = Vec::with_capacity(n);
    let mut mask = Vec::with_capacity(n);
    for i in 0..n {
        let digits = register.decode(i)?;
        if digits.contains(&2) {
            indices.push(0i64);
            mask.push(0.);
        } else {
            indices.push(input.register().encode(&digits)? as i64);
            mask.push(1.);
        }
    }
    let device = input.device();
    let dtype = input.amplitudes().dtype();
    let idx = Tensor::<1, Int>::from_data(TensorData::new(indices, [n]), &device);
    let mask = Tensor::<2>::from_data(TensorData::new(mask, [1, n]), (&device, dtype));
    StateVector::from_amplitudes(
        register,
        ComplexTensor::new(
            input.amplitudes().real().select(1, idx.clone()) * mask.clone(),
            input.amplitudes().imag().select(1, idx) * mask,
        )?,
        engine.limits,
    )
}
