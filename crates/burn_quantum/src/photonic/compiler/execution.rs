use super::*;
use crate::{
    ComplexTensor, DenseSimulator, Error, Register, Result, Rotation, Unitary, gate::Axis,
};
use burn::tensor::{DType, Int, TensorData};

/// Exact ideal measurement-pattern reference with bounded branch enumeration.
///
/// Live state width is logical width plus one prepared ancilla, independently
/// of total pattern length. Exact classical transcripts still grow exponentially.
#[derive(Clone, Copy, Debug, Default)]
pub struct PatternSimulator {
    /// Dense-state and transcript enumeration budgets.
    pub limits: PatternSimulationLimits,
}
impl PatternSimulator {
    /// Execute every corrected outcome with absolute subnormalized amplitudes.
    pub fn execute_all(
        &self,
        pattern: &MeasurementPattern,
        input: StateVector,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
    ) -> Result<Vec<PatternBranch>> {
        self.execute(pattern, input, parameters, inputs, None)
    }
    /// Execute one explicit transcript, retaining its physical branch probability.
    /// This does not sample or claim deterministic hardware success.
    pub fn execute_branch(
        &self,
        pattern: &MeasurementPattern,
        input: StateVector,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
        outcomes: &[bool],
    ) -> Result<PatternBranch> {
        let mut result = self.execute(pattern, input, parameters, inputs, Some(outcomes))?;
        Ok(result.remove(0))
    }
    fn execute(
        &self,
        pattern: &MeasurementPattern,
        input: StateVector,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
        selected: Option<&[bool]>,
    ) -> Result<Vec<PatternBranch>> {
        if input.register() != pattern.source.register() {
            return Err(Error::Shape("pattern and input registers differ".into()));
        }
        if selected.is_some_and(|x| x.len() != pattern.measurements) {
            return Err(Error::Shape("pattern transcript length differs".into()));
        }
        // Retain absolute weights instead of silently underflowing a long selected
        // transcript. A future log-weight trajectory API can normalize each step.
        let exponent_budget = if input.amplitudes().dtype() == DType::F32 {
            120
        } else {
            1000
        };
        if pattern
            .measurements
            .saturating_add(input.register().wires())
            > exponent_budget
        {
            return Err(Error::Unsupported(
                "absolute pattern branch weights exceed the dtype exponent budget".into(),
            ));
        }
        let required = if selected.is_some() {
            1
        } else {
            1usize
                .checked_shl(u32::try_from(pattern.measurements).map_err(|_| Error::Overflow)?)
                .ok_or(Error::Overflow)?
        };
        if required > self.limits.max_branches {
            return Err(Error::ResourceLimit {
                resource: "pattern branches",
                required,
                limit: self.limits.max_branches,
            });
        }
        let width = input.register().wires() + usize::from(pattern.measurements > 0);
        let largest = Register::qubits(width)?.size();
        self.limits.dense.check_state(input.batch_size(), largest)?;
        self.limits.dense.check_plan(largest)?;
        let engine = DenseSimulator::new(self.limits.dense);
        // Native binding validation, including dtype/context compatibility, before execution.
        let validator = Circuit::new(
            input.register().clone(),
            pattern.source.num_parameters(),
            pattern.source.num_inputs(),
        );
        let input = validator.run_from(&engine, input, parameters.clone(), inputs.clone())?;
        let mut branches = vec![PatternBranch {
            outcomes: vec![],
            state: input,
        }];
        let mut active: Vec<_> = (0..pattern.source.register().wires())
            .map(PatternSite)
            .collect();
        for command in &pattern.commands {
            match command {
                PatternCommand::PreparePlus { site } => {
                    for branch in &mut branches {
                        let state = &branch.state;
                        let [b, n] = state.amplitudes().dims();
                        let re = state.amplitudes().real().reshape([b, n, 1])
                            * std::f64::consts::FRAC_1_SQRT_2;
                        let im = state.amplitudes().imag().reshape([b, n, 1])
                            * std::f64::consts::FRAC_1_SQRT_2;
                        branch.state = StateVector::from_amplitudes(
                            Register::qubits(active.len() + 1)?,
                            ComplexTensor::new(
                                Tensor::cat(vec![re.clone(), re], 2).reshape([b, n * 2]),
                                Tensor::cat(vec![im.clone(), im], 2).reshape([b, n * 2]),
                            )?,
                            self.limits.dense,
                        )?;
                    }
                    active.push(*site);
                }
                PatternCommand::Entangle { sites } => {
                    let targets = sites.map(|site| {
                        active
                            .iter()
                            .position(|x| *x == site)
                            .expect("compiled live edge")
                    });
                    for branch in &mut branches {
                        branch.state = engine.apply(
                            &branch.state,
                            &targets,
                            &Unitary::z().controlled(2, 1)?,
                        )?;
                    }
                }
                PatternCommand::Measure { site, angle, bit } => {
                    let position = active
                        .iter()
                        .position(|x| x == site)
                        .expect("compiled live measurement");
                    let mut next =
                        Vec::with_capacity(branches.len() * if selected.is_some() { 1 } else { 2 });
                    for branch in branches {
                        let mut measurement = Circuit::new(
                            branch.state.register().clone(),
                            pattern.source.num_parameters(),
                            pattern.source.num_inputs(),
                        );
                        measurement.rotate(position, Rotation::qubit(Axis::Z, angle.clone()))?;
                        measurement.unitary(vec![position], Unitary::h())?;
                        let rotated = measurement.run_from(
                            &engine,
                            branch.state,
                            parameters.clone(),
                            inputs.clone(),
                        )?;
                        for outcome in [false, true] {
                            if selected.is_some_and(|record| record[*bit] != outcome) {
                                continue;
                            }
                            let state = project(&rotated, position, outcome, self.limits.dense)?;
                            let mut outcomes = branch.outcomes.clone();
                            outcomes.push(outcome);
                            next.push(PatternBranch { state, outcomes });
                        }
                    }
                    branches = next;
                    active.remove(position);
                }
                PatternCommand::CorrectX { site, bit } => {
                    let position = active
                        .iter()
                        .position(|x| x == site)
                        .expect("compiled live correction");
                    for branch in &mut branches {
                        if branch.outcomes[*bit] {
                            branch.state =
                                engine.apply(&branch.state, &[position], &Unitary::x())?;
                        }
                    }
                }
            }
        }
        let output_order: Vec<_> = pattern
            .outputs
            .iter()
            .map(|site| {
                active
                    .iter()
                    .position(|x| x == site)
                    .expect("compiled output live")
            })
            .collect();
        for branch in &mut branches {
            let state = &branch.state;
            let n = state.register().size();
            let (forward, _) = state.register().permutation(&output_order)?;
            let indices =
                Tensor::<1, Int>::from_data(TensorData::new(forward, [n]), &state.device());
            branch.state = StateVector::from_amplitudes(
                state.register().clone(),
                ComplexTensor::new(
                    state.amplitudes().real().select(1, indices.clone()),
                    state.amplitudes().imag().select(1, indices),
                )?,
                self.limits.dense,
            )?;
        }
        Ok(branches)
    }
}
fn project(
    state: &StateVector,
    wire: usize,
    outcome: bool,
    limits: ResourceLimits,
) -> Result<StateVector> {
    let n = state.register().size() / 2;
    let shift = state.register().wires() - 1 - wire;
    let mask = (1usize << shift) - 1;
    let selected: Vec<i64> = (0..n)
        .map(|x| {
            (((x >> shift) << (shift + 1)) | (usize::from(outcome) << shift) | (x & mask)) as i64
        })
        .collect();
    let index = Tensor::<1, Int>::from_data(TensorData::new(selected, [n]), &state.device());
    StateVector::from_amplitudes(
        Register::qubits(state.register().wires() - 1)?,
        ComplexTensor::new(
            state.amplitudes().real().select(1, index.clone()),
            state.amplitudes().imag().select(1, index),
        )?,
        limits,
    )
}
