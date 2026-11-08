//! Structured controlled gates without exponentially large local matrices.
use crate::{ComplexTensor, DenseSimulator, Error, Register, Result, StateVector};
use burn::tensor::{Int, Tensor, TensorData};

pub(crate) fn validate_controls(register: &Register, controls: &[(usize, usize)]) -> Result<()> {
    for (i, &(wire, level)) in controls.iter().enumerate() {
        if wire >= register.wires() || controls[..i].iter().any(|c| c.0 == wire) {
            return Err(Error::Wires("invalid or repeated control wire".into()));
        }
        if level >= register.dimensions()[wire] {
            return Err(Error::Basis);
        }
    }
    Ok(())
}

pub(crate) fn validate_x(
    register: &Register,
    controls: &[(usize, usize)],
    target: usize,
) -> Result<()> {
    validate_controls(register, controls)?;
    if target >= register.wires() || controls.iter().any(|c| c.0 == target) {
        return Err(Error::Wires("target must be distinct from controls".into()));
    }
    if register.dimensions()[target] != 2 {
        return Err(Error::Unsupported(
            "controlled X requires a qubit target".into(),
        ));
    }
    Ok(())
}

fn stride(register: &Register, wire: usize) -> usize {
    register.dimensions()[wire + 1..].iter().product()
}

fn conditions(register: &Register, controls: &[(usize, usize)]) -> Vec<(usize, usize, usize)> {
    controls
        .iter()
        .map(|&(w, level)| (stride(register, w), register.dimensions()[w], level))
        .collect()
}

fn matches(index: usize, conditions: &[(usize, usize, usize)]) -> bool {
    conditions
        .iter()
        .all(|&(stride, dimension, level)| index / stride % dimension == level)
}

impl DenseSimulator {
    /// Flip a qubit if every `(wire, level)` control matches. Controls may be qudits.
    /// Empty controls give ordinary X. Uses a length-N permutation, never an N² matrix.
    pub fn controlled_x(
        &self,
        state: &StateVector,
        controls: &[(usize, usize)],
        target: usize,
    ) -> Result<StateVector> {
        let register = state.register();
        validate_x(register, controls, target)?;
        let n = register.size();
        self.limits.check_state(state.batch_size(), n)?;
        self.limits.check_plan(n)?;
        if n > i32::MAX as usize {
            return Err(Error::Unsupported(
                "dense gather indices exceed signed 32-bit range".into(),
            ));
        }
        let controls = conditions(register, controls);
        let step = stride(register, target);
        // X is self-inverse, so the forward permutation is also the gather map.
        let indices: Vec<i64> = (0..n)
            .map(|i| {
                let source = if matches(i, &controls) {
                    if (i / step).is_multiple_of(2) {
                        i + step
                    } else {
                        i - step
                    }
                } else {
                    i
                };
                source as i64
            })
            .collect();
        let index = Tensor::<1, Int>::from_data(TensorData::new(indices, [n]), &state.device());
        StateVector::from_amplitudes(
            register.clone(),
            ComplexTensor::new(
                state.amplitudes().real().select(1, index.clone()),
                state.amplitudes().imag().select(1, index),
            )?,
            self.limits,
        )
    }

    /// Multiply amplitudes by -1 where every `(wire, level)` control matches.
    /// Empty controls give a global minus sign. The diagonal needs only N scalars.
    pub fn phase_flip(
        &self,
        state: &StateVector,
        controls: &[(usize, usize)],
    ) -> Result<StateVector> {
        let register = state.register();
        validate_controls(register, controls)?;
        let n = register.size();
        self.limits.check_state(state.batch_size(), n)?;
        self.limits.check_plan(n)?;
        let controls = conditions(register, controls);
        let signs: Vec<f64> = (0..n)
            .map(|i| if matches(i, &controls) { -1.0 } else { 1.0 })
            .collect();
        let signs = Tensor::from_data(
            TensorData::new(signs, [1, n]),
            (&state.device(), state.amplitudes().dtype()),
        );
        StateVector::from_amplitudes(
            register.clone(),
            ComplexTensor::new(
                state.amplitudes().real() * signs.clone(),
                state.amplitudes().imag() * signs,
            )?,
            self.limits,
        )
    }
}
