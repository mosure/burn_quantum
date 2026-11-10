//! Bound expensive checked core deserialization before constructing any Unitary.
use super::{IoResult, MAX_UNITARY_VALIDATION_WORK};
use crate::simulation::TraceLimits;
use burn_quantum::Register;
use serde_json::Value;

fn array(value: &Value) -> IoResult<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| "experiment preflight expected an array".into())
}
fn integer(value: &Value) -> IoResult<usize> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| "experiment preflight expected a bounded unsigned integer".into())
}
fn bounded_size(actual: usize, limit: usize) -> IoResult<()> {
    if actual > limit {
        Err("experiment preflight array or dimension exceeds its budget".into())
    } else {
        Ok(())
    }
}
fn exact_size(value: &Value, expected: usize) -> IoResult<()> {
    if array(value)?.len() != expected {
        return Err("experiment preflight array shape differs from its register".into());
    }
    Ok(())
}

pub(super) fn check(value: &Value) -> IoResult<()> {
    let limits = TraceLimits::default();
    let circuit = &value["circuit"];
    let dimensions = array(&circuit["register"])?;
    bounded_size(dimensions.len(), limits.max_wires)?;
    let mut dims = Vec::with_capacity(dimensions.len());
    for dimension in dimensions {
        let d = integer(dimension)?;
        bounded_size(d, limits.max_local_dimension)?;
        dims.push(d);
    }
    let register = Register::new(dims)?;
    let frames = array(&value["trace"]["frames"])?;
    limits.check(&register, frames.len())?;
    let parameters = integer(&circuit["parameters"])?;
    let inputs = integer(&circuit["inputs"])?;
    bounded_size(parameters, 64)?;
    bounded_size(inputs, 256)?;
    exact_size(&value["parameters"], parameters)?;
    exact_size(&value["inputs"], inputs)?;
    let operations = array(&circuit["operations"])?;
    bounded_size(operations.len(), limits.max_frames.saturating_sub(1))?;
    exact_size(&value["operation_labels"], operations.len())?;
    if !value["protocol"].is_null() {
        let protocol = &value["protocol"];
        bounded_size(
            array(&protocol["program"]["instructions"])?.len(),
            limits.max_frames.saturating_sub(1),
        )?;
        bounded_size(array(&protocol["outcomes"])?.len(), limits.max_frames)?;
        exact_size(&protocol["probabilities"], frames.len())?;
        for instruction in array(&protocol["program"]["instructions"])? {
            if !instruction["Gate"].is_null() {
                bounded_size(
                    array(&instruction["Gate"]["conditions"])?.len(),
                    limits.max_frames,
                )?;
            }
        }
    }
    let mut work = 0usize;
    for operation in operations {
        let object = operation
            .as_object()
            .filter(|object| object.len() == 1)
            .ok_or("experiment preflight expected one operation variant")?;
        let (kind, fields) = object.iter().next().unwrap();
        match kind.as_str() {
            "Unitary" => {
                bounded_size(array(&fields["wires"])?.len(), limits.max_wires)?;
                let gate = &fields["gate"];
                let dimension = integer(&gate["dimension"])?;
                let entries = dimension
                    .checked_mul(dimension)
                    .ok_or("experiment preflight matrix dimension overflow")?;
                bounded_size(entries, limits.max_matrix_elements)?;
                exact_size(&gate["entries"], entries)?;
                work = entries
                    .checked_mul(dimension)
                    .and_then(|terms| work.checked_add(terms))
                    .ok_or("experiment preflight unitary validation work overflow")?;
                if work > MAX_UNITARY_VALIDATION_WORK {
                    return Err(
                        "experiment preflight aggregate unitary validation work exceeds its budget"
                            .into(),
                    );
                }
            }
            "ControlledX" | "PhaseFlip" => {
                bounded_size(array(&fields["controls"])?.len(), limits.max_wires)?;
            }
            "Rotate" => {}
            _ => return Err("experiment preflight unknown operation variant".into()),
        }
    }
    for frame in frames {
        exact_size(&frame["probabilities"], register.size())?;
        if !frame["amplitudes"].is_null() {
            exact_size(&frame["amplitudes"], register.size())?;
        }
        exact_size(&frame["subsystems"], register.wires())?;
        for (subsystem, &d) in array(&frame["subsystems"])?
            .iter()
            .zip(register.dimensions())
        {
            exact_size(&subsystem["probabilities"], d)?;
            exact_size(&subsystem["density"], d * d)?;
        }
    }
    bounded_size(array(&value["optimization"])?.len(), limits.max_frames)?;
    Ok(())
}
