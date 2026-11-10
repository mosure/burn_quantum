//! Reusable quantum feature encoding, circuit templates, kernels, and losses.
//!
//! Tensor functions validate metadata without host synchronization. Their input
//! states/distributions must already be physically valid. Call state `validate`
//! at explicit boundaries when numerical validation is needed.

use crate::complex::compatible;
use crate::gate::Axis;
use crate::state::compatible_states;
use crate::{Angle, Circuit, Error, ResourceLimits, Result, Rotation, StateVector, Unitary};
use burn::tensor::Tensor;

#[cfg(feature = "benchmarks")]
pub mod benchmarks;
pub mod preprocessing;

fn qubit_targets(circuit: &Circuit, wires: &[usize]) -> Result<()> {
    circuit.register().local_size(wires)?;
    if wires
        .iter()
        .any(|&w| circuit.register().dimensions()[w] != 2)
    {
        return Err(Error::InvalidDimensions(
            "this template requires qubit targets; use explicit level rotations for qudits".into(),
        ));
    }
    Ok(())
}

/// Append one angle-encoding rotation per ordered qubit target.
///
/// Wire `wires[j]` receives `Angle::input(input_offset + j)`. The caller declares
/// input counts on the circuit. Validation is atomic: failure leaves it unchanged.
pub fn angle_embedding(
    circuit: &mut Circuit,
    wires: &[usize],
    axis: Axis,
    input_offset: usize,
) -> Result<()> {
    qubit_targets(circuit, wires)?;
    let end = input_offset
        .checked_add(wires.len())
        .ok_or(Error::Overflow)?;
    if end > circuit.num_inputs() {
        return Err(Error::Shape("insufficient circuit input features".into()));
    }
    let mut candidate = circuit.clone();
    for (j, &wire) in wires.iter().enumerate() {
        candidate.rotate(wire, Rotation::qubit(axis, Angle::input(input_offset + j)))?;
    }
    *circuit = candidate;
    Ok(())
}

/// Append trainable RY rotations followed by an ordered nearest-neighbor CNOT chain.
///
/// This real-amplitude variational layer uses `wires.len()` parameters starting
/// at `parameter_offset`. Repeating a parameter offset deliberately ties weights.
/// It is a specific reusable ansatz, not a claim of universality at fixed depth.
/// Failure leaves the circuit unchanged.
pub fn ry_cnot_layer(
    circuit: &mut Circuit,
    wires: &[usize],
    parameter_offset: usize,
) -> Result<()> {
    qubit_targets(circuit, wires)?;
    let end = parameter_offset
        .checked_add(wires.len())
        .ok_or(Error::Overflow)?;
    if end > circuit.num_parameters() {
        return Err(Error::Shape("insufficient circuit parameters".into()));
    }
    let mut candidate = circuit.clone();
    for (j, &wire) in wires.iter().enumerate() {
        candidate.rotate(
            wire,
            Rotation::qubit(Axis::Y, Angle::parameter(parameter_offset + j)),
        )?;
    }
    for pair in wires.windows(2) {
        candidate.unitary(pair.to_vec(), Unitary::cnot())?;
    }
    *circuit = candidate;
    Ok(())
}

/// Convert physical Pauli-Z expectations `[batch]` to probabilities `[batch, 2]`.
/// Columns are P(0) = (1+z)/2 and P(1) = (1-z)/2. No clipping or readback occurs.
pub fn pauli_z_probabilities(expectations: Tensor<1>) -> Tensor<2> {
    let [batch] = expectations.dims();
    let zero = (expectations.clone() + 1.) / 2.;
    let one = (-expectations + 1.) / 2.;
    Tensor::cat(vec![zero.reshape([batch, 1]), one.reshape([batch, 1])], 1)
}

/// Mean categorical cross-entropy from probabilities and normalized soft targets.
///
/// Both arguments have shape `[batch, classes]`, with finite, nonnegative rows
/// summing to one. One-hot targets implement ordinary categorical NLL. Only
/// metadata is checked; values stay on device. `epsilon` clamps the logarithm's
/// argument to avoid log(0); gradients below this floor are zero. This function
/// consumes probabilities, not logits (which use Burn's standard losses).
pub fn distribution_cross_entropy(
    probabilities: Tensor<2>,
    targets: Tensor<2>,
    epsilon: f64,
) -> Result<Tensor<1>> {
    let [batch, classes] = probabilities.dims();
    if batch == 0 || classes < 2 || targets.dims() != [batch, classes] {
        return Err(Error::Shape(
            "cross-entropy needs matching nonempty [batch, classes>=2] tensors".into(),
        ));
    }
    compatible(&probabilities, &targets)?;
    crate::backend::supported_dtype(probabilities.dtype())?;
    crate::error::tolerance(epsilon)?;
    // An f64 epsilon must remain positive after conversion to the tensor dtype.
    if probabilities.dtype() == burn::tensor::DType::F32 && (epsilon as f32) == 0. {
        return Err(Error::InvalidValue(
            "epsilon underflows the probability dtype".into(),
        ));
    }
    Ok(-(targets * probabilities.clamp_min(epsilon).log())
        .sum_dim(1)
        .mean())
}

/// Fidelity Gram/cross-kernel `K[i,j] = |<left_i | right_j>|²`.
///
/// Returns `[left_batch, right_batch]` on the input device and retains gradients.
/// Valid normalized states produce a PSD Gram matrix when left == right. Input
/// registers, dtypes and execution contexts must match; batch counts may differ.
/// `limits` bounds each input and the pairwise output, not total peak memory.
pub fn fidelity_kernel(
    left: &StateVector,
    right: &StateVector,
    limits: ResourceLimits,
) -> Result<Tensor<2>> {
    compatible_states(left, right)?;
    limits.check_state(left.batch_size(), left.register().size())?;
    limits.check_state(right.batch_size(), right.register().size())?;
    limits.check_state(left.batch_size(), right.batch_size())?;
    let a = left.amplitudes();
    let b = right.amplitudes();
    let re = crate::precision::matmul2(a.real(), b.real().transpose())
        + crate::precision::matmul2(a.imag(), b.imag().transpose());
    let im = crate::precision::matmul2(a.real(), b.imag().transpose())
        - crate::precision::matmul2(a.imag(), b.real().transpose());
    Ok(re.square() + im.square())
}
