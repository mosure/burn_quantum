use super::*;
/// Odd-length repetition encoding, injected X errors, and a coherent majority decoder.
/// Returns the recovered logical state on q0, with syndrome information on the remaining wires.
/// Corrects up to (n-1)/2 bit flips; it does not correct phase flips or implement repeated syndrome rounds.
pub fn bit_flip_code(
    qubits: usize,
    errors: &[usize],
    theta: f64,
    phi: f64,
) -> Result<AlgorithmCircuit> {
    width(qubits, 3, 9)?;
    if qubits.is_multiple_of(2)
        || !theta.is_finite()
        || !phi.is_finite()
        || errors
            .iter()
            .enumerate()
            .any(|(i, &w)| w >= qubits || errors[..i].contains(&w))
    {
        return Err(Error::InvalidValue(
            "odd repetition size, finite angles and distinct error wires required".into(),
        ));
    }
    let mut a = AlgorithmCircuit::new(qubits, vec![])?;
    a.rotate(0, Axis::Y, Angle::Constant(theta), "Prepare logical Ry")?;
    a.rotate(0, Axis::Z, Angle::Constant(phi), "Prepare logical Rz")?;
    for w in 1..qubits {
        a.cx(0, w)?;
    }
    for &w in errors {
        a.gate(vec![w], Unitary::x(), format!("Injected X error q{w}"))?;
    }
    for w in 1..qubits {
        a.cx(0, w)?;
    }
    for pattern in 0usize..1usize << (qubits - 1) {
        if pattern.count_ones() as usize > qubits / 2 {
            a.push(
                Operation::ControlledX {
                    controls: (1..qubits)
                        .map(|w| (w, pattern >> (qubits - 1 - w) & 1))
                        .collect(),
                    target: 0,
                },
                format!(
                    "Coherent majority recovery, syndrome {pattern:0width$b}",
                    width = qubits - 1
                ),
            )?;
        }
    }
    Ok(a)
}
