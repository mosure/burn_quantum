use super::*;
/// Near-optimal iteration count for one marked item in a nontrivial qubit search space.
pub fn grover_iterations(qubits: usize) -> Result<usize> {
    width(qubits, 1, 16)?;
    let theta = (1. / ((1usize << qubits) as f64).sqrt()).asin();
    Ok((std::f64::consts::PI / (4. * theta) - 0.5).round().max(0.) as usize)
}
/// Exact single-target amplitude amplification. The diffuser differs from 2|s><s|-I by a global -1.
pub fn grover(qubits: usize, marked: usize, iterations: usize) -> Result<AlgorithmCircuit> {
    width(qubits, 1, 16)?;
    if marked >= 1usize << qubits {
        return Err(Error::Basis);
    }
    width(iterations, 0, 256)?;
    let mut a = AlgorithmCircuit::new(qubits, vec![])?;
    for w in 0..qubits {
        a.gate(vec![w], Unitary::h(), format!("Uniform H q{w}"))?;
    }
    for round in 0..iterations {
        a.push(
            Operation::PhaseFlip {
                controls: (0..qubits)
                    .map(|w| (w, marked >> (qubits - 1 - w) & 1))
                    .collect(),
            },
            format!("Oracle marks {marked}; round {}", round + 1),
        )?;
        for w in 0..qubits {
            a.gate(vec![w], Unitary::h(), format!("Diffusion H q{w}"))?;
        }
        a.push(
            Operation::PhaseFlip {
                controls: (0..qubits).map(|w| (w, 0)).collect(),
            },
            "Reflect |0…0>",
        )?;
        for w in 0..qubits {
            a.gate(vec![w], Unitary::h(), format!("Diffusion H q{w}"))?;
        }
    }
    Ok(a)
}
