use super::*;
/// Promise oracle family. Any nonzero parity mask is balanced; zero is constant.
pub fn deutsch_jozsa(inputs: usize, mask: usize, offset: bool) -> Result<AlgorithmCircuit> {
    width(inputs, 1, 15)?;
    if mask >= 1usize << inputs {
        return Err(Error::Basis);
    }
    let mut a = AlgorithmCircuit::new(inputs + 1, vec![])?;
    a.gate(vec![inputs], Unitary::x(), "Prepare oracle ancilla |1>")?;
    for w in 0..=inputs {
        a.gate(vec![w], Unitary::h(), format!("H q{w}"))?;
    }
    for w in 0..inputs {
        if mask >> (inputs - 1 - w) & 1 == 1 {
            a.cx(w, inputs)?;
        }
    }
    if offset {
        a.gate(vec![inputs], Unitary::x(), "Oracle constant offset")?;
    }
    for w in 0..inputs {
        a.gate(vec![w], Unitary::h(), format!("Decode input q{w}"))?;
    }
    Ok(a)
}
