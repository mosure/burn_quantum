use super::*;
use std::f64::consts::TAU;
/// QPE for U=P(2π phase) and eigenstate |1>, with most-significant counting wire first.
pub fn phase_estimation(precision: usize, phase: f64) -> Result<AlgorithmCircuit> {
    width(precision, 1, 15)?;
    if !phase.is_finite() || !(0.0..1.0).contains(&phase) {
        return Err(Error::InvalidValue("phase must be in [0,1)".into()));
    }
    let mut a = AlgorithmCircuit::new(precision + 1, vec![])?;
    a.gate(vec![precision], Unitary::x(), "Prepare eigenstate |1>")?;
    for w in 0..precision {
        a.gate(vec![w], Unitary::h(), format!("H counting q{w}"))?;
    }
    for w in 0..precision {
        let power = 1usize << (precision - 1 - w);
        a.gate(
            vec![w, precision],
            Unitary::phase(TAU * phase * power as f64)?.controlled(2, 1)?,
            format!("Controlled U^{power}; phase {phase}"),
        )?;
    }
    let before = a.circuit.operations().len();
    append_qft(&mut a.circuit, &(0..precision).collect::<Vec<_>>(), true)?;
    a.labels.extend(
        (before..a.circuit.operations().len()).map(|i| format!("Inverse QFT {}", i - before + 1)),
    );
    Ok(a)
}
