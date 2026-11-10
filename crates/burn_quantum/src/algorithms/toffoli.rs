use super::*;
/// Toffoli for two controls, generalized to n controls without allocating a 2^(n+1) gate matrix.
/// Prepare controls in |+> to expose the target's correlations in the default demonstration.
pub fn controlled_x_demo(controls: usize, superposition: bool) -> Result<AlgorithmCircuit> {
    width(controls, 1, 15)?;
    let mut a = AlgorithmCircuit::new(controls + 1, vec![])?;
    for w in 0..controls {
        a.gate(
            vec![w],
            if superposition {
                Unitary::h()
            } else {
                Unitary::x()
            },
            format!("Prepare control q{w}"),
        )?;
    }
    a.push(
        Operation::ControlledX {
            controls: (0..controls).map(|w| (w, 1)).collect(),
            target: controls,
        },
        format!("{}-controlled X", controls),
    )?;
    Ok(a)
}
