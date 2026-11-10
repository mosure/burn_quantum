use super::*;
/// Independent parallel teleportations, with actual measurements and classical X/Z feed-forward.
/// For transfer j: input=3j, Alice resource=3j+1, Bob=3j+2; classical bits 2j,2j+1.
pub fn teleportation(transfers: usize, theta: f64, phi: f64) -> Result<AlgorithmCircuit> {
    width(transfers, 1, 5)?;
    if !theta.is_finite() || !phi.is_finite() {
        return Err(Error::InvalidValue("finite input angles required".into()));
    }
    let mut a = AlgorithmCircuit::new(3 * transfers, vec![])?;
    a.protocol = Some(Protocol::default());
    for j in 0..transfers {
        let q = 3 * j;
        a.rotate(
            q,
            Axis::Y,
            Angle::Constant(theta),
            format!("Prepare input {} Ry", j + 1),
        )?;
        a.rotate(
            q,
            Axis::Z,
            Angle::Constant(phi),
            format!("Prepare input {} Rz", j + 1),
        )?;
        a.gate(vec![q + 1], Unitary::h(), "Create shared Bell resource")?;
        a.cx(q + 1, q + 2)?;
        a.cx(q, q + 1)?;
        a.gate(vec![q], Unitary::h(), "Alice Bell-basis rotation")?;
        a.protocol.as_mut().unwrap().instructions.extend([
            Instruction::Measure {
                wire: q,
                bit: 2 * j,
            },
            Instruction::Measure {
                wire: q + 1,
                bit: 2 * j + 1,
            },
        ]);
        for (gate, bit, name) in [(Unitary::x(), 2 * j + 1, "X"), (Unitary::z(), 2 * j, "Z")] {
            a.gate(vec![q + 2], gate, format!("Bob {name} if c{bit}=1"))?;
            if let Some(Instruction::Gate { conditions, .. }) =
                a.protocol.as_mut().unwrap().instructions.last_mut()
            {
                conditions.push((bit, 1));
            }
        }
    }
    Ok(a)
}
