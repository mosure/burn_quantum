use super::*;
use std::f64::consts::TAU;
/// Append the positive-sign Fourier transform (or its inverse), including final bit reversal.
/// `|x> -> sum_y exp(2π i xy/2^n)|y>/sqrt(2^n)`. Targets are ordered most significant first.
pub fn append_qft(circuit: &mut Circuit, wires: &[usize], inverse: bool) -> Result<()> {
    circuit.register().local_size(wires)?;
    if wires
        .iter()
        .any(|&w| circuit.register().dimensions()[w] != 2)
    {
        return Err(Error::InvalidDimensions("QFT requires qubits".into()));
    }
    let mut ops = Vec::new();
    for (i, &wire) in wires.iter().enumerate() {
        ops.push(Operation::Unitary {
            wires: vec![wire],
            gate: Unitary::h(),
        });
        for (j, &control) in wires.iter().enumerate().skip(i + 1) {
            ops.push(Operation::Unitary {
                wires: vec![control, wire],
                gate: Unitary::phase(TAU / (1usize << (j - i + 1)) as f64)?.controlled(2, 1)?,
            });
        }
    }
    for i in 0..wires.len() / 2 {
        ops.push(Operation::Unitary {
            wires: vec![wires[i], wires[wires.len() - 1 - i]],
            gate: Unitary::swap(2)?,
        });
    }
    if inverse {
        ops = ops
            .into_iter()
            .rev()
            .map(|op| match op {
                Operation::Unitary { wires, gate } => Operation::Unitary {
                    wires,
                    gate: gate.adjoint(),
                },
                _ => unreachable!(),
            })
            .collect();
    }
    let mut candidate = circuit.clone();
    for op in ops {
        candidate.push(op)?;
    }
    *circuit = candidate;
    Ok(())
}
/// Prepare an integer basis input, then apply QFT or inverse QFT.
pub fn qft(qubits: usize, input: usize, inverse: bool) -> Result<AlgorithmCircuit> {
    width(qubits, 1, 16)?;
    if input >= 1usize << qubits {
        return Err(Error::Basis);
    }
    let mut a = AlgorithmCircuit::new(qubits, vec![])?;
    for wire in 0..qubits {
        if input >> (qubits - 1 - wire) & 1 == 1 {
            a.gate(
                vec![wire],
                Unitary::x(),
                format!("Prepare input bit q{wire}"),
            )?;
        }
    }
    let before = a.circuit.operations().len();
    append_qft(&mut a.circuit, &(0..qubits).collect::<Vec<_>>(), inverse)?;
    a.labels
        .extend((before..a.circuit.operations().len()).map(|i| {
            format!(
                "{} operation {}",
                if inverse { "Inverse QFT" } else { "QFT" },
                i - before + 1
            )
        }));
    Ok(a)
}
