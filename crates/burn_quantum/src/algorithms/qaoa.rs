use super::*;
/// MaxCut QAOA for an arbitrary simple graph, with (gamma,beta) bindings per layer.
/// Cost C=sum_edges(1-Zu Zv)/2; cost unitary exp(-i gamma C), mixer exp(-i beta sum X).
/// Constant cost phases are omitted. Every edge uses CX, Rz(-gamma), CX.
pub fn qaoa_maxcut(
    qubits: usize,
    edges: &[(usize, usize)],
    angles: &[(f64, f64)],
) -> Result<AlgorithmCircuit> {
    width(qubits, 2, 16)?;
    width(angles.len(), 1, 16)?;
    if angles
        .iter()
        .any(|&(g, b)| !g.is_finite() || !b.is_finite())
        || edges.iter().enumerate().any(|(i, &(u, v))| {
            u >= qubits
                || v >= qubits
                || u == v
                || edges[..i]
                    .iter()
                    .any(|&(a, b)| (a == u && b == v) || (a == v && b == u))
        })
    {
        return Err(Error::InvalidValue(
            "invalid MaxCut graph or QAOA angles".into(),
        ));
    }
    let mut a = AlgorithmCircuit::new(qubits, angles.iter().flat_map(|&(g, b)| [g, b]).collect())?;
    for w in 0..qubits {
        a.gate(vec![w], Unitary::h(), format!("H q{w}"))?;
    }
    for layer in 0..angles.len() {
        for &(u, v) in edges {
            a.cx(u, v)?;
            a.rotate(
                v,
                Axis::Z,
                Angle::Parameter {
                    index: 2 * layer,
                    scale: -1.,
                    offset: 0.,
                },
                format!("Cost edge {u}–{v}, layer {}", layer + 1),
            )?;
            a.cx(u, v)?;
        }
        for w in 0..qubits {
            a.rotate(
                w,
                Axis::X,
                Angle::Parameter {
                    index: 2 * layer + 1,
                    scale: 2.,
                    offset: 0.,
                },
                format!("Mixer q{w}, layer {}", layer + 1),
            )?;
        }
    }
    Ok(a)
}
/// Classical cut size of one big-endian basis assignment; useful for independent readout.
pub fn cut_value(qubits: usize, edges: &[(usize, usize)], basis: usize) -> Result<usize> {
    width(qubits, 1, 16)?;
    if basis >= 1usize << qubits || edges.iter().any(|&(u, v)| u >= qubits || v >= qubits) {
        return Err(Error::Basis);
    }
    Ok(edges
        .iter()
        .filter(|&&(u, v)| ((basis >> (qubits - 1 - u)) ^ (basis >> (qubits - 1 - v))) & 1 == 1)
        .count())
}
