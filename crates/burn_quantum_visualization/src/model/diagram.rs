//! Semantic circuit notation, derived from checked operations rather than prose labels.
use burn_quantum::{Angle, Operation, Unitary};
/// Canonical drawing primitive for a quantum operation.
#[derive(Clone, Debug, PartialEq)]
pub enum GateSymbol {
    /// One operator box; ordered targets are retained even when nonadjacent.
    Box {
        /// Ordered target wires.
        wires: Vec<usize>,
        /// Short mathematical name.
        label: String,
    },
    /// Open/closed level controls connected to a circled-plus target.
    ControlledX {
        /// Level conditions.
        controls: Vec<(usize, usize)>,
        /// Target wire.
        target: usize,
    },
    /// A diagonal minus sign on the indicated subspace.
    PhaseFlip(Vec<(usize, usize)>),
    /// Two crosses joined by a vertical line.
    Swap([usize; 2]),
    /// A filled control connected to a P(angle) target box.
    ControlledPhase {
        /// Control wire.
        control: usize,
        /// Phase target wire.
        target: usize,
        /// Phase in radians, principal branch.
        angle: f64,
    },
}
fn equal(a: &Unitary, b: &Unitary) -> bool {
    a.matrix().dimension() == b.matrix().dimension()
        && a.matrix()
            .entries()
            .iter()
            .zip(b.matrix().entries())
            .all(|(a, b)| (*a - *b).norm() < 1e-12)
}
fn phase(g: &Unitary) -> Option<f64> {
    let n = g.matrix().dimension();
    let e = g.matrix().entries();
    e.iter()
        .enumerate()
        .all(|(i, v)| {
            if i == n * n - 1 {
                true
            } else if i / n == i % n {
                (v.re - 1.).abs() < 1e-12 && v.im.abs() < 1e-12
            } else {
                v.norm() < 1e-12
            }
        })
        .then(|| e[n * n - 1].arg())
}
/// Classify exact built-ins or equivalent matrices within 1e-12, otherwise retain one generic U box.
pub fn symbol(op: &Operation) -> GateSymbol {
    match op {
        Operation::ControlledX { controls, target } => GateSymbol::ControlledX {
            controls: controls.clone(),
            target: *target,
        },
        Operation::PhaseFlip { controls } => GateSymbol::PhaseFlip(controls.clone()),
        Operation::Rotate { wire, rotation } => {
            let angle = match rotation.angle {
                Angle::Constant(v) => format!("{v:.2}"),
                Angle::Parameter {
                    index,
                    scale,
                    offset,
                } => {
                    if offset == 0. {
                        format!("{scale}θ{index}")
                    } else {
                        format!("θ{index}…")
                    }
                }
                Angle::Input { index, .. } => format!("x{index}"),
            };
            GateSymbol::Box {
                wires: vec![*wire],
                label: format!("R{:?}({angle})", rotation.axis),
            }
        }
        Operation::Unitary { wires, gate } => {
            if wires.len() == 2 && equal(gate, &Unitary::cnot()) {
                return GateSymbol::ControlledX {
                    controls: vec![(wires[0], 1)],
                    target: wires[1],
                };
            }
            if wires.len() == 2 && equal(gate, &Unitary::swap(2).unwrap()) {
                return GateSymbol::Swap([wires[0], wires[1]]);
            }
            if wires.len() == 2
                && gate.matrix().dimension() == 4
                && let Some(angle) = phase(gate)
            {
                return GateSymbol::ControlledPhase {
                    control: wires[0],
                    target: wires[1],
                    angle,
                };
            }
            let mut label = format!("U{}", gate.matrix().dimension());
            for (name, builtin) in [
                ("H", Unitary::h()),
                ("X", Unitary::x()),
                ("Y", Unitary::y()),
                ("Z", Unitary::z()),
                ("S", Unitary::s()),
                ("T", Unitary::t()),
            ] {
                if equal(gate, &builtin) {
                    label = name.into();
                    break;
                }
            }
            if label.starts_with('U') && wires.len() == 1 {
                if let Some(angle) = phase(gate) {
                    label = format!("P({angle:.2})");
                } else if let Ok(f) = Unitary::fourier(gate.matrix().dimension()) {
                    if equal(gate, &f) {
                        label = format!("F{}", gate.matrix().dimension())
                    } else if equal(gate, &f.adjoint()) {
                        label = format!("F{}†", gate.matrix().dimension())
                    }
                }
            }
            GateSymbol::Box {
                wires: wires.clone(),
                label,
            }
        }
    }
}
impl GateSymbol {
    /// Quantum wires touched by the symbol, preserving operator target order.
    pub fn wires(&self) -> Vec<usize> {
        match self {
            Self::Box { wires, .. } => wires.clone(),
            Self::ControlledX { controls, target } => {
                controls.iter().map(|c| c.0).chain([*target]).collect()
            }
            Self::PhaseFlip(c) => {
                if c.is_empty() {
                    vec![0]
                } else {
                    c.iter().map(|c| c.0).collect()
                }
            }
            Self::Swap(w) => w.to_vec(),
            Self::ControlledPhase {
                control, target, ..
            } => vec![*control, *target],
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn symbols_follow_matrices_and_order_not_labels() {
        assert_eq!(
            symbol(&Operation::Unitary {
                wires: vec![2, 0],
                gate: Unitary::cnot()
            }),
            GateSymbol::ControlledX {
                controls: vec![(2, 1)],
                target: 0
            }
        );
        assert_eq!(
            symbol(&Operation::Unitary {
                wires: vec![3, 1],
                gate: Unitary::swap(2).unwrap()
            }),
            GateSymbol::Swap([3, 1])
        );
        assert!(matches!(
            symbol(&Operation::Unitary {
                wires: vec![0, 2],
                gate: Unitary::phase(0.3).unwrap().controlled(2, 1).unwrap()
            }),
            GateSymbol::ControlledPhase {
                control: 0,
                target: 2,
                ..
            }
        ));
        assert!(
            matches!(symbol(&Operation::Unitary{wires:vec![2,0],gate:Unitary::sum(3).unwrap()}),GateSymbol::Box{wires,..} if wires==[2,0])
        );
    }
}
