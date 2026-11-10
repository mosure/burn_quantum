//! Brief scientific descriptions shared by circuit tooltips and other frontends.
use super::{GateSymbol, phase, symbol};
use burn_quantum::{Angle, Operation};

/// A gate name and a short explanation derived from its checked operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateExplanation {
    /// Mathematical name and, where useful, its common alias.
    pub title: String,
    /// Brief effect, including relevant wire order, levels or angle expression.
    pub summary: String,
}

impl GateExplanation {
    fn new(title: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            summary: summary.into(),
        }
    }
}

/// Explain a checked circuit operation without consulting free-form labels.
///
/// Symbolic angles retain their affine expression rather than inventing a bound
/// value. Numeric values are rounded for display and angles are in radians.
/// Long wire/control lists show their first three entries and remaining count.
/// This function describes the operation; it does not validate or execute it.
pub fn explain(operation: &Operation) -> GateExplanation {
    if let Operation::Rotate { wire, rotation } = operation {
        let axis = format!("{:?}", rotation.axis);
        let [a, b] = rotation.levels;
        return GateExplanation::new(
            format!("R{axis} · {axis} rotation"),
            format!(
                "Rotates q{wire} about {axis} by {} rad in the ordered |{a}>, |{b}> subspace. Other levels are unchanged.",
                angle(&rotation.angle)
            ),
        );
    }
    match symbol(operation) {
        GateSymbol::ControlledX { controls, target } => {
            if controls.is_empty() {
                return GateExplanation::new(
                    "X · bit flip",
                    format!("Swaps |0> and |1> on q{target}."),
                );
            }
            let title = match controls.as_slice() {
                [(_, 1)] => "Controlled X · CNOT on qubits",
                [(_, 0)] => "Open-controlled X",
                [(_, 1), (_, 1)] => "Two-control X · Toffoli on qubits",
                [_] => "Level-controlled X",
                _ => "Multi-controlled X",
            };
            GateExplanation::new(
                title,
                format!(
                    "Flips q{target} (|0> ↔ |1>) only when all controls match: {}. Control basis values are preserved.",
                    conditions(&controls)
                ),
            )
        }
        GateSymbol::PhaseFlip(controls) if controls.is_empty() => GateExplanation::new(
            "−I · global phase",
            "Multiplies every amplitude by −1. This global phase leaves all measurement probabilities unchanged.",
        ),
        GateSymbol::PhaseFlip(controls) => GateExplanation::new(
            "Conditional phase flip",
            format!(
                "Adds a π phase (a minus sign) only to amplitudes where all controls match: {}. No basis values are flipped.",
                conditions(&controls)
            ),
        ),
        GateSymbol::Swap([a, b]) => GateExplanation::new(
            "SWAP",
            format!(
                "Exchanges the quantum states of q{a} and q{b}, including their correlations with other wires."
            ),
        ),
        GateSymbol::ControlledPhase {
            control,
            target,
            angle,
        } => GateExplanation::new(
            "Controlled phase",
            format!(
                "Adds {} rad of phase only to |11> on ordered wires q{control}, q{target}. Other basis amplitudes are unchanged.",
                number(angle)
            ),
        ),
        GateSymbol::Box { wires, label } => {
            let Operation::Unitary { gate, .. } = operation else {
                unreachable!("rotations were handled before symbol classification");
            };
            let dimension = gate.matrix().dimension();
            let names = wire_names(&wires);
            match label.as_str() {
                "H" => GateExplanation::new(
                    "H · Hadamard",
                    format!(
                        "Maps |0> to (|0> + |1>)/√2 and |1> to (|0> − |1>)/√2 on {names}. Changes between Z and X bases."
                    ),
                ),
                "X" => GateExplanation::new(
                    "X · Pauli X",
                    format!("Swaps |0> and |1> on {names}; a bit flip."),
                ),
                "Y" => GateExplanation::new(
                    "Y · Pauli Y",
                    format!(
                        "Maps |0> to i|1> and |1> to −i|0> on {names}; a bit flip with opposite phases."
                    ),
                ),
                "Z" => GateExplanation::new(
                    "Z · Pauli Z",
                    format!(
                        "Adds a minus sign to |1> on {names}, leaving |0> unchanged; a relative phase flip."
                    ),
                ),
                "S" => GateExplanation::new(
                    "S · quarter-turn phase",
                    format!("Multiplies |1> by i (π/2 phase) on {names}, leaving |0> unchanged."),
                ),
                "T" => GateExplanation::new(
                    "T · eighth-turn phase",
                    format!("Adds π/4 phase to |1> on {names}, leaving |0> unchanged."),
                ),
                _ if label.starts_with("P(") => GateExplanation::new(
                    "P · phase",
                    format!(
                        "Adds {} rad of phase to |{}> on {names}. Other basis amplitudes are unchanged.",
                        number(phase(gate).expect("P symbols are checked phase matrices")),
                        dimension - 1,
                    ),
                ),
                _ if label.starts_with('F') => {
                    let inverse = label.ends_with('†');
                    GateExplanation::new(
                        if inverse {
                            "Inverse Fourier transform"
                        } else {
                            "Fourier transform"
                        },
                        format!(
                            "Mixes the {dimension} levels of {names} with equal-magnitude Fourier amplitudes and {} phase winding. {}",
                            if inverse { "negative" } else { "positive" },
                            if inverse {
                                "Undoes the forward transform."
                            } else {
                                "Maps each basis state to a phase-patterned superposition."
                            },
                        ),
                    )
                }
                _ => GateExplanation::new(
                    format!("U{dimension} · unitary operator"),
                    format!(
                        "Applies a {dimension} × {dimension} reversible matrix to ordered targets {names}. Numbered ports specify local basis order, not intermediate wires."
                    ),
                ),
            }
        }
    }
}

fn number(value: f64) -> String {
    let full = value.to_string();
    if full.len() <= 10 {
        full
    } else {
        format!("{value:.4e}")
    }
}

fn angle(value: &Angle) -> String {
    match value {
        Angle::Constant(value) => number(*value),
        Angle::Parameter {
            index,
            scale,
            offset,
        } => affine(&format!("θ{index}"), *scale, *offset),
        Angle::Input {
            index,
            scale,
            offset,
        } => affine(&format!("x{index}"), *scale, *offset),
    }
}

fn affine(variable: &str, scale: f64, offset: f64) -> String {
    if scale == 0. {
        return number(offset);
    }
    let scaled = match scale {
        1. => variable.to_owned(),
        -1. => format!("−{variable}"),
        _ => format!("{} × {variable}", number(scale)),
    };
    if offset == 0. {
        scaled
    } else {
        format!(
            "({scaled} {} {})",
            if offset < 0. { "−" } else { "+" },
            number(offset.abs())
        )
    }
}

fn bounded_list<T>(values: &[T], format: impl Fn(&T) -> String) -> String {
    let mut result = values
        .iter()
        .take(3)
        .map(format)
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > 3 {
        result.push_str(&format!(", … ({} more)", values.len() - 3));
    }
    result
}

fn wire_names(wires: &[usize]) -> String {
    bounded_list(wires, |wire| format!("q{wire}"))
}

fn conditions(controls: &[(usize, usize)]) -> String {
    bounded_list(controls, |(wire, level)| format!("q{wire} = |{level}>"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_quantum::{Matrix, Rotation, Unitary, gate::Axis};

    fn unitary(wires: Vec<usize>, gate: Unitary) -> GateExplanation {
        explain(&Operation::Unitary { wires, gate })
    }

    #[test]
    fn checked_matrices_supply_names_and_keep_ordered_targets() {
        assert!(unitary(vec![4], Unitary::h()).title.contains("Hadamard"));
        assert!(unitary(vec![4], Unitary::y()).summary.contains("−i|0>"));
        assert!(unitary(vec![4], Unitary::s()).summary.contains("π/2"));
        assert!(unitary(vec![4], Unitary::t()).summary.contains("π/4"));
        let cnot = unitary(vec![7, 2], Unitary::cnot());
        assert!(cnot.title.contains("CNOT"));
        assert!(cnot.summary.contains("Flips q2"));
        assert!(cnot.summary.contains("q7 = |1>"));
        let swap = unitary(vec![7, 2], Unitary::swap(2).unwrap());
        assert!(swap.summary.contains("q7 and q2"));
        let generic = unitary(vec![7, 2], Unitary::sum(3).unwrap());
        assert_eq!(generic.title, "U9 · unitary operator");
        assert!(generic.summary.contains("ordered targets q7, q2"));
    }

    #[test]
    fn controls_describe_levels_and_distinguish_global_phase() {
        let controlled = explain(&Operation::ControlledX {
            controls: vec![(4, 0), (2, 2)],
            target: 0,
        });
        assert!(controlled.summary.contains("q4 = |0>, q2 = |2>"));
        assert!(controlled.summary.contains("all controls"));
        let phase = explain(&Operation::PhaseFlip {
            controls: vec![(5, 0)],
        });
        assert!(phase.summary.contains("q5 = |0>"));
        assert!(phase.summary.contains("No basis values are flipped"));
        let global = explain(&Operation::PhaseFlip { controls: vec![] });
        assert!(global.title.contains("global phase"));
        assert!(global.summary.contains("every amplitude"));
        let unconditional = explain(&Operation::ControlledX {
            controls: vec![],
            target: 2,
        });
        assert_eq!(unconditional.summary, "Swaps |0> and |1> on q2.");
        let many = explain(&Operation::ControlledX {
            controls: (0..1000).map(|i| (i, 1)).collect(),
            target: 1000,
        });
        assert!(many.summary.contains("997 more"));
        assert!(many.summary.len() < 220);
    }

    #[test]
    fn embedded_rotations_keep_level_order_and_affine_bindings() {
        let description = explain(&Operation::Rotate {
            wire: 3,
            rotation: Rotation {
                axis: Axis::Y,
                levels: [2, 0],
                angle: Angle::Parameter {
                    index: 4,
                    scale: -2.,
                    offset: 0.25,
                },
            },
        });
        assert!(description.summary.contains("(-2 × θ4 + 0.25) rad"));
        assert!(description.summary.contains("ordered |2>, |0>"));
        assert!(description.summary.contains("Other levels are unchanged"));
        assert_eq!(
            angle(&Angle::Input {
                index: 1,
                scale: 0.5,
                offset: -0.25
            }),
            "(0.5 × x1 − 0.25)"
        );
        assert_eq!(
            angle(&Angle::Parameter {
                index: 9,
                scale: 0.,
                offset: 2.
            }),
            "2"
        );
    }

    #[test]
    fn phase_and_fourier_descriptions_respect_dimension_and_sign() {
        let qudit_phase = unitary(
            vec![6],
            Unitary::new(Matrix::real(3, vec![1., 0., 0., 0., 1., 0., 0., 0., -1.]).unwrap())
                .unwrap(),
        );
        assert_eq!(qudit_phase.title, "P · phase");
        assert!(qudit_phase.summary.contains("phase to |2> on q6"));
        let controlled = unitary(
            vec![3, 1],
            Unitary::phase(-0.25).unwrap().controlled(2, 1).unwrap(),
        );
        assert!(controlled.summary.contains("-0.25 rad"));
        assert!(controlled.summary.contains("|11> on ordered wires q3, q1"));
        let forward = unitary(vec![2], Unitary::fourier(3).unwrap());
        let inverse = unitary(vec![2], Unitary::fourier(3).unwrap().adjoint());
        assert!(forward.summary.contains("3 levels of q2"));
        assert!(forward.summary.contains("positive phase"));
        assert!(inverse.summary.contains("negative phase"));
        assert!(inverse.title.contains("Inverse"));
    }
}
