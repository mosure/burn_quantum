use crate::{
    app::{Explorer, ViewMode},
    model::VisualizationDocument,
};
use burn_quantum::{Angle, Operation};

pub(super) fn describe(state: &Explorer, document: Option<&VisualizationDocument>) -> String {
    let Some(doc) = document else {
        return "The simulation is loading.\n\nAll views share one\nvalidated experiment.".into();
    };
    let Some(frame) = doc.trace.frames.get(state.frame) else {
        return String::new();
    };
    match state.view {
        ViewMode::Circuit => {
            let Some(index) = state.gate else {
                return "INITIAL STATE\n\nThe register starts before\nany operation is applied."
                    .into();
            };
            let op_index = if let Some(record) = &doc.protocol {
                match record.program.instructions.get(index) {
                    Some(burn_quantum::protocol::Instruction::Measure { wire, bit }) => {
                        return format!(
                            "PROJECTIVE MEASUREMENT\n\nq{wire} → c{bit}={}\n\nCumulative branch probability {:.6}\n\nThe state is collapsed and normalized. Classical bits control Bob's corrections.",
                            record.outcomes[*bit],
                            record.probabilities[(index + 1).min(record.probabilities.len() - 1)]
                        );
                    }
                    Some(burn_quantum::protocol::Instruction::Gate { operation, .. }) => *operation,
                    None => return "Select a protocol step.".into(),
                }
            } else {
                index
            };
            let Some(op) = doc.circuit.operations().get(op_index) else {
                return "Select a circuit gate.".into();
            };
            let parameters = doc
                .optimization
                .get(state.frame)
                .map(|p| p.parameters.as_slice())
                .unwrap_or(&doc.parameters);
            let operation = match op {
                Operation::Unitary { wires, gate } => format!(
                    "Unitary on {wires:?}\n{} × {} matrix",
                    gate.matrix().dimension(),
                    gate.matrix().dimension()
                ),
                Operation::Rotate { wire, rotation } => {
                    let angle = match rotation.angle {
                        Angle::Constant(v) => v,
                        Angle::Parameter {
                            index,
                            scale,
                            offset,
                        } => scale * parameters[index] + offset,
                        Angle::Input {
                            index,
                            scale,
                            offset,
                        } => scale * doc.inputs[index] + offset,
                    };
                    format!(
                        "R{:?} on q{wire}\nLevels {:?}\nAngle {angle:+.5} rad",
                        rotation.axis, rotation.levels
                    )
                }
                Operation::ControlledX { controls, target } => {
                    format!("Controlled X → q{target}\nControls (wire, level)\n{controls:?}")
                }
                Operation::PhaseFlip { controls } => {
                    format!("Phase −1\nControls (wire, level)\n{controls:?}")
                }
            };
            format!(
                "GATE {:02}\n\n{}\n\n{operation}\n\nNorm {:.6}",
                index + 1,
                doc.operation_labels
                    .get(op_index)
                    .map(String::as_str)
                    .unwrap_or("Unlabeled operation"),
                frame.norm
            )
        }
        ViewMode::Fabric => {
            let Some(fabric) = &doc.fabric else {
                return "No physical schedule.\n\nChoose Photonic routing.".into();
            };
            if let Some(node) = state
                .node
                .and_then(|id| fabric.nodes.iter().find(|n| n.id == id))
            {
                let resources = fabric
                    .resources
                    .iter()
                    .filter(|r| r.source == node.id)
                    .map(|r| {
                        format!(
                            "Photons {:?}\nGraph edges {:?}\nLocal H {:?}",
                            r.photons, r.edges, r.hadamards
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let routes = fabric
                    .routes
                    .iter()
                    .filter(|r| {
                        r.from == node.id
                            || r.to == node.id
                            || r.hops.iter().any(|h| {
                                fabric.links[h.link].from == node.id
                                    || fabric.links[h.link].to == node.id
                            })
                    })
                    .take(6)
                    .map(|r| {
                        format!(
                            "p{}: t{} → {}\nRoute rails {:.2}% / {:.2}%",
                            r.photon,
                            r.release,
                            r.arrival,
                            r.transmission[0] * 100.,
                            r.transmission[1] * 100.
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                format!(
                    "{}\n\n{}\n\n{resources}\n\n{routes}\n\nStorage / tick\nRail 0 {:.2}% · Rail 1 {:.2}%\n\nRoute transmission includes links and intervening waits. It excludes analyzer/detector loss. Outputs are held at their sources. No fusion outcome is inferred.",
                    node.label,
                    node.kind,
                    node.storage[0] * 100.,
                    node.storage[1] * 100.
                )
            } else {
                format!(
                    "PHYSICAL NETWORK\n\n{} resource states\n{} photons\n{} routing links\n{} fusion events\n\nIdeally supplied resources.\nSchedule metadata only;\nno fusion outcome assumed.\n\nSelect a node to inspect.",
                    fabric.resource_count,
                    fabric.photon_count,
                    fabric.links.len(),
                    fabric.fusions.len()
                )
            }
        }
        ViewMode::Learning => {
            let Some(point) = doc.optimization.get(state.frame) else {
                return "Choose an optimization\nexperiment to inspect\nparameters and metric."
                    .into();
            };
            let parameters = point
                .parameters
                .iter()
                .take(4)
                .enumerate()
                .map(|(i, v)| format!("θ{i}  {v:+.5}"))
                .collect::<Vec<_>>()
                .join("\n");
            let diagonal = (0..point.parameters.len().min(4))
                .filter_map(|i| {
                    point
                        .metric
                        .get(i * point.parameters.len() + i)
                        .map(|v| format!("g{i}  {v:.5}"))
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "ITERATE {}\n\nParameters{}\n{parameters}\n\nMetric diagonal\n{diagonal}\n\nObjective {:.6}\n\n{} loss probes\n{} fidelity probes",
                point.iteration,
                if point.parameters.len() > 4 {
                    " (first 4)"
                } else {
                    ""
                },
                point.loss,
                point.loss_evaluations,
                point.fidelity_evaluations
            )
        }
        ViewMode::State => {
            let Some(q) = frame.subsystems.get(state.wire) else {
                return String::new();
            };
            let xyz = q
                .bloch
                .map(|v| {
                    let r = crate::model::bloch::BlochReadout::from_vector(v);
                    let phase = r.phase.map(|p| format!("{:.1}°", p.to_degrees())).unwrap_or_else(|| "undefined".into());
                    format!("P(0)  {:.2}%\nP(1)  {:.2}%\nPhase  {phase}\nCoherence  {:.3}\n\nx  {:+.3}\ny  {:+.3}\nz  {:+.3}",
                        r.probabilities[0]*100., r.probabilities[1]*100., r.coherence, v[0], v[1], v[2])
                })
                .unwrap_or_else(|| {
                    "Qudits use level\npopulations, not a\nqubit Bloch sphere.".into()
                });
            format!(
                "SUBSYSTEM q{}\n\n{} levels\nPurity {:.4}\n\n{xyz}\n\nState norm {:.6}",
                q.wire, q.dimension, q.purity, frame.norm
            )
        }
    }
}
