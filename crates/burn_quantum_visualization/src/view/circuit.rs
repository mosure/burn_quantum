use super::{
    plot::{Hit, PlotCache, short},
    theme::*,
};
use crate::{
    app::Explorer,
    model::{TraceKind, VisualizationDocument},
};
use bevy::prelude::*;
use burn_quantum::circuit::Operation;
pub(super) fn build(
    p: &mut PlotCache,
    bounds: Rect,
    state: &Explorer,
    doc: &VisualizationDocument,
) {
    let count = doc.circuit.register().dimensions().len().min(16);
    let row = (bounds.height() - 130.).max(100.) / count.max(1) as f32;
    let start = bounds.min + Vec2::new(95., 80.) + state.pan;
    let col = 104. * state.zoom;
    p.text(
        Vec2::new(bounds.center().x, bounds.min.y + 20.),
        if doc.trace.kind == TraceKind::Optimization {
            "Parameterized ansatz · state follows the selected optimizer iterate"
        } else {
            "Circuit execution · wire 0 is the most significant subsystem"
        },
        13.,
        MUTED,
    );
    for wire in 0..count {
        let y = start.y + wire as f32 * row;
        let mut cursor = bounds.min.x + 50.;
        for (index, op) in doc.circuit.operations().iter().take(256).enumerate() {
            if let Some(half) = box_half_width(op, wire, col) {
                let x = start.x + index as f32 * col;
                let left = (x - half).clamp(bounds.min.x + 50., bounds.max.x - 15.);
                let right = (x + half).clamp(bounds.min.x + 50., bounds.max.x - 15.);
                if left > cursor {
                    p.line(Vec2::new(cursor, y), Vec2::new(left, y), GUIDE);
                }
                cursor = cursor.max(right);
            }
        }
        if cursor < bounds.max.x - 15. {
            p.line(
                Vec2::new(cursor, y),
                Vec2::new(bounds.max.x - 15., y),
                GUIDE,
            );
        }
        p.text(
            Vec2::new(bounds.min.x + 24., y),
            format!("q{wire}"),
            14.,
            if state.wire == wire { CYAN } else { MUTED },
        );
        p.hit(
            Rect::from_center_size(Vec2::new(bounds.min.x + 24., y), Vec2::splat(38.)),
            Hit::Wire(wire),
        );
    }
    let mut visible = 0;
    for (i, op) in doc.circuit.operations().iter().take(256).enumerate() {
        let x = start.x + i as f32 * col;
        if x < bounds.min.x + 58. || x > bounds.max.x - 25. {
            continue;
        }
        visible += 1;
        let wires = match op {
            Operation::Unitary { wires, .. } => wires.clone(),
            Operation::Rotate { wire, .. } => vec![*wire],
            Operation::ControlledX { controls, target } => {
                controls.iter().map(|(w, _)| *w).chain([*target]).collect()
            }
            Operation::PhaseFlip { controls } => {
                if controls.is_empty() {
                    vec![0]
                } else {
                    controls.iter().map(|(w, _)| *w).collect()
                }
            }
        };
        let wires = wires.into_iter().filter(|w| *w < count).collect::<Vec<_>>();
        if wires.is_empty() {
            continue;
        }
        let min = *wires.iter().min().unwrap();
        let max = *wires.iter().max().unwrap();
        let executed = doc.trace.kind == TraceKind::Optimization || state.frame > i;
        let color = if state.gate == Some(i) {
            GOLD
        } else if executed {
            CYAN
        } else {
            MUTED
        };
        if matches!(op, Operation::Unitary { .. } | Operation::Rotate { .. })
            && box_half_width(op, min, col).is_some()
        {
            let mut sorted = wires.clone();
            sorted.sort_unstable();
            for pair in sorted.windows(2) {
                let a = start.y + pair[0] as f32 * row + 19.;
                let b = start.y + pair[1] as f32 * row - 19.;
                if b > a {
                    p.line(Vec2::new(x, a), Vec2::new(x, b), color);
                }
            }
        } else {
            p.line(
                Vec2::new(x, start.y + min as f32 * row),
                Vec2::new(x, start.y + max as f32 * row),
                color,
            );
        }
        let location = |wire: usize| Vec2::new(x, start.y + wire as f32 * row);
        let controlled = match op {
            Operation::ControlledX { controls, target } => Some((controls.clone(), *target)),
            Operation::Unitary { wires, gate }
                if wires.len() == 2 && gate.matrix() == burn_quantum::Unitary::cnot().matrix() =>
            {
                Some((vec![(wires[0], 1)], wires[1]))
            }
            _ => None,
        };
        if let Some((controls, target)) = controlled {
            for (wire, value) in controls {
                control(p, location(wire), value, color);
            }
            let at = location(target);
            p.circle(at, 13., color);
            p.line(at - Vec2::X * 10., at + Vec2::X * 10., color);
            p.line(at - Vec2::Y * 10., at + Vec2::Y * 10., color);
        } else if let Operation::PhaseFlip { controls } = op {
            if controls.is_empty() {
                p.text(location(0), "global −1", 13., color);
            }
            for &(wire, value) in controls {
                control(p, location(wire), value, color);
            }
            p.text(
                Vec2::new(x, start.y + max as f32 * row + 27.),
                "phase −1",
                11.,
                color,
            );
        } else {
            for &wire in &wires {
                let center = location(wire);
                let rect = Rect::from_center_size(center, Vec2::new((col * 0.78).min(90.), 38.));
                p.rect(rect, if executed { INSET } else { PANEL });
                p.outline(rect, color);
                let label = doc
                    .operation_labels
                    .get(i)
                    .map(String::as_str)
                    .unwrap_or("U");
                p.text(center, short(label, 10), 13., color);
            }
        }
        for &wire in &wires {
            p.hit(
                Rect::from_center_size(location(wire), Vec2::new((col * 0.8).min(90.), 38.)),
                Hit::Gate(i),
            );
        }
        if state.grid {
            p.text(
                Vec2::new(x, start.y - 34.),
                format!("{:02}", i + 1),
                11.,
                MUTED,
            );
        }
    }
    p.description = format!(
        "{} visible gates / {} total\n{} visible wires · cap 256 gates",
        visible,
        doc.circuit.operations().len(),
        count
    );
    p.text(
        Vec2::new(bounds.center().x, bounds.max.y - 25.),
        format!(
            "{}   ·   {}",
            doc.trace
                .frames
                .get(state.frame)
                .map(|f| f.label.as_str())
                .unwrap_or("Initial state"),
            if state.gate.is_some() {
                "Selected gate in amber"
            } else {
                "Click a gate to seek"
            }
        ),
        13.,
        TEXT,
    );
}

fn control(p: &mut PlotCache, at: Vec2, value: usize, color: Color) {
    if value <= 1 {
        p.rect(Rect::from_center_size(at, Vec2::splat(16.)), BACKGROUND);
        p.circle(at, 6., color);
        if value == 1 {
            p.circle(at, 4., color);
            p.circle(at, 2., color);
        }
    } else {
        p.rect(Rect::from_center_size(at, Vec2::splat(25.)), PANEL);
        p.outline(Rect::from_center_size(at, Vec2::splat(25.)), color);
        p.text(at, value.to_string(), 12., color);
    }
}

fn box_half_width(op: &Operation, wire: usize, col: f32) -> Option<f32> {
    match op {
        Operation::Unitary { wires, gate }
            if wires.contains(&wire)
                && !(wires.len() == 2
                    && gate.matrix() == burn_quantum::Unitary::cnot().matrix()) =>
        {
            Some((col * 0.78).min(90.) * 0.5 + 2.)
        }
        Operation::Rotate { wire: w, .. } if *w == wire => Some((col * 0.78).min(90.) * 0.5 + 2.),
        Operation::PhaseFlip { controls } if controls.is_empty() && wire == 0 => Some(38.),
        Operation::ControlledX { controls, .. } | Operation::PhaseFlip { controls }
            if controls.iter().any(|&(w, l)| w == wire && l > 1) =>
        {
            Some(14.)
        }
        _ => None,
    }
}
