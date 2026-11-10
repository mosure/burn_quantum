//! Canonical symbols, conservative parallel columns and explicit classical wires.
use super::{
    plot::{Hit, PlotCache, short},
    theme::*,
};
use crate::{
    app::Explorer,
    model::{
        TraceKind, VisualizationDocument,
        diagram::{self, GateSymbol},
    },
};
use bevy::prelude::*;
use burn_quantum::protocol::Instruction;
struct Step {
    symbol: Option<GateSymbol>,
    measurement: Option<(usize, usize)>,
    conditions: Vec<(usize, usize)>,
    column: usize,
}
fn steps(doc: &VisualizationDocument) -> Vec<Step> {
    let instructions = doc
        .protocol
        .as_ref()
        .map(|p| p.program.instructions.clone())
        .unwrap_or_else(|| {
            (0..doc.circuit.operations().len())
                .map(|operation| Instruction::Gate {
                    operation,
                    conditions: vec![],
                })
                .collect()
        });
    let mut next = vec![
        0;
        doc.circuit.register().wires()
            + doc.protocol.as_ref().map(|p| p.outcomes.len()).unwrap_or(0)
    ];
    let n = doc.circuit.register().wires();
    instructions
        .iter()
        .map(|i| {
            let (symbol, measurement, conditions, wires) = match i {
                Instruction::Gate {
                    operation,
                    conditions,
                } => {
                    let s = diagram::symbol(&doc.circuit.operations()[*operation]);
                    let mut w = s.wires();
                    w.extend(conditions.iter().map(|(b, _)| n + b));
                    (Some(s), None, conditions.clone(), w)
                }
                Instruction::Measure { wire, bit } => {
                    (None, Some((*wire, *bit)), vec![], vec![*wire, n + bit])
                }
            };
            let min = *wires.iter().min().unwrap();
            let max = *wires.iter().max().unwrap();
            // Reserve the entire vertical span, avoiding ambiguous connector crossings.
            let column = *next[min..=max].iter().max().unwrap();
            next[min..=max].fill(column + 1);
            Step {
                symbol,
                measurement,
                conditions,
                column,
            }
        })
        .collect()
}
pub(super) fn build(
    p: &mut PlotCache,
    bounds: Rect,
    state: &Explorer,
    doc: &VisualizationDocument,
) {
    let diagram = steps(doc);
    let n = doc.circuit.register().wires();
    let bits = doc.protocol.as_ref().map(|p| p.outcomes.len()).unwrap_or(0);
    let rows = n + bits;
    let row = ((bounds.height() - 112.) / rows.max(1) as f32).clamp(25., 66.);
    let spacing = 88. * state.zoom;
    let active = state.gate.unwrap_or(0).min(diagram.len().saturating_sub(1));
    let columns = diagram.iter().map(|s| s.column + 1).max().unwrap_or(1);
    let automatic = if state.pan.x == 0. && state.frame > 0 {
        (diagram.get(active).map(|s| s.column).unwrap_or(0) as f32 * spacing
            - (bounds.width() - 180.))
            .max(0.)
    } else {
        0.
    };
    let start = bounds.min + Vec2::new(88. - automatic, 65.) + state.pan;
    let at = |col: usize, wire: usize| {
        Vec2::new(start.x + col as f32 * spacing, start.y + wire as f32 * row)
    };
    let half = (spacing * 0.42).min(37.);
    let height = (row * 0.64).min(32.);
    let rect = |col: usize, wire: usize| {
        Rect::from_center_size(at(col, wire), Vec2::new(2. * half, height))
    };
    p.text(
        Vec2::new(bounds.center().x, bounds.min.y + 15.),
        if bits > 0 {
            "Quantum wires + double classical wires · explicit measurement branch"
        } else {
            "Wire 0 is most significant · disjoint gates share columns"
        },
        12.,
        MUTED,
    );
    for wire in 0..rows {
        let y = at(0, wire).y;
        let mut spans = vec![];
        for step in &diagram {
            if blocks(step, wire) {
                let r = rect(step.column, wire);
                spans.push((r.min.x - 1., r.max.x + 1.));
            }
        }
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut cursor = if wire < n {
            bounds.min.x + 52.
        } else {
            diagram
                .iter()
                .find(|s| s.measurement.is_some_and(|(_, b)| n + b == wire))
                .map(|s| at(s.column, wire).x)
                .unwrap_or(bounds.max.x)
        };
        let line = |p: &mut PlotCache, a: f32, b: f32| {
            if b > a {
                if wire < n {
                    p.line(Vec2::new(a, y), Vec2::new(b, y), GUIDE)
                } else {
                    double(p, Vec2::new(a, y), Vec2::new(b, y), GOLD.with_alpha(0.6))
                }
            }
        };
        for (left, right) in spans {
            let left = left.clamp(bounds.min.x + 52., bounds.max.x - 10.);
            let right = right.clamp(bounds.min.x + 52., bounds.max.x - 10.);
            line(p, cursor, left);
            cursor = cursor.max(right);
        }
        line(p, cursor.max(bounds.min.x + 52.), bounds.max.x - 10.);
        p.text(
            Vec2::new(bounds.min.x + 24., y),
            if wire < n {
                format!("q{wire}")
            } else {
                format!("c{}", wire - n)
            },
            12.,
            if wire < n { CYAN } else { GOLD },
        );
        if wire < n {
            p.hit(
                Rect::from_center_size(Vec2::new(bounds.min.x + 24., y), Vec2::splat(28.)),
                Hit::Wire(wire),
            );
        }
    }
    let mut visible = 0;
    let mut labeled_columns = std::collections::BTreeSet::new();
    for (index, step) in diagram.iter().enumerate() {
        let x = at(step.column, 0).x;
        if x < bounds.min.x + 55. || x > bounds.max.x - half {
            continue;
        }
        visible += 1;
        let color = if state.gate == Some(index) {
            GOLD
        } else if state.frame > index || doc.trace.kind == TraceKind::Optimization {
            CYAN
        } else {
            MUTED
        };
        let box_at = |p: &mut PlotCache, wire: usize, label: String| {
            let r = rect(step.column, wire);
            p.rect(r, PANEL);
            p.outline(r, color);
            p.text(r.center(), label, 11., color);
        };
        if let Some((wire, bit)) = step.measurement {
            box_at(p, wire, "M".into());
            double(
                p,
                at(step.column, wire) + Vec2::Y * height / 2.,
                at(step.column, n + bit),
                GOLD,
            );
            let value = doc.protocol.as_ref().unwrap().outcomes[bit];
            p.text(
                at(step.column, n + bit) + Vec2::new(18., -10.),
                value.to_string(),
                10.,
                GOLD,
            );
        }
        if let Some(symbol) = &step.symbol {
            match symbol {
                GateSymbol::ControlledX { controls, target } => {
                    let wires = symbol.wires();
                    connector(p, &wires, |w| at(step.column, w), color);
                    for &(w, v) in controls {
                        control(p, at(step.column, w), v, color);
                    }
                    let pos = at(step.column, *target);
                    p.circle(pos, 10., color);
                    p.line(pos - Vec2::X * 10., pos + Vec2::X * 10., color);
                    p.line(pos - Vec2::Y * 10., pos + Vec2::Y * 10., color);
                }
                GateSymbol::Swap(w) => {
                    connector(p, w, |w| at(step.column, w), color);
                    for &wire in w {
                        let pos = at(step.column, wire);
                        p.line(pos - Vec2::splat(7.), pos + Vec2::splat(7.), color);
                        p.line(pos + Vec2::new(-7., 7.), pos + Vec2::new(7., -7.), color);
                    }
                }
                GateSymbol::ControlledPhase {
                    control: c,
                    target,
                    angle,
                } => {
                    let from = at(step.column, *c);
                    let to = at(step.column, *target);
                    p.line(
                        from,
                        to - Vec2::Y * (to.y - from.y).signum() * height / 2.,
                        color,
                    );
                    control(p, from, 1, color);
                    box_at(p, *target, format!("P({angle:.2})"));
                }
                GateSymbol::PhaseFlip(controls) => {
                    if controls.is_empty() {
                        box_at(p, 0, "−I".into())
                    } else {
                        connector(p, &symbol.wires(), |w| at(step.column, w), color);
                        for &(w, v) in controls {
                            control(p, at(step.column, w), v, color);
                        }
                    }
                }
                GateSymbol::Box { wires, label } => {
                    if wires.len() == 1 {
                        box_at(p, wires[0], label.clone());
                    } else {
                        let low = *wires.iter().min().unwrap();
                        let high = *wires.iter().max().unwrap();
                        let r = Rect::from_corners(
                            rect(step.column, low).min,
                            rect(step.column, high).max,
                        );
                        p.rect(r, PANEL);
                        p.outline(r, color);
                        p.text(r.center(), label, 12., color);
                        for (port, &wire) in wires.iter().enumerate() {
                            p.text(
                                at(step.column, wire) - Vec2::X * (half - 9.),
                                port.to_string(),
                                9.,
                                color,
                            );
                        }
                    }
                }
            }
            for &(bit, value) in &step.conditions {
                let wire = *symbol.wires().iter().max().unwrap();
                let a = at(step.column, n + bit);
                let b = at(step.column, wire) + Vec2::Y * height / 2.;
                double(p, a, b, GOLD);
                control(p, a, value, GOLD);
            }
        }
        let wires = step
            .symbol
            .as_ref()
            .map(GateSymbol::wires)
            .unwrap_or_else(|| vec![step.measurement.unwrap().0]);
        for wire in wires {
            p.hit(rect(step.column, wire), Hit::Gate(index));
        }
        if state.grid && labeled_columns.insert(step.column) {
            p.text(
                Vec2::new(x, start.y - 29.),
                format!("{}", step.column + 1),
                10.,
                MUTED,
            );
        }
    }
    let branch = doc
        .protocol
        .as_ref()
        .map(|p| format!(" · branch likelihood {:.4}", p.probabilities[state.frame]))
        .unwrap_or_default();
    p.text(
        Vec2::new(bounds.center().x, bounds.max.y - 12.),
        format!(
            "{}{}",
            short(
                &doc.trace.frames[state.frame].label,
                if bounds.width() < 500. { 38 } else { 95 }
            ),
            branch
        ),
        11.,
        TEXT,
    );
    p.description = format!(
        "{visible} visible / {} steps · {columns} columns\n{n} quantum + {bits} classical wires\nDrag to pan · scroll to zoom · timeline follows execution",
        diagram.len()
    );
}
fn blocks(step: &Step, wire: usize) -> bool {
    if step.measurement.is_some_and(|(w, _)| w == wire) {
        return true;
    }
    match &step.symbol {
        Some(GateSymbol::Box { wires, .. }) => {
            wire >= *wires.iter().min().unwrap() && wire <= *wires.iter().max().unwrap()
        }
        Some(GateSymbol::ControlledPhase { target, .. }) => wire == *target,
        Some(GateSymbol::PhaseFlip(c)) => c.is_empty() && wire == 0,
        _ => false,
    }
}
fn connector(p: &mut PlotCache, wires: &[usize], at: impl Fn(usize) -> Vec2, color: Color) {
    if let (Some(a), Some(b)) = (wires.iter().min(), wires.iter().max()) {
        p.line(at(*a), at(*b), color);
    }
}
fn double(p: &mut PlotCache, a: Vec2, b: Vec2, color: Color) {
    let v = b - a;
    let side = Vec2::new(-v.y, v.x).normalize_or_zero() * 2.;
    p.line(a + side, b + side, color);
    p.line(a - side, b - side, color);
}
fn control(p: &mut PlotCache, at: Vec2, value: usize, color: Color) {
    if value <= 1 {
        p.rect(Rect::from_center_size(at, Vec2::splat(12.)), BACKGROUND);
        p.circle(at, 5., color);
        if value == 1 {
            p.circle(at, 3., color);
            p.circle(at, 1., color);
        }
    } else {
        p.text(at, format!("={value}"), 11., color);
    }
}
