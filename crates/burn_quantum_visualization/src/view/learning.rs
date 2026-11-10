use super::{plot::PlotCache, theme::*};
use crate::{app::Explorer, model::VisualizationDocument};
use bevy::prelude::*;
pub(super) fn build(
    p: &mut PlotCache,
    bounds: Rect,
    state: &Explorer,
    doc: &VisualizationDocument,
) {
    if doc.optimization.is_empty() {
        p.text(
            bounds.center() + Vec2::new(0., -25.),
            "Watch a quantum model learn.",
            25.,
            TEXT,
        );
        p.text(
            bounds.center() + Vec2::new(0., 15.),
            "Choose the optimization experiment for an actual QN-SPSA trajectory.",
            13.,
            MUTED,
        );
        p.description = "No optimization history attached".into();
        return;
    }
    let points = &doc.optimization;
    let classification = doc
        .study
        .as_ref()
        .and_then(|study| study.classification.as_ref());
    let validation = classification
        .map(|c| c.validation_loss.as_slice())
        .unwrap_or(&[]);
    let area = Rect::from_corners(
        bounds.min + Vec2::new(75., 85.),
        bounds.max - Vec2::new(35., 100.),
    );
    let scale = points
        .iter()
        .map(|p| p.loss.abs())
        .chain(validation.iter().map(|v| v.abs()))
        .fold(1f64, f64::max);
    let lo = points
        .iter()
        .map(|x| x.loss / scale)
        .chain(validation.iter().map(|v| v / scale))
        .fold(f64::INFINITY, f64::min)
        .min(0.);
    let hi = points
        .iter()
        .map(|x| x.loss / scale)
        .chain(validation.iter().map(|v| v / scale))
        .fold(f64::NEG_INFINITY, f64::max)
        .max(lo + 1e-5);
    let range = (hi - lo) * 1.15;
    let to_point = |i: usize, loss: f64| {
        let raw = Vec2::new(
            area.min.x + area.width() * i as f32 / points.len().saturating_sub(1).max(1) as f32,
            area.max.y - area.height() * ((loss - lo) / range) as f32,
        );
        area.center() + (raw - area.center()) * state.zoom + state.pan
    };
    p.text(
        Vec2::new(bounds.center().x, bounds.min.y + 25.),
        "Optimization history",
        21.,
        TEXT,
    );
    p.text(
        Vec2::new(bounds.center().x, bounds.min.y + 51.),
        if scale > 1e5 {
            format!("Objective axis / {scale:.3e}")
        } else {
            if classification.is_some() && bounds.width() < 560. {
                "Train / validation · selected checkpoint"
            } else if classification.is_some() {
                "Training objective · validation loss · fixed selected checkpoint"
            } else {
                "Recorded objective at each parameter iterate"
            }
            .into()
        },
        12.,
        MUTED,
    );
    if state.grid {
        for i in 0..=4 {
            let v = lo + range * i as f64 / 4.;
            let y = to_point(0, v).y;
            p.line(Vec2::new(area.min.x, y), Vec2::new(area.max.x, y), GUIDE);
            p.text(
                Vec2::new(area.min.x - 35., y),
                if scale > 1e5 {
                    format!("{v:.2}")
                } else {
                    format!("{:.2}", v * scale)
                },
                11.,
                MUTED,
            );
        }
    }
    for i in 0..=4 {
        let index = i * (points.len() - 1) / 4;
        let x = to_point(index, lo).x;
        p.text(
            Vec2::new(x, area.max.y + 13.),
            points[index].iteration.to_string(),
            11.,
            MUTED,
        );
    }
    p.text(
        Vec2::new(area.center().x, area.max.y + 47.),
        "Completed optimizer updates",
        11.,
        MUTED,
    );
    let stride = points.len().div_ceil(256).max(1);
    let mut last = None;
    for (i, point) in points.iter().enumerate().step_by(stride) {
        let xy = to_point(i, point.loss / scale);
        if let Some(prev) = last {
            p.line(prev, xy, BLUE.with_alpha(0.3));
        }
        last = Some(xy);
    }
    let current = state.frame.min(points.len() - 1);
    // A circular marker is 128 line segments; bound markers separately from
    // curve samples so long histories cannot exhaust PlotCache before the
    // validation curve and selected-checkpoint marker are emitted.
    let marker_stride = points.len().div_ceil(24).max(1);
    let mut last = None;
    for (i, point) in points.iter().enumerate().take(current + 1).step_by(stride) {
        let xy = to_point(i, point.loss / scale);
        if let Some(prev) = last {
            p.line(prev, xy, ACCENT);
        }
        if i.is_multiple_of(marker_stride) || i == current {
            p.dot(xy, 2., ACCENT);
        }
        last = Some(xy);
    }
    let mut last = None;
    for (i, &loss) in validation.iter().enumerate().step_by(stride) {
        let xy = to_point(i, loss / scale);
        if let Some(prev) = last {
            p.line(
                prev,
                xy,
                GOLD.with_alpha(if i <= current { 0.95 } else { 0.3 }),
            );
        }
        last = Some(xy);
    }
    if let Some(c) = classification {
        let checkpoint = c.selected_iteration.min(points.len() - 1);
        let x = to_point(checkpoint, lo).x;
        for i in 0..12 {
            let y = area.min.y + area.height() * i as f32 / 12.;
            p.line(Vec2::new(x, y), Vec2::new(x, y + area.height() / 24.), BLUE);
        }
        p.dot(
            to_point(checkpoint, validation[checkpoint] / scale),
            6.,
            BLUE,
        );
        let label_y = bounds.min.y + 70.;
        p.text(
            Vec2::new(area.min.x + area.width() * 0.16, label_y),
            "Training",
            11.,
            ACCENT,
        );
        p.text(Vec2::new(area.center().x, label_y), "Validation", 11., GOLD);
        p.text(
            Vec2::new(area.min.x + area.width() * 0.84, label_y),
            format!("Selected {checkpoint}"),
            11.,
            BLUE,
        );
    }
    let selected = &points[current];
    let xy = to_point(current, selected.loss / scale);
    p.line(
        Vec2::new(xy.x, area.min.y),
        Vec2::new(xy.x, area.max.y),
        GOLD.with_alpha(0.5),
    );
    p.dot(xy, 6., GOLD);
    p.text(
        Vec2::new(area.center().x, area.max.y + 70.),
        if bounds.width() < 560. && !validation.is_empty() {
            format!(
                "Iter {} · train {:.3} · val {:.3}",
                selected.iteration, selected.loss, validation[current]
            )
        } else {
            format!(
                "Iteration {} · objective {}{}",
                selected.iteration,
                if selected.loss.abs() > 1e5 {
                    format!("{:.3e}", selected.loss)
                } else {
                    format!("{:.5}", selected.loss)
                },
                validation
                    .get(current)
                    .map(|v| format!(" · validation {v:.5}"))
                    .unwrap_or_default()
            )
        },
        15.,
        TEXT,
    );
    p.text(
        Vec2::new(area.center().x, area.max.y + 92.),
        format!(
            "{} loss evaluations  +  {} fidelity evaluations",
            selected.loss_evaluations, selected.fidelity_evaluations
        ),
        12.,
        MUTED,
    );
    p.description = format!(
        "{} optimizer iterates\n≤256 points per curve · actual evaluation costs",
        points.len()
    );
}
