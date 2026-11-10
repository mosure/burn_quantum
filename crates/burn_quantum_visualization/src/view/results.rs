//! Compact task-first outcomes; the Details panel retains every metric and assumption.
use super::{
    plot::{PlotCache, short},
    theme::*,
};
use crate::{app::Explorer, model::*};
use bevy::prelude::*;

pub(super) fn build(p: &mut PlotCache, bounds: Rect, _: &Explorer, doc: &VisualizationDocument) {
    let Some(study) = &doc.study else {
        return;
    };
    let area = Rect::from_corners(bounds.min + Vec2::splat(12.), bounds.max - Vec2::splat(12.));
    let compact = area.height() < 360.;
    let header = if area.width() < 520. { 17. } else { 21. };
    let mut y = paragraph(p, area, area.min.y + 12., &study.title, header, TEXT, 2);
    y = paragraph(
        p,
        area,
        y + 6.,
        &study.summary,
        12.,
        MUTED,
        if compact { 2 } else { 3 },
    ) + 12.;
    let count = study.metrics.len().min(if compact { 2 } else { 4 });
    let columns = if area.width() >= 760. { 4 } else { 2 }.min(count.max(1));
    let rows = count.div_ceil(columns);
    let gap = 10.;
    let width = (area.width() - gap * (columns - 1) as f32) / columns as f32;
    for (i, metric) in study.metrics.iter().take(count).enumerate() {
        let min = Vec2::new(
            area.min.x + (i % columns) as f32 * (width + gap),
            y + (i / columns) as f32 * 80.,
        );
        metric_card(
            p,
            Rect::from_corners(min, min + Vec2::new(width, 70.)),
            metric,
            i == 0,
        );
    }
    y += rows as f32 * 80. + 8.;
    let body = Rect::from_corners(Vec2::new(area.min.x, y), area.max - Vec2::Y * 16.);
    if body.height() >= 100. {
        if let Some(classification) = &study.classification {
            classifier(p, body, classification);
        } else if let Some(sat) = &study.sat {
            satisfiability(p, body, sat);
        } else if let Some(photonic) = &study.photonic {
            photons(p, body, photonic);
        } else {
            // VQE and other studies can use the same view without a new UI schema.
            for metric in study
                .metrics
                .iter()
                .skip(count)
                .take((body.height() / 32.) as usize)
            {
                y = paragraph(
                    p,
                    body,
                    y + 4.,
                    &format!("{}: {} {}", metric.label, number(metric.value), metric.unit),
                    13.,
                    TEXT,
                    1,
                ) + 12.;
            }
        }
    }
    p.text(
        Vec2::new(area.center().x, area.max.y - 1.),
        "Details: all metrics, references and assumptions",
        10.,
        MUTED,
    );
    p.description = format!(
        "{} metrics · explicit task readouts\nResults are independent of the selected display frame",
        study.metrics.len()
    );
}

fn metric_card(p: &mut PlotCache, rect: Rect, metric: &StudyMetric, primary: bool) {
    p.rect(
        rect,
        if primary {
            ACCENT_SOFT.with_alpha(0.35)
        } else {
            PANEL
        },
    );
    p.line(
        rect.min,
        Vec2::new(rect.max.x, rect.min.y),
        if primary { ACCENT } else { BORDER },
    );
    paragraph(p, rect, rect.min.y + 13., &metric.label, 11., MUTED, 2);
    let value = if metric.unit.is_empty() {
        number(metric.value)
    } else {
        format!("{} {}", number(metric.value), metric.unit)
    };
    p.text(
        Vec2::new(rect.center().x, rect.max.y - 18.),
        short(&value, (rect.width() / 9.).max(8.) as usize),
        19.,
        if primary { ACCENT } else { TEXT },
    );
    p.hover(
        rect,
        &metric.label,
        format!("{:.8} {}", metric.value, metric.unit),
    );
}

fn classifier(p: &mut PlotCache, area: Rect, result: &ClassificationSummary) {
    let n = result.classes.len();
    let side = (area.height() - 70.)
        .min(area.width() - 80.)
        .clamp(48., 230.);
    let cell = side / n as f32;
    let top = area.min.y + 27.;
    let left = area.center().x - side / 2. + 16.;
    p.text(
        Vec2::new(area.center().x, area.min.y + 4.),
        "Untouched test set · true class ↓ / predicted →",
        12.,
        TEXT,
    );
    let max = result
        .confusion
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1);
    for (i, row) in result.confusion.iter().enumerate() {
        p.text(
            Vec2::new(left - 20., top + (i as f32 + 0.5) * cell),
            (i + 1).to_string(),
            12.,
            MUTED,
        );
        p.text(
            Vec2::new(left + (i as f32 + 0.5) * cell, top + side + 11.),
            (i + 1).to_string(),
            12.,
            MUTED,
        );
        for (j, &count) in row.iter().enumerate() {
            let min = Vec2::new(left + j as f32 * cell, top + i as f32 * cell);
            let rect = Rect::from_corners(min + Vec2::splat(2.), min + Vec2::splat(cell - 2.));
            let color = if i == j { ACCENT } else { GOLD };
            p.rect(
                rect,
                color.with_alpha(0.07 + 0.45 * count as f32 / max as f32),
            );
            p.text(
                rect.center(),
                count.to_string(),
                if cell > 45. { 21. } else { 13. },
                if count == 0 { MUTED } else { TEXT },
            );
            p.hover(rect, format!("True {} → predicted {}", result.classes[i], result.classes[j]), format!("{count} held-out test samples. Rows are true labels; columns are model predictions."));
        }
    }
    let key = result
        .classes
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{} {}", i + 1, c))
        .collect::<Vec<_>>()
        .join(" · ");
    paragraph(p, area, top + side + 32., &key, 11., MUTED, 2);
    if area.height() > side + 120. {
        paragraph(
            p,
            area,
            area.max.y - 18.,
            &format!(
                "Train / validation / test: {} / {} / {} · selected update {}",
                result.split_sizes[0],
                result.split_sizes[1],
                result.split_sizes[2],
                result.selected_iteration
            ),
            11.,
            MUTED,
            2,
        );
    }
}

fn satisfiability(p: &mut PlotCache, area: Rect, sat: &SatSummary) {
    let y = area.min.y + 7.;
    p.text(
        Vec2::new(area.center().x, y),
        format!(
            "3SAT · {} variables · {} clauses",
            sat.variables,
            sat.clauses.len()
        ),
        14.,
        TEXT,
    );
    let labels = [
        ("Quantum satisfying mass", sat.success_probability, ACCENT),
        ("Uniform assignment", sat.uniform_probability, BLUE),
    ];
    let track = Rect::from_corners(
        Vec2::new(area.min.x + 12., y + 36.),
        Vec2::new(area.max.x - 12., y + 46.),
    );
    for (i, (label, value, color)) in labels.into_iter().enumerate() {
        let r = Rect::from_corners(
            track.min + Vec2::Y * (i as f32 * 38.),
            track.max + Vec2::Y * (i as f32 * 38.),
        );
        p.rect(r, PANEL);
        p.rect(
            Rect::from_corners(
                r.min,
                Vec2::new(r.min.x + r.width() * value.clamp(0., 1.) as f32, r.max.y),
            ),
            color,
        );
        p.text(
            Vec2::new(area.center().x, r.min.y - 12.),
            format!("{label} · {:.2}%", 100. * value),
            11.,
            color,
        );
    }
    let witness = sat.witness.as_ref().map(|bits| {
        bits.iter()
            .enumerate()
            .map(|(i, b)| format!("x{}={}", i + 1, usize::from(*b)))
            .collect::<Vec<_>>()
            .join("  ")
    });
    let mut y = paragraph(
        p,
        area,
        y + 105.,
        &witness
            .map(|w| format!("Verified witness: {w}"))
            .unwrap_or_else(|| {
                "No satisfying assignment in the exact classical enumeration.".into()
            }),
        12.,
        TEXT,
        2,
    ) + 14.;
    let available = ((area.max.y - y) / 20.).floor().max(0.) as usize;
    for (i, clause) in sat.clauses.iter().take(available.min(8)).enumerate() {
        let text = clause
            .iter()
            .map(|literal| {
                format!(
                    "{}x{}",
                    if *literal < 0 { "!" } else { "" },
                    literal.unsigned_abs()
                )
            })
            .collect::<Vec<_>>()
            .join(" OR ");
        p.text(
            Vec2::new(area.center().x, y),
            format!("C{}: ({text})", i + 1),
            12.,
            MUTED,
        );
        y += 20.;
    }
    if available < sat.clauses.len() || sat.clauses.len() > 8 {
        p.text(
            Vec2::new(area.center().x, area.max.y - 5.),
            "Full formula and witness in Details / JSON",
            10.,
            MUTED,
        );
    }
}

fn photons(p: &mut PlotCache, area: Rect, data: &PhotonicSummary) {
    let counts = [
        data.delivered,
        data.attempts.saturating_sub(data.delivered + data.erased),
        data.erased,
    ];
    let names = ["Delivered pair", "Ordinary failure", "Loss / erasure"];
    let colors = [ACCENT, BLUE, GOLD];
    p.text(
        Vec2::new(area.center().x, area.min.y + 6.),
        format!(
            "Every attempt counts · {} unpostselected trials",
            data.attempts
        ),
        13.,
        TEXT,
    );
    let track = Rect::from_corners(
        area.min + Vec2::new(12., 30.),
        Vec2::new(area.max.x - 12., area.min.y + 56.),
    );
    let mut x = track.min.x;
    for i in 0..3 {
        let width = track.width() * counts[i] as f32 / data.attempts.max(1) as f32;
        let rect = Rect::from_corners(Vec2::new(x, track.min.y), Vec2::new(x + width, track.max.y));
        p.rect(rect, colors[i].with_alpha(0.75));
        p.hover(
            rect,
            names[i],
            format!("{} / {} attempts", counts[i], data.attempts),
        );
        x += width;
        p.text(
            Vec2::new(area.center().x, area.min.y + 80. + 22. * i as f32),
            format!(
                "{}  ·  {}  ({:.1}%)",
                names[i],
                counts[i],
                100. * counts[i] as f64 / data.attempts.max(1) as f64
            ),
            12.,
            colors[i],
        );
    }
    if area.height() > 175. {
        paragraph(
            p,
            area,
            area.min.y + 157.,
            if data.includes_vacuum {
                "|10> = logical 0    |01> = logical 1    |00> = vacuum"
            } else {
                "Two optical rails encode one logical qubit: logical 0 = |10>, logical 1 = |01>."
            },
            12.,
            TEXT,
            2,
        );
    }
    if area.height() > 225. {
        paragraph(
            p,
            area,
            area.min.y + 198.,
            "The vacuum level tracks photon loss; it is not a third logical level. State shows resource preparation; Fabric shows the compiled schedule.",
            11.,
            MUTED,
            3,
        );
    }
}

/// Centered bounded wrapping keeps long provenance outside the plot, in Details.
fn paragraph(
    p: &mut PlotCache,
    area: Rect,
    y: f32,
    text: &str,
    size: f32,
    color: Color,
    max_lines: usize,
) -> f32 {
    let chars = ((area.width() - 12.) / (size * 0.54)).max(8.) as usize;
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + word.chars().count() + 1 > chars {
            lines.push(line);
            line = String::new();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    let visible = lines.len().min(max_lines);
    for (i, line) in lines.iter().take(visible).enumerate() {
        let text = if i + 1 == visible && visible < lines.len() {
            format!("{}…", short(line, chars.saturating_sub(1)))
        } else {
            short(line, chars)
        };
        p.text(
            Vec2::new(area.center().x, y + i as f32 * (size + 4.)),
            text,
            size,
            color,
        );
    }
    y + visible.saturating_sub(1) as f32 * (size + 4.) + size / 2.
}

fn number(value: f64) -> String {
    if value.abs() >= 1e6 || (value != 0. && value.abs() < 0.0001) {
        format!("{value:.2e}")
    } else if value.fract().abs() < 1e-8 {
        format!("{value:.0}")
    } else if value.abs() >= 100. {
        format!("{value:.1}")
    } else {
        format!("{value:.3}")
    }
}
