//! Adaptive, bounded panels for an entire mixed-dimensional register.
use super::bloch;
use crate::{
    app::Explorer,
    model::{SubsystemState, VisualizationDocument, aggregate_probabilities},
    view::{
        plot::{Hit, PlotCache},
        theme::*,
    },
};
use bevy::prelude::*;
use std::ops::Range;

struct Layout {
    capacity: usize,
    range: Range<usize>,
    cards: Vec<Rect>,
}
fn layout(bounds: Rect, count: usize, selected: usize, planar: bool) -> Layout {
    let area = Rect::from_corners(
        bounds.min + Vec2::new(2., 30.),
        bounds.max - Vec2::new(2., 24.),
    );
    let max_cols = ((area.width() + 10.) / 125.).floor().max(1.) as usize;
    let minimum_height = if planar { 140. } else { 124. };
    let max_rows = ((area.height() + 10.) / (minimum_height + 10.))
        .floor()
        .max(1.) as usize;
    let capacity = (max_cols * max_rows).max(1).min(count.max(1));
    let start = selected.min(count.saturating_sub(1)) / capacity * capacity;
    let end = (start + capacity).min(count);
    let visible = end - start;
    let gap = 10.;
    let (cols, _) = (1..=visible.max(1).min(max_cols))
        .filter(|&cols| visible.max(1).div_ceil(cols) <= max_rows)
        .map(|cols| {
            let rows = visible.max(1).div_ceil(cols);
            let size = (area.size() - Vec2::new((cols - 1) as f32, (rows - 1) as f32) * gap)
                / Vec2::new(cols as f32, rows as f32);
            let diameter = (size.x - 32.).min(size.y - 52.);
            let row_penalty = if rows == 1 && visible >= 5 && max_rows >= 2 {
                diameter * 0.30
            } else {
                0.
            };
            (
                cols,
                diameter - row_penalty - (cols * rows - visible) as f32 * 3.,
            )
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let rows = visible.max(1).div_ceil(cols);
    let slot = (area.size() - Vec2::new((cols - 1) as f32, (rows - 1) as f32) * gap)
        / Vec2::new(cols as f32, rows as f32);
    let diameter = (slot.x - 32.).min(slot.y - 52.).max(12.);
    let size = Vec2::new(
        (diameter + 32.).max(90.).min(slot.x),
        (diameter + 52.).max(80.).min(slot.y),
    );
    let top = (area.height() - (size.y * rows as f32 + gap * (rows - 1) as f32)) * 0.5;
    let cards = (0..visible)
        .map(|i| {
            let row = i / cols;
            let row_count = (visible - row * cols).min(cols);
            let offset =
                (area.width() - row_count as f32 * size.x - (row_count - 1) as f32 * gap) * 0.5;
            let min = area.min
                + Vec2::new(
                    offset + (i % cols) as f32 * (size.x + gap),
                    top + row as f32 * (size.y + gap),
                );
            Rect::from_corners(min, min + size)
        })
        .collect();
    Layout {
        capacity,
        range: start..end,
        cards,
    }
}

pub(super) fn build(
    p: &mut PlotCache,
    bounds: Rect,
    state: &Explorer,
    doc: &VisualizationDocument,
) {
    let frame = &doc.trace.frames[state.frame];
    let grid = layout(
        bounds,
        frame.subsystems.len(),
        state.wire,
        state.bloch_mode != crate::model::bloch::BlochMode::Spatial,
    );
    p.overview_capacity = grid.capacity;
    p.text(
        Vec2::new(bounds.center().x, bounds.min.y + 12.),
        if grid.range.len() == frame.subsystems.len() {
            format!(
                "{} subsystems · click a panel to select",
                frame.subsystems.len()
            )
        } else {
            format!(
                "Subsystems {}–{} / {} · select or change group",
                grid.range.start + 1,
                grid.range.end,
                frame.subsystems.len()
            )
        },
        12.,
        MUTED,
    );
    for (wire, rect) in grid.range.clone().zip(grid.cards) {
        let q = &frame.subsystems[wire];
        let selected = state.wire == wire;
        let color = [CYAN, BLUE, GOLD, ACCENT][wire % 4];
        if selected {
            p.rect(rect, ACCENT_SOFT.with_alpha(0.18));
            // Small corner brackets preserve selection without boxing in every sphere.
            for (corner, dx, dy) in [
                (rect.min, 1., 1.),
                (Vec2::new(rect.max.x, rect.min.y), -1., 1.),
                (rect.max, -1., -1.),
                (Vec2::new(rect.min.x, rect.max.y), 1., -1.),
            ] {
                p.line(corner, corner + Vec2::X * dx * 12., ACCENT.with_alpha(0.65));
                p.line(corner, corner + Vec2::Y * dy * 12., ACCENT.with_alpha(0.65));
            }
        }
        p.text(
            Vec2::new(rect.center().x, rect.min.y + 15.),
            if q.dimension == 2 {
                format!("q{wire}")
            } else {
                format!("q{wire} · {} levels", q.dimension)
            },
            13.,
            if selected { ACCENT } else { TEXT },
        );
        if q.bloch.is_some() {
            let reserve = 88.;
            let maximum = ((rect.width() - 26.).min(rect.height() - reserve)) * 0.5;
            let radius = (maximum * state.zoom).min(maximum).max(4.);
            bloch::draw(
                p,
                wire,
                state,
                doc,
                bloch::Style {
                    center: rect.center() + Vec2::new(0., -6.),
                    radius,
                    color,
                    axes: radius > 38.,
                    trail: if state.trails { 32 } else { 0 },
                },
            );
        } else {
            populations(p, rect, q, color);
        }
        p.text(
            Vec2::new(rect.center().x, rect.max.y - 13.),
            format!("purity {:.2}", q.purity),
            11.,
            color,
        );
        p.hit(rect, Hit::Wire(wire));
    }
    p.text(Vec2::new(bounds.center().x,bounds.max.y-9.),
        if state.bloch_mode == crate::model::bloch::BlochMode::Planar { "Area = P(1) · hand = phase / coherence" }
        else if bounds.width() < 500. { "Shared timeline · Focus for joint probabilities" }
        else { "Reduced states share one timeline and camera. Local states alone do not show all joint correlations." },
        10., MUTED);
    p.description = format!(
        "{} / {} reduced subsystems · {} states\nSelected q{} · norm {:.5}\nFocus shows the joint distribution",
        grid.range.len(),
        frame.subsystems.len(),
        frame.probabilities.len(),
        state.wire,
        frame.norm
    );
}

fn populations(p: &mut PlotCache, rect: Rect, q: &SubsystemState, color: Color) {
    let chart = Rect::from_corners(
        rect.min + Vec2::new(14., 36.),
        rect.max - Vec2::new(14., 40.),
    );
    let max_bins = (chart.width() / 26.).floor().max(1.) as usize;
    let Ok(bins) = aggregate_probabilities(&q.probabilities, max_bins) else {
        return;
    };
    let slot = chart.width() / bins.len() as f32;
    p.line(Vec2::new(chart.min.x, chart.max.y), chart.max, GUIDE);
    for (index, bin) in bins.iter().enumerate() {
        let x = chart.min.x + slot * index as f32;
        let top = chart.max.y - (chart.height() * bin.probability as f32).max(1.);
        p.rect(
            Rect::from_corners(
                Vec2::new(x + slot * 0.18, top),
                Vec2::new(x + slot * 0.82, chart.max.y),
            ),
            color,
        );
        let label = if bin.end - bin.start == 1 {
            bin.start.to_string()
        } else {
            format!("{}–{}", bin.start, bin.end - 1)
        };
        p.text(
            Vec2::new(x + slot * 0.5, chart.max.y + 10.),
            label,
            10.,
            MUTED,
        );
        if chart.height() > 80. {
            p.text(
                Vec2::new(x + slot * 0.5, top - 10.),
                format!("{:.0}%", bin.probability * 100.),
                10.,
                TEXT,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adaptive_groups_keep_selection_visible_and_cards_disjoint() {
        for size in [
            Vec2::new(1118., 396.),
            Vec2::new(334., 270.),
            Vec2::new(640., 200.),
        ] {
            let bounds = Rect::from_corners(Vec2::ZERO, size);
            for count in [2, 4, 6, 12, 16] {
                for selected in 0..count {
                    let grid = layout(bounds, count, selected, selected % 2 == 0);
                    assert!(grid.range.contains(&selected));
                    assert_eq!(grid.cards.len(), grid.range.len());
                    for (i, card) in grid.cards.iter().enumerate() {
                        assert!(
                            card.min.cmpge(bounds.min).all() && card.max.cmple(bounds.max).all()
                        );
                        assert!(card.width() >= 90. && card.height() >= 80.);
                        for other in &grid.cards[i + 1..] {
                            assert!(
                                (card.max.min(other.max) - card.min.max(other.min)).min_element()
                                    <= 0.
                            );
                        }
                    }
                }
            }
        }
    }
}
