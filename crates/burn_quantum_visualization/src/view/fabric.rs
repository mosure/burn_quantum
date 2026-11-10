use super::{
    plot::{Hit, PlotCache, short},
    theme::*,
};
use crate::{app::Explorer, model::VisualizationDocument};
use bevy::prelude::*;
use std::collections::BTreeMap;
pub(super) fn build(
    p: &mut PlotCache,
    bounds: Rect,
    state: &Explorer,
    doc: &VisualizationDocument,
) {
    let Some(fabric) = &doc.fabric else {
        p.text(
            bounds.center() + Vec2::new(0., -25.),
            "A quantum circuit is not a physical routing plan.",
            21.,
            TEXT,
        );
        p.text(
            bounds.center() + Vec2::new(0., 12.),
            "Choose Photonic routing to explore source, switch and fusion nodes.",
            13.,
            MUTED,
        );
        p.description = "No physical schedule attached".into();
        return;
    };
    if fabric.nodes.is_empty() {
        p.description = "Empty physical network".into();
        return;
    }
    let mut min = Vec2::splat(f32::INFINITY);
    let mut max = Vec2::splat(f32::NEG_INFINITY);
    for n in &fabric.nodes {
        let v = Vec2::new(n.position[0], n.position[1]);
        min = min.min(v);
        max = max.max(v);
    }
    let extent = (max - min).max(Vec2::splat(1.));
    let area = Rect::from_corners(
        bounds.min + Vec2::new(65., 95.),
        bounds.max - Vec2::new(65., 95.),
    );
    let scale = (area.width() / extent.x).min(area.height() / extent.y) * 0.85 * state.zoom;
    let center = (min + max) * 0.5;
    let project = |n: &crate::model::FabricNodeView| {
        area.center() + (Vec2::new(n.position[0], n.position[1]) - center) * scale + state.pan
    };
    let mut candidates = fabric
        .nodes
        .iter()
        .filter(|n| {
            let pos = project(n);
            let min = pos - Vec2::new(42., 25.);
            let max = pos + Vec2::new(42., 50.);
            min.x < bounds.max.x
                && max.x > bounds.min.x
                && min.y < bounds.max.y
                && max.y > bounds.min.y
        })
        .collect::<Vec<_>>();
    let visible = candidates.len();
    candidates.sort_by(|a, b| {
        project(a)
            .distance_squared(bounds.center())
            .total_cmp(&project(b).distance_squared(bounds.center()))
            .then(a.id.cmp(&b.id))
    });
    let nodes = candidates.into_iter().take(96).collect::<Vec<_>>();
    let points: BTreeMap<usize, Vec2> = nodes.iter().map(|n| (n.id, project(n))).collect();
    p.text(
        Vec2::new(bounds.center().x, bounds.min.y + 22.),
        format!("Schedule tick {:.1} / {}", state.tick, fabric.finish_tick),
        18.,
        TEXT,
    );
    p.text(
        Vec2::new(bounds.center().x, bounds.min.y + 48.),
        "Compiled photon routes · sources → switches → fusion and output",
        12.,
        MUTED,
    );
    let mut links = 0;
    for link in fabric
        .links
        .iter()
        .filter(|l| points.contains_key(&l.from) && points.contains_key(&l.to))
        .take(256)
    {
        if let (Some(&a), Some(&b)) = (points.get(&link.from), points.get(&link.to)) {
            let survival = (link.transmission[0] + link.transmission[1]) * 0.5;
            let color = if survival < 0.99 {
                GOLD.with_alpha(0.55)
            } else {
                GUIDE
            };
            p.line(a, b, color);
            p.arrow(a.lerp(b, 0.48), a.lerp(b, 0.65), color);
            if state.grid {
                p.text(
                    (a + b) * 0.5 + Vec2::new(0., -14.),
                    format!(
                        "{}t · rails {:.0}/{:.0}%",
                        link.ticks,
                        link.transmission[0] * 100.,
                        link.transmission[1] * 100.
                    ),
                    10.,
                    MUTED,
                );
            }
            links += 1;
        }
    }
    let mut photons = 0;
    for route in fabric.routes.iter().take(256) {
        for hop in &route.hops {
            let t = state.tick as f64;
            if t >= hop.departure as f64
                && t < hop.arrival as f64
                && let Some(link) = fabric.links.iter().find(|l| l.id == hop.link)
                && let (Some(&a), Some(&b)) = (points.get(&link.from), points.get(&link.to))
            {
                let progress = ((t - hop.departure as f64)
                    / (hop.arrival - hop.departure).max(1) as f64)
                    .clamp(0., 1.) as f32;
                let position = a.lerp(b, progress);
                p.line(a, b, CYAN.with_alpha(0.5));
                p.dot(position, 4., GOLD);
                photons += 1;
            }
        }
    }
    for n in nodes.iter() {
        let center = points[&n.id];
        let color = if n.kind.to_lowercase().contains("source") {
            CYAN
        } else if n.kind.to_lowercase().contains("fusion") {
            GOLD
        } else {
            BLUE
        };
        let rect = Rect::from_center_size(center, Vec2::new(84., 50.));
        p.rect(rect, PANEL);
        p.outline(
            rect,
            if state.node == Some(n.id) {
                TEXT
            } else {
                color
            },
        );
        p.text(center, short(&n.label, 12), 13., color);
        p.text(center + Vec2::new(0., 39.), short(&n.kind, 16), 10., MUTED);
        p.hit(rect, Hit::Node(n.id));
    }
    p.text(
        Vec2::new(bounds.center().x, bounds.max.y - 30.),
        "Scheduled travel; no survival or fusion outcome inferred.",
        11.,
        MUTED,
    );
    p.description = format!(
        "{} drawn / {} visible / {} total nodes\n{} / {} links · {} in-transit photons\nPan / zoom to inspect · caps 96 nodes, 256 links/routes",
        nodes.len(),
        visible,
        fabric.nodes.len(),
        links,
        fabric.links.len(),
        photons
    );
}
