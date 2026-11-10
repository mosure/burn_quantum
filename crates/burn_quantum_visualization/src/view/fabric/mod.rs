//! Physical photonic schematic: resources, paired rails, packet occupancy and destructive analyzers.
mod components;
mod timeline;
use super::{
    plot::{Hit, PlotCache},
    theme::*,
};
use crate::{
    app::Explorer,
    model::{PhotonLocation, VisualizationDocument},
};
use bevy::prelude::*;
use std::collections::BTreeMap;
pub(super) fn build(
    p: &mut PlotCache,
    bounds: Rect,
    state: &Explorer,
    doc: &VisualizationDocument,
) {
    let Some(f) = &doc.fabric else {
        p.text(
            bounds.center() - Vec2::Y * 18.,
            "No physical photonic plan attached",
            20.,
            TEXT,
        );
        p.text(
            bounds.center() + Vec2::Y * 15.,
            "Choose Photonic routing to inspect resources, dual rails and fusion.",
            12.,
            MUTED,
        );
        p.description = "A gate circuit does not specify optical routing.".into();
        return;
    };
    let compact = bounds.width() < 620.;
    let footer = if bounds.height() > 300. { 102. } else { 64. };
    let area = Rect::from_corners(
        bounds.min + Vec2::new(if compact { 54. } else { 100. }, 95.),
        bounds.max - Vec2::new(if compact { 52. } else { 90. }, footer + 60.),
    );
    let (min, max) = f.nodes.iter().fold(
        (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
        |(min, max), n| {
            let v = Vec2::new(n.position[0], n.position[1]);
            (min.min(v), max.max(v))
        },
    );
    let extent = (max - min).max(Vec2::ONE);
    let center = (max + min) * 0.5;
    // Compact glyphs preserve their text size and fit the source-to-fusion flow on a phone.
    let scale = Vec2::new(
        (area.width() / extent.x).max(if compact { 24. } else { 55. }),
        (area.height() / extent.y).max(48.),
    ) * state.zoom;
    let project = |n: &crate::model::FabricNodeView| {
        area.center() + (Vec2::new(n.position[0], n.position[1]) - center) * scale + state.pan
    };
    let points: BTreeMap<_, _> = f
        .nodes
        .iter()
        .filter(|n| bounds.inflate(110.).contains(project(n)))
        .take(64)
        .map(|n| (n.id, project(n)))
        .collect();
    let header = if compact {
        format!(
            "{} resources → {} fusions · t {:.1}/{}",
            f.resource_count,
            f.fusions.len(),
            state.tick,
            f.finish_tick
        )
    } else {
        format!(
            "SMALL RESOURCE STATES     →     DUAL-RAIL ROUTING     →     DESTRUCTIVE FUSION     ·     t {:.1}/{}",
            state.tick, f.finish_tick
        )
    };
    p.text(
        Vec2::new(bounds.center().x, bounds.min.y + 15.),
        header,
        if compact { 11. } else { 12. },
        TEXT,
    );
    let relevant = |link: &crate::model::FabricLinkView| {
        state.node.is_none_or(|id| {
            id == link.from
                || id == link.to
                || f.routes.iter().any(|r| {
                    (r.from == id || r.to == id) && r.hops.iter().any(|h| h.link == link.id)
                })
        })
    };
    let mut links = 0;
    let mut paths = BTreeMap::new();
    for link in f.links.iter().take(256) {
        let (Some(&a), Some(&b)) = (points.get(&link.from), points.get(&link.to)) else {
            continue;
        };
        let direction = (b - a).normalize_or_zero();
        let from = a + direction * components::radius(&f.nodes[link.from].kind, compact);
        let to = b - direction * components::radius(&f.nodes[link.to].kind, compact);
        let side = Vec2::new(-direction.y, direction.x) * 3.;
        let selected = relevant(link);
        let occupied=f.routes.iter().any(|r|matches!(f.photon_location(r.photon,state.tick as f64),Some(PhotonLocation::Link{link:id,..}) if id==link.id));
        let color = if !selected {
            GUIDE.with_alpha(0.25)
        } else if occupied {
            CYAN
        } else {
            GUIDE
        };
        for sign in [-1., 1.] {
            p.line(from + side * sign, to + side * sign, color);
        }
        if selected {
            p.arrow(from.lerp(to, 0.49), from.lerp(to, 0.60), color);
        }
        if state.grid && selected && (!compact || state.node.is_some()) {
            let loss = (1. - (link.transmission[0] + link.transmission[1]) * 0.5) * 100.;
            p.text(
                from.lerp(to, 0.45) - Vec2::Y * 15.,
                format!("{} ticks · {:.1}% loss", link.ticks, loss),
                10.,
                if loss > 1. { GOLD } else { MUTED },
            );
        }
        paths.insert(link.id, (from, to));
        links += 1;
    }
    for n in f.nodes.iter().filter(|n| points.contains_key(&n.id)) {
        let at = points[&n.id];
        components::node(p, at, n, f, state, compact);
        p.hit(
            Rect::from_center_size(
                at,
                Vec2::new(components::radius(&n.kind, compact) * 2., 88.),
            ),
            Hit::Node(n.id),
        );
    }
    let mut traveling = 0;
    let mut waiting = 0;
    let mut consumed = 0;
    for photon in 0..f.photon_count.min(256) {
        match f.photon_location(photon, state.tick as f64) {
            Some(PhotonLocation::Link { link, progress }) => {
                if let Some(&(a, b)) = paths.get(&link) {
                    let at = a.lerp(b, progress);
                    p.circle(at, 7., GOLD);
                    p.circle(at, 5., GOLD);
                    p.text(at + Vec2::Y * 18., format!("p{photon}"), 11., GOLD);
                    traveling += 1;
                }
            }
            Some(PhotonLocation::Node(node)) => {
                if f.routes.iter().any(|r| r.photon == photon)
                    && let Some(&at) = points.get(&node)
                {
                    waiting += 1;
                    if f.nodes[node].kind == "Source" {
                        continue;
                    }
                    let local=f.routes.iter().filter(|r|r.photon<photon&&matches!(f.photon_location(r.photon,state.tick as f64),Some(PhotonLocation::Node(id)) if id==node)).count();
                    p.text(
                        at + Vec2::new((local as f32 - 0.5) * 28., -55.),
                        format!("p{photon}"),
                        11.,
                        GOLD,
                    );
                }
            }
            Some(PhotonLocation::Consumed) => consumed += 1,
            None => {}
        }
    }
    let timeline_area = Rect::from_corners(
        Vec2::new(bounds.min.x + 12., bounds.max.y - footer),
        bounds.max - Vec2::new(12., 26.),
    );
    timeline::draw(p, timeline_area, f, state);
    p.text(Vec2::new(bounds.center().x,bounds.max.y-9.),if compact{"Paired lines = two modes · markers = photon packets"}else{"Dual rail = one photon across two modes. Packet position is scheduled; survival and fusion outcomes are not sampled."},10.,MUTED);
    p.description = format!(
        "{} / {} components · {links} paired-rail links\n{traveling} traveling · {waiting} waiting · {consumed} consumed\n{} held outputs · select a component to trace its paths",
        points.len(),
        f.nodes.len(),
        f.outputs.len()
    );
}
