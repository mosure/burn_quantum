use super::*;
use crate::model::{FabricNodeView, FabricSnapshot};
pub(super) fn radius(kind: &str, compact: bool) -> f32 {
    if compact {
        return match kind {
            "Source" => 52.,
            "Fusion" => 50.,
            _ => 22.,
        };
    }
    match kind {
        "Source" => 75.,
        "Fusion" => 58.,
        _ => 35.,
    }
}
pub(super) fn node(
    p: &mut PlotCache,
    at: Vec2,
    n: &FabricNodeView,
    f: &FabricSnapshot,
    state: &Explorer,
    compact: bool,
) {
    let color = if state.node == Some(n.id) {
        TEXT
    } else {
        match n.kind.as_str() {
            "Source" => CYAN,
            "Fusion" => GOLD,
            _ => BLUE,
        }
    };
    match n.kind.as_str() {
        "Source" => {
            let r = Rect::from_center_size(
                at,
                if compact {
                    Vec2::new(104., 76.)
                } else {
                    Vec2::new(150., 82.)
                },
            );
            p.rect(r, PANEL);
            p.outline(r, color);
            p.text(
                at - Vec2::Y * 28.,
                if compact {
                    format!("Source {}", n.id)
                } else {
                    format!("RESOURCE SOURCE {}", n.id)
                },
                10.,
                color,
            );
            let resources = f
                .resources
                .iter()
                .filter(|r| r.source == n.id)
                .collect::<Vec<_>>();
            if let Some(resource) = resources.first() {
                let count = resource.photons.len().min(6);
                let point = |i: usize| {
                    at + Vec2::new(
                        (i as f32 - (count - 1) as f32 * 0.5) * if compact { 17. } else { 26. },
                        -3.,
                    )
                };
                for &(a, b) in &resource.edges {
                    if a < count && b < count {
                        let start = point(a);
                        let end = point(b);
                        for j in 0..6 {
                            p.line(
                                start.lerp(end, j as f32 / 6.),
                                start.lerp(end, (j as f32 + 0.45) / 6.),
                                color.with_alpha(0.65),
                            );
                        }
                    }
                }
                for (i, &photon) in resource.photons.iter().take(count).enumerate() {
                    let pos = point(i);
                    p.dot(pos,6.,if matches!(f.photon_location(photon,state.tick as f64),Some(PhotonLocation::Node(id)) if id==n.id){CYAN}else{GUIDE});
                    p.text(pos + Vec2::Y * 15., format!("p{photon}"), 10., MUTED);
                }
                let held = resource
                    .photons
                    .iter()
                    .filter(|id| f.outputs.contains(id))
                    .map(|id| format!("p{id}"))
                    .collect::<Vec<_>>()
                    .join(",");
                p.text(
                    at + Vec2::Y * 31.,
                    if held.is_empty() {
                        "All photons routed".into()
                    } else {
                        if compact {
                            format!("Output {held}")
                        } else {
                            format!("Held output: {held}")
                        }
                    },
                    10.,
                    CYAN,
                );
            } else {
                p.text(at, "Ideal resource supply", 11., MUTED);
            }
        }
        "Fusion" => {
            let r = Rect::from_center_size(
                at,
                if compact {
                    Vec2::new(100., 76.)
                } else {
                    Vec2::new(116., 84.)
                },
            );
            p.rect(r, PANEL);
            p.outline(r, color);
            let active = f.fusions.iter().any(|e| {
                e.node == n.id && state.tick >= e.tick as f32 && state.tick < (e.tick + 1) as f32
            });
            p.text(
                at - Vec2::Y * 29.,
                if compact {
                    "Fusion analyzer"
                } else {
                    "FUSION ANALYZER"
                },
                10.,
                color,
            );
            // Schematic beam splitter + two detector outputs; not a claimed optical decomposition.
            p.line(at + Vec2::new(-33., -10.), at + Vec2::new(25., 10.), color);
            p.line(at + Vec2::new(-33., 10.), at + Vec2::new(25., -10.), color);
            p.line(at + Vec2::new(-6., 12.), at + Vec2::new(6., -12.), color);
            for y in [-10., 10.] {
                p.outline(
                    Rect::from_center_size(at + Vec2::new(32., y), Vec2::new(11., 11.)),
                    if active { TEXT } else { GOLD },
                );
            }
            let events = f
                .fusions
                .iter()
                .filter(|e| e.node == n.id)
                .collect::<Vec<_>>();
            let label = if active {
                "Measuring".into()
            } else if let Some(e) = events.first() {
                format!("p{} + p{} · t={}", e.photons[0], e.photons[1], e.tick)
            } else {
                "Destructive measurement".into()
            };
            p.text(at + Vec2::Y * 30., label, 10., color);
        }
        _ => {
            let scale = if compact { 0.64 } else { 1. };
            p.rect(Rect::from_center_size(at, Vec2::splat(65. * scale)), PANEL);
            for (a, b) in [
                (Vec2::new(-34., 0.), Vec2::new(0., -28.)),
                (Vec2::new(0., -28.), Vec2::new(34., 0.)),
                (Vec2::new(34., 0.), Vec2::new(0., 28.)),
                (Vec2::new(0., 28.), Vec2::new(-34., 0.)),
            ] {
                p.line(at + a * scale, at + b * scale, color);
            }
            p.line(
                at - Vec2::X * 18. * scale,
                at + Vec2::new(18., -10.) * scale,
                color,
            );
            p.line(
                at + Vec2::new(-18., 8.) * scale,
                at + Vec2::new(18., 10.) * scale,
                color,
            );
            p.text(
                at + Vec2::Y * if compact { 32. } else { 42. },
                format!("{} {}", n.kind, n.id),
                11.,
                color,
            );
        }
    }
}
