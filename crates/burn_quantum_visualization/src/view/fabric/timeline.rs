use super::*;
use crate::model::FabricSnapshot;
pub(super) fn draw(p: &mut PlotCache, area: Rect, f: &FabricSnapshot, state: &Explorer) {
    let left = area.min.x + 72.;
    let right = area.max.x - 10.;
    let duration = f.finish_tick.max(1) as f32;
    let x = |tick: f32| left + (right - left) * tick / duration;
    let count = f.routes.len().min(if area.height() > 60. { 3 } else { 1 });
    let height = (area.height() - 20.) / count.max(1) as f32;
    p.text(
        Vec2::new(area.min.x + 35., area.min.y + 5.),
        "SCHEDULE",
        9.,
        MUTED,
    );
    for (i, r) in f.routes.iter().take(count).enumerate() {
        let y = area.min.y + 20. + height * i as f32;
        p.text(
            Vec2::new(area.min.x + 29., y),
            format!("p{} → F", r.photon),
            10.,
            MUTED,
        );
        p.line(Vec2::new(left, y), Vec2::new(right, y), GUIDE);
        for h in &r.hops {
            let bar = Rect::from_corners(
                Vec2::new(x(h.departure as f32), y - 4.),
                Vec2::new(x(h.arrival as f32), y + 4.),
            );
            p.rect(bar, CYAN.with_alpha(0.5));
        }
        if let Some(event) = f.fusions.iter().find(|e| e.photons.contains(&r.photon)) {
            p.rect(
                Rect::from_corners(
                    Vec2::new(x(event.tick as f32), y - 6.),
                    Vec2::new(
                        x((event.tick + 1) as f32).max(x(event.tick as f32) + 3.),
                        y + 6.,
                    ),
                ),
                GOLD,
            );
        }
    }
    let cursor = x(state.tick);
    p.line(
        Vec2::new(cursor, area.min.y + 12.),
        Vec2::new(cursor, area.max.y),
        TEXT,
    );
    p.text(
        Vec2::new((left + right) * 0.5, area.min.y + 4.),
        "cyan: travel   ·   gaps: storage/wait   ·   gold: fusion",
        9.,
        MUTED,
    );
}
