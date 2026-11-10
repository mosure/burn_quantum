//! Shared spatial/projection mathematics for focused and parallel reduced-qubit views.
use crate::{
    app::Explorer,
    model::{
        VisualizationDocument,
        bloch::{BlochMode, BlochReadout},
    },
    view::{plot::PlotCache, theme::*},
};
use bevy::prelude::*;
pub(super) struct Style {
    pub center: Vec2,
    pub radius: f32,
    pub color: Color,
    pub axes: bool,
    pub trail: usize,
}
pub(super) fn draw(
    p: &mut PlotCache,
    wire: usize,
    state: &Explorer,
    doc: &VisualizationDocument,
    style: Style,
) {
    let Some(vector) = doc
        .trace
        .frames
        .get(state.frame)
        .and_then(|f| f.subsystems.get(wire))
        .and_then(|q| q.bloch)
    else {
        return;
    };
    let Style {
        center,
        radius,
        color,
        axes,
        trail,
    } = style;
    let readout = BlochReadout::from_vector(vector);
    if state.bloch_mode == BlochMode::Planar {
        probability_disk(p, center, radius, color, axes && state.grid, &readout);
        if trail > 0 {
            let start = (state.frame + 1).saturating_sub(trail);
            let mut last = None;
            for frame in &doc.trace.frames[start..=state.frame] {
                if let Some(v) = frame.subsystems.get(wire).and_then(|q| q.bloch) {
                    let tip = center + Vec2::new(v[0] as f32, -v[1] as f32) * radius;
                    if let Some(previous) = last {
                        p.line(previous, tip, color.with_alpha(0.20));
                    }
                    last = Some(tip);
                }
            }
        }
        return;
    }
    let spatial = state.bloch_mode == BlochMode::Spatial;
    let rotate =
        |v: Vec3| Quat::from_rotation_x(state.pitch) * (Quat::from_rotation_z(state.yaw) * v);
    let project = |v: Vec3| {
        if spatial {
            let v = rotate(v);
            center + Vec2::new(v.x, -v.z) * radius
        } else {
            let v = state
                .bloch_plane
                .project([v.x as f64, v.y as f64, v.z as f64]);
            center + Vec2::new(v[0] as f32, -v[1] as f32) * radius
        }
    };
    p.disc(
        center,
        radius,
        if spatial {
            color
        } else {
            color.with_alpha(0.12)
        },
        spatial,
    );
    p.circle(center, radius, color.with_alpha(0.35));
    if spatial {
        if state.grid {
            let segments = 64;
            for axis in 0..3 {
                for i in 0..segments {
                    let point = |j: usize| {
                        let a = j as f32 * std::f32::consts::TAU / segments as f32;
                        match axis {
                            0 => Vec3::new(a.cos(), a.sin(), 0.),
                            1 => Vec3::new(a.cos(), 0., a.sin()),
                            _ => Vec3::new(0., a.cos(), a.sin()),
                        }
                    };
                    let a = point(i);
                    let b = point(i + 1);
                    let front = rotate((a + b) * 0.5).y < 0.;
                    if front || i % 2 == 0 {
                        p.line(
                            project(a),
                            project(b),
                            color.with_alpha(if front { 0.34 } else { 0.10 }),
                        );
                    }
                }
            }
        }
        for (axis, label) in [
            (Vec3::Z, "|0>"),
            (-Vec3::Z, "|1>"),
            (Vec3::X, "+x"),
            (Vec3::Y, "+y"),
        ] {
            let tip = project(axis);
            if state.grid {
                p.line(center, tip, color.with_alpha(0.12));
            }
            if axes {
                p.text(center + (tip - center) * 1.12, label, 10., MUTED);
            }
        }
    } else {
        let labels = state.bloch_plane.axes();
        if state.grid {
            p.line(
                center - Vec2::X * radius,
                center + Vec2::X * radius,
                color.with_alpha(0.25),
            );
            p.line(
                center - Vec2::Y * radius,
                center + Vec2::Y * radius,
                color.with_alpha(0.25),
            );
            p.circle(center, radius * 0.5, color.with_alpha(0.10));
        }
        if axes {
            p.text(center + Vec2::new(radius + 10., 0.), labels[0], 10., MUTED);
            p.text(center - Vec2::new(0., radius + 10.), labels[1], 10., MUTED);
        }
        let hidden = state.bloch_plane.project(vector)[2];
        p.text(
            center + Vec2::Y * (radius + 14.),
            format!("{} {hidden:+.2} out of plane", labels[2]),
            9.,
            MUTED,
        );
    }
    if spatial {
        let z_tip = project(Vec3::new(0., 0., vector[2] as f32));
        if state.grid {
            let tip = project(Vec3::new(
                vector[0] as f32,
                vector[1] as f32,
                vector[2] as f32,
            ));
            p.line(z_tip, tip, color.with_alpha(0.25));
        }
        p.text(
            center + Vec2::Y * (radius + 19.),
            format!("P(1) {:.0}%", readout.probabilities[1] * 100.),
            10.,
            color,
        );
    }
    if trail > 0 {
        let start = (state.frame + 1).saturating_sub(trail);
        let mut last = None;
        for frame in &doc.trace.frames[start..=state.frame] {
            if let Some(v) = frame.subsystems.get(wire).and_then(|q| q.bloch) {
                let pt = project(Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32));
                if let Some(a) = last {
                    p.line(a, pt, color.with_alpha(0.2));
                }
                last = Some(pt);
            }
        }
    }
    let tip = project(Vec3::new(
        vector[0] as f32,
        vector[1] as f32,
        vector[2] as f32,
    ));
    if tip.distance_squared(center) > 1. {
        p.arrow(center, tip, color);
    }
    p.dot(center, 1.5, MUTED);
    p.dot(tip, if radius < 40. { 2.5 } else { 3.5 }, color);
}

/// A probability disk is a different encoding from a Cartesian Bloch projection.
/// Filled area / reference area = P(1), and the hand is the XY coherence vector.
fn probability_disk(
    p: &mut PlotCache,
    center: Vec2,
    radius: f32,
    color: Color,
    guides: bool,
    r: &BlochReadout,
) {
    p.disc(center, radius, color.with_alpha(0.08), false);
    p.disc(
        center,
        radius * r.probability_radius as f32,
        color.with_alpha(0.50),
        false,
    );
    p.circle(center, radius, color.with_alpha(0.45));
    if guides {
        // Equal-probability area contours, not equally spaced radii.
        p.circle(center, radius * 0.5_f32.sqrt(), color.with_alpha(0.15));
        for (angle, label) in [
            (0., "0"),
            (std::f32::consts::FRAC_PI_2, "π/2"),
            (std::f32::consts::PI, "π"),
        ] {
            let axis = Vec2::new(angle.cos(), -angle.sin());
            p.line(
                center + axis * (radius - 3.),
                center + axis * (radius + 3.),
                MUTED,
            );
            p.text(center + axis * (radius + 11.), label, 9., MUTED);
        }
    }
    if let Some(phase) = r.phase {
        let direction = Vec2::new(phase.cos() as f32, -phase.sin() as f32);
        let tip = center + direction * (radius * r.coherence as f32);
        if tip.distance(center) > 10. {
            p.arrow(center, tip, color);
        } else {
            p.line(center, tip, color);
        }
        p.dot(tip, 2.5, color);
    }
    p.dot(center, 1.5, MUTED);
    let phase = r
        .phase
        .map(|v| format!("φ {:.0}°", v.to_degrees()))
        .unwrap_or_else(|| "φ —".into());
    p.text(
        center + Vec2::Y * (radius + 16.),
        format!("P(1) {:.0}% · {phase}", r.probabilities[1] * 100.),
        10.,
        color,
    );
}
