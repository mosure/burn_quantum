mod bloch;
mod overview;
pub(crate) mod surface;

use super::{
    plot::{Hit, PlotCache},
    theme::*,
};
use crate::{
    app::Explorer,
    model::{VisualizationDocument, aggregate_probabilities},
};
use bevy::prelude::*;
pub(super) fn build(
    p: &mut PlotCache,
    bounds: Rect,
    state: &Explorer,
    doc: &VisualizationDocument,
) {
    let Some(frame) = doc.trace.frames.get(state.frame) else {
        return;
    };
    if state.state_overview && frame.subsystems.len() > 1 {
        overview::build(p, bounds, state, doc);
        return;
    }
    let Some(q) = frame.subsystems.get(state.wire) else {
        return;
    };
    let wide = bounds.width() > 660. || bounds.height() < 360.;
    let sphere = if wide {
        Rect::from_corners(
            bounds.min,
            bounds.min
                + Vec2::new(
                    bounds.width() * if bounds.width() < 660. { 0.45 } else { 0.53 },
                    bounds.height(),
                ),
        )
    } else {
        Rect::from_corners(
            bounds.min,
            bounds.min + Vec2::new(bounds.width(), bounds.height() * 0.58),
        )
    };
    let bars = if wide {
        Rect::from_corners(
            Vec2::new(sphere.max.x + 20., bounds.min.y + 35.),
            bounds.max - Vec2::new(16., 30.),
        )
    } else {
        Rect::from_corners(
            Vec2::new(bounds.min.x + 15., sphere.max.y + 15.),
            bounds.max - Vec2::new(15., 15.),
        )
    };
    p.text(
        Vec2::new(sphere.center().x, sphere.min.y + 20.),
        if sphere.width() < 230. {
            format!("q{} · {} levels", q.wire, q.dimension)
        } else {
            format!("q{} · {}-level subsystem", q.wire, q.dimension)
        },
        16.,
        TEXT,
    );
    if let Some(bloch) = q.bloch {
        let radius = (sphere.width().min(sphere.height() - 95.) * 0.36 * state.zoom).max(25.);
        let center = sphere.center() + Vec2::new(0., -12.) + state.pan;
        let v = Vec3::new(bloch[0] as f32, bloch[1] as f32, bloch[2] as f32);
        bloch::draw(
            p,
            q.wire,
            state,
            doc,
            bloch::Style {
                center,
                radius,
                color: CYAN,
                axes: true,
                trail: if state.trails { 64 } else { 0 },
            },
        );
        p.text(
            Vec2::new(sphere.center().x, sphere.max.y - 53.),
            if sphere.width() < 230. {
                format!("r {:.2} · purity {:.2}", v.length(), q.purity)
            } else {
                format!("Bloch length {:.3} · purity {:.3}", v.length(), q.purity)
            },
            13.,
            CYAN,
        );
        p.text(
            Vec2::new(sphere.center().x, sphere.max.y - 30.),
            if !state.trails {
                if state.bloch_mode == crate::model::bloch::BlochMode::Spatial {
                    "3D sphere · enable Trails for history"
                } else {
                    "2D projection · omitted axis labeled"
                }
            } else if sphere.width() < 230. {
                "Faint trail: prior states"
            } else if v.length() < 0.02 {
                "Center = mixed state · faint trail = earlier states"
            } else {
                "Faint trail = earlier states · dot = current state"
            },
            11.,
            MUTED,
        );
        p.hit(
            Rect::from_center_size(center, Vec2::splat(radius * 2.)),
            Hit::Wire(q.wire),
        );
    } else {
        let available = (sphere.height() - 65.).max(30.);
        let max_levels = ((available / 32.) as usize).clamp(1, 12);
        if let Ok(levels) = aggregate_probabilities(&q.probabilities, max_levels) {
            let row = available / levels.len().max(1) as f32;
            for (i, bin) in levels.iter().enumerate() {
                let y = sphere.min.y + 55. + i as f32 * row;
                let width = (sphere.width() - 90.).max(30.);
                let x = sphere.min.x + 45.;
                p.rect(
                    Rect::from_corners(Vec2::new(x, y), Vec2::new(x + width, y + 10.)),
                    PANEL,
                );
                p.rect(
                    Rect::from_corners(
                        Vec2::new(x, y),
                        Vec2::new(x + width * bin.probability as f32, y + 10.),
                    ),
                    CYAN,
                );
                let label = if bin.end - bin.start == 1 {
                    format!("|{}>", bin.start)
                } else {
                    format!("levels {}–{}", bin.start, bin.end - 1)
                };
                p.text(
                    Vec2::new(x + width / 2., y - 11.),
                    format!("{label}  {:.1}%", bin.probability * 100.),
                    11.,
                    TEXT,
                );
            }
        }
    }
    p.text(
        Vec2::new(bars.center().x, bars.min.y),
        if bars.width() < 250. {
            "Basis probabilities"
        } else {
            "Computational basis"
        },
        if bars.width() < 250. { 13. } else { 17. },
        TEXT,
    );
    p.text(
        Vec2::new(bars.center().x, bars.min.y + 23.),
        "Bar height = probability",
        11.,
        MUTED,
    );
    // Budget for the longest displayed label before aggregating. Large basis
    // ranges need substantially more room than short qubit kets.
    let states = frame.probabilities.len();
    let label_pixels = if states > 16 {
        let digits = states.saturating_sub(1).to_string().len();
        (2 * digits + 1) as f32 * 6.5 + 20.
    } else {
        basis_label(states.saturating_sub(1), doc).chars().count() as f32 * 5.5 + 12.
    };
    let max_bins = ((bars.width() / label_pixels) as usize).clamp(1, 16);
    let Ok(bins) = aggregate_probabilities(&frame.probabilities, max_bins) else {
        return;
    };
    let top = bars.min.y + 55.;
    let baseline = bars.max.y - 68.;
    let height = (baseline - top).max(30.);
    let slot = bars.width() / bins.len().max(1) as f32;
    let max = 1f64;
    if state.grid {
        for level in [0., 0.5, 1.] {
            let y = baseline - height * level;
            p.line(
                Vec2::new(bars.min.x, y),
                Vec2::new(bars.max.x, y),
                GUIDE.with_alpha(0.55),
            );
        }
    }
    for (i, bin) in bins.iter().enumerate() {
        let x = bars.min.x + i as f32 * slot + slot * 0.15;
        let w = slot * 0.7;
        let h = (bin.probability / max) as f32 * height;
        let color = if bin.probability < 1e-14 {
            MUTED
        } else if bin.end - bin.start == 1 {
            frame
                .amplitudes
                .as_ref()
                .and_then(|a| a.get(bin.start))
                .map(|a| phase_color(a.im.atan2(a.re)))
                .unwrap_or(CYAN)
        } else {
            BLUE
        };
        p.rect(
            Rect::from_corners(
                Vec2::new(x, baseline - h.max(1.)),
                Vec2::new(x + w, baseline),
            ),
            color,
        );
        let label = if bin.end - bin.start == 1 {
            basis_label(bin.start, doc)
        } else {
            format!("{}–{}", bin.start, bin.end - 1)
        };
        p.text(Vec2::new(x + w / 2., baseline + 15.), label, 11., MUTED);
        p.text(
            Vec2::new(x + w / 2., baseline - h - 12.),
            probability_label(bin.probability),
            11.,
            TEXT,
        );
    }
    if bins.len() == frame.probabilities.len() && frame.amplitudes.is_some() {
        let phases = [
            (-std::f64::consts::PI, "−π"),
            (-std::f64::consts::FRAC_PI_2, "−π/2"),
            (0., "0"),
            (std::f64::consts::FRAC_PI_2, "π/2"),
            (std::f64::consts::PI, "π"),
        ];
        for (i, (phase, label)) in phases.into_iter().enumerate() {
            let x = bars.min.x + bars.width() * (i as f32 + 0.5) / 5.;
            let y = bars.max.y - 23.;
            p.rect(
                Rect::from_center_size(Vec2::new(x, y), Vec2::new(22., 5.)),
                phase_color(phase),
            );
            p.text(Vec2::new(x, y + 13.), label, 10., MUTED);
        }
        p.text(
            Vec2::new(bars.center().x, bars.max.y - 39.),
            "Amplitude phase (radians)",
            10.,
            MUTED,
        );
    } else {
        p.text(
            Vec2::new(bars.center().x, bars.max.y - 15.),
            if bins.len() < frame.probabilities.len() {
                "Adjacent states combined · all probability retained"
            } else {
                "Probabilities · amplitudes not supplied"
            },
            10.,
            MUTED,
        );
    }
    p.description = format!(
        "{} probability bins / {} states\nSelected q{} · norm {:.5}",
        bins.len(),
        frame.probabilities.len(),
        q.wire,
        frame.norm
    );
}
fn basis_label(index: usize, doc: &VisualizationDocument) -> String {
    let dims = doc.circuit.register().dimensions();
    let mut n = index;
    let mut digits = vec![0; dims.len()];
    for i in (0..dims.len()).rev() {
        digits[i] = n % dims[i];
        n /= dims[i];
    }
    if digits.len() <= 5 {
        format!(
            "|{}>",
            digits
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(if dims.iter().any(|d| *d > 10) {
                    ","
                } else {
                    ""
                })
        )
    } else {
        index.to_string()
    }
}

fn probability_label(probability: f64) -> String {
    let percent = probability * 100.;
    if percent > 0. && percent < 0.005 {
        "<0.01%".into()
    } else if (percent - percent.round()).abs() < 0.005 {
        format!("{percent:.0}%")
    } else {
        format!("{percent:.2}%")
    }
}
