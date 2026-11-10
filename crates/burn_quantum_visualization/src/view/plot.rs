use super::{theme::*, ui::Canvas};
use crate::app::{Experiment, Explorer, ViewMode};
use bevy::{feathers::constants::fonts, prelude::*};
#[derive(Resource, Default, Clone, Copy, PartialEq)]
pub(super) struct CanvasBounds {
    pub rect: Rect,
    pub window: Vec2,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Hit {
    Wire(usize),
    Gate(usize),
    Node(usize),
}
#[derive(Component)]
pub(super) struct PlotEntity;
#[derive(Clone)]
struct Label {
    position: Vec2,
    text: String,
    size: f32,
    color: Color,
}
#[derive(Resource, Default)]
pub(super) struct PlotCache {
    revision: u64,
    bounds: CanvasBounds,
    lines: Vec<(Vec2, Vec2, Color)>,
    boxes: Vec<(Rect, Color)>,
    labels: Vec<Label>,
    pub hits: Vec<(Rect, Hit)>,
    pub description: String,
}
impl PlotCache {
    pub fn line(&mut self, a: Vec2, b: Vec2, color: Color) {
        if self.lines.len() >= 8192 {
            return;
        }
        if let Some((a, b)) = clip(a, b, self.bounds.rect) {
            self.lines.push((a, b, color));
        }
    }
    pub fn rect(&mut self, rect: Rect, color: Color) {
        let min = rect.min.max(self.bounds.rect.min);
        let max = rect.max.min(self.bounds.rect.max);
        if min.cmplt(max).all() && self.boxes.len() < 512 {
            self.boxes.push((Rect::from_corners(min, max), color));
        }
    }
    pub fn outline(&mut self, rect: Rect, color: Color) {
        let p = [
            rect.min,
            Vec2::new(rect.max.x, rect.min.y),
            rect.max,
            Vec2::new(rect.min.x, rect.max.y),
        ];
        for i in 0..4 {
            self.line(p[i], p[(i + 1) % 4], color);
        }
    }
    pub fn text(&mut self, position: Vec2, text: impl Into<String>, size: f32, color: Color) {
        if self.labels.len() < 160 && self.bounds.rect.contains(position) {
            self.labels.push(Label {
                position,
                text: text.into(),
                size,
                color,
            });
        }
    }
    pub fn circle(&mut self, center: Vec2, radius: f32, color: Color) {
        let mut last = center + Vec2::X * radius;
        for i in 1..=64 {
            let a = i as f32 * std::f32::consts::TAU / 64.;
            let p = center + Vec2::new(a.cos(), a.sin()) * radius;
            self.line(last, p, color);
            last = p;
        }
    }
    pub fn dot(&mut self, center: Vec2, radius: f32, color: Color) {
        self.circle(center, radius, color);
        self.circle(center, (radius - 1.).max(0.5), color);
    }
    pub fn arrow(&mut self, a: Vec2, b: Vec2, color: Color) {
        self.line(a, b, color);
        let v = (b - a).normalize_or_zero();
        let side = Vec2::new(-v.y, v.x);
        self.line(b, b - v * 10. + side * 4., color);
        self.line(b, b - v * 10. - side * 4., color);
    }
    pub fn hit(&mut self, rect: Rect, hit: Hit) {
        self.hits.push((rect, hit));
    }
}
pub(super) fn measure_canvas(
    window: Single<&Window>,
    canvas: Query<(&ComputedNode, &UiGlobalTransform), With<Canvas>>,
    mut bounds: ResMut<CanvasBounds>,
) {
    if let Ok((node, transform)) = canvas.single() {
        let scale = node.inverse_scale_factor();
        let center = transform.affine().translation * scale;
        let size = node.size() * scale;
        let next = CanvasBounds {
            rect: Rect::from_center_size(center, size),
            window: Vec2::new(window.width(), window.height()),
        };
        if *bounds != next {
            *bounds = next;
        }
    }
}
pub(super) fn rebuild(
    mut commands: Commands,
    state: Res<Explorer>,
    experiment: Res<Experiment>,
    bounds: Res<CanvasBounds>,
    mut cache: ResMut<PlotCache>,
    old: Query<Entity, With<PlotEntity>>,
    assets: Res<AssetServer>,
) {
    if cache.revision == state.revision && cache.bounds == *bounds {
        return;
    }
    for e in &old {
        commands.entity(e).despawn();
    }
    cache.lines.clear();
    cache.boxes.clear();
    cache.labels.clear();
    cache.hits.clear();
    cache.revision = state.revision;
    cache.bounds = *bounds;
    cache.description = String::new();
    if bounds.rect.width() < 10. || bounds.rect.height() < 10. {
        return;
    }
    if let Some(doc) = &experiment.document {
        match state.view {
            ViewMode::State => super::state::build(&mut cache, bounds.rect, &state, doc),
            ViewMode::Circuit => super::circuit::build(&mut cache, bounds.rect, &state, doc),
            ViewMode::Fabric => super::fabric::build(&mut cache, bounds.rect, &state, doc),
            ViewMode::Learning => super::learning::build(&mut cache, bounds.rect, &state, doc),
        }
    } else {
        cache.text(
            bounds.rect.center(),
            "Preparing a quantum experiment",
            20.,
            MUTED,
        );
    }
    let world = |p: Vec2| Vec2::new(p.x - bounds.window.x / 2., bounds.window.y / 2. - p.y);
    for (layer, (rect, color)) in cache.boxes.iter().enumerate() {
        commands.spawn((
            Sprite::from_color(*color, rect.size()),
            // Preserve painter order for overlapping tracks and fills. Equal-Z
            // sprite sorting can otherwise hide a probability bar behind its
            // track after entities are recycled during a view change.
            Transform::from_translation(world(rect.center()).extend(-2. + layer as f32 * 0.001)),
            PlotEntity,
        ));
    }
    for label in &cache.labels {
        commands.spawn((
            Text2d::new(&label.text),
            TextFont {
                font: assets.load(fonts::REGULAR).into(),
                ..TextFont::from_font_size(label.size)
            },
            TextColor(label.color),
            TextLayout::justify(Justify::Center),
            Transform::from_translation(world(label.position).extend(1.)),
            PlotEntity,
        ));
    }
    let count = cache.lines.len();
    cache.description = format!(
        "{}\n{} line segments · {} labels",
        cache.description,
        count,
        cache.labels.len()
    );
}
pub(super) fn draw(mut gizmos: Gizmos, cache: Res<PlotCache>) {
    let world = |p: Vec2| {
        Vec2::new(
            p.x - cache.bounds.window.x / 2.,
            cache.bounds.window.y / 2. - p.y,
        )
    };
    for &(a, b, c) in &cache.lines {
        gizmos.line_2d(world(a), world(b), c);
    }
}
fn clip(a: Vec2, b: Vec2, rect: Rect) -> Option<(Vec2, Vec2)> {
    let d = b - a;
    let mut lo = 0f32;
    let mut hi = 1f32;
    for (p, q) in [
        (-d.x, a.x - rect.min.x),
        (d.x, rect.max.x - a.x),
        (-d.y, a.y - rect.min.y),
        (d.y, rect.max.y - a.y),
    ] {
        if p.abs() < 1e-6 {
            if q < 0. {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0. {
                lo = lo.max(r);
            } else {
                hi = hi.min(r);
            }
            if lo > hi {
                return None;
            }
        }
    }
    Some((a + d * lo, a + d * hi))
}
pub(super) fn short(value: &str, max: usize) -> String {
    let mut chars = value.chars();
    let text = chars.by_ref().take(max).collect::<String>();
    if chars.next().is_some() {
        format!("{text}…")
    } else {
        text
    }
}
