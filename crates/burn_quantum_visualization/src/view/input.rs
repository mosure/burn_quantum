use super::{
    plot::{CanvasBounds, Hit, PlotCache},
    ui::{self, Action},
};
use crate::app::{Experiment, Explorer, ViewMode};
use bevy::{
    input::mouse::{MouseScrollUnit, MouseWheel},
    prelude::*,
};
#[derive(Default)]
pub(super) struct Drag {
    last: Option<Vec2>,
    active: bool,
}
#[allow(clippy::too_many_arguments)] // Independent Bevy input resources and view state.
pub(super) fn pointer_keyboard(
    window: Single<&Window>,
    keys: Res<ButtonInput<KeyCode>>,
    touches: Res<Touches>,
    mut focus: ResMut<bevy::input_focus::InputFocus>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut wheel: MessageReader<MouseWheel>,
    bounds: Res<CanvasBounds>,
    cache: Res<PlotCache>,
    mut drag: Local<Drag>,
    mut state: ResMut<Explorer>,
    experiment: Res<Experiment>,
) {
    if state.view == ViewMode::Results {
        wheel.clear();
        drag.active = false;
        return;
    }
    for (key, action) in [
        (KeyCode::Space, Action::Play),
        (KeyCode::ArrowLeft, Action::Step(-1)),
        (KeyCode::ArrowRight, Action::Step(1)),
        (KeyCode::KeyR, Action::Reset),
        (KeyCode::KeyG, Action::Grid),
    ] {
        if keys.just_pressed(key)
            && (focus.get().is_none() || matches!(key, KeyCode::KeyR | KeyCode::KeyG))
        {
            ui::apply(action, &mut state, &experiment, cache.overview_capacity);
        }
    }
    if touches.iter().next().is_some() {
        drag.active = false;
        wheel.clear();
        return;
    }
    let position = window.cursor_position();
    if buttons.just_pressed(MouseButton::Left) || buttons.just_pressed(MouseButton::Right) {
        drag.active = position.is_some_and(|p| bounds.rect.contains(p));
        if drag.active {
            focus.clear();
            if buttons.just_pressed(MouseButton::Left)
                && let Some(p) = position
            {
                select(p, &cache, &mut state, &experiment);
            }
        }
    }

    if drag.active
        && (buttons.pressed(MouseButton::Left) || buttons.pressed(MouseButton::Right))
        && let (Some(now), Some(last)) = (position, drag.last)
    {
        let delta = now - last;
        if delta.length_squared() > 0. {
            if state.view == ViewMode::State
                && state.bloch_mode == crate::model::bloch::BlochMode::Spatial
                && buttons.pressed(MouseButton::Left)
            {
                state.yaw += delta.x * 0.008;
                state.pitch = (state.pitch + delta.y * 0.008).clamp(-1.25, 1.25);
            } else {
                state.pan += delta;
            }
            state.touch();
        }
    }
    if buttons.just_released(MouseButton::Left) || buttons.just_released(MouseButton::Right) {
        drag.active = false;
    }
    drag.last = position;
    for event in wheel.read() {
        if position.is_some_and(|p| bounds.rect.contains(p)) {
            let amount = match event.unit {
                MouseScrollUnit::Line => event.y * 0.10,
                MouseScrollUnit::Pixel => event.y * 0.002,
            };
            state.zoom = (state.zoom * amount.exp()).clamp(0.35, 3.);
            state.touch();
        }
    }
}

fn select(position: Vec2, cache: &PlotCache, state: &mut Explorer, experiment: &Experiment) {
    if let Some((_, hit)) = cache
        .hits
        .iter()
        .rev()
        .find(|(rect, _)| rect.contains(position))
    {
        match *hit {
            Hit::Wire(wire) => state.wire = wire,
            Hit::Node(node) => state.node = Some(node),
            Hit::Gate(gate) => {
                state.gate = Some(gate);
                if let Some(doc) = &experiment.document
                    && doc.trace.kind != crate::model::TraceKind::Optimization
                {
                    state.seek(gate + 1, doc.trace.frames.len());
                }
            }
        }
        state.touch();
    }
}
#[derive(Default)]
pub(super) struct TouchGesture {
    active: bool,
    scroll: Option<Entity>,
    last: Vec<(u64, Vec2)>,
}
pub(super) fn touch(
    touches: Res<Touches>,
    bounds: Res<CanvasBounds>,
    cache: Res<PlotCache>,
    mut gesture: Local<TouchGesture>,
    mut scrolls: Query<
        (
            Entity,
            &ComputedNode,
            &UiGlobalTransform,
            &mut ScrollPosition,
        ),
        With<bevy::ui_widgets::ScrollArea>,
    >,
    mut state: ResMut<Explorer>,
    experiment: Res<Experiment>,
) {
    let mut current = touches
        .iter()
        .map(|t| (t.id(), t.position()))
        .collect::<Vec<_>>();
    current.sort_by_key(|x| x.0);
    current.truncate(2);
    if current.is_empty() {
        gesture.active = false;
        gesture.scroll = None;
        gesture.last.clear();
        return;
    }
    if gesture.last.is_empty() {
        gesture.scroll = scrolls.iter().find_map(|(entity, node, transform, _)| {
            let size = node.size() * node.inverse_scale_factor();
            let center = transform.affine().translation * node.inverse_scale_factor();
            (size.min_element() > 1. && Rect::from_center_size(center, size).contains(current[0].1))
                .then_some(entity)
        });
        gesture.active = state.view != ViewMode::Results
            && gesture.scroll.is_none()
            && bounds.rect.contains(current[0].1);
        if gesture.active {
            select(current[0].1, &cache, &mut state, &experiment);
        }
    } else if let Some(entity) = gesture.scroll {
        if let Some((_, old)) = gesture.last.iter().find(|(id, _)| *id == current[0].0)
            && let Ok((_, _, _, mut position)) = scrolls.get_mut(entity)
        {
            position.y -= (current[0].1 - *old).y;
        }
    } else if gesture.active
        && current.len() == gesture.last.len()
        && current.iter().zip(&gesture.last).all(|(a, b)| a.0 == b.0)
    {
        if current.len() == 2 {
            let now = current[0].1.distance(current[1].1);
            let old = gesture.last[0].1.distance(gesture.last[1].1);
            if old > 1. && now > 1. {
                let ratio = now / old;
                let delta =
                    (current[0].1 + current[1].1 - gesture.last[0].1 - gesture.last[1].1) * 0.5;
                if (ratio - 1.).abs() > 1e-5 || delta.length_squared() > 0. {
                    state.zoom = (state.zoom * ratio).clamp(0.35, 3.);
                    state.pan += delta;
                    state.touch();
                }
            }
        } else {
            let delta = current[0].1 - gesture.last[0].1;
            if delta.length_squared() > 0. {
                if state.view == ViewMode::State
                    && state.bloch_mode == crate::model::bloch::BlochMode::Spatial
                {
                    state.yaw += delta.x * 0.008;
                    state.pitch = (state.pitch + delta.y * 0.008).clamp(-1.25, 1.25);
                } else {
                    state.pan += delta;
                }
                state.touch();
            }
        }
    }
    gesture.last = current;
}
