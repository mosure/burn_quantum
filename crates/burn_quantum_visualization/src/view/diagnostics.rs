//! A read-only snapshot for accessibility tooling and browser interaction tests.
use crate::app::{Experiment, Explorer, ViewMode};
use bevy::prelude::*;
use std::cell::RefCell;

type ControlNode = (
    &'static ComputedNode,
    &'static UiGlobalTransform,
    Option<&'static super::ui::Action>,
    Option<&'static super::ui::Scrubber>,
    Has<bevy::ui::Pressed>,
    Has<bevy::ui::InteractionDisabled>,
);

thread_local! {
    static SNAPSHOT: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Read a QA-only snapshot without triggering simulation or rendering work.
/// Control rectangles use logical pixels relative to the Bevy canvas; this is
/// diagnostic data, not a stable scientific interchange schema.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
pub fn diagnostics_json() -> String {
    SNAPSHOT.with(|snapshot| snapshot.borrow().clone())
}

#[allow(clippy::too_many_arguments)] // Independent Bevy system parameters.
pub(super) fn update(
    _main: bevy::ecs::system::NonSendMarker,
    state: Res<Explorer>,
    experiment: Res<Experiment>,
    cache: Res<super::plot::PlotCache>,
    bounds: Res<super::plot::CanvasBounds>,
    entities: Query<Entity>,
    controls: Query<ControlNode>,
    scrolls: Query<
        (&ComputedNode, &UiGlobalTransform, &ScrollPosition),
        With<bevy::ui_widgets::ScrollArea>,
    >,
    adapter: Option<Res<bevy::render::renderer::RenderAdapterInfo>>,
    time: Res<Time>,
    mut clock: Local<(u64, f64, u64)>,
) {
    clock.0 += 1;
    let now = time.elapsed_secs_f64();
    if clock.0 > 1 && now - clock.1 < 0.25 && clock.2 == state.revision {
        return;
    }
    clock.1 = now;
    clock.2 = state.revision;
    let controls=controls.iter().filter_map(|(node,transform,action,slider,pressed,disabled)|{
        let size=node.size()*node.inverse_scale_factor();if size.x<1. || size.y<1. || (action.is_none() && slider.is_none()){return None;}
        let center=transform.affine().translation*node.inverse_scale_factor();let rect=Rect::from_center_size(center,size);
        Some(serde_json::json!({"pressed":pressed,"disabled":disabled,"action":action.map(|a|format!("{a:?}")).unwrap_or_else(||"Scrubber".into()),"rect":[rect.min.x,rect.min.y,rect.max.x,rect.max.y]}))
    }).collect::<Vec<_>>();
    let scroll_regions=scrolls.iter().filter_map(|(node,transform,offset)|{
        let scale=node.inverse_scale_factor();let size=node.size()*scale;
        if size.min_element()<1.{return None;}
        let center=transform.affine().translation*scale;let rect=Rect::from_center_size(center,size);
        Some(serde_json::json!({"rect":[rect.min.x,rect.min.y,rect.max.x,rect.max.y],"viewport_height":size.y,"content_height":node.content_size().y*scale,"offset_y":offset.y}))
    }).collect::<Vec<_>>();
    let value = serde_json::json!({
        "scroll_regions":scroll_regions,
        "controls":controls,
        "plot_hits":cache.hits.iter().map(|(r,h)|serde_json::json!({"action":format!("{h:?}"),"rect":[r.min.x,r.min.y,r.max.x,r.max.y]})).collect::<Vec<_>>(),
        "app_frames":clock.0,"frame_time_seconds":time.delta_secs_f64(),
        "rendering_adapter":adapter.as_ref().map(|a|serde_json::json!({"name":a.name,"backend":format!("{:?}",a.backend),"device_type":format!("{:?}",a.device_type)})),
        "demo": state.demo.id(),
        "config": state.config,
        "active_config": experiment.source_config,
        "source": if experiment.imported {"imported"} else {"catalog"},
        "query": crate::app::query::QueryState::capture(&state, &experiment).encode(),
        "config_valid":state.config_valid,
        "config_note":state.config_note,
        "view": match state.view {ViewMode::State=>"state",ViewMode::Circuit=>"circuit",ViewMode::Fabric=>"fabric",ViewMode::Learning=>"learning"},
        "ready": experiment.document.is_some(),
        "frame": state.frame,
        "tick": state.tick,
        "qubit": state.wire,
        "state_overview": state.state_overview,
        "bloch_mode":state.bloch_mode,
        "bloch_plane":state.bloch_plane,
        "trails":state.trails,
        "show_settings":state.show_settings,
        "overview_capacity": cache.overview_capacity,
        "gate": state.gate,
        "node": state.node,
        "playing": state.playing,
        "speed": state.speed,
        "zoom": state.zoom,
        "pan": [state.pan.x,state.pan.y],
        "orbit": [state.yaw,state.pitch],
        "guides": state.grid,
        "show_details":state.show_details,
        "inspection_text":super::inspection::describe(&state,experiment.document.as_ref()),
        "revision": state.revision,
        "entity_count": entities.iter().len(),
        "window_logical": [bounds.window.x,bounds.window.y],
        "canvas": [bounds.rect.min.x,bounds.rect.min.y,bounds.rect.max.x,bounds.rect.max.y],
        "drawing": cache.description,
        "title": experiment.document.as_ref().map(|d|d.title.as_str()),
        "status":state.status,
    });
    SNAPSHOT.with(|snapshot| *snapshot.borrow_mut() = value.to_string());
}
