//! Native Bevy UI and bounded, cached scientific drawing.
mod circuit;
pub(crate) mod diagnostics;
mod fabric;
mod input;
mod inspection;
mod learning;
mod plot;
mod state;
pub(crate) mod theme;
mod tooltip;
mod ui;
use bevy::prelude::*;
/// Installs responsive Feathers controls and scientific views.
pub(crate) struct ExplorerViewPlugin;
impl Plugin for ExplorerViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<plot::CanvasBounds>()
            .init_resource::<plot::PlotCache>()
            .init_resource::<tooltip::HoverState>()
            .init_resource::<state::surface::BlochSurfaces>()
            .add_systems(Startup, (ui::setup, tooltip::setup))
            .add_systems(
                Update,
                (
                    ui::responsive,
                    input::pointer_keyboard,
                    input::touch,
                    ui::sync_controls,
                    plot::rebuild,
                    tooltip::update,
                    plot::draw,
                )
                    .chain(),
            )
            .add_systems(
                PostUpdate,
                (plot::measure_canvas, diagnostics::update)
                    .chain()
                    .after(bevy::ui::UiSystems::Layout),
            );
    }
}
