//! Consistent visual tokens shared by native UI and scientific plots.
use bevy::prelude::*;
pub const BACKGROUND: Color = Color::srgb(0.037, 0.055, 0.082);
pub const PANEL: Color = Color::srgb(0.062, 0.084, 0.118);
pub const INSET: Color = Color::srgb(0.080, 0.108, 0.148);
pub const BORDER: Color = Color::srgb(0.15, 0.20, 0.27);
pub const TEXT: Color = Color::srgb(0.91, 0.94, 0.98);
pub const MUTED: Color = Color::srgb(0.49, 0.58, 0.70);
pub const CYAN: Color = Color::srgb(0.23, 0.87, 0.81);
pub const GOLD: Color = Color::srgb(1.0, 0.70, 0.34);
pub const BLUE: Color = Color::srgb(0.40, 0.60, 1.0);
pub fn phase_color(angle: f64) -> Color {
    Color::hsl((angle.to_degrees() as f32 + 360.) % 360., 0.68, 0.64)
}
