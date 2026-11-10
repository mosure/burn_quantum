//! Controller palette, mirrored in `web/index.html` for the browser shell.
//!
//! Surfaces and accents follow burn_gekko's `burn_ui/assets/theme.css`.
//! Scientific colors remain distinct from selection and keyboard focus.
use bevy::{
    feathers::{
        dark_theme::create_dark_theme,
        theme::{SemanticToken, UiTheme},
        tokens::{self, semantic},
    },
    prelude::*,
};
pub const BACKGROUND: Color = Color::srgb_u8(0x11, 0x10, 0x14);
pub const PANEL: Color = Color::srgb_u8(0x19, 0x18, 0x1e);
pub const INSET: Color = Color::srgb_u8(0x14, 0x13, 0x18);
pub const BORDER: Color = Color::srgb_u8(0x33, 0x30, 0x39);
pub const TEXT: Color = Color::srgb_u8(0xf2, 0xed, 0xef);
pub const MUTED: Color = Color::srgb_u8(0xaa, 0xa3, 0xaf);
pub const ACCENT: Color = Color::srgb_u8(0xd8, 0x7c, 0x7c);
pub const ACCENT_SOFT: Color = Color::srgb_u8(0x28, 0x20, 0x25);
pub const ACCENT_LINE: Color = Color::srgb_u8(0x62, 0x40, 0x44);
const HOVER: Color = Color::srgb_u8(0x23, 0x21, 0x29);
const PRESSED: Color = Color::srgb_u8(0x2d, 0x29, 0x32);
pub const GUIDE: Color = Color::srgb_u8(0x4c, 0x47, 0x51);
pub const CYAN: Color = Color::srgb_u8(0x8e, 0xca, 0xcb);
pub const GOLD: Color = Color::srgb_u8(0xe7, 0xc1, 0x8c);
pub const BLUE: Color = Color::srgb_u8(0xa5, 0xad, 0xc9);

/// Keep Feathers' complete token map and widget state handling, with controller
/// colors for every surface, hover/press/disabled state and focus indicator.
pub fn controller_theme() -> UiTheme {
    let mut theme = create_dark_theme();
    // The controller has a single control palette on all surface levels.
    theme.semantic_overrides.clear();
    theme.semantic_base.extend([
        (semantic::SURFACE_WINDOW, BACKGROUND),
        (semantic::SURFACE_PANE_BODY, PANEL),
        (semantic::SURFACE_PANE_HEADER, INSET),
        (semantic::SURFACE_SUBPANE_BODY, INSET),
        (semantic::SURFACE_SUBPANE_HEADER, PANEL),
        (semantic::SURFACE_GROUP, PANEL),
        (semantic::SURFACE_DIALOG, PANEL),
        (semantic::SURFACE_MENU, PANEL),
        (semantic::BORDER_DEFAULT, BORDER),
        (semantic::TEXT_DEFAULT, TEXT),
        (semantic::TEXT_DIM, MUTED),
        (semantic::TEXT_DISABLED, MUTED.with_alpha(0.4)),
        (semantic::TEXT_ON_ACCENT, BACKGROUND),
        (semantic::FILL_ACCENT_DEFAULT, ACCENT),
        (
            semantic::FILL_ACCENT_HOVER,
            Color::srgb_u8(0xe4, 0x90, 0x90),
        ),
        (
            semantic::FILL_ACCENT_PRESSED,
            Color::srgb_u8(0xc9, 0x6e, 0x6e),
        ),
        (semantic::FILL_ACCENT_DISABLED, ACCENT_LINE),
        (semantic::FILL_SOLID_DEFAULT, PANEL),
        (semantic::FILL_SOLID_HOVER, HOVER),
        (semantic::FILL_SOLID_PRESSED, PRESSED),
        (semantic::FILL_SOLID_DISABLED, INSET),
        (semantic::FILL_ITEM_HOVER, HOVER),
        (semantic::FILL_ITEM_PRESSED, PRESSED),
        (semantic::FILL_FIELD_DEFAULT, INSET),
        (semantic::FILL_FIELD_HOVER, HOVER),
        (semantic::FILL_FIELD_PRESSED, PRESSED),
        (semantic::FOCUS_RING, ACCENT),
    ]);
    // Soft selection backgrounds leave saturated color for the timeline/data.
    let selected = SemanticToken::new_static("quantum.selection");
    let selected_hover = SemanticToken::new_static("quantum.selection.hover");
    let selected_pressed = SemanticToken::new_static("quantum.selection.pressed");
    theme.semantic_base.extend([
        (selected.clone(), ACCENT_SOFT),
        (selected_hover.clone(), Color::srgb_u8(0x35, 0x26, 0x2b)),
        (selected_pressed.clone(), Color::srgb_u8(0x42, 0x2e, 0x33)),
    ]);
    theme.token_assignments.extend([
        (tokens::BUTTON_BG, semantic::FILL_SOLID_DEFAULT),
        (tokens::BUTTON_TEXT, semantic::TEXT_DEFAULT),
        (tokens::BUTTON_PRIMARY_BG, selected),
        (tokens::BUTTON_PRIMARY_BG_HOVER, selected_hover),
        (tokens::BUTTON_PRIMARY_BG_PRESSED, selected_pressed),
        (tokens::BUTTON_PRIMARY_TEXT, semantic::FILL_ACCENT_DEFAULT),
        (tokens::SLIDER_TEXT, semantic::TEXT_DEFAULT),
        (tokens::MENUITEM_TEXT, semantic::TEXT_DEFAULT),
        (tokens::TEXT_INPUT_TEXT, semantic::TEXT_DEFAULT),
        (tokens::LISTROW_TEXT, semantic::TEXT_DEFAULT),
    ]);
    UiTheme(theme)
}

pub fn phase_color(angle: f64) -> Color {
    // Retain a cyclic phase legend; zero phase shares the muted rose accent.
    Color::hsl((angle.to_degrees() as f32 + 360.) % 360., 0.54, 0.667)
}
