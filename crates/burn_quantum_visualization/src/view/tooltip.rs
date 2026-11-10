//! One persistent, noninteractive UI overlay; hovering never changes experiment state.
use super::{
    plot::{CanvasBounds, PlotCache},
    theme::*,
};
use bevy::{feathers::constants::fonts, prelude::*};

#[derive(Clone, PartialEq)]
pub(super) struct HoverRegion {
    pub rect: Rect,
    pub title: String,
    pub summary: String,
}

#[derive(Resource, Default)]
pub(super) struct HoverState {
    candidate: Option<(u64, usize)>,
    since: f64,
    anchor: Vec2,
    pub shown: Option<HoverRegion>,
    pub rect: Option<Rect>,
}
#[derive(Component)]
pub(super) struct Tooltip;
#[derive(Component)]
pub(super) struct TooltipTitle;
#[derive(Component)]
pub(super) struct TooltipSummary;

pub(super) fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    let font = assets.load(fonts::REGULAR);
    let root = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                width: px(320),
                padding: UiRect::all(px(12)),
                border: UiRect::all(px(1)),
                border_radius: BorderRadius::all(px(7)),
                flex_direction: FlexDirection::Column,
                row_gap: px(5),
                ..default()
            },
            BackgroundColor(PANEL),
            BorderColor::all(ACCENT_LINE),
            GlobalZIndex(100),
            Visibility::Hidden,
            Pickable::IGNORE,
            Tooltip,
        ))
        .id();
    commands.spawn((
        Text::new(""),
        TextFont {
            font: font.clone().into(),
            ..TextFont::from_font_size(14.)
        },
        TextColor(TEXT),
        Pickable::IGNORE,
        ChildOf(root),
        TooltipTitle,
    ));
    commands.spawn((
        Text::new(""),
        TextFont {
            font: font.into(),
            ..TextFont::from_font_size(12.)
        },
        TextColor(MUTED),
        Pickable::IGNORE,
        ChildOf(root),
        TooltipSummary,
    ));
}

#[allow(clippy::too_many_arguments)] // Pointer inputs and one persistent UI overlay.
pub(super) fn update(
    window: Single<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    touches: Res<Touches>,
    time: Res<Time>,
    bounds: Res<CanvasBounds>,
    cache: Res<PlotCache>,
    mut hover: ResMut<HoverState>,
    mut popup: Single<(&mut Node, &ComputedNode, &mut Visibility), With<Tooltip>>,
    mut title: Single<&mut Text, (With<TooltipTitle>, Without<TooltipSummary>)>,
    mut summary: Single<&mut Text, (With<TooltipSummary>, Without<TooltipTitle>)>,
) {
    let (node, computed, visibility) = &mut *popup;
    let pointer = window.cursor_position().filter(|p| {
        window.focused
            && bounds.rect.contains(*p)
            && !buttons.pressed(MouseButton::Left)
            && !buttons.pressed(MouseButton::Right)
            && touches.iter().next().is_none()
    });
    let candidate = pointer
        .and_then(|p| cache.hover_regions.iter().rposition(|h| h.rect.contains(p)))
        .map(|index| (cache.generation, index));
    if candidate != hover.candidate {
        hover.candidate = candidate;
        hover.since = time.elapsed_secs_f64();
        hover.anchor = pointer.unwrap_or_default();
        hover.shown = None;
        hover.rect = None;
        **visibility = Visibility::Hidden;
    }
    let Some((_, index)) = candidate else { return };
    if time.elapsed_secs_f64() - hover.since < 0.25 {
        return;
    }
    let region = &cache.hover_regions[index];
    let width = 320_f32.min((bounds.window.x - 16.).max(1.));
    if hover.shown.as_ref() != Some(region) {
        title.0 = region.title.clone();
        summary.0 = region.summary.clone();
        node.width = px(width);
        hover.shown = Some(region.clone());
        // Lay out the new text while hidden, then position using its actual size.
        return;
    }
    let size = computed.size() * computed.inverse_scale_factor();
    let position = position(hover.anchor, size, bounds.window);
    // A stationary tooltip must not invalidate the entire UI layout each frame.
    if node.left != px(position.x) {
        node.left = px(position.x);
    }
    if node.top != px(position.y) {
        node.top = px(position.y);
    }
    if **visibility != Visibility::Visible {
        **visibility = Visibility::Visible;
    }
    hover.rect = Some(Rect::from_corners(position, position + size));
}

fn position(anchor: Vec2, size: Vec2, window: Vec2) -> Vec2 {
    let right = anchor.x + 16.;
    let below = anchor.y + 18.;
    Vec2::new(
        if right + size.x <= window.x - 8. {
            right
        } else {
            anchor.x - size.x - 12.
        },
        if below + size.y <= window.y - 8. {
            below
        } else {
            anchor.y - size.y - 12.
        },
    )
    .clamp(
        Vec2::splat(8.),
        (window - size - Vec2::splat(8.)).max(Vec2::splat(8.)),
    )
}
