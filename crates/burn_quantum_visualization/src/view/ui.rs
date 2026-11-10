use super::{plot::PlotCache, theme::*};
use crate::model::bloch::BlochMode;
use crate::{
    app::{Experiment, Explorer, ViewMode},
    demo::{DemoConfig, DemoKind},
};
use bevy::{
    feathers::{
        constants::fonts,
        controls::{ButtonVariant, FeathersButton, FeathersSlider},
        display::caption,
    },
    prelude::*,
    ui_widgets::{Activate, SliderPrecision, SliderRange, SliderStep, SliderValue, ValueChange},
};
#[derive(Component)]
pub(super) struct Canvas;
#[derive(Component, Clone, Copy)]
pub(super) enum Responsive {
    Sidebar,
    Inspector,
    InspectorExtras,
    CompactContext,
    Plot,
    NarrowHide,
    CompactToolbar,
    Header,
    Main,
    Footer,
    Description,
    Note,
    StateTools,
    FocusOnly,
    OverviewGroups,
    Configuration,
    DepthSetting,
    VariantSetting,
    SettingsPanel,
    StateToolbar,
    FocusToggle,
    Title,
    SmallNavigation,
    ViewTab(ViewMode),
    PlaneSetting,
    BlochTools,
}
#[derive(Component)]
pub(super) struct BrandTitle;
#[derive(Component, Default, Clone)]
pub(super) struct Scrubber;
#[derive(Component, Clone, Copy)]
pub(super) enum Field {
    Title,
    Description,
    Status,
    Inspection,
    Context,
    Step,
    ViewNote,
    Stats,
    OverviewGroup,
    ConfigSize,
    ConfigDepth,
    ConfigVariant,
    ConfigNote,
    PlaneName,
    ViewSummary,
}
#[derive(Component, Clone, Copy, Debug)]
pub(super) enum Action {
    Demo(DemoKind),
    View(ViewMode),
    Play,
    Step(i32),
    Reset,
    Wire(i32),
    Speed,
    Grid,
    NextDemo,
    Inspect,
    StateGroup(i32),
    Size(i32),
    Depth(i32),
    Variant,
    Run,
    Settings,
    ToggleFocus,
    BlochMode(BlochMode),
    Plane,
    Trails,
}
fn text(
    commands: &mut Commands,
    parent: Entity,
    content: impl Into<String>,
    size: f32,
    color: Color,
    font: Handle<Font>,
) -> Entity {
    commands
        .spawn((
            Text::new(content),
            TextFont {
                font: font.into(),
                ..TextFont::from_font_size(size)
            },
            TextColor(color),
            ChildOf(parent),
        ))
        .id()
}
fn node(commands: &mut Commands, parent: Entity, style: Node, color: Option<Color>) -> Entity {
    let mut e = commands.spawn((style, ChildOf(parent)));
    if let Some(c) = color {
        e.insert(BackgroundColor(c));
    }
    e.id()
}
fn button(
    commands: &mut Commands,
    parent: Entity,
    label: String,
    action: Action,
    primary: bool,
    width: Val,
) -> Entity {
    let tab = matches!(action, Action::View(_));
    let navigation = matches!(action, Action::Demo(_));
    let border = if tab {
        UiRect::bottom(px(2))
    } else if navigation {
        UiRect::ZERO
    } else {
        UiRect::all(px(1))
    };
    let radius = BorderRadius::all(px(if tab { 0. } else { 6. }));
    // Navigation is a text list: a shared leading edge and content-sized rows
    // keep long experiment titles legible instead of centering or squeezing them.
    if navigation {
        return commands.spawn_scene(bsn!{
            @FeathersButton { @variant:{ButtonVariant::selected(primary)}, @caption:bsn!{
                @caption(label)
                Node {width:percent(100),min_width:px(0)}
                TextLayout::justify(Justify::Left)
            } }
            Node {height:Val::Auto,min_height:px(36),flex_shrink:0.,width:{width},padding:UiRect::axes(px(12),px(8)),justify_content:JustifyContent::FlexStart,border:{border},border_radius:{radius}}
            BorderColor::all(BORDER)
            action
            ChildOf(parent)
            on(activate)
        }).id();
    }
    commands.spawn_scene(bsn!{
        @FeathersButton { @variant:{ButtonVariant::selected(primary)}, @caption:bsn!{@caption(label)} }
        Node {height:px(36),min_height:px(36),width:{width},padding:UiRect::horizontal(px(12)),border:{border},border_radius:{radius}}
        BorderColor::all(BORDER)
        action
        ChildOf(parent)
        on(activate)
    }).id()
}

pub(super) fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    commands.spawn((Camera2d, IsDefaultUiCamera));
    let font = assets.load(fonts::REGULAR);
    let mono = assets.load(fonts::MONO);
    let root = commands
        .spawn(Node {
            display: Display::Grid,
            width: percent(100),
            height: percent(100),
            // A zero-minimum middle track keeps long inspector content from
            // taking space reserved for the header and playback controls.
            grid_template_rows: vec![GridTrack::auto(), GridTrack::flex(1.), GridTrack::auto()],
            ..default()
        })
        .id();
    let header = node(
        &mut commands,
        root,
        Node {
            grid_row: GridPlacement::start(1),
            height: px(88),
            min_height: px(88),
            padding: UiRect::axes(px(24), px(16)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::SpaceBetween,
            border: UiRect::bottom(px(1)),
            ..default()
        },
        Some(INSET),
    );
    commands
        .entity(header)
        .insert((BorderColor::all(BORDER), Responsive::Header));
    let brand = node(
        &mut commands,
        header,
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: px(3),
            ..default()
        },
        None,
    );
    text(
        &mut commands,
        brand,
        "BURN / QUANTUM",
        11.,
        MUTED,
        mono.clone(),
    );
    let brand_title = text(
        &mut commands,
        brand,
        "Quantum explorer",
        25.,
        TEXT,
        font.clone(),
    );
    commands.entity(brand_title).insert(BrandTitle);
    let headright = node(
        &mut commands,
        header,
        Node {
            align_items: AlignItems::Center,
            column_gap: px(12),
            ..default()
        },
        None,
    );
    let next_demo = button(
        &mut commands,
        headright,
        "Next experiment".into(),
        Action::NextDemo,
        false,
        Val::Auto,
    );
    commands
        .entity(next_demo)
        .insert(Responsive::SmallNavigation);
    let status = text(
        &mut commands,
        headright,
        "LOCAL SIMULATION",
        11.,
        MUTED,
        mono.clone(),
    );
    commands
        .entity(status)
        .insert((Field::Status, Responsive::NarrowHide));
    let body = node(
        &mut commands,
        root,
        Node {
            grid_row: GridPlacement::start(2),
            flex_grow: 1.,
            flex_basis: px(0),
            min_height: px(0),
            ..default()
        },
        None,
    );
    let sidebar = node(
        &mut commands,
        body,
        Node {
            overflow: Overflow::scroll_y(),
            width: px(218),
            min_width: px(218),
            min_height: px(0),
            padding: UiRect::all(px(18)),
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            border: UiRect::right(px(1)),
            ..default()
        },
        Some(PANEL),
    );
    commands.entity(sidebar).insert((
        Responsive::Sidebar,
        BorderColor::all(BORDER),
        bevy::ui_widgets::ScrollArea,
    ));
    let list_heading = text(
        &mut commands,
        sidebar,
        "EXPERIMENTS",
        11.,
        MUTED,
        mono.clone(),
    );
    commands.entity(list_heading).insert(Node {
        flex_shrink: 0.,
        margin: UiRect::bottom(px(8)),
        ..default()
    });
    for &kind in DemoKind::all() {
        button(
            &mut commands,
            sidebar,
            kind.title().into(),
            Action::Demo(kind),
            kind == DemoKind::Bell,
            percent(100),
        );
    }
    let main = node(
        &mut commands,
        body,
        Node {
            flex_grow: 1.,
            min_width: px(0),
            min_height: px(0),
            padding: UiRect::all(px(24)),
            flex_direction: FlexDirection::Column,
            row_gap: px(16),
            ..default()
        },
        None,
    );
    commands.entity(main).insert(Responsive::Main);
    let titles = node(
        &mut commands,
        main,
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: px(5),
            ..default()
        },
        None,
    );
    let title = text(
        &mut commands,
        titles,
        "Preparing your experiment",
        26.,
        TEXT,
        font.clone(),
    );
    commands
        .entity(title)
        .insert((Field::Title, Responsive::Title));
    let description = text(
        &mut commands,
        titles,
        "Explore a state, follow its circuit, and inspect what changes.",
        13.,
        MUTED,
        font.clone(),
    );
    commands
        .entity(description)
        .insert((Field::Description, Responsive::Description));
    let settings = node(
        &mut commands,
        main,
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            ..default()
        },
        None,
    );
    commands.entity(settings).insert(Responsive::Configuration);
    text(
        &mut commands,
        settings,
        "EXPERIMENT",
        11.,
        MUTED,
        mono.clone(),
    );
    let settings_row = node(
        &mut commands,
        settings,
        Node {
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(6),
            row_gap: px(4),
            align_items: AlignItems::Center,
            ..default()
        },
        None,
    );
    button(
        &mut commands,
        settings_row,
        "−".into(),
        Action::Size(-1),
        false,
        Val::Auto,
    );
    let size_label = text(&mut commands, settings_row, "", 12., TEXT, mono.clone());
    commands.entity(size_label).insert(Field::ConfigSize);
    button(
        &mut commands,
        settings_row,
        "+".into(),
        Action::Size(1),
        false,
        Val::Auto,
    );
    let depth = node(
        &mut commands,
        settings_row,
        Node {
            column_gap: px(5),
            align_items: AlignItems::Center,
            ..default()
        },
        None,
    );
    commands.entity(depth).insert(Responsive::DepthSetting);
    button(
        &mut commands,
        depth,
        "− Layer".into(),
        Action::Depth(-1),
        false,
        Val::Auto,
    );
    let depth_label = text(&mut commands, depth, "", 12., TEXT, mono.clone());
    commands.entity(depth_label).insert(Field::ConfigDepth);
    button(
        &mut commands,
        depth,
        "+ Layer".into(),
        Action::Depth(1),
        false,
        Val::Auto,
    );
    let variant = node(
        &mut commands,
        settings_row,
        Node {
            column_gap: px(5),
            align_items: AlignItems::Center,
            flex_wrap: FlexWrap::Wrap,
            ..default()
        },
        None,
    );
    commands.entity(variant).insert(Responsive::VariantSetting);
    button(
        &mut commands,
        variant,
        "Variant".into(),
        Action::Variant,
        false,
        Val::Auto,
    );
    let variant_label = text(&mut commands, variant, "", 11., MUTED, font.clone());
    commands.entity(variant_label).insert(Field::ConfigVariant);
    button(
        &mut commands,
        settings_row,
        "Run".into(),
        Action::Run,
        true,
        Val::Auto,
    );
    let config_note = text(&mut commands, settings, "", 11., MUTED, font.clone());
    commands.entity(config_note).insert(Field::ConfigNote);
    let tabs = node(
        &mut commands,
        main,
        Node {
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(7),
            row_gap: px(6),
            ..default()
        },
        None,
    );
    for mode in ViewMode::ALL {
        let tab = button(
            &mut commands,
            tabs,
            mode.title().into(),
            Action::View(mode),
            mode == ViewMode::State,
            Val::Auto,
        );
        commands.entity(tab).insert(Responsive::ViewTab(mode));
    }
    button(
        &mut commands,
        tabs,
        "Settings".into(),
        Action::Settings,
        false,
        Val::Auto,
    );
    let details = button(
        &mut commands,
        tabs,
        "Details".into(),
        Action::Inspect,
        false,
        Val::Auto,
    );
    commands.entity(details).insert(Responsive::CompactToolbar);
    let view_tools = node(
        &mut commands,
        main,
        Node {
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(10),
            row_gap: px(6),
            align_items: AlignItems::Center,
            ..default()
        },
        None,
    );
    commands.entity(view_tools).insert(Responsive::StateToolbar);
    let focus = button(
        &mut commands,
        view_tools,
        "Focus".into(),
        Action::ToggleFocus,
        false,
        Val::Auto,
    );
    commands.entity(focus).insert(Responsive::FocusToggle);
    let summary = text(&mut commands, view_tools, "", 11., MUTED, mono.clone());
    commands.entity(summary).insert(Field::ViewSummary);
    let state_tools = node(
        &mut commands,
        view_tools,
        Node {
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(6),
            row_gap: px(4),
            ..default()
        },
        None,
    );
    commands.entity(state_tools).insert(Responsive::StateTools);
    text(
        &mut commands,
        state_tools,
        "Projection",
        12.,
        MUTED,
        font.clone(),
    );
    let mode_tools = node(
        &mut commands,
        state_tools,
        Node {
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(4),
            row_gap: px(4),
            align_items: AlignItems::Center,
            ..default()
        },
        None,
    );
    commands.entity(mode_tools).insert(Responsive::BlochTools);
    for (label, mode) in [
        ("3D", BlochMode::Spatial),
        ("2D probability", BlochMode::Planar),
        ("Projection", BlochMode::Projection),
    ] {
        button(
            &mut commands,
            mode_tools,
            label.into(),
            Action::BlochMode(mode),
            mode == BlochMode::Spatial,
            Val::Auto,
        );
    }
    let plane = node(
        &mut commands,
        mode_tools,
        Node {
            column_gap: px(4),
            align_items: AlignItems::Center,
            ..default()
        },
        None,
    );
    commands.entity(plane).insert(Responsive::PlaneSetting);
    button(
        &mut commands,
        plane,
        "Plane".into(),
        Action::Plane,
        false,
        Val::Auto,
    );
    let plane_name = text(&mut commands, plane, "", 11., MUTED, mono.clone());
    commands.entity(plane_name).insert(Field::PlaneName);
    let groups = node(
        &mut commands,
        view_tools,
        Node {
            column_gap: px(6),
            align_items: AlignItems::Center,
            ..default()
        },
        None,
    );
    commands.entity(groups).insert(Responsive::OverviewGroups);
    button(
        &mut commands,
        groups,
        "←".into(),
        Action::StateGroup(-1),
        false,
        Val::Auto,
    );
    let group_label = text(&mut commands, groups, "", 11., MUTED, mono.clone());
    commands.entity(group_label).insert(Field::OverviewGroup);
    button(
        &mut commands,
        groups,
        "→".into(),
        Action::StateGroup(1),
        false,
        Val::Auto,
    );
    let compact = node(
        &mut commands,
        view_tools,
        Node {
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(6),
            row_gap: px(4),
            ..default()
        },
        None,
    );

    let next_wire = button(
        &mut commands,
        view_tools,
        "Next subsystem".into(),
        Action::Wire(1),
        false,
        Val::Auto,
    );
    commands.entity(next_wire).insert(Responsive::FocusOnly);
    button(
        &mut commands,
        compact,
        "Reset".into(),
        Action::Reset,
        false,
        Val::Auto,
    );
    button(
        &mut commands,
        compact,
        "Guides".into(),
        Action::Grid,
        false,
        Val::Auto,
    );
    let trails = button(
        &mut commands,
        compact,
        "Trails".into(),
        Action::Trails,
        false,
        Val::Auto,
    );
    commands.entity(trails).insert(Responsive::StateTools);
    button(
        &mut commands,
        compact,
        "Playback speed".into(),
        Action::Speed,
        false,
        Val::Auto,
    );
    let work = node(
        &mut commands,
        main,
        Node {
            flex_grow: 1.,
            flex_basis: px(0),
            min_height: px(0),
            column_gap: px(18),
            ..default()
        },
        None,
    );
    let options = node(
        &mut commands,
        work,
        Node {
            position_type: PositionType::Absolute,
            top: px(0),
            left: px(0),
            width: percent(100),
            height: percent(100),
            min_height: px(0),
            overflow: Overflow::scroll_y(),
            flex_direction: FlexDirection::Column,
            padding: UiRect::all(px(20)),
            row_gap: px(18),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(10)),
            ..default()
        },
        Some(PANEL),
    );
    commands.entity(options).insert((
        Responsive::SettingsPanel,
        bevy::ui_widgets::ScrollArea,
        BorderColor::all(BORDER),
    ));
    text(
        &mut commands,
        options,
        "VIEW SETTINGS",
        11.,
        MUTED,
        mono.clone(),
    );
    commands.entity(state_tools).insert(ChildOf(options));
    commands.entity(compact).insert(ChildOf(options));
    commands.entity(settings).insert(ChildOf(options));
    let canvas = node(
        &mut commands,
        work,
        Node {
            flex_grow: 1.,
            min_width: px(0),
            min_height: px(0),
            ..default()
        },
        None,
    );
    commands.entity(canvas).insert((Canvas, Responsive::Plot));
    let inspector = node(
        &mut commands,
        work,
        Node {
            width: px(240),
            min_width: px(240),
            padding: UiRect::all(px(18)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(12)),
            flex_direction: FlexDirection::Column,
            row_gap: px(15),
            ..default()
        },
        Some(PANEL),
    );
    commands.entity(inspector).insert((
        Responsive::Inspector,
        bevy::ui_widgets::ScrollArea,
        BorderColor::all(BORDER),
    ));
    let inspector_heading = text(
        &mut commands,
        inspector,
        "INSPECTOR",
        11.,
        MUTED,
        mono.clone(),
    );
    commands.entity(inspector_heading).insert(Node {
        flex_shrink: 0.,
        ..default()
    });
    let inspect = text(
        &mut commands,
        inspector,
        "Select a qubit or gate\nto see its state.",
        14.,
        TEXT,
        font.clone(),
    );
    commands.entity(inspect).insert((
        Field::Inspection,
        Node {
            flex_shrink: 0.,
            ..default()
        },
    ));
    let context = text(&mut commands, inspector, "", 12., MUTED, font.clone());
    commands.entity(context).insert((
        Field::Context,
        Responsive::CompactContext,
        Node {
            flex_shrink: 0.,
            ..default()
        },
    ));
    let extras = node(
        &mut commands,
        inspector,
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: px(15),
            ..default()
        },
        None,
    );
    commands.entity(extras).insert(Responsive::InspectorExtras);
    text(
        &mut commands,
        extras,
        "3D: drag to orbit.\nFocus / diagrams: drag to pan.\nScroll to zoom.\nSpace to play · ← → to step.",
        12.,
        MUTED,
        font.clone(),
    );
    let stats = text(
        &mut commands,
        extras,
        "Bounded drawing",
        11.,
        MUTED,
        mono.clone(),
    );
    commands.entity(stats).insert(Field::Stats);
    let note = text(
        &mut commands,
        main,
        "The Bloch ball shows one reduced qubit; entanglement can place it inside the sphere.",
        12.,
        MUTED,
        font.clone(),
    );
    commands
        .entity(note)
        .insert((Field::ViewNote, Responsive::Note));
    let footer = node(
        &mut commands,
        root,
        Node {
            grid_row: GridPlacement::start(3),
            min_height: px(112),
            padding: UiRect::axes(px(24), px(18)),
            border: UiRect::top(px(1)),
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            ..default()
        },
        Some(PANEL),
    );
    commands
        .entity(footer)
        .insert((BorderColor::all(BORDER), Responsive::Footer));
    let controls = node(
        &mut commands,
        footer,
        Node {
            column_gap: px(7),
            row_gap: px(6),
            flex_wrap: FlexWrap::Wrap,
            align_items: AlignItems::Center,
            ..default()
        },
        None,
    );
    button(
        &mut commands,
        controls,
        "←".into(),
        Action::Step(-1),
        false,
        Val::Auto,
    );
    let play = button(
        &mut commands,
        controls,
        "Play / pause".into(),
        Action::Play,
        true,
        px(120),
    );
    let _ = play;
    button(
        &mut commands,
        controls,
        "→".into(),
        Action::Step(1),
        false,
        Val::Auto,
    );
    let timeline = text(&mut commands, controls, "Step 0 / 0", 13., TEXT, mono);
    commands.entity(timeline).insert(Field::Step);
    commands.spawn_scene(bsn!{@FeathersSlider{@min:0.,@max:1.} Node{width:percent(100),height:px(22)} SliderPrecision(0) SliderStep(1.) Scrubber ChildOf(footer) on(scrub)});
}
fn activate(
    event: On<Activate>,
    actions: Query<&Action>,
    mut state: ResMut<Explorer>,
    experiment: Res<Experiment>,
    cache: Res<PlotCache>,
) {
    if let Ok(action) = actions.get(event.entity) {
        apply(*action, &mut state, &experiment, cache.overview_capacity);
    }
}
pub(super) fn apply(
    action: Action,
    state: &mut Explorer,
    experiment: &Experiment,
    capacity: usize,
) {
    match action {
        Action::Settings => {
            state.show_settings = !state.show_settings;
            state.show_details = false;
            state.touch();
        }
        Action::ToggleFocus => {
            state.state_overview = !state.state_overview;
            state.reset_camera();
        }
        Action::BlochMode(mode) => {
            state.bloch_mode = mode;
            state.touch();
        }
        Action::Plane => {
            state.bloch_plane = state.bloch_plane.next();
            state.touch();
        }
        Action::Trails => {
            state.trails = !state.trails;
            state.touch();
        }
        Action::Size(delta) => {
            let (min, max, step) = DemoConfig::range(state.demo);
            state.config.size = state
                .config
                .size
                .saturating_add_signed(delta as isize * step as isize)
                .clamp(min, max);
            state.preview_config();
        }
        Action::Depth(delta) => {
            let maximum = DemoConfig::max_depth(state.demo);
            state.config.depth = state
                .config
                .depth
                .saturating_add_signed(delta as isize)
                .clamp(1, maximum);
            state.preview_config();
        }
        Action::Variant => {
            state.config.variant = (state.config.variant + 1) % DemoConfig::variants(state.demo);
            state.preview_config();
        }
        Action::Run => {
            if state.config_valid {
                state.requested = true;
                state.show_settings = false;
                state.frame = 0;
                state.show_details = false;
                state.touch();
            }
        }
        Action::StateGroup(delta) => {
            if let Some(doc) = &experiment.document {
                let capacity = capacity.max(1);
                let pages = doc.circuit.register().wires().div_ceil(capacity);
                state.wire = ((state.wire / capacity) as i32 + delta).rem_euclid(pages as i32)
                    as usize
                    * capacity;
                state.touch();
            }
        }
        Action::Inspect => {
            state.show_details = !state.show_details;
            state.show_settings = false;
            state.touch();
        }
        Action::Demo(kind) => {
            state.show_details = false;
            state.show_settings = false;
            state.demo = kind;
            state.config = DemoConfig::for_demo(kind);
            state.preview_config();
            state.requested = true;
            state.frame = 0;
            state.tick = 0.;
            state.gate = None;
            state.node = None;
            state.touch();
        }
        Action::View(mode) => {
            state.show_details = false;
            state.show_settings = false;
            state.view = mode;
            state.playing = false;
            state.reset_camera();
        }
        Action::Play => {
            if let Some(doc) = &experiment.document {
                let len = timeline_len(state, doc);
                if cursor(state) + 1 >= len {
                    seek(state, doc, 0);
                }
                state.playing = !state.playing;
                state.touch();
            }
        }
        Action::Step(delta) => {
            if let Some(doc) = &experiment.document {
                state.playing = false;
                let next = if delta == i32::MIN {
                    0
                } else {
                    cursor(state).saturating_add_signed(delta as isize)
                };
                seek(state, doc, next);
            }
        }
        Action::Reset => state.reset_camera(),
        Action::Grid => {
            state.grid = !state.grid;
            state.touch();
        }
        Action::Wire(delta) => {
            if let Some(frame) = experiment
                .document
                .as_ref()
                .and_then(|d| d.trace.frames.get(state.frame))
            {
                let len = frame.subsystems.len();
                if len > 0 {
                    state.wire = (state.wire as i32 + delta).rem_euclid(len as i32) as usize;
                    state.touch();
                }
            }
        }
        Action::Speed => {
            state.speed = if state.speed >= 4. {
                0.5
            } else {
                state.speed * 2.
            };
            state.touch();
        }
        Action::NextDemo => {
            let all = DemoKind::all();
            let current = all.iter().position(|k| *k == state.demo).unwrap_or(0);
            apply(
                Action::Demo(all[(current + 1) % all.len()]),
                state,
                experiment,
                capacity,
            );
        }
    }
}
fn scrub(change: On<ValueChange<f32>>, mut state: ResMut<Explorer>, experiment: Res<Experiment>) {
    if let Some(doc) = &experiment.document {
        state.playing = false;
        seek(&mut state, doc, change.value.round().max(0.) as usize);
    }
}
pub(super) fn timeline_len(state: &Explorer, doc: &crate::model::VisualizationDocument) -> usize {
    if state.view == ViewMode::Fabric
        && let Some(f) = &doc.fabric
    {
        return (f.finish_tick as usize).saturating_add(1);
    }
    doc.trace.frames.len()
}
pub(super) fn cursor(state: &Explorer) -> usize {
    if state.view == ViewMode::Fabric {
        state.tick.round() as usize
    } else {
        state.frame
    }
}
pub(super) fn seek(state: &mut Explorer, doc: &crate::model::VisualizationDocument, index: usize) {
    if state.view == ViewMode::Fabric {
        state.tick = index.min(timeline_len(state, doc).saturating_sub(1)) as f32;
        state.touch();
    } else {
        state.seek(index, doc.trace.frames.len());
    }
}
pub(super) fn responsive(
    window: Single<&Window>,
    state: Res<Explorer>,
    experiment: Res<Experiment>,
    cache: Res<PlotCache>,
    mut nodes: Query<(&Responsive, &mut Node)>,
    mut brands: Query<&mut TextFont, With<BrandTitle>>,
) {
    let width = window.width();
    let height = window.height();
    let inspector_visible = width >= 1180. && height >= 850.;
    let compact = width < 540. || height < 760.;
    for (kind, mut node) in &mut nodes {
        match kind {
            Responsive::Sidebar
            | Responsive::Inspector
            | Responsive::InspectorExtras
            | Responsive::CompactContext
            | Responsive::Plot
            | Responsive::NarrowHide
            | Responsive::CompactToolbar
            | Responsive::Description
            | Responsive::Note
            | Responsive::StateTools
            | Responsive::FocusOnly
            | Responsive::OverviewGroups
            | Responsive::Configuration
            | Responsive::DepthSetting
            | Responsive::VariantSetting
            | Responsive::SettingsPanel
            | Responsive::StateToolbar
            | Responsive::FocusToggle
            | Responsive::Title
            | Responsive::SmallNavigation
            | Responsive::ViewTab(_)
            | Responsive::PlaneSetting
            | Responsive::BlochTools => {
                let visible = match kind {
                    Responsive::FocusToggle => {
                        state.view == ViewMode::State
                            && experiment
                                .document
                                .as_ref()
                                .is_some_and(|d| d.circuit.register().wires() > 1)
                    }
                    Responsive::SettingsPanel => state.show_settings,
                    Responsive::StateToolbar => !state.show_settings && !state.show_details,
                    Responsive::Title => !cfg!(target_arch = "wasm32"),
                    Responsive::SmallNavigation => width < 780. || height < 700.,
                    Responsive::ViewTab(mode) => match mode {
                        ViewMode::Fabric => experiment
                            .document
                            .as_ref()
                            .is_some_and(|d| d.fabric.is_some()),
                        ViewMode::Learning => experiment
                            .document
                            .as_ref()
                            .is_some_and(|d| !d.optimization.is_empty()),
                        ViewMode::Results => experiment
                            .document
                            .as_ref()
                            .is_some_and(|d| d.study.is_some()),
                        _ => true,
                    },
                    Responsive::BlochTools => experiment
                        .document
                        .as_ref()
                        .is_some_and(|d| d.circuit.register().dimensions().contains(&2)),
                    Responsive::PlaneSetting => state.bloch_mode == BlochMode::Projection,
                    Responsive::Configuration => {
                        DemoConfig::range(state.demo).0 != DemoConfig::range(state.demo).1
                            && !state.show_details
                            && state.show_settings
                    }
                    Responsive::DepthSetting => matches!(
                        state.demo,
                        DemoKind::Qaoa | DemoKind::WineClassifier | DemoKind::Vqe
                    ),
                    Responsive::VariantSetting => DemoConfig::variants(state.demo) > 1,
                    Responsive::StateTools => {
                        state.view == ViewMode::State
                            && experiment
                                .document
                                .as_ref()
                                .is_some_and(|d| d.circuit.register().dimensions().contains(&2))
                    }
                    Responsive::FocusOnly => {
                        state.view == ViewMode::Circuit
                            || (state.view == ViewMode::State && !state.state_overview)
                    }
                    Responsive::OverviewGroups => {
                        state.state_overview
                            && cache.overview_capacity > 0
                            && experiment.document.as_ref().is_some_and(|d| {
                                d.circuit.register().wires() > cache.overview_capacity
                            })
                    }
                    Responsive::Sidebar => {
                        width >= 780. && height >= 700. && !cfg!(target_arch = "wasm32")
                    }
                    Responsive::Inspector => {
                        !state.show_settings && (inspector_visible || state.show_details)
                    }
                    Responsive::InspectorExtras => inspector_visible,
                    Responsive::CompactContext => !inspector_visible,
                    Responsive::Plot => {
                        !state.show_settings && (inspector_visible || !state.show_details)
                    }
                    Responsive::NarrowHide => width >= 1050.,
                    Responsive::CompactToolbar => !inspector_visible,
                    Responsive::Description => width >= 540. && height >= 760.,
                    Responsive::Note => height >= 760.,
                    _ => true,
                };
                let display = if visible {
                    Display::Flex
                } else {
                    Display::None
                };
                if node.display != display {
                    node.display = display;
                }
                if matches!(kind, Responsive::Inspector) {
                    let width = if inspector_visible {
                        px(240)
                    } else {
                        percent(100)
                    };
                    let min = if inspector_visible { px(240) } else { px(0) };
                    if node.width != width {
                        node.width = width;
                    }
                    if node.min_width != min {
                        node.min_width = min;
                    }
                    node.min_height = px(0);
                    node.overflow = Overflow::scroll_y();
                    // Remove compact details from intrinsic flex sizing: its content
                    // scrolls within the same finite viewport used by the diagram.
                    node.position_type = if inspector_visible {
                        PositionType::Relative
                    } else {
                        PositionType::Absolute
                    };
                    node.top = if inspector_visible { Val::Auto } else { px(0) };
                    node.left = if inspector_visible { Val::Auto } else { px(0) };
                    node.height = if inspector_visible {
                        Val::Auto
                    } else {
                        percent(100)
                    };
                    node.max_height = percent(100);
                }
            }
            Responsive::Main => {
                let padding = UiRect::all(px(if compact { 12. } else { 24. }));
                if node.padding != padding {
                    node.padding = padding;
                }
                let gap = px(if compact { 8. } else { 16. });
                if node.row_gap != gap {
                    node.row_gap = gap;
                }
            }
            Responsive::Header => {
                let display = if cfg!(target_arch = "wasm32") {
                    Display::None
                } else {
                    Display::Flex
                };
                if node.display != display {
                    node.display = display;
                }
                let size = px(if height < 760. { 64. } else { 88. });
                if node.height != size {
                    node.height = size;
                    node.min_height = size;
                }
            }
            Responsive::Footer => {
                node.display = if state.view == ViewMode::Results {
                    Display::None
                } else {
                    Display::Flex
                };
                node.flex_shrink = 0.;
                let padding = UiRect::axes(px(if compact { 12. } else { 24. }), px(16.));
                if node.padding != padding {
                    node.padding = padding;
                }
            }
        }
    }
    for mut font in &mut brands {
        let size = bevy::text::FontSize::Px(if width < 540. { 19. } else { 25. });
        if font.font_size != size {
            font.font_size = size;
        }
    }
}
pub(super) fn sync_controls(
    mut commands: Commands,
    state: Res<Explorer>,
    experiment: Res<Experiment>,
    cache: Res<PlotCache>,
    mut fields: Query<(&Field, &mut Text)>,
    mut buttons: Query<(&Action, &mut ButtonVariant, &mut BorderColor)>,
    sliders: Query<(Entity, &SliderValue, &SliderRange), With<Scrubber>>,
) {
    let doc = experiment.document.as_ref();
    for (field, mut text) in &mut fields {
        let next=match field{
        Field::Title=>doc.map(|d|d.title.clone()).unwrap_or_else(||state.demo.title().into()),Field::Description=>doc.map(|d|d.description.clone()).unwrap_or_else(||state.status.clone()),Field::Status=>if state.status.contains("Flex(Cpu)"){"Simulation · CPU (Flex)".into()}else{state.status.clone()},
        Field::Step=>{let count=doc.map(|d|timeline_len(&state,d)).unwrap_or(1);format!("{} {} / {}   ·   {}   ·   ×{}",if state.view==ViewMode::Fabric{"Schedule tick"}else{"Step"},cursor(&state),count.saturating_sub(1),if state.playing{"Playing"}else{"Paused"},state.speed)},
        Field::Inspection=>super::inspection::describe(&state,doc),
        Field::Context=>doc.map(|d|format!("CONTEXT\n\n{}",d.description)).unwrap_or_default(),
        Field::ViewNote=>match state.view{ViewMode::State=>if state.bloch_mode == BlochMode::Planar {"Disk area = P(1); hand angle = phase, length = coherence. Purity distinguishes mixtures. Focus reveals joint probabilities."}else{"Each panel is a reduced subsystem. Qubits show Bloch vectors; qudits show level populations. Focus reveals joint probabilities."},ViewMode::Circuit=>if doc.is_some_and(|d|d.trace.kind==crate::model::TraceKind::Optimization){"Parameterized ansatz at the selected optimizer iterate. Hover for gate help; click to inspect its actual parameter binding."}else{"Read left to right; the highlighted prefix has executed. Hover for gate help; click to inspect and seek. Drag to pan."},ViewMode::Fabric=>"Physical schedule time is independent of the circuit prefix. Moving dots follow compiled routes; links show rail transmission.",ViewMode::Learning=>"Actual parameter iterates, objective values and evaluation costs. Scrub to compare the current iterate with its history.",ViewMode::Results=>"Task outcomes retain their baselines, survival conditions and resource costs. Details contains every metric and model assumption."}.into(),
        Field::Stats=>cache.description.clone(),
        Field::ViewSummary=>match state.view {
            ViewMode::State=>format!("{} · {}",if state.state_overview { "Register".into() } else { format!("q{}",state.wire) }, if doc.is_some_and(|d|d.circuit.register().dimensions().contains(&2)) { match state.bloch_mode {BlochMode::Spatial=>"3D spheres".into(),BlochMode::Planar=>"2D probability / phase".into(),BlochMode::Projection=>format!("Projection {}",state.bloch_plane.axes()[..2].join("").to_uppercase())} } else {"Populations".into()}),
            ViewMode::Circuit=>"Hover symbols for help".into(),
            ViewMode::Results=>"Measured outcomes · references · scope in Details".into(),
            _=>"Drag to pan · scroll to zoom".into(),
        },
        Field::PlaneName=>state.bloch_plane.axes()[..2].join("").to_uppercase(),
        Field::ConfigSize=>format!("{} {}",state.config.size,DemoConfig::size_label(state.demo)),
        Field::ConfigDepth=>state.config.depth.to_string(),
        Field::ConfigVariant=>state.config.variant_label(state.demo),
        Field::ConfigNote=>format!("{} · {}",state.config_note,if state.config_valid{"edit settings, then Run"}else{"Run unavailable"}),
        Field::OverviewGroup=>{let cap=cache.overview_capacity.max(1);format!("{} / {}",state.wire/cap+1,doc.map(|d|d.circuit.register().wires().div_ceil(cap)).unwrap_or(1))},
    };
        if text.0 != next {
            text.0 = next;
        }
    }
    for (action, mut variant, mut border) in &mut buttons {
        let selected = match action {
            Action::Demo(k) => *k == state.demo,
            Action::View(v) => *v == state.view,
            Action::Play => state.playing,
            Action::Inspect => state.show_details,
            Action::ToggleFocus => !state.state_overview,
            Action::Grid => state.grid,
            Action::BlochMode(mode) => *mode == state.bloch_mode,
            Action::Trails => state.trails,
            Action::Settings => state.show_settings,
            _ => false,
        };
        let next = if !selected && matches!(action, Action::Demo(_) | Action::View(_)) {
            ButtonVariant::Plain
        } else {
            ButtonVariant::selected(selected)
        };
        if *variant != next {
            *variant = next;
        }
        let color = if matches!(action, Action::View(_)) {
            if selected { ACCENT } else { Color::NONE }
        } else if selected {
            ACCENT_LINE
        } else {
            BORDER
        };
        let next_border = BorderColor::all(color);
        if *border != next_border {
            *border = next_border;
        }
    }
    for (entity, value, range) in &sliders {
        let max = doc
            .map(|d| timeline_len(&state, d).saturating_sub(1))
            .unwrap_or(1)
            .max(1) as f32;
        let next = cursor(&state) as f32;
        if value.0 != next {
            commands.entity(entity).insert(SliderValue(next));
        }
        if range.end() != max {
            commands.entity(entity).insert(SliderRange::new(0., max));
        }
    }
}
