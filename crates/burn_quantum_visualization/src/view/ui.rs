use super::{plot::PlotCache, theme::*};
use crate::{
    app::{Experiment, Explorer, ViewMode},
    demo::DemoKind,
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
    commands.spawn_scene(bsn!{
        @FeathersButton { @variant:{ButtonVariant::selected(primary)}, @caption:bsn!{@caption(label)} }
        Node {height:px(36),min_height:px(36),width:{width},padding:UiRect::horizontal(px(12)),}
        action
        ChildOf(parent)
        on(activate)
    }).id()
}
pub(super) fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    commands.spawn((Camera2d, IsDefaultUiCamera));
    let font = assets.load(fonts::REGULAR);
    let bold = assets.load(fonts::BOLD);
    let mono = assets.load(fonts::MONO);
    let root = commands
        .spawn(Node {
            width: percent(100),
            height: percent(100),
            flex_direction: FlexDirection::Column,
            ..default()
        })
        .id();
    let header = node(
        &mut commands,
        root,
        Node {
            height: px(88),
            min_height: px(88),
            padding: UiRect::axes(px(24), px(16)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::SpaceBetween,
            border: UiRect::bottom(px(1)),
            ..default()
        },
        Some(PANEL),
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
        CYAN,
        mono.clone(),
    );
    let brand_title = text(
        &mut commands,
        brand,
        "Quantum explorer",
        25.,
        TEXT,
        bold.clone(),
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
    button(
        &mut commands,
        headright,
        "Next experiment".into(),
        Action::NextDemo,
        false,
        Val::Auto,
    );
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
            width: px(218),
            min_width: px(218),
            padding: UiRect::all(px(18)),
            flex_direction: FlexDirection::Column,
            row_gap: px(9),
            border: UiRect::right(px(1)),
            ..default()
        },
        Some(PANEL),
    );
    commands
        .entity(sidebar)
        .insert((Responsive::Sidebar, BorderColor::all(BORDER)));
    text(
        &mut commands,
        sidebar,
        "EXPERIMENTS",
        11.,
        MUTED,
        mono.clone(),
    );
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
    let spacer = node(
        &mut commands,
        sidebar,
        Node {
            flex_grow: 1.,
            ..default()
        },
        None,
    );
    let _ = spacer;
    text(
        &mut commands,
        sidebar,
        "A small lab for\nlarge quantum ideas.",
        18.,
        TEXT,
        font.clone(),
    );
    text(
        &mut commands,
        sidebar,
        "Every state comes from Burn.\nProbability, phase and reduced\nstates are shown separately.",
        12.,
        MUTED,
        font.clone(),
    );
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
        bold.clone(),
    );
    commands.entity(title).insert(Field::Title);
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
        button(
            &mut commands,
            tabs,
            mode.title().into(),
            Action::View(mode),
            mode == ViewMode::State,
            Val::Auto,
        );
    }
    let compact = node(
        &mut commands,
        main,
        Node {
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(6),
            row_gap: px(4),
            ..default()
        },
        None,
    );
    commands.entity(compact).insert(Responsive::CompactToolbar);
    button(
        &mut commands,
        compact,
        "Next qubit".into(),
        Action::Wire(1),
        false,
        Val::Auto,
    );
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
    button(
        &mut commands,
        compact,
        "Details".into(),
        Action::Inspect,
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
            border_radius: BorderRadius::all(px(12)),
            flex_direction: FlexDirection::Column,
            row_gap: px(15),
            ..default()
        },
        Some(PANEL),
    );
    commands
        .entity(inspector)
        .insert((Responsive::Inspector, bevy::ui_widgets::ScrollArea));
    let inspector_heading = text(
        &mut commands,
        inspector,
        "INSPECTOR",
        11.,
        CYAN,
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
    let wires = node(
        &mut commands,
        extras,
        Node {
            column_gap: px(7),
            ..default()
        },
        None,
    );
    button(
        &mut commands,
        wires,
        "← Qubit".into(),
        Action::Wire(-1),
        false,
        Val::Auto,
    );
    button(
        &mut commands,
        wires,
        "Qubit →".into(),
        Action::Wire(1),
        false,
        Val::Auto,
    );
    text(
        &mut commands,
        extras,
        "VIEW CONTROLS",
        11.,
        MUTED,
        mono.clone(),
    );
    button(
        &mut commands,
        extras,
        "Reset view  ·  R".into(),
        Action::Reset,
        false,
        percent(100),
    );
    button(
        &mut commands,
        extras,
        "Toggle guides  ·  G".into(),
        Action::Grid,
        false,
        percent(100),
    );
    text(
        &mut commands,
        extras,
        "Drag to orbit the sphere.\nDrag to pan other views.\nScroll to zoom.\nSpace to play · ← → to step.",
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
        "|←".into(),
        Action::Step(i32::MIN),
        false,
        Val::Auto,
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
    button(
        &mut commands,
        controls,
        "Speed".into(),
        Action::Speed,
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
) {
    if let Ok(action) = actions.get(event.entity) {
        apply(*action, &mut state, &experiment);
    }
}
pub(super) fn apply(action: Action, state: &mut Explorer, experiment: &Experiment) {
    match action {
        Action::Inspect => {
            state.show_details = !state.show_details;
            state.touch();
        }
        Action::Demo(kind) => {
            state.show_details = false;
            state.demo = kind;
            state.requested = true;
            state.frame = 0;
            state.tick = 0.;
            state.gate = None;
            state.node = None;
            state.touch();
        }
        Action::View(mode) => {
            state.show_details = false;
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
            | Responsive::Note => {
                let visible = match kind {
                    Responsive::Sidebar => {
                        width >= 780. && height >= 700. && !cfg!(target_arch = "wasm32")
                    }
                    Responsive::Inspector => inspector_visible || state.show_details,
                    Responsive::InspectorExtras => inspector_visible,
                    Responsive::CompactContext => !inspector_visible,
                    Responsive::Plot => inspector_visible || !state.show_details,
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
    mut buttons: Query<(&Action, &mut ButtonVariant)>,
    sliders: Query<(Entity, &SliderValue, &SliderRange), With<Scrubber>>,
) {
    let doc = experiment.document.as_ref();
    for (field, mut text) in &mut fields {
        let next=match field{
        Field::Title=>doc.map(|d|d.title.clone()).unwrap_or_else(||state.demo.title().into()),Field::Description=>doc.map(|d|d.description.clone()).unwrap_or_else(||state.status.clone()),Field::Status=>if state.status.contains("Flex(Cpu)"){"Simulation · CPU (Flex)".into()}else{state.status.clone()},
        Field::Step=>{let count=doc.map(|d|timeline_len(&state,d)).unwrap_or(1);format!("{} {} / {}   ·   {}   ·   ×{}",if state.view==ViewMode::Fabric{"Schedule tick"}else{"Step"},cursor(&state),count.saturating_sub(1),if state.playing{"Playing"}else{"Paused"},state.speed)},
        Field::Inspection=>super::inspection::describe(&state,doc),
        Field::Context=>doc.map(|d|format!("CONTEXT\n\n{}",d.description)).unwrap_or_default(),
        Field::ViewNote=>match state.view{ViewMode::State=>"Length encodes local purity. A mixed reduced state lies inside the Bloch sphere; probability is separate from phase.",ViewMode::Circuit=>if doc.is_some_and(|d|d.trace.kind==crate::model::TraceKind::Optimization){"Parameterized ansatz at the selected optimizer iterate. Click a gate to inspect its actual parameter binding."}else{"Read left to right. The highlighted prefix has already executed. Click a gate to inspect and seek; drag to pan."},ViewMode::Fabric=>"Physical schedule time is independent of the circuit prefix. Moving dots follow compiled routes; links show rail transmission.",ViewMode::Learning=>"Actual parameter iterates, objective values and evaluation costs. Scrub to compare the current iterate with its history."}.into(),
        Field::Stats=>cache.description.clone(),
    };
        if text.0 != next {
            text.0 = next;
        }
    }
    for (action, mut variant) in &mut buttons {
        let selected = match action {
            Action::Demo(k) => *k == state.demo,
            Action::View(v) => *v == state.view,
            Action::Play => state.playing,
            Action::Inspect => state.show_details,
            _ => false,
        };
        let next = ButtonVariant::selected(selected);
        if *variant != next {
            *variant = next;
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
