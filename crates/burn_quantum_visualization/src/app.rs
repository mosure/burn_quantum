//! Native/WASM application, asynchronous document loading and interaction state.
use crate::{
    demo::{self, DemoConfig, DemoKind},
    model::{
        VisualizationDocument,
        bloch::{BlochMode, BlochPlane},
    },
    simulation::TraceLimits,
    view,
};
use bevy::{
    feathers::FeathersPlugins,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, futures_lite::future},
    window::WindowResolution,
};

/// The scientific lens shown in the main canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    /// Reduced states and measured probabilities.
    State,
    /// Ordered circuit operations.
    Circuit,
    /// Physical photonic routing.
    Fabric,
    /// Optimization objective history.
    Learning,
}
impl ViewMode {
    pub(crate) const ALL: [Self; 4] = [Self::State, Self::Circuit, Self::Fabric, Self::Learning];
    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::State => "State",
            Self::Circuit => "Circuit",
            Self::Fabric => "Fabric",
            Self::Learning => "Learning",
        }
    }
    fn parse(value: &str) -> Self {
        match value {
            "circuit" => Self::Circuit,
            "fabric" => Self::Fabric,
            "learning" | "optimization" => Self::Learning,
            _ => Self::State,
        }
    }
}
/// Bounded view state. Scientific data remain immutable in `Experiment`.
#[derive(Resource)]
pub(crate) struct Explorer {
    pub demo: DemoKind,
    pub config: DemoConfig,
    pub config_note: String,
    pub config_valid: bool,
    pub view: ViewMode,
    pub frame: usize,
    pub wire: usize,
    pub state_overview: bool,
    pub bloch_mode: BlochMode,
    pub bloch_plane: BlochPlane,
    pub trails: bool,
    pub gate: Option<usize>,
    pub node: Option<usize>,
    pub tick: f32,
    pub playing: bool,
    pub speed: f32,
    pub elapsed: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan: Vec2,
    pub grid: bool,
    pub show_details: bool,
    pub show_settings: bool,
    pub revision: u64,
    pub requested: bool,
    pub status: String,
}
impl Default for Explorer {
    fn default() -> Self {
        Self {
            demo: DemoKind::Bell,
            config: DemoConfig::for_demo(DemoKind::Bell),
            config_note: String::new(),
            config_valid: true,
            view: ViewMode::State,
            frame: 0,
            wire: 0,
            state_overview: true,
            bloch_mode: BlochMode::Spatial,
            bloch_plane: BlochPlane::Xz,
            trails: false,
            gate: None,
            node: None,
            tick: 0.,
            playing: false,
            speed: 1.,
            elapsed: 0.,
            yaw: -0.65,
            pitch: 0.35,
            zoom: 1.,
            pan: Vec2::ZERO,
            grid: true,
            show_details: false,
            show_settings: false,
            revision: 1,
            requested: true,
            status: "Preparing the experiment…".into(),
        }
    }
}
impl Explorer {
    pub fn preview_config(&mut self) {
        match demo::preview(self.demo, self.config, TraceLimits::default()) {
            Ok(Some(p)) => {
                self.config_valid = true;
                self.config_note = format!(
                    "{} qubits · {} states · {} steps · {:.1} MiB numeric snapshots",
                    p.wires,
                    p.dimension,
                    p.steps,
                    p.snapshot_bytes as f64 / 1048576.
                );
            }
            Ok(None) => {
                self.config_valid = true;
                self.config_note = "Fixed-size foundational experiment".into();
            }
            Err(e) => {
                self.config_valid = false;
                self.config_note = format!("Budget check: {e}. Reduce size or depth.");
            }
        }
        self.touch();
    }
    pub fn touch(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
    pub fn reset_camera(&mut self) {
        self.yaw = -0.65;
        self.pitch = 0.35;
        self.zoom = 1.;
        self.pan = Vec2::ZERO;
        self.touch();
    }
    pub fn seek(&mut self, frame: usize, len: usize) {
        let frame = frame.min(len.saturating_sub(1));
        if self.frame != frame {
            self.frame = frame;
            self.gate = frame.checked_sub(1);
            self.touch();
        }
    }
}
#[derive(Resource, Default)]
pub(crate) struct Experiment {
    pub document: Option<VisualizationDocument>,
    task: Option<Task<Result<VisualizationDocument, String>>>,
    generation: u64,
}
#[derive(Resource, Default)]
struct Launch {
    #[cfg(not(target_arch = "wasm32"))]
    screenshot: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    input: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    export: Option<String>,
    step: Option<usize>,
    ready_frames: u32,
    #[cfg(not(target_arch = "wasm32"))]
    capture_started: bool,
    done: bool,
}
/// Launch the interactive Bevy/Feathers explorer. Native QA flags include
/// `--demo`, `--view`, `--step`, `--screenshot`, `--input` and `--export`.
pub fn run() {
    let mut explorer = Explorer::default();
    let mut launch = Launch::default();
    #[cfg(not(target_arch = "wasm32"))]
    let mut size = (1440, 960);
    #[cfg(target_arch = "wasm32")]
    let size = (1440, 960);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--demo" => {
                    if let Some(v) = args.next() {
                        explorer.demo = parse_demo(&v);
                        explorer.config = DemoConfig::for_demo(explorer.demo);
                    }
                }
                "--view" => {
                    if let Some(v) = args.next() {
                        explorer.view = ViewMode::parse(&v);
                    }
                }
                "--bloch" => {
                    explorer.bloch_mode = if args.next().as_deref() == Some("2d") {
                        BlochMode::Planar
                    } else {
                        BlochMode::Spatial
                    };
                }
                "--plane" => {
                    explorer.bloch_plane = match args.next().as_deref() {
                        Some("xy") => BlochPlane::Xy,
                        Some("yz") => BlochPlane::Yz,
                        _ => BlochPlane::Xz,
                    };
                }
                "--trails" => explorer.trails = true,
                "--size" => {
                    if let Some(n) = args.next().and_then(|v| v.parse().ok()) {
                        explorer.config.size = n;
                    }
                }
                "--layers" => {
                    if let Some(n) = args.next().and_then(|v| v.parse().ok()) {
                        explorer.config.depth = n;
                    }
                }
                "--variant" => {
                    if let Some(n) = args.next().and_then(|v| v.parse().ok()) {
                        explorer.config.variant = n;
                    }
                }
                "--step" => launch.step = args.next().and_then(|s| s.parse().ok()),
                "--width" => {
                    size.0 = args
                        .next()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(1440)
                        .max(320)
                }
                "--height" => {
                    size.1 = args
                        .next()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(960)
                        .max(420)
                }
                "--screenshot" => launch.screenshot = args.next(),
                "--input" => launch.input = args.next(),
                "--export" => launch.export = args.next(),
                _ => {}
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Ok(search) = window.location().search() {
                if let Ok(params) = web_sys::UrlSearchParams::new_with_str(&search) {
                    if let Some(v) = params.get("demo") {
                        explorer.demo = parse_demo(&v);
                        explorer.config = DemoConfig::for_demo(explorer.demo);
                    }
                    if let Some(v) = params.get("view") {
                        explorer.view = ViewMode::parse(&v);
                    }
                    if let Some(n) = params.get("size").and_then(|v| v.parse().ok()) {
                        explorer.config.size = n;
                    }
                    if let Some(n) = params.get("layers").and_then(|v| v.parse().ok()) {
                        explorer.config.depth = n;
                    }
                    if let Some(n) = params.get("variant").and_then(|v| v.parse().ok()) {
                        explorer.config.variant = n;
                    }
                    explorer.bloch_mode = if params.get("bloch").as_deref() == Some("2d") {
                        BlochMode::Planar
                    } else {
                        BlochMode::Spatial
                    };
                    explorer.bloch_plane = match params.get("plane").as_deref() {
                        Some("xy") => BlochPlane::Xy,
                        Some("yz") => BlochPlane::Yz,
                        _ => BlochPlane::Xz,
                    };
                    explorer.trails = params.get("trails").as_deref() == Some("1");
                    launch.step = params.get("step").and_then(|v| v.parse().ok());
                }
            }
        }
    }
    explorer.preview_config();
    App::new()
        .insert_resource(ClearColor(view::theme::BACKGROUND))
        .insert_resource(explorer)
        .insert_resource(launch)
        .init_resource::<Experiment>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Quantum explorer · burn_quantum".into(),
                resolution: WindowResolution::new(size.0, size.1),
                canvas: Some("#quantum-canvas".into()),
                fit_canvas_to_parent: true,
                prevent_default_event_handling: true,
                ..default()
            }),
            ..default()
        }))
        .insert_non_send(UiMainThread(std::marker::PhantomData))
        .add_plugins(FeathersPlugins)
        .insert_resource(view::theme::controller_theme())
        .insert_resource(bevy::render::error_handler::RenderErrorHandler(
            report_render_error,
        ))
        .add_plugins(view::ExplorerViewPlugin)
        .add_systems(
            Update,
            (load_documents, poll_document, playback, automation).chain(),
        )
        .add_systems(Last, report_exit)
        .run();
}
// Renderer errors can arrive after the Last schedule, so notify the browser
// directly before the standard handler terminates the application.
fn report_render_error(
    error: &bevy::render::error_handler::RenderError,
    main_world: &mut World,
    render_world: &mut World,
) -> bevy::render::error_handler::RenderErrorPolicy {
    #[cfg(target_arch = "wasm32")]
    if let (Some(window), Ok(event)) = (
        web_sys::window(),
        web_sys::CustomEvent::new("burn-quantum-error"),
    ) {
        let _ = window.dispatch_event(&event);
    }
    (bevy::render::error_handler::RenderErrorHandler::default().0)(error, main_world, render_world)
}
fn parse_demo(value: &str) -> DemoKind {
    DemoKind::from_id(value).unwrap_or(DemoKind::Bell)
}
struct UiMainThread(std::marker::PhantomData<std::rc::Rc<()>>);
fn load_documents(
    _main: NonSend<UiMainThread>,
    mut state: ResMut<Explorer>,
    mut experiment: ResMut<Experiment>,
    mut launch: ResMut<Launch>,
) {
    if let Some(document) = crate::io::take_pending_document() {
        experiment.task = None;
        accept_document(document, &mut state, &mut experiment, &mut launch);
        state.requested = false;
    }
    if !state.requested {
        return;
    }
    state.requested = false;
    state.playing = false;
    state.status = "Simulating with Burn…".into();
    experiment.generation += 1;
    experiment.document = None;
    state.touch();
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = launch.input.take() {
        match crate::io::read_input(&path) {
            Ok(document) => accept_document(document, &mut state, &mut experiment, &mut launch),
            Err(error) => {
                state.status = format!("Import failed: {error}");
                state.touch();
            }
        }
        return;
    }
    let kind = state.demo;
    let config = state.config;
    experiment.task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let device = burn::tensor::Device::flex();
        demo::build_configured(kind, config, &device, TraceLimits::default())
            .await
            .map_err(|e| e.to_string())
    }));
}
fn poll_document(
    _main: NonSend<UiMainThread>,
    mut state: ResMut<Explorer>,
    mut experiment: ResMut<Experiment>,
    mut launch: ResMut<Launch>,
) {
    let Some(task) = experiment.task.as_mut() else {
        return;
    };
    if let Some(result) = future::block_on(future::poll_once(task)) {
        experiment.task = None;
        match result {
            Ok(document) => accept_document(document, &mut state, &mut experiment, &mut launch),
            Err(error) => {
                state.status = format!("Could not simulate: {error}");
                state.touch();
            }
        }
    }
}
fn accept_document(
    document: VisualizationDocument,
    state: &mut Explorer,
    experiment: &mut Experiment,
    launch: &mut Launch,
) {
    state.tick = 0.;
    state.node = None;
    let initial = launch.step.take().unwrap_or(0);
    if (state.view == ViewMode::Fabric && document.fabric.is_none())
        || (state.view == ViewMode::Learning && document.optimization.is_empty())
    {
        state.view = ViewMode::State;
    }
    if state.view == ViewMode::Fabric {
        state.tick = initial.min(
            document
                .fabric
                .as_ref()
                .map(|f| f.finish_tick as usize)
                .unwrap_or(0),
        ) as f32;
    }
    state.frame = if state.view == ViewMode::Fabric {
        0
    } else {
        initial.min(document.trace.frames.len().saturating_sub(1))
    };
    state.wire = 0;
    state.state_overview = true;
    state.gate = state.frame.checked_sub(1);
    state.status = document.trace.backend.clone();
    state.elapsed = 0.;
    state.reset_camera();
    crate::io::set_current_document(&document);
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = launch.export.take()
        && let Err(error) = crate::io::write_export(&path, &document)
    {
        state.status = format!("Export failed: {error}");
    }
    experiment.document = Some(document);
    launch.ready_frames = 0;
}
fn playback(time: Res<Time>, mut state: ResMut<Explorer>, experiment: Res<Experiment>) {
    if !state.playing {
        return;
    }
    let Some(doc) = &experiment.document else {
        return;
    };
    if state.view == ViewMode::Fabric
        && let Some(fabric) = &doc.fabric
    {
        state.tick =
            (state.tick + time.delta_secs() * state.speed * 2.).min(fabric.finish_tick as f32);
        if state.tick >= fabric.finish_tick as f32 {
            state.playing = false;
        }
        state.touch();
        return;
    }
    state.elapsed += time.delta_secs() * state.speed;
    if state.elapsed >= 0.8 {
        state.elapsed = 0.;
        let len = doc.trace.frames.len();
        if state.frame + 1 < len {
            let next = state.frame + 1;
            state.seek(next, len);
        } else {
            state.playing = false;
            state.touch();
        }
    }
}
fn automation(mut commands: Commands, experiment: Res<Experiment>, mut launch: ResMut<Launch>) {
    if experiment.document.is_none() {
        return;
    }
    launch.ready_frames += 1;
    #[cfg(target_arch = "wasm32")]
    if launch.ready_frames == 3 {
        if let (Some(window), Ok(event)) = (
            web_sys::window(),
            web_sys::CustomEvent::new("burn-quantum-ready"),
        ) {
            let _ = window.dispatch_event(&event);
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    if launch.ready_frames >= 60
        && !launch.capture_started
        && let Some(path) = launch.screenshot.clone()
    {
        launch.capture_started = true;
        commands
            .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
            .observe(bevy::render::view::screenshot::save_to_disk(path))
            .observe(
                |_: On<bevy::render::view::screenshot::ScreenshotCaptured>,
                 mut launch: ResMut<Launch>| {
                    launch.done = true;
                },
            );
    }
    if launch.done {
        commands.write_message(bevy::app::AppExit::Success);
    }
}

/// Validate and open an existing scientific document in the explorer.
pub fn run_document(document: VisualizationDocument) -> crate::io::IoResult<()> {
    crate::io::queue_document(document)?;
    run();
    Ok(())
}

/// Read-only diagnostics for browser automation; no scientific work is performed.
pub use crate::view::diagnostics::diagnostics_json;

fn report_exit(
    _main: bevy::ecs::system::NonSendMarker,
    mut exits: MessageReader<bevy::app::AppExit>,
) {
    for exit in exits.read() {
        #[cfg(target_arch = "wasm32")]
        if let (Some(window), Ok(event)) = (
            web_sys::window(),
            web_sys::CustomEvent::new(if exit.is_error() {
                "burn-quantum-error"
            } else {
                "burn-quantum-stopped"
            }),
        ) {
            let _ = window.dispatch_event(&event);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = exit;
    }
}
