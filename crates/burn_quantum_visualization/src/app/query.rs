//! Versioned, bounded URL view state. Scientific document payloads travel as JSON,
//! never as query strings. Parsing does not allocate simulation tensors.
use super::*;
use std::collections::BTreeMap;

/// All durable explorer settings; renderer timings and transient gestures are excluded.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct QueryState {
    pub demo: DemoKind,
    pub config: DemoConfig,
    pub draft_config: DemoConfig,
    pub view: ViewMode,
    pub frame: usize,
    pub wire: usize,
    pub overview: bool,
    pub mode: BlochMode,
    pub plane: BlochPlane,
    pub trails: bool,
    pub gate: Option<usize>,
    pub node: Option<usize>,
    pub tick: f32,
    pub playing: bool,
    pub speed: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan: Vec2,
    pub guides: bool,
    pub details: bool,
    pub settings: bool,
    pub imported: bool,
}
impl QueryState {
    pub fn capture(state: &Explorer, experiment: &Experiment) -> Self {
        Self {
            demo: state.demo,
            config: experiment.source_config.unwrap_or(state.config),
            draft_config: state.config,
            view: state.view,
            frame: state.frame,
            wire: state.wire,
            overview: state.state_overview,
            mode: state.bloch_mode,
            plane: state.bloch_plane,
            trails: state.trails,
            gate: state.gate,
            node: state.node,
            tick: state.tick,
            playing: state.playing,
            speed: state.speed,
            yaw: state.yaw,
            pitch: state.pitch,
            zoom: state.zoom,
            pan: state.pan,
            guides: state.grid,
            details: state.show_details,
            settings: state.show_settings,
            imported: experiment.imported,
        }
    }
    /// First duplicate wins, matching URLSearchParams::get. Unknown keys are ignored.
    pub fn parse(pairs: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut values = BTreeMap::new();
        for (key, value) in pairs {
            values.entry(key).or_insert(value);
        }
        let get = |key: &str| values.get(key).map(String::as_str);
        let integer = |key: &str, fallback| {
            get(key)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(fallback)
        };
        let scalar = |key: &str, fallback: f32, min: f32, max: f32| {
            get(key)
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| v.is_finite())
                .unwrap_or(fallback)
                .clamp(min, max)
        };
        let flag = |key: &str, fallback| match get(key) {
            Some("1" | "true") => true,
            Some("0" | "false") => false,
            _ => fallback,
        };
        let demo = get("demo").map(parse_demo).unwrap_or(DemoKind::Bell);
        let defaults = DemoConfig::for_demo(demo);
        let config = DemoConfig {
            size: integer("size", defaults.size),
            depth: integer("layers", defaults.depth),
            variant: integer("variant", defaults.variant),
        };
        let view = get("view").map(ViewMode::parse).unwrap_or(ViewMode::State);
        let step = integer("step", 0);
        let frame = integer("frame", if view == ViewMode::Fabric { 0 } else { step });
        let index = |key: &str, fallback| match get(key) {
            Some("none") => None,
            Some(v) => v.parse::<usize>().ok(),
            None => fallback,
        };
        Self {
            demo,
            config,
            draft_config: DemoConfig {
                size: integer("draft_size", config.size),
                depth: integer("draft_layers", config.depth),
                variant: integer("draft_variant", config.variant),
            },
            view,
            frame,
            wire: integer("wire", 0),
            overview: flag("overview", true),
            mode: match get("bloch") {
                Some("2d") => BlochMode::Planar,
                Some("projection") => BlochMode::Projection,
                _ => BlochMode::Spatial,
            },
            plane: match get("plane") {
                Some("xy") => BlochPlane::Xy,
                Some("yz") => BlochPlane::Yz,
                _ => BlochPlane::Xz,
            },
            trails: flag("trails", false),
            gate: index("gate", frame.checked_sub(1)),
            node: index("node", None),
            tick: scalar(
                "tick",
                if view == ViewMode::Fabric {
                    step as f32
                } else {
                    0.
                },
                0.,
                1e9,
            ),
            playing: flag("playing", false),
            speed: scalar("speed", 1., 0.25, 4.),
            yaw: scalar("yaw", -0.65, -1e6, 1e6),
            pitch: scalar("pitch", 0.35, -1.25, 1.25),
            zoom: scalar("zoom", 1., 0.35, 3.),
            pan: Vec2::new(
                scalar("pan_x", 0., -1e6, 1e6),
                scalar("pan_y", 0., -1e6, 1e6),
            ),
            guides: flag("guides", true),
            details: flag("details", false),
            settings: flag("settings", false),
            imported: get("source") == Some("imported"),
        }
    }
    /// Canonical explicit values prevent default changes from altering shared views.
    /// Values are exclusively URL-safe identifiers, finite numbers, and booleans.
    pub fn encode(&self) -> String {
        let mut pairs = Vec::new();
        macro_rules! put {
            ($key:literal, $value:expr) => {
                pairs.push(format!("{}={}", $key, $value));
            };
        }
        put!("v", 1);
        put!("demo", self.demo.id());
        put!("size", self.config.size);
        put!("layers", self.config.depth);
        put!("variant", self.config.variant);
        if self.draft_config != self.config {
            put!("draft_size", self.draft_config.size);
            put!("draft_layers", self.draft_config.depth);
            put!("draft_variant", self.draft_config.variant);
        }
        put!(
            "view",
            match self.view {
                ViewMode::State => "state",
                ViewMode::Circuit => "circuit",
                ViewMode::Fabric => "fabric",
                ViewMode::Learning => "learning",
            }
        );
        put!(
            "step",
            if self.view == ViewMode::Fabric {
                self.tick as usize
            } else {
                self.frame
            }
        );
        put!("frame", self.frame);
        put!("tick", self.tick);
        put!("wire", self.wire);
        put!(
            "gate",
            self.gate
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".into())
        );
        put!(
            "node",
            self.node
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".into())
        );
        put!("overview", u8::from(self.overview));
        put!(
            "bloch",
            match self.mode {
                BlochMode::Spatial => "3d",
                BlochMode::Planar => "2d",
                BlochMode::Projection => "projection",
            }
        );
        put!(
            "plane",
            match self.plane {
                BlochPlane::Xy => "xy",
                BlochPlane::Xz => "xz",
                BlochPlane::Yz => "yz",
            }
        );
        put!("trails", u8::from(self.trails));
        put!("playing", u8::from(self.playing));
        put!("speed", self.speed);
        put!("yaw", self.yaw);
        put!("pitch", self.pitch);
        put!("zoom", self.zoom);
        put!("pan_x", self.pan.x);
        put!("pan_y", self.pan.y);
        put!("guides", u8::from(self.guides));
        put!("details", u8::from(self.details));
        put!("settings", u8::from(self.settings));
        put!("source", if self.imported { "imported" } else { "catalog" });
        pairs.join("&")
    }
    pub fn prepare(&self, state: &mut Explorer) {
        state.demo = self.demo;
        state.config = self.config;
        state.view = self.view;
        state.requested = !self.imported;
        if self.imported {
            state.status = "Import the JSON document associated with this link.".into();
        }
    }
    /// Apply after loading, so generic document initialization cannot erase the URL.
    /// Clamp against the actual document rather than trusting query indices.
    pub fn restore(&self, state: &mut Explorer, doc: &VisualizationDocument) {
        state.config = self.draft_config;
        state.preview_config();
        state.frame = self.frame.min(doc.trace.frames.len().saturating_sub(1));
        state.wire = self
            .wire
            .min(doc.circuit.register().wires().saturating_sub(1));
        let steps = doc
            .protocol
            .as_ref()
            .map_or(doc.circuit.operations().len(), |p| {
                p.program.instructions.len()
            });
        state.gate = self.gate.filter(|&g| g < steps);
        state.node = self.node.filter(|&n| {
            doc.fabric
                .as_ref()
                .is_some_and(|f| f.nodes.iter().any(|node| node.id == n))
        });
        state.tick = self
            .tick
            .min(doc.fabric.as_ref().map_or(0., |f| f.finish_tick as f32));
        state.state_overview = self.overview;
        state.bloch_mode = self.mode;
        state.bloch_plane = self.plane;
        state.trails = self.trails;
        state.playing = self.playing;
        state.speed = self.speed;
        state.yaw = self.yaw;
        state.pitch = self.pitch;
        state.zoom = self.zoom;
        state.pan = self.pan;
        state.grid = self.guides;
        state.show_details = self.details;
        state.show_settings = self.settings;
        state.touch();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn parse(query: &str) -> QueryState {
        QueryState::parse(
            query
                .split('&')
                .filter_map(|pair| pair.split_once('=').map(|(k, v)| (k.into(), v.into()))),
        )
    }
    #[test]
    fn every_persistent_field_round_trips_and_drafts_do_not_run() {
        let query = "demo=qaoa&size=8&layers=3&variant=0&draft_size=10&draft_layers=5&draft_variant=0&view=fabric&frame=7&tick=3.75&wire=6&gate=4&node=2&overview=0&bloch=projection&plane=yz&trails=1&playing=1&speed=2&yaw=-2.7&pitch=0.9&zoom=1.8&pan_x=-183.25&pan_y=78.5&guides=0&details=1&settings=1&source=imported";
        let state = parse(query);
        assert_eq!(state, parse(&state.encode()));
        let mut explorer = Explorer::default();
        state.prepare(&mut explorer);
        assert_eq!(explorer.config.size, 8);
        assert_eq!(state.draft_config.size, 10);
        assert!(!explorer.requested);
    }
    #[test]
    fn legacy_step_and_malformed_parameters_are_bounded() {
        let state = parse(
            "demo=photonic&view=fabric&step=4&zoom=NaN&pitch=999&pan_y=inf&speed=-2&wire=-1&gate=none",
        );
        assert_eq!((state.frame, state.tick), (0, 4.));
        assert_eq!(state.zoom, 1.);
        assert_eq!(state.pitch, 1.25);
        assert_eq!(state.pan.y, 0.);
        assert_eq!(state.speed, 0.25);
        assert_eq!(state.wire, 0);
        assert_eq!(state.gate, None);
        assert_eq!(parse("step=7").frame, 7);
        assert_eq!(parse("demo=invalid&demo=qft").demo, DemoKind::Bell);
        for mode in ["2d", "3d", "projection"] {
            let q = parse(&format!("bloch={mode}"));
            assert_eq!(q, parse(&q.encode()));
        }
    }
    #[test]
    fn protocol_selection_uses_instruction_count_including_measurements() {
        let doc = pollster::block_on(demo::build(
            DemoKind::Teleportation,
            &burn::tensor::Device::flex(),
            TraceLimits::default(),
        ))
        .unwrap();
        let last = doc.protocol.as_ref().unwrap().program.instructions.len() - 1;
        let q = parse(&format!(
            "demo=teleportation&gate={last}&frame={}",
            last + 1
        ));
        let mut state = Explorer::default();
        q.restore(&mut state, &doc);
        assert_eq!(state.gate, Some(last));
    }
    #[test]
    fn restoration_is_after_document_load_and_checks_selection_bounds() {
        let doc = pollster::block_on(demo::build(
            DemoKind::Bell,
            &burn::tensor::Device::flex(),
            TraceLimits::default(),
        ))
        .unwrap();
        let q = parse(
            "demo=bell&frame=9999&wire=9999&gate=9999&node=9999&tick=999&overview=0&bloch=2d&plane=xy&pan_x=100&zoom=2&playing=1&details=1",
        );
        let mut state = Explorer::default();
        q.restore(&mut state, &doc);
        assert_eq!(state.frame, doc.trace.frames.len() - 1);
        assert_eq!(state.wire, 1);
        assert_eq!(state.gate, None);
        assert_eq!(state.node, None);
        assert_eq!(state.tick, 0.);
        assert!(!state.state_overview);
        assert!(state.playing && state.show_details);
        assert_eq!(state.pan.x, 100.);
        assert_eq!(state.zoom, 2.);
    }
}
