# burn_quantum_visualization

A Bevy 0.20 / Feathers workbench for actual `burn_quantum` experiments, with a
shared native and WebGPU interface. Open the
[live gallery](https://mosure.github.io/burn_quantum/) or run it locally:

```sh
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo photonic --view fabric
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo optimization --view learning
cargo run --locked -p burn_quantum_visualization --example embedded
```

Rust 1.97.1 is required by the viewer. The core library keeps its Rust 1.95 MSRV.
Install the native application from crates.io:

```sh
cargo install burn_quantum_visualization --version 0.3.0 --locked
quantum_explorer
```

To embed the viewer, add `burn_quantum_visualization = "0.3.0"` to your dependencies.

## Explore

Six canonical experiments cover Bell/GHZ entanglement, phase interference,
qutrit Fourier gates, compiled photonic routing and quantum natural SPSA.
Each experiment is built once through core Burn APIs. Changing views or scrubbing
a cached timeline does not rerun its simulation.

| View | Meaning |
| --- | --- |
| State | Exact local partial traces, qubit Bloch ball, purity, computational-basis probability and complex phase |
| Circuit | Ordered mixed-radix wires and operations, selected gate and execution prefix |
| Photonic fabric | Resource nodes, dual-rail transmission, compiled routes and independent physical schedule time |
| Learning | Actual QN-SPSA parameter iterates, objective history and cumulative probe costs |

Entangled qubits can lie inside the Bloch sphere. Qudits use populations instead
of a misleading three-dimensional qubit representation. Adjacent probability
bins retain all mass; aggregation does not pretend to retain one complex phase.
The photonic demo separates resource preparation from physical routing: moving
dots show scheduled travel, not sampled photon detection or a successful fusion.

Use the view tabs, experiment selector, qubit selector and timeline. Click the
plot to use global keyboard shortcuts; focused Feathers widgets retain their
normal keyboard behavior. Space toggles
playback, left/right arrows step, R resets the view and G toggles guides. Drag the
sphere to orbit; drag other diagrams to pan; scroll to zoom. The speed control
cycles through 0.5–4×. The timeline stops at the final frame. Fabric playback uses
schedule ticks independently of quantum circuit steps.

The native controls and browser shell follow the `burn_gekko` controller theme:
warm charcoal surfaces, muted rose selections, thin borders and compact labels.
[`view/theme.rs`](https://github.com/mosure/burn_quantum/blob/main/crates/burn_quantum_visualization/src/view/theme.rs) owns the Feathers semantic colors, including
hover, pressed, disabled and keyboard-focus states; [`web/index.html`](https://github.com/mosure/burn_quantum/blob/main/web/index.html)
mirrors the surface and accent tokens. Scientific plots retain separate cyan,
amber and blue series and a cyclic phase legend. Plot guides use a brighter
neutral than panel borders so fine geometry remains visible.

## Integrate an experiment

The core simulation crate has no Bevy dependency. `model::VisualizationDocument`
is a versioned, owned record containing a `Circuit`, parameter/input bindings,
labels and an `ExperimentTrace`; optional fabric/optimization records attach
physical or learning context. Build a document and call `app::run_document`.
The complete [embedded example](https://github.com/mosure/burn_quantum/blob/main/crates/burn_quantum_visualization/examples/embedded.rs) shows the integration.

Use `simulation::simulate` for actual circuit-prefix execution, or
`snapshot_pure` / `snapshot_density` to adapt a state from another experiment.
These APIs accept a real Burn `Device` and use asynchronous tensor readback.
`snapshot_density` does not invent pure amplitudes for mixed states. Validation
checks bounded sizes and scientific consistency; it cannot authenticate the
provenance of an externally supplied trace.

For analysis and interchange without a renderer:

```toml
burn_quantum_visualization = { version = "0.3.0", default-features = false }
```

The browser's Import/Export buttons exchange validated JSON. Native equivalents:

```sh
cargo run -p burn_quantum_visualization --bin quantum_explorer -- --demo bell --export bell.json
cargo run -p burn_quantum_visualization --bin quantum_explorer -- --input bell.json
```

`io::parse_document`, `document_json`, `read_input` and `write_export` are reusable.
JSON is capped at 16 MiB. Imported values, register order, reduced states, norms,
operation counts, optimization records and fabric references are checked before
replacing the selected document. Invalid input leaves the current experiment intact.
Before checked core deserialization, JSON preflight also bounds aggregate dense
unitary-validation work: `sum(dimension^3) <= 16,777,216`. Records without global
amplitudes validate local PSD states and marginals; these summaries neither
reconstruct nor prove compatibility of a complete mixed global density matrix.

## Budgets and GPU interoperability

Default `TraceLimits` bound wires (16), local dimension (16), global dimension
(65,536), frames (257), aggregate retained complex entries (1,048,576), mixed-state
source entries (65,536) and local matrix entries (65,536). These limits combine;
not every maximum can occur simultaneously. They do not bound total process RAM.
The 16 MiB JSON limit is separate: a valid in-memory trace can exceed its export
budget. Dense state preparation is still exponential in the number of qubits.

Rendering uses bounded probability bins, trails, graph elements and viewport
culling. Geometry/text are rebuilt on selection or camera changes, rather than
recapturing tensors every frame. Drawing budgets are reported in the inspector.

Enable `gpu-bridge` for `bridge::SharedGpuBridge`. Register the renderer's actual
WGPU instance, adapter, device and queue once, then create Burn tensors on
`bridge.device()`. `export(tensor, shape)` validates an eager contiguous, untiled
F32 tensor and exposes read-only storage. Its `TensorBufferLease` retains both
the tensor and allocator ownership; `submit_read` keeps that ownership until
GPU work completes. Other devices, dtypes, shapes and incompatible layouts are
rejected. The caller supplies the shader/pipeline and must not write the buffer.

This optional path avoids a host value readback for eligible existing GPU data.
The tutorial gallery uses Flex CPU simulation and cached host snapshots. It does
not automatically turn arbitrary dense readouts into zero-copy textures. The
[render proof](https://github.com/mosure/burn_quantum/blob/main/crates/burn_quantum_visualization/tests/gpu_bridge.rs) computes a Burn color, renders directly from
that allocation, and reads back only the final image to verify its pixels.

## Browser build and validation

```sh
rustup target add wasm32-unknown-unknown
# Match the wasm-bindgen version in Cargo.lock.
cargo install wasm-bindgen-cli --version 0.2.129 --locked
scripts/build_visualization_web.sh
python3 -m http.server --directory target/visualization-web 8080
```

Open `http://localhost:8080`. HTTPS/localhost and a WebGPU-capable browser/adapter
are required; unsupported graphics produce an explicit fallback screen. URL
parameters `demo`, `view` and `step` select the initial experiment. The Pages
workflow deploys the actual WASM build on `main` and publishes build provenance.

```sh
cargo test --locked -p burn_quantum_visualization --no-default-features
cargo clippy --locked -p burn_quantum_visualization --all-targets -- -D warnings
cargo test --locked -p burn_quantum_visualization --features gpu-bridge --test gpu_bridge -- --ignored --nocapture
cargo run --locked -p burn_quantum_visualization --no-default-features --example trace_profile
```

Native screenshot automation uses `--screenshot /absolute/path.png` (captures the
application after initialization and exits). Architecture, acceptance gates and
further scale-out work are recorded in the
[plan](https://github.com/mosure/burn_quantum/blob/main/docs/visualization-plan.md).

Browser regression checks use `scripts/validate_visualization.cjs` with Playwright
1.63.0 and a working WebGPU adapter. Configure `PLAYWRIGHT_MODULE` if Playwright
is installed outside the repository, `CHROME_BIN` for a browser executable,
`BROWSER_HEADLESS=false` for a visible window, and `BROWSER_ARGS` for a JSON array
of platform-specific launch flags. The default `web-release` profile uses size
optimization and thin LTO; `BURN_QUANTUM_WEB_PROFILE=release` selects the standard
release profile for comparison. `VISUALIZATION_URL` selects a local or deployed
site. The script saves all six rendered experiments, checks real widget clicks,
keyboard focus, import/export, bounded entity counts, narrow layout and unsupported
graphics, and writes measured results under `artifacts/visualization/browser`.
See the [validation record](https://github.com/mosure/burn_quantum/blob/main/docs/visualization-validation.md) for the tested
adapter, numerical tolerances and performance evidence.
