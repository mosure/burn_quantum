# burn_quantum_visualization

A Bevy 0.20 / Feathers workbench for `burn_quantum` circuits, states, photonic
fabrics and optimization traces. It shares one interface across native and WebGPU
runtimes. Open the [live explorer](https://mosure.github.io/burn_quantum/) or
install the native application:

```sh
cargo install burn_quantum_visualization --version 0.4.0 --locked
quantum_explorer
quantum_explorer --demo mixed --step 6 --bloch 2d
```

This crate requires Rust 1.97.1, including builds without the viewer. The separate
`burn_quantum` core crate supports Rust 1.95. To embed the workbench, add
`burn_quantum_visualization = "0.4.0"` to your dependencies.

From a repository checkout:

```sh
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo photonic --view fabric
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo optimization --view learning
cargo run --locked -p burn_quantum_visualization --example embedded
```

## Explore

Fifteen experiments cover Bell/GHZ states, mixed qubit/qutrit registers, interference,
teleportation, Toffoli, QFT, Deutsch–Jozsa, QPE, Grover, QAOA MaxCut, bit-flip
correction, compiled photonic routing and quantum natural SPSA.
Each loaded configuration is simulated once through core Burn APIs. Changing
views or scrubbing a cached timeline does not rerun its simulation.

| View | Meaning |
| --- | --- |
| State | Parallel reduced-state panels, direct subsystem selection, qudit populations; focused joint probabilities and amplitude phase |
| Circuit | Semantic gate symbols, conservative parallel columns, SWAPs, control levels, measurement and double classical wires |
| Photonic fabric | Resource ownership, held outputs, paired optical rails, switches, destructive fusion analyzers and per-photon travel/wait timelines |
| Learning | Actual QN-SPSA parameter iterates, objective history and cumulative probe costs |

Entangled qubits can lie inside the Bloch sphere. Qudits use populations instead
of a misleading three-dimensional qubit representation. Adjacent probability
bins retain all mass; aggregation does not pretend to retain one complex phase.
The photonic demo separates resource preparation from physical routing: moving
dots show scheduled travel, not sampled photon detection or a successful fusion.

Use the experiment list, view tabs, clickable subsystem panels and timeline.
Narrow browser windows use an experiment selector; compact native windows use
**Next experiment**. The list keeps titles left-aligned and
lets wrapped rows grow. Register overview displays many reduced states together;
small viewports divide them into selectable groups. **Focus** shows the selected
subsystem alongside the joint computational-basis distribution.

### Bloch representations

- **3D:** an orthographic sphere with shaded surfaces, depth-cued great circles,
  computational poles `|0>` / `|1>`, and an orbitable Bloch vector inside the unit ball.
- **2D probability:** colored area fraction is `P(1) = (1-z)/2`, so the fill radius
  is `R * sqrt(P(1))`. The hand angle is the coherence phase `atan2(y,x)`; its
  length is `R * sqrt(x²+y²) = R * 2|rho01|`. No hand is drawn when coherence is
  below display precision. Probability, phase and purity are separate readouts:
  a 50/50 pure superposition and a maximally mixed state have equal fill areas
  but different hands and purity. Trails trace earlier hand tips. This follows
  the local-state [phase-disk convention](https://quantum.cloud.ibm.com/docs/en/guides/composer#phase-disk), with explicit area scaling.
- **Projection:** XY/XZ/YZ coordinate disks retain the omitted coordinate as a
  label. A short projected vector need not be mixed; use the full purity and
  omitted component to interpret it.

Qudits retain population charts in all three modes. Local reduced states alone
do not describe joint correlations. Two shared analytic surface textures are
uploaded once; the panels introduce no per-qubit render targets, cameras or
tensor readbacks.

### Controls

**Settings** groups representation modes, projection planes, guides, trails,
playback speed and experiment configuration. **Details** shows the inspector on
compact layouts; large native windows keep it visible. Secondary controls stay
out of the default toolbar.

Left-drag a 3D sphere to orbit; right-drag a focused state view, or drag a circuit
or fabric diagram, to pan. Scroll to zoom. Orbit and zoom are shared across the
visible register panels. Click the plot to use
keyboard shortcuts: **Space** toggles playback, **Left/Right** step, **R** resets
the camera and **G** toggles guides. Focused Feathers widgets retain their normal
keyboard behavior. Playback cycles through 0.5–4× and stops at the last frame.
Fabric schedule ticks remain independent of quantum circuit steps.

The native controls and browser shell follow the `burn_gekko` controller theme:
charcoal surfaces, rose selections, thin borders and compact labels.
[`view/theme.rs`](https://github.com/mosure/burn_quantum/blob/main/crates/burn_quantum_visualization/src/view/theme.rs)
owns the Feathers semantic colors;
[`web/index.html`](https://github.com/mosure/burn_quantum/blob/main/web/index.html)
mirrors the surface and accent tokens. Scientific plots retain distinct series
colors and a cyclic amplitude-phase legend.

## Configurable algorithms

Open **Settings**, use the size/variant controls (and layers for QAOA), inspect
the resource preview, then press **Run**. Editing settings does not recompute the displayed experiment.
The preview rejects excessive wire, frame and retained-state budgets before any
state allocation. Native and browser defaults use the same portable budgets;
applications can supply their own `TraceLimits` based on measured device resources.
The UI does not guess available VRAM or silently reduce a requested circuit.

```sh
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo ghz --size 12 --step 12
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo mixed --step 6 --bloch 2d
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo qpe --size 6 --variant 1 --view circuit
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo qaoa --size 8 --layers 3 --view circuit
```

Browser links accept the same settings, for example
[twelve qubits in 3D](https://mosure.github.io/burn_quantum/?demo=ghz&size=12&step=12)
or [a mixed register in 2D](https://mosure.github.io/burn_quantum/?demo=mixed&step=6&bloch=2d).

### Share a view

The address bar tracks the complete durable view state. Copying the URL restores
the selected example, applied configuration, pending edits, both timelines,
selection, camera, playback, and display settings after the simulation loads.
Back/Forward restores through the same initialization path. Display preferences
also survive switching experiments; experiment-specific settings reset to that
example's defaults.

| Query parameters | Meaning |
| --- | --- |
| `demo`, `size`, `layers`, `variant` | Applied experiment and configuration |
| `draft_size`, `draft_layers`, `draft_variant` | Unrun Settings edits, when different from the applied configuration |
| `view`, `frame`, `tick` | Scientific view, quantum frame, and independent physical fabric time |
| `step` | Legacy shorthand for the active timeline; explicit `frame`/`tick` take precedence |
| `wire`, `gate`, `node` | Selected subsystem, operation, and fabric component (`none` clears an optional selection) |
| `overview`, `bloch`, `plane`, `trails` | Register/focus mode, `3d`/`2d`/`projection`, projection plane, and trails |
| `playing`, `speed` | Playback state and multiplier |
| `yaw`, `pitch`, `zoom`, `pan_x`, `pan_y` | Shared camera, with finite bounded values |
| `guides`, `details`, `settings` | Guide and panel visibility |
| `source`, `v` | `catalog`/`imported` document source and query format version |

Boolean values are `0` or `1`. `bloch=2d` selects probability/phase disks;
`bloch=projection&plane=xy&trails=1` selects a Cartesian projection. Native
flags include `--bloch projection --plane xy --trails`; `--query '<query>'`
accepts the canonical query emitted by the viewer. For an imported experiment,
share its exported JSON alongside the URL. The link explicitly asks for that
file before restoring the view, since scientific payloads are not embedded in
query parameters. Native imports can combine `--query` with `--input`.

`demo::build_configured` and `demo::preview` expose the catalog
without UI boilerplate. Algorithm definitions live in `burn_quantum::algorithms`;
`burn_quantum::protocol::BranchExecution` supplies real projective collapse and
classical feed-forward. Teleportation displays an explicitly selected branch and
its likelihood, and tests check every outcome for one/two parallel transfers.
QAOA displays a fixed-angle ansatz evaluation; the examples crate separately trains
MaxCut with the same builder. Bit-flip correction uses a coherent majority decoder,
retaining syndrome qubits, rather than claiming repeated noisy syndrome extraction.

Circuit boxes are classified from operations/matrices, independent of educational
labels. Ordered multi-wire operators get one box with local port indices. QFT
includes bit-reversal SWAPs; all register/readout ordering is big-endian. Dynamic
protocol JSON uses document schema 2; schema 1 remains readable.

The fabric diagram distinguishes graph-resource connections from physical rails.
Unconsumed outputs remain at their actual source buffers. Photon packets pause at
routing/synchronization waits and disappear after destructive measurement. Link
loss labels show mean rail intensity loss; the inspector retains both rail values
and route transmission. The analyzer icon is schematic, not a claimed optical
component decomposition. Neither routing animation nor loss metadata asserts a
sampled fusion outcome.

## Integrate an experiment

The core simulation crate has no Bevy dependency. `model::VisualizationDocument`
is a versioned, owned record containing a `Circuit`, parameter/input bindings,
labels and an `ExperimentTrace`; optional fabric/optimization records attach
physical or learning context. Build a document and call `app::run_document`.
The complete [embedded example](https://github.com/mosure/burn_quantum/blob/main/crates/burn_quantum_visualization/examples/embedded.rs) shows the integration.

Use `simulation::simulate` for actual circuit-prefix execution, or
`snapshot_pure` / `snapshot_density` to adapt a state from another experiment.
`simulate` accepts a Burn `Device`; the snapshot adapters read states on their
existing device. Tensor readback is asynchronous; each snapshot displays one
batch row. `snapshot_density` does not invent pure amplitudes for mixed states. Validation
checks bounded sizes and scientific consistency; it cannot authenticate the
provenance of an externally supplied trace.

For analysis and interchange without a renderer:

```toml
burn_quantum_visualization = { version = "0.4.0", default-features = false }
```

The browser's Import/Export buttons exchange validated JSON. Native equivalents:

```sh
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo bell --export bell.json
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --input bell.json
```

`io::parse_document`, `document_json`, `read_input` and `write_export` are reusable.
The JSON representation is capped at 16 MiB for both import and export. Imported
values, register order, reduced states, norms, operation counts, optimization
records and fabric references are checked before
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
recapturing tensors every frame. The inspector reports the selected scientific
state; `app::diagnostics_json` exposes drawing/entity statistics for tooling.

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
state uses the [query parameters above](#share-a-view). The Pages workflow deploys
the actual WASM build on `main` and publishes a `build.json` provenance manifest.

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
site. The script renders all fifteen experiments and checks real widget clicks,
keyboard focus, import/export, complete URL restoration, bounded entity counts,
responsive layouts, touch input, unsupported graphics and device loss. It writes
results under `artifacts/visualization/browser`; `VISUALIZATION_ARTIFACTS` changes
the output directory. To include the larger imported-register checks, generate
the fixture expected by the script:

```sh
mkdir -p artifacts/visualization
cargo run --locked -p burn_quantum_visualization --no-default-features --example trace_profile -- artifacts/visualization/twelve-qubits.json
VISUALIZATION_URL=http://localhost:8080/ LARGE_TRACE_JSON="$PWD/artifacts/visualization/twelve-qubits.json" node scripts/validate_visualization.cjs
```

See the [validation record](https://github.com/mosure/burn_quantum/blob/main/docs/visualization-validation.md) for the tested
adapter, numerical tolerances and performance evidence.
