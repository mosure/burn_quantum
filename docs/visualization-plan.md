# Quantum visualization architecture and acceptance plan

The workbench is a separate `burn_quantum_visualization` crate. Core simulation
remains independent of Bevy. A serializable, bounded experiment document is the
boundary between simulation, scientific inspection, rendering and saved runs.
The viewer must label exact amplitudes, reduced states, marginal/aggregated
probabilities, physical records and illustrative layout distinctly.

## Data and execution

1. Capture actual circuit prefixes through the existing Burn backend, preserving
   parameter/input bindings, register dimensions and operation order. Store
   snapshots once; scrubbing never reruns a solver or performs GPU readback.
2. Derive each qubit's Bloch vector from its reduced density matrix. Entangled
   qubits belong inside the sphere; the global state vector is not a single
   qubit. Qudits get population/phase views and no misleading Bloch sphere.
3. Keep sampling, conditioning, losses and ideal references explicit. Physical
   fabric traces retain topology, routes, time and resource assumptions.
4. Plot actual objective/probe histories for optimization, retaining evaluation
   counts and the parameter state associated with a selected iteration.
5. Validate loaded documents, cap input bytes, frames, amplitudes, graph elements
   and render work. Keep rendering limits independent of simulation limits;
   a future summary-only trace must not require allocating dense Hilbert space.

## Views and interaction

The application uses Bevy UI and genuine Feathers widgets. A persistent example
selector, view tabs, timeline and selected-object inspector provide consistent
navigation. The canvas carries only the current view's diagram; legends and
scientific context stay in adjacent panels. Color uses a stable phase/role
meaning, accompanied by text, values or shape rather than color alone.

- **State:** rotatable projected Bloch sphere, qubit selector, reduced purity,
  populations and complex amplitude/phase readouts.
- **Circuit:** wire order, mixed-radix labels, gate/control structure, operation
  cursor, and a synchronized input/output inspector. Pan/zoom for long circuits.
- **Fabric:** resource/analyzer nodes, directed routes, loss/capacity annotations
  and selected-time highlighting. Geometry is an inspection layout, not an
  invented hardware floorplan.
- **Learning:** objective versus iteration, parameter/probe metrics and selected
  iteration; make missing and nonfinite values visible or reject them.
- **Timeline:** play/pause, scrub, first/last and single-step. Keyboard controls
  must agree with buttons, stop at the end, and never change selected data merely
  because a camera moved.
- **Input/output:** validated JSON import/export and native CLI loading, with a
  reusable library integration path rather than copying viewer code per example.

## Rendering and scaling

Render budgets limit glyphs, nodes, edges and amplitude bins; omitted detail is
reported. UI data updates only when selection or trace changes. Dense results
should use aggregated summaries or GPU storage/textures, not millions of UI
entities. Camera operations transform rendering rather than recapturing physics.

An optional shared-WGPU module must use the same device and queue as Bevy,
validate dtype/shape/strides/offsets/usages, flush pending Burn operations and
retain allocation ownership until rendering completes. A buffer export API alone
does not prove zero-host-copy rendering. Qualification must include an actual
consumer or explicitly state what is still unsupported. The bounded tutorial
traces may use one-time host snapshots; they must not be marketed as zero-copy.

## Native and browser delivery

Bevy 0.20 requires Rust 1.97.1; the core library's declared MSRV remains 1.95.
The browser build uses WebGPU rendering with an explicit unsupported/loading/error
state and relative asset URLs under the repository's GitHub Pages prefix.
The landing gallery supplies high-level routes and explanation; quantum diagrams
are rendered by the Bevy application. Native and browser share the same model,
widgets and view code.

## Acceptance gates

- Mathematical tests: known Bloch axes, Bell/GHZ mixed reductions, qutrit norms,
  gate-prefix ordering, input bindings, bounded documents, objective histories.
- Integration: model-only/native/optional-bridge/WASM compilation, strict linting,
  deterministic example serialization and load/export round trips.
- Visual review: actual native and browser screenshots at desktop and narrow
  sizes, each view, middle/end timeline states, camera/reset, tab/example switching
  and imported traces. Inspect overlap, clipping, contrast and misleading labels.
- Performance: bounded draw complexity and trace storage, frame timing after
  warm-up, allocation changes across repeated navigation, and documented GPU
  readback/copy boundaries. Never infer performance from compilation or startup.
- Deployment: build the actual WASM package, deploy through GitHub Pages,
  verify the exact deployed build and exercise the live page.

Further scale-out should add tiled probability textures, graph clustering and
selection-aware streaming behind these interfaces. Distributed execution,
arbitrary measurement-branch inspection and provider hardware control need
their own explicit execution/data contracts rather than speculative UI switches.

## Implemented structure and next scale-out milestones

The first delivery implements the bounded scientific document, actual Burn trace
adapters, six canonical experiments, four synchronized Bevy/Feathers views,
compact scrollable inspection, JSON preflight, native/WASM runtimes and deployment.
The optional GPU bridge has a real shader-consumer test. The
[validation record](visualization-validation.md) separates mathematical checks,
actual device execution, render review and deployment evidence.

These extensions have concrete boundaries rather than speculative switches:

| Extension | Required contract and acceptance evidence |
| --- | --- |
| Summary-only and streamed experiments | A new versioned trace representation for local observables, selected amplitudes, MPS/Clifford summaries and paged checkpoints. Replay and cancellation must preserve provenance without a dense global vector. |
| Dense GPU heatmaps and instances | Render adapters consuming checked `TensorBufferLease` storage; define shape, complex phase mapping, precision, ownership and cross-queue ordering. Compare actual pixels/instances with a CPU oracle and measure transfer bytes. |
| Larger network overview | Spatial indexing and clustered edges/nodes beyond the current 256-node import / 96-drawn-node budgets. Selection and pan/zoom must reveal every node without silently dropping topology. |
| Branches, noise and measurement | Explicit branch probabilities, conditional versus unconditional outcomes, sample counts and uncertainty; never substitute routing metadata for physical loss or detector results. |
| Distributed/remote jobs | Incremental records with job/partition identifiers, cancellation, revision ordering and resumable storage. No provider hardware execution or scale-out throughput is implied by this local viewer. |
| General training dashboards | Typed objective/parameter/metric channels, comparison runs, uncertainty and wall-clock/evaluation axes. Preserve algorithm identity rather than labeling imported histories as QN-SPSA. |

Bevy UI belongs only in the viewer feature. Core physics and scientific records
remain independently testable. Changes to document semantics require a schema
version change and migration policy; QA diagnostics intentionally have no stable
scientific-schema guarantee.
