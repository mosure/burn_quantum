# Visualization validation

This report records completed local checks on 2026-11-10 for
`burn_quantum_visualization`. The committed [result record](visualization-results.json)
contains source hashes and compact measurements. Full logs and captures are local,
ignored files under `artifacts/visualization`; their links below refer to that
checkout. This is not a crates.io publication claim.
The viewer uses Rust 1.97.1, Burn 0.22.1 and Bevy 0.20.0. The core library retains
its Rust 1.95 MSRV.

## Completed checks

| Check | Recorded result | Evidence |
| --- | --- | --- |
| Full workspace regression | 347 tests passed across 104 test executables/suites | [Workspace log](../artifacts/visualization/workspace-tests.log) |
| Headless scientific model and interchange | 9 passed, 0 failed, 0 ignored: 7 scientific tests and 2 interchange tests | [Model log](../artifacts/visualization/model-tests.log) |
| Strict Clippy | Headless, full workspace and optional GPU bridge passed with `-D warnings` | [Workspace log](../artifacts/visualization/workspace-clippy.log), [bridge log](../artifacts/visualization/bridge-clippy.log) |
| Actual WGPU scientific parity | All six demos passed on a discrete NVIDIA GPU | [GPU log](../artifacts/visualization/gpu-tests.log) |
| Shared Burn/renderer allocation | Actual rendered pixels passed the exact RGBA oracle | [GPU log](../artifacts/visualization/gpu-tests.log), [test](../crates/burn_quantum_visualization/tests/gpu_bridge.rs) |
| Native Bevy rendering | Vulkan adapter initialized, window rendered, fabric screenshot captured | [Native log](../artifacts/visualization/native-fabric-run.log), [capture](../artifacts/visualization/native-fabric.png) |
| Core Rust 1.95 check | `burn_quantum` compiled successfully; this is a compile check, not a core test-suite rerun | [MSRV log](../artifacts/visualization/core-msrv.log) |
| WASM compilation | Standard release, optimized web-release and optional bridge compile passed | [Size/build record](../artifacts/visualization/wasm-size-comparison.json) |
| Documentation and repository checks | Doctests, warning-free docs, five Python tests and WGPU/CUDA/fusion compilation passed | [Docs](../artifacts/visualization/docs.log), [GPU compile](../artifacts/visualization/gpu-compile.log) |
| Browser execution | Six experiments, desktop/narrow layout, actual controls and I/O passed on NVIDIA WebGPU | [Browser record](../artifacts/visualization/browser-final/results.json) |

The nine headless tests exercise Bloch X/Y/Z signs, Bell/GHZ mixed reduced
states, an amplitude-damped density matrix, qutrit interference, mixed-radix
endianness and input bindings, timeline ordering, actual QN-SPSA iterates and
evaluation costs, 12-qubit retention and probability aggregation, compiled
photonic schedules, malformed states and route records, JSON round trips,
atomic rejected imports, and dense-gate validation budgets. The headless run
does not execute the GPU-only targets; the two explicitly invoked GPU tests
have their own results below.

## Actual CPU/GPU scientific comparison

The adapter was **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, discrete
GPU, Vulkan, NVIDIA driver **610.43.02**. The scientific parity test explicitly
requires a discrete WGPU adapter; it does not accept CPU fallback.

The test builds each complete document independently on Flex CPU and WGPU,
validates both, and compares all scientific fields. Structure, labels and
integer counts are exact. Every floating-point field uses the same absolute
tolerance, **5e-4**, including the optimizer's complete parameter, metric and
loss trajectory. Only compute-device provenance is excluded from equality.

| Demo | Frames | Maximum absolute numeric difference |
| --- | ---: | ---: |
| Bell | 3 | 0 |
| GHZ | 5 | 0 |
| Phase interference | 4 | 8.742277657347586e-8 |
| Qutrit Fourier | 4 | 0 |
| Photonic resource preparation and schedule | 9 | 0 |
| Quantum natural SPSA | 25 | 1.8395522278780163e-5 |

These maxima cover amplitudes, probabilities, local density matrices, Bloch
vectors, purity, bindings, optimization records and attached fabric data, as
applicable. The photonic frames simulate ideal resource preparation; the
schedule comparison does not assert a sampled fusion success or a lossy output
quantum state. The optimizer uses exact simulated expectations and squared
overlaps without shot noise. Its candidate objective is evaluated at each
actual iterate, separately from the perturbed probes.

This qualifies the tested native WGPU path. It does not establish browser GPU
compute, CUDA interoperability, every possible circuit, or every mixed-state
channel on every GPU configuration.

## Real shared-buffer rendering

The separate GPU bridge test creates a renderer device/queue, registers that
same setup with Burn, computes `[0.1, 0.2, 0.3, 0.5] * 2`, and binds the
exported Burn allocation directly as fragment-shader storage. Every pixel in
the final 4-by-4 RGBA8 image is checked exactly against
`[51, 102, 153, 255]`. Only the finished image is read back for the assertion;
the intermediate color tensor is not copied through host values.

The test also rejects incompatible shape, F16 dtype and a transposed layout,
and checks that transforming an alias after export does not overwrite the
leased allocation. This is an actual shader-consumer proof for the optional
shared-WGPU path, not a claim that all viewer state plots use that path. The
tutorial gallery currently uses Flex CPU simulation and cached host records.
Eligible buffer exports require the checked eager, contiguous, untiled F32
layout and allocation lifetime contract documented by
[`SharedGpuBridge`](../crates/burn_quantum_visualization/src/bridge.rs).

## CPU trace profile

The [recorded profile](../artifacts/visualization/trace-profile.json) runs the
[`trace_profile` example](../crates/burn_quantum_visualization/examples/trace_profile.rs)
with Flex CPU on an Intel Core i9-12900K in the **development profile**. Each circuit applies one
Hadamard per qubit and retains the input plus every gate output. The timer
covers trace simulation and its awaited scientific readbacks; JSON
serialization and final histogram aggregation occur afterward.

| Qubits | State dimension | Frames | Simulation seconds | Serialized trace bytes | Probability bins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 16 | 5 | 0.009213799 | 8,566 | 16 |
| 8 | 256 | 9 | 0.041853624 | 91,537 | 64 |
| 12 | 4,096 | 13 | 0.488972904 | 1,603,146 | 64 |

Every final histogram retains probability mass
`0.9999997615814351`, consistent with the F32 simulation tolerance. Adjacent
basis states are summed; low-probability states are not discarded. The
12-qubit trace retains 53,248 global amplitudes across its 13 frames, alongside
local reduced states and probabilities.

These are individual observed timings, not throughput guarantees, GPU
benchmarks, release-profile results, frame-rate measurements or memory
measurements. There is no timing assertion in the tests. Dense simulation
remains exponential in qubit count; bounded rendering does not change that
simulation cost.

## Scientific and resource boundaries

Default trace limits are 16 wires, local dimension 16, global dimension 65,536,
257 frames, 1,048,576 retained complex entries according to the trace budget,
65,536 mixed-source density entries and 65,536 entries per dense gate. These
limits combine and do not represent a total-process RAM bound. Display limits
are separate: probability bins retain all mass, while graph/gate/trail views
report their drawing caps.

JSON import and export are capped at 16 MiB. Before typed circuit
deserialization can validate a dense unitary, a JSON preflight checks register,
frame, array, operation and matrix sizes and caps aggregate unitary validation
work at **16,777,216 complex inner-product terms**, computed as
`sum(gate_dimension^3)`. A valid in-memory document may exceed the separate
JSON size or validation-work export budget. Rejected imports do not replace
the pending experiment.

Imported pure amplitudes are checked against probabilities and local partial
traces. When amplitudes are absent, validation checks local density positivity
within the declared 5e-4 consistency tolerance,
Hermiticity, trace, purity, Bloch vectors and probability marginals. Those local
summaries do **not** reconstruct global mixed-state coherences or prove global
compatibility. Validation checks consistency, not experiment provenance or
whether the supplied circuit was actually executed.

Fabric snapshots preserve compiled routes, resource counts and fusion times.
Route attenuation must agree with its listed links and intervening storage
waits. It excludes pre-release and final synchronization/detector attenuation.
Drawing coordinates are schematic and bounded for rendering; they are not
optical path lengths. Animation represents scheduled travel, not observed
photon survival or an automatically successful fusion.

## Reproduce

From the repository root, using the checked-in Rust 1.97.1 toolchain:

```sh
cargo test --locked -p burn_quantum_visualization --no-default-features
cargo clippy --locked -p burn_quantum_visualization --no-default-features --all-targets -- -D warnings
cargo test --locked -p burn_quantum_visualization --features gpu-bridge --test gpu_bridge --test gpu_scientific -- --ignored --nocapture
cargo run --locked -p burn_quantum_visualization --no-default-features --example trace_profile
cargo +1.95.0 check --locked -p burn_quantum --lib
```

The GPU command requires an actual accessible native discrete adapter. Native
render capture requires a graphical session:

```sh
cargo run --locked -p burn_quantum_visualization --bin quantum_explorer -- --demo photonic --view fabric --screenshot "$PWD/artifacts/visualization/native-fabric.png"
```

Build and serve the actual browser application with a matching wasm-bindgen
CLI; the build script verifies its version against `Cargo.lock`:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
scripts/build_visualization_web.sh
python3 -m http.server --directory target/visualization-web 8080
```

## Actual browser and visual qualification

Chrome **156.0.8078.79** ran the final `web-release` artifact on NVIDIA Blackwell
WebGPU in a visible browser window. The Linux test session enabled Vulkan and
WebGPU with explicit browser flags; this is not a claim about every browser's
default GPU configuration. SwiftShader startup encountered device loss, so
software-rendered browser compatibility is not claimed.

The automated [browser runner](../scripts/validate_visualization.cjs) checks the
actual Bevy canvas and real Feathers controls, not an HTML mockup. All six demos
render and export scientifically valid traces. Qualification includes button and
keyboard stepping with explicit focus, independent physical-tick playback,
qubit selection, camera zoom/reset, guide toggles, 16 repeated view switches,
validated JSON download/import, rejection preserving the selected document,
390-pixel layout, compact Details, actual bounded scrolling, unsupported graphics,
and deliberately destroying the real WebGPU device to verify the error overlay.
A separate two-finger touch exercise changed zoom from 1 to 1.8 and activated
Details through a real touch event; the runner includes this check for future runs.

Actual screenshots were reviewed for all four views and all six examples. Review
led to canonical CNOT symbols, supported ket glyphs, a phase legend, explicit
snapshot-trail semantics, separate rail transmission values, independent schedule
time, bounded inspector scrolling and width-aware probability bins. The final
12-qubit import shows all 4,096 outcomes in six legible adjacent bins and **538
entities**, with the full 13-frame timeline. Binning retains all probability mass.

After warm-up, 120 requestAnimationFrame intervals on the paused Bell scene had
median **6.9 ms** and p95 **7.1 ms**. Bevy continued updating, the scientific/view
revision stayed unchanged, and entity count returned to **542** after repeated
tab switches. These are one workstation's scheduler/frame-interval observations,
not a universal frame-rate guarantee or a total-allocation measurement.

## Web payload and deployment

A dedicated `web-release` Cargo profile uses size optimization, thin LTO and one
codegen unit; it leaves native numerical build profiles unchanged. Comparing the
same source tree after wasm-bindgen name-section removal:

| Payload | Standard release | web-release | Reduction |
| --- | ---: | ---: | ---: |
| WASM bytes | 57,828,667 | 35,686,826 | 38.3% |
| Deterministic gzip-9 bytes | 11,522,542 | 7,808,242 | 32.2% |

The gzip figures measure compressibility; they do not assert a hosting server's
Content-Encoding. Local startup timings and exact artifact hashes are retained
in the result record. Network download time depends on hosting and connectivity.

The [Pages workflow](../.github/workflows/pages.yml) builds the actual executable
and deploys it at <https://mosure.github.io/burn_quantum/>. Its generated
[`build.json`](https://mosure.github.io/burn_quantum/build.json) is the authoritative
commit/toolchain/profile/file-hash record for the deployed artifact. Deployment
closeout verifies that commit against the remote branch and repeats browser
checks against the live URL; local build success alone does not establish that.

To repeat browser checks, install Playwright 1.63.0, serve the build and configure
the executable/launch arguments appropriate to the machine. `PLAYWRIGHT_MODULE`
can point to an existing installation; `BROWSER_HEADLESS=false` selects a visible
browser. The Linux NVIDIA run used Vulkan, `--enable-unsafe-webgpu`,
`--ignore-gpu-blocklist`, and `--disable-vulkan-surface`.

```sh
cargo run -p burn_quantum_visualization --no-default-features --example trace_profile -- artifacts/visualization/twelve-qubits.json
VISUALIZATION_URL=http://localhost:8080/ LARGE_TRACE_JSON="$PWD/artifacts/visualization/twelve-qubits.json" node scripts/validate_visualization.cjs
```
