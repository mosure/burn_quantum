# CPU and GPU runtime validation

Hardware execution recorded on **2026-11-07**, with Rust 1.95.0, Burn 0.22.1,
CubeCL 0.11.3, and the repository's locked dependency graph.

This is the original seven-example qualification snapshot. Subsequent additions
have separate records: [Grover SAT](sat.md) and [quantum natural SPSA](qnspsa.md),
including their CPU counts and targeted hardware checks.

## Device selection and reproducibility

All seven `burn_quantum_example` binaries and the basic simulator demos select
their device through `crates/burn_quantum/support/device.rs`. Selection is explicit and cached once
per process. `BURN_QUANTUM_DEVICE` defaults to `cpu` (Flex); `wgpu` selects a
discrete GPU, `wgpu-integrated` selects an integrated GPU, and `cuda` selects a
CUDA device. `BURN_QUANTUM_DEVICE_INDEX` defaults to zero. WGPU adapter type is
checked after acquisition. Missing features, unavailable requested hardware,
and unknown backend names fail the run instead of falling back or skipping.
The dedicated `wgpu_smoke` example deliberately selects discrete WGPU adapter
zero directly; it is separate from the selectable example suite.

`wgpu-cpu` is a separate, explicit software-adapter diagnostic mode. It cannot
satisfy hardware qualification. Cargo backend features enable a device factory;
the environment selector determines which device actually executes the tests.

```sh
bash scripts/validate.sh
bash scripts/run_examples.sh cpu
bash scripts/validate_backend.sh wgpu
bash scripts/validate_backend.sh wgpu --fusion
bash scripts/validate_backend.sh cuda
bash scripts/validate_backend.sh cuda --fusion
python3 scripts/compare_backends.py artifacts/examples/cpu \
    artifacts/examples/wgpu artifacts/examples/wgpu-fusion \
    artifacts/examples/cuda artifacts/examples/cuda-fusion
```

Use `CARGO_NET_OFFLINE=true` when dependencies are already cached. This session
used the standalone Rust toolchain on PATH because the sandbox's snap shim was
unavailable. GPU processes ran outside the filesystem sandbox: the host exposes
an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**, driver **610.43.02**,
with CUDA **13.3**. WGPU reported a **DiscreteGpu / Vulkan** adapter by that name.
The sandbox itself did not expose a usable GPU; that was an access boundary,
not a driver failure on the host.

## Coverage and precision

The complete CPU suite contains **51 tests** plus one doctest: the original 35
core integration tests, nine new portable backend tests, and seven algorithm
acceptance tests. The strict F64 CPU reference checks remain unchanged. GPU
qualification uses F32, matching the examples and portable shader support.

Each hardware configuration runs nine tests in `crates/burn_quantum/tests/backend_parity.rs` and
the seven example acceptance tests, then executes the seven binaries and the
basic demos. It does not relabel CPU-only F64 tests as GPU coverage. The portable
tests exercise both the selected backend and Flex, with independent references:

- Qubit/qudit entanglement, probabilities, reduced density matrices, partial
  traces and seeded terminal measurement; independent scalar complex outer
  products also validate the complete density matrix.
- Batched complex contractions on mixed registers `[2,3,2]` and `[3,2,4]`,
  including reversed and nonadjacent targets, against a scalar complex oracle.
- X/Y/Z rotations on dimensions 2/3/4, shared affine parameters, native
  gradients and occurrence-wise parameter shifts against F64 finite differences.
- Batched classical-input gradients and Burn gradient checkpointing against
  analytic sine/cosine values.
- Amplitude damping, depolarization and differentiation through pure-to-density
  conversion and a noise channel.
- Complex fidelity kernels, probability cross-entropy, and parameter/input
  gradients against CPU execution.
- Adam training curves, burnpack weight records, GPU-to-CPU module movement,
  explicit complex-state transfers and rejection of cross-device components.

Forward comparisons use absolute tolerance `2e-5`; gradient comparisons use
`5e-5`. The 20-step optimizer trajectory comparison uses `2e-4`. These tolerances
are confined to F32 backend tests. The existing F64 oracle suite still uses its
original `1e-7` to `1e-12` bounds.

The seven example binaries use identical data, initial parameters, training
budgets, seeds and acceptance criteria across devices. JSON reports include
backend, adapter, F32 precision and the example package's fusion feature.
`compare_backends.py` requires matching configurations/metric names and finite
values, then checks `abs(gpu-cpu) <= 5e-5 + 2e-4*abs(cpu)`. Sampled total variation
has a separate statistical tolerance of `0.02`; each Born-machine run also must
meet its own absolute sampling acceptance bound.

Hybrid parameter-change diagnostics have an absolute comparison tolerance of
`0.005`: native Burn `Linear` layers can use TF32 on CUDA, so optimizer paths
differ slightly even when final task losses agree. The hybrid loss comparison
and each run's held-out MSE/three-stage-update acceptance criteria remain
unchanged. Quantum gradient and optimizer parity tests retain their tighter
bounds above.

## Numerical correction found by hardware execution

CUDA initially produced `0.8258399963` for a circuit expectation whose analytic
value is `cos(0.6) = 0.8253356149`, failing the `2e-5` tolerance. Inspection of the
locked CubeCL/Cubek matrix kernel showed that its accelerator selection converts
F32 operands to TF32. The quantum simulator now uses dtype-preserving
multiply/reduce contractions on non-Flex devices, implemented in
[`precision.rs`](../crates/burn_quantum/src/precision.rs). Flex retains its native GEMM path.
This applies to local gates, reduced states, pure-to-density conversion and
fidelity kernels. No device readback, custom autodiff bridge, or dependency fork
was added.

The first direct-broadcast outer-product prototype also failed CUDA fusion's
complex-density checks. The final implementation evaluates outer products
through the same column-wise contraction path, and a new scalar-complex density
test guards that regression. Both fused and unfused CUDA now pass all nine
portable tests. This correctness path may launch more kernels than an optimized
GEMM; it preserves the existing per-allocation bounds and mathematical complexity
but is not a GPU performance optimization.

## Results

The final matrix passed on the corrected implementation:

| Configuration | Tests | Canonical executables | Basic demos | CPU metric comparison |
| --- | --- | --- | --- | --- |
| Flex CPU | 51 + 1 doctest | 7/7 | Pass | Reference |
| WGPU / Vulkan, fusion off | 16/16 | 7/7 | 5/5 | Pass |
| WGPU / Vulkan, fusion on | 16/16 | 7/7 | 5/5 | Pass |
| CUDA, fusion off | 16/16 | 7/7 | 5/5 | Pass |
| CUDA, fusion on | 16/16 | 7/7 | 5/5 | Pass |

Basic GPU demos include `bell`, `qudit`, `train`, `hybrid`, and the bounded
4/8/12-qubit `profile` program. The separate WGPU smoke executable also passed:
expectation `0.7648422`, derivative `-0.64421755`, matching cosine/negative sine
at angle `0.7`.

| Backend | VQE energy (reference -3) | QAOA expected cut (reference 2) | Born total variation | Hybrid test MSE |
| --- | --- | --- | --- | --- |
| CPU | -2.99999952 | 1.99999952 | 1.138e-6 | 0.00009461 |
| WGPU | -2.99999762 | 1.99999690 | 1.165e-6 | 0.00009461 |
| WGPU + fusion | -2.99999762 | 1.99999690 | 1.237e-6 | 0.00009461 |
| CUDA | -3.00000095 | 1.99999952 | 1.099e-6 | 0.00009725 |
| CUDA + fusion | -3.00000072 | 1.99999952 | 1.033e-6 | 0.00009773 |

All three classifiers reached 100% accuracy on their stated disjoint synthetic
test sets in every configuration. Small excursions beyond exact energy bounds
are F32 roundoff and remain within each example's declared acceptance tolerance.
Maximum absolute reported-metric differences from CPU were `2.742e-6` for WGPU,
`2.623e-6` for WGPU fusion, `0.001385` for CUDA, and `0.001415` for CUDA fusion.
The larger CUDA differences are the hybrid encoder's parameter-change diagnostic;
the table shows the corresponding task-loss agreement.

Strict Clippy, formatting, default/all-feature documentation, isolated core
feature checks, and independent core-package compilation also passed. The report
comparator was checked to reject a deliberately corrupted energy even with
Python optimization enabled. Full example metrics, hardware metadata, comparison
results and the lockfile hash are preserved in
[`backend-results.json`](backend-results.json). Raw runtime logs/reports remain
under ignored `artifacts/validation/` and `artifacts/examples/` directories.

## Automation and boundaries

Ordinary CI runs the CPU suite and compiles both packages with GPU features.
`.github/workflows/gpu.yml` adds manual hardware execution on a runner labeled
`self-hosted, linux, x64, gpu`, covering WGPU/CUDA with fusion off/on. The job
runs on a host with a usable NVIDIA graphics driver and CUDA toolkit/runtime
(including NVRTC and CUDA headers), required by the CUDA half of the matrix. It
runs the CPU baseline, hardware tests and example comparisons, fails on missing
hardware, and uploads logs/reports. No remote CI run is implied by local results.

This is numerical/runtime qualification of small dense simulations and learning
examples on one NVIDIA device. It does not measure GPU speedup or certify other
vendors, Metal, DirectX, browser WebGPU, integrated adapters, F64 GPU execution,
or large-state memory behavior. Hardware was concurrently used by unrelated
applications; no performance conclusions are drawn from these runs. Dense state
storage remains exponential, and these checks do not establish quantum advantage.
