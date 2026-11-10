# Documentation

The [interactive quantum workbench](../crates/burn_quantum_visualization/README.md)
provides Bevy/Feathers probability/phase disks, Bloch spheres, circuit diagrams,
photonic fabrics and learning views. View settings and timeline selections are
shareable through URLs. Open the [live explorer](https://mosure.github.io/burn_quantum/)
or see the [configurable catalog](visualization-catalog.md). Its
[architecture and acceptance plan](visualization-plan.md) records the scientific
contracts, native/browser delivery and GPU interoperability boundaries.

- [Current photonic implementation](photonic-status.md): local simulation expansion,
  PsiQuantum direction, acceptance evidence and explicit model boundaries.

## Design and development

- [Repository organization](organization.md): module ownership, public facades,
  test structure and automatic example discovery.
- [Architecture](architecture.md): Burn integration, mathematical contracts,
  execution layers and resource boundaries.
- [Roadmap](roadmap.md): implemented capabilities and future engineering work.
- [Photonic/FBQC roadmap](photonic-roadmap.md): representation, compiler, optical
  physics, resource factories, QEC, Quantum AI/ML and scale-out milestones.
- [Contributing](../CONTRIBUTING.md): where to put changes and how to validate them.
- [Changelog](../CHANGELOG.md): release contents and migration notes.

## Algorithms and usage

- [Core package](../crates/burn_quantum/README.md): library usage, features and packaged demos.
- [Example suite](../crates/burn_quantum_example/README.md): one file per canonical application.
- [Application benchmarks](application-benchmarks.md): real Wine classification,
  Ising VQE, exact-3SAT, photonic logical qubits, baselines and reproducible limits.
- [Boolean SAT](sat.md): CNF/DIMACS, reversible oracles and bounded search semantics.
- [Quantum natural SPSA](qnspsa.md): fidelity geometry, optimizer API, checkpoints and numerical limits.
- [Photonic foundations](photonic-fbqc.md): dual-rail encoding, loss, resource/fusion
  programs, fabric scheduling and exact branch semantics.
- [Optical models](photonic-optics.md): multimode Fock execution, detectors, sources, calibration and measured learning protocols.
- [Photonic compiler](photonic-compiler.md): causal measurement patterns, physical gate instruments, synthesis and encoded retries.
- [Clifford and QEC](photonic-stabilizer.md): scalable frame sampling, four-star/six-ring networks, encoded fusion, finite boundaries and planar logical references.
- [Growing-distance surgery](photonic-surgery.md): local merge/split/CNOT, transition detectors, physical recovery and model-specific qualification.
- [Six-ring surgery](photonic-fusion-surgery.md): actual optical instruments, boundary networks, signed logical channels and physical yield.
- [Threshold decoders](photonic-threshold-decoder.md): sparse graphs, exact erasures, union-find, fixed-shot statistics and explicit noise models.
- [Tensor networks](tensor-network.md): bounded qudit MPS, approximation accounting and distributed entangled-state ownership.
- [PsiQDK interchange](photonic-interop.md): bounded OpenQASM export and actual pinned local SDK fixtures.
- [Magic states](photonic-magic.md): measured injection/distillation channels, resource costs and declared noise assumptions.
- [Physical runtime](photonic-runtime.md): finite buffers, correlated factories,
  causal event windows, bounded retries and process workers.

## Validation evidence

Each record applies to its stated source snapshot, commands and hardware. An
older record's file paths and hashes can predate subsequent reorganizations.

| Record | Scope |
| --- | --- |
| [Surgery and thresholds](photonic-surgery-validation.md), [raw results](photonic-surgery-results.json) | Growing-distance logical instruments, noisy transitions, fixed-shot threshold families, physical six-ring boundaries and CPU/GPU oracles |
| [Local completion](photonic-completion-validation.md), [raw results](photonic-completion-results.json) | Integrated local M0–M7 paths, pinned SDK interchange, full CPU/GPU/example/package matrix at its stated snapshot |
| [Photonic milestones](photonic-milestones-validation.md), [raw results](photonic-milestones-results.json) | Physical gates, calibrated optics, four-star QEC, finite control and process execution |
| [Photonic expansion](photonic-expansion-validation.md), [raw results](photonic-expansion-results.json) | Fock/Clifford/compilation/runtime expansion, external fixtures, local CPU/GPU validation |
| [Photonic qualification](photonic-validation.md), [raw results](photonic-results.json) | Encoded photonic model, routing/compiler contracts, conditional gradients, CPU and four GPU configurations |
| [Workspace relocation](organization.md#workspace-relocation-verification) | Virtual workspace, unchanged source/dependency contracts, CPU/GPU regressions and standalone package checks |
| [Organization regression results](organization-results.json) | Internal module split before the crate relocation, CPU/GPU regressions, example discovery and source provenance |
| [QN-SPSA results](qnspsa-results.json) | Analytic geometry, optimizer semantics and four GPU configurations when introduced |
| [SAT results](sat-results.json) | Oracle/search qualification when introduced |
| [Backend qualification](backend-validation.md), [raw results](backend-results.json) | Original seven-example CPU/WGPU/CUDA runtime matrix and precision corrections |
| [Example qualification](examples-validation.md) | Original canonical QML suite |
| [Initial simulator validation](validation.md) | Initial implementation and mathematical reference checks |

Local command output and generated reports live under ignored `artifacts/`;
compiled code and packaged archives live under ignored `target/`.

The [visualization validation record](visualization-validation.md) records numerical,
GPU bridge, render and browser evidence with explicit resource limits.
