# Documentation

## Design and development

- [Repository organization](organization.md): module ownership, public facades,
  test structure and automatic example discovery.
- [Architecture](architecture.md): Burn integration, mathematical contracts,
  execution layers and resource boundaries.
- [Roadmap](roadmap.md): implemented capabilities and future engineering work.
- [Contributing](../CONTRIBUTING.md): where to put changes and how to validate them.
- [Changelog](../CHANGELOG.md): release contents and migration notes.

## Algorithms and usage

- [Example suite](../crates/burn_quantum_example/README.md): one file per canonical application.
- [Boolean SAT](sat.md): CNF/DIMACS, reversible oracles and bounded search semantics.
- [Quantum natural SPSA](qnspsa.md): fidelity geometry, optimizer API, checkpoints and numerical limits.

## Validation evidence

Each record applies to its stated source snapshot, commands and hardware. An
older record's file paths and hashes can predate subsequent reorganizations.

| Record | Scope |
| --- | --- |
| [Organization regression results](organization-results.json) | Current module split, CPU/GPU regressions, example discovery and source provenance |
| [QN-SPSA results](qnspsa-results.json) | Analytic geometry, optimizer semantics and four GPU configurations when introduced |
| [SAT results](sat-results.json) | Oracle/search qualification when introduced |
| [Backend qualification](backend-validation.md), [raw results](backend-results.json) | Original seven-example CPU/WGPU/CUDA runtime matrix and precision corrections |
| [Example qualification](examples-validation.md) | Original canonical QML suite |
| [Initial simulator validation](validation.md) | Initial implementation and mathematical reference checks |

Local command output and generated reports live under ignored `artifacts/`;
compiled code and packaged archives live under ignored `target/`.
