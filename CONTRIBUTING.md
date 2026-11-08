# Contributing

Start with the [repository map](docs/organization.md) and
[architecture](docs/architecture.md). The core library is the root package;
`crates/burn_quantum_example` is the executable application suite.

## Place changes by responsibility

- Add reusable quantum operations, mathematical contracts and validation to
  the relevant core module. Keep task-specific models, data, training loops and
  evaluation in a single example binary.
- Extend private implementation modules behind the existing public exports.
  Keep validation next to the type or operation whose invariants it protects.
  Share helpers when their mathematical contract is actually common.
- Keep device selection and environment variables in executable/test support.
  Library callers pass native Burn tensors and devices explicitly.
- Keep independent test oracles outside production code. Test observable
  behavior, physical invariants, invalid inputs, gradients and numerical
  agreement rather than reproducing implementation details.
- Circuit or checkpoint schema changes need explicit version handling and
  compatibility checks. Moving implementation files does not require a new schema.

The workspace defines common package metadata, dependency versions and lints.
Select execution features on the package that needs them; check the core
separately so example/dev features do not hide a missing library feature.

## Add an application example

Create `crates/burn_quantum_example/src/bin/<name>.rs`. Keep its problem, circuit,
training/fitting/search procedure, independent evaluation, `main`, and acceptance
test together. Use shared device selection and JSON reporting from the example
library. Return an error when acceptance criteria fail, and make seeds, budgets
and reference values explicit. Reusable quantum preprocessing/readouts belong
in the core; generic experiment reporting remains in the example library.

Add the example to its README table. Cargo discovers the binary, and both runner
and comparison scripts discover Cargo's binary targets automatically. Add any
new metric-specific comparison tolerance only with an explained numerical or
statistical reason.

## Validate

```sh
bash scripts/validate.sh
bash scripts/run_examples.sh cpu
```

The first command checks formatting, strict Clippy, all workspace tests,
doctests, isolated core feature combinations, documentation and basic demos.
The second exercises the current Cargo-discovered example catalog and saves
JSON reports. Focused tests can use `--test sat` or `--test qnspsa`, optionally
with their module names as filters.

When changing device execution, run the hardware matrix on available hardware:

```sh
bash scripts/validate_backend.sh wgpu
bash scripts/validate_backend.sh wgpu --fusion
bash scripts/validate_backend.sh cuda
bash scripts/validate_backend.sh cuda --fusion
python3 -B scripts/compare_backends.py artifacts/examples/cpu \
    artifacts/examples/wgpu artifacts/examples/wgpu-fusion \
    artifacts/examples/cuda artifacts/examples/cuda-fusion
```

Feature compilation alone does not qualify GPU execution. Record the actual
adapter, precision, fusion settings, commands and limits of each validation run.
Keep local logs in `artifacts/`; commit concise evidence in `docs/` when it
substantiates a capability or numerical claim. Preserve historical results as
snapshots rather than relabeling them after later source changes.

Verify the standalone library package when changing workspace or packaging rules:

```sh
cargo package --locked -p burn_quantum
```

For a local uncommitted verification, add `--allow-dirty`; publication is a
separate operation. The example package remains `publish = false`.
