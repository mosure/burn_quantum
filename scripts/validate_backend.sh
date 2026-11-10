#!/usr/bin/env bash
# Runtime qualification. A requested unavailable GPU is an error, never a skip.
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-cpu}"
features=()
case "$backend" in
    cpu) ;;
    wgpu|wgpu-integrated|wgpu-cpu) features=(--features wgpu) ;;
    cuda) features=(--features cuda) ;;
    *) echo "Unknown backend: $backend" >&2; exit 2 ;;
esac
suffix=""
if [[ "${2:-}" == "--fusion" ]]; then
    features+=(--features fusion)
    suffix="-fusion"
elif [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/validate_backend.sh [cpu|wgpu|wgpu-integrated|wgpu-cpu|cuda] [--fusion]" >&2
    exit 2
fi
export BURN_QUANTUM_DEVICE="$backend"
output="artifacts/validation/$backend$suffix"
mkdir -p "$output"
# Discover tests that explicitly select the requested runtime device. Host-only
# protocol/routing/decoder tests are covered by the full workspace CPU run.
# Cargo auto-discovers new test files; keep the GPU matrix equally automatic.
mapfile -t runtime_tests < <(python3 - <<'PYTESTS'
from pathlib import Path
for path in sorted(Path("crates/burn_quantum/tests").glob("*.rs")):
    if "support/device.rs" in path.read_text():
        print(path.stem)
PYTESTS
)
test_args=()
for name in "${runtime_tests[@]}"; do test_args+=(--test "$name"); done
if [[ ${#test_args[@]} -eq 0 ]]; then
    echo "No runtime device tests discovered" >&2
    exit 2
fi
cargo test --locked -p burn_quantum "${features[@]}" "${test_args[@]}" -- --test-threads=1 --nocapture 2>&1 | tee "$output/parity.log"
cargo test --locked -p burn_quantum_example "${features[@]}" --bins -- --test-threads=1 --nocapture 2>&1 | tee "$output/example-tests.log"
bash scripts/run_examples.sh "$@" 2>&1 | tee "$output/examples.log"
for example in bell qudit train hybrid profile; do
    cargo run --locked -p burn_quantum "${features[@]}" --example "$example" 2>&1 | tee "$output/$example.log"
done
