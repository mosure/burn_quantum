#!/usr/bin/env bash
# Run each single-file QML implementation and retain its JSON report.
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-${BURN_QUANTUM_DEVICE:-cpu}}"
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
    echo "Usage: bash scripts/run_examples.sh [cpu|wgpu|wgpu-integrated|wgpu-cpu|cuda] [--fusion]" >&2
    exit 2
fi
export BURN_QUANTUM_DEVICE="$backend"
output="artifacts/examples/$backend$suffix"
mkdir -p "$output"
example_catalog="$(python3 scripts/example_catalog.py)"
while IFS= read -r example; do
    cargo run --locked --quiet -p burn_quantum_example "${features[@]}" --bin "$example" | tee "$output/$example.json"
done <<< "$example_catalog"
