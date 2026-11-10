#!/usr/bin/env bash
# Build the executable WebGPU gallery. No deployment or Pages configuration change.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
# Avoid serializing native application builds behind the long cross-target build.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/viz-wasm}"
web_output="${1:-target/visualization-web}"
web_profile="${BURN_QUANTUM_WEB_PROFILE:-web-release}"
if [[ $# -gt 1 ]]; then
  echo "usage: scripts/build_visualization_web.sh [output-directory]" >&2
  exit 2
fi
case "$web_profile" in
  release|web-release) ;;
  *) echo "BURN_QUANTUM_WEB_PROFILE must be release or web-release" >&2; exit 2 ;;
esac
command -v cargo >/dev/null || { echo "cargo is required" >&2; exit 1; }
command -v wasm-bindgen >/dev/null || { echo "Install wasm-bindgen-cli matching Cargo.lock before building." >&2; exit 1; }
locked_bindgen="$(python3 - <<'PY'
import tomllib
with open('Cargo.lock','rb') as f:
    lock=tomllib.load(f)
versions={p['version'] for p in lock['package'] if p['name']=='wasm-bindgen'}
if len(versions)!=1:
    raise SystemExit('expected one locked wasm-bindgen version')
print(versions.pop())
PY
)"
installed_bindgen="$(wasm-bindgen --version)"
if [[ "$installed_bindgen" != "wasm-bindgen $locked_bindgen" ]]; then
  echo "wasm-bindgen-cli mismatch: Cargo.lock requires $locked_bindgen, found $installed_bindgen" >&2
  echo "Install with: cargo install wasm-bindgen-cli --version $locked_bindgen --locked" >&2
  exit 1
fi
cargo build --locked --profile "$web_profile" -p burn_quantum_visualization --bin quantum_explorer --target wasm32-unknown-unknown
mkdir -p "$web_output/pkg"
wasm-bindgen --target web --no-typescript --remove-name-section \
  --out-dir "$web_output/pkg" --out-name burn_quantum_visualization \
  "${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/$web_profile/quantum_explorer.wasm"
cp web/index.html "$web_output/index.html"
touch "$web_output/.nojekyll"
python3 - "$web_output" "$web_profile" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);files=['index.html','pkg/burn_quantum_visualization.js','pkg/burn_quantum_visualization_bg.wasm']
record={'schema':1,'cargo_profile':sys.argv[2],'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),
        'wasm_bindgen':subprocess.check_output(['wasm-bindgen','--version'],text=True).strip(),
        'files':{name:{'bytes':(out/name).stat().st_size,'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()} for name in files}}
(out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
print(f"Built {out.resolve()}")
print(f"Serve with: python3 -m http.server --directory {out} 8080")
PY
