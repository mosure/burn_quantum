#!/usr/bin/env bash
# Deterministic local qualification; logs are ignored build artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/validation
cargo fmt --all --check 2>&1 | tee artifacts/validation/fmt.log
cargo clippy --locked --workspace --all-targets -- -D warnings 2>&1 | tee artifacts/validation/clippy.log
cargo test --locked --workspace --all-targets 2>&1 | tee artifacts/validation/tests.log
cargo test --locked --workspace --doc 2>&1 | tee artifacts/validation/doctests.log
cargo check --locked -p burn_quantum --no-default-features 2>&1 | tee artifacts/validation/no-backend.log
cargo check --locked -p burn_quantum --no-default-features --features flex 2>&1 | tee artifacts/validation/inference.log
cargo check --locked -p burn_quantum --features wgpu,fusion --all-targets 2>&1 | tee artifacts/validation/wgpu-compile.log
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps 2>&1 | tee artifacts/validation/docs.log
cargo run --locked -p burn_quantum --example bell 2>&1 | tee artifacts/validation/bell.log
cargo run --locked -p burn_quantum --example qudit 2>&1 | tee artifacts/validation/qudit.log
cargo run --locked -p burn_quantum --example train 2>&1 | tee artifacts/validation/train.log
cargo run --locked -p burn_quantum --example hybrid 2>&1 | tee artifacts/validation/hybrid.log
