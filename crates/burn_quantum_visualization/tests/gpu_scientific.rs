//! Explicit real-device comparison of every scientific demo field.
#![cfg(all(feature = "gpu-bridge", not(target_arch = "wasm32")))]

use burn::tensor::{Device, DeviceKind, wgpu::wgpu::DeviceType};
use burn_quantum_visualization::{
    demo::{self, DemoKind},
    simulation::TraceLimits,
};
use serde_json::Value;

/// Compare complete documents, retaining exact structure and integer counters.
/// F32 numeric fields receive the same absolute tolerance, including the full
/// QN-SPSA parameter/metric trajectory rather than only its final energy.
fn compare(left: &Value, right: &Value, path: &str, max_error: &mut f64) {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            let error = (a - b).abs();
            assert!(a.is_finite() && b.is_finite());
            assert!(error <= 5e-4, "{path}: CPU {a}, GPU {b}, error {error}");
            *max_error = max_error.max(error);
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "array length at {path}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                compare(a, b, &format!("{path}[{index}]"), max_error);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (key, a) in a {
                compare(a, &b[key], &format!("{path}.{key}"), max_error);
            }
        }
        _ => assert_eq!(left, right, "exact field at {path}"),
    }
}

#[test]
#[ignore = "requires an actual native discrete WGPU device; run explicitly with --ignored"]
fn all_six_scientific_demo_traces_match_flex_on_real_wgpu() {
    let (gpu, setup) = Device::wgpu_options()
        .device_kind(DeviceKind::DiscreteGpu(0))
        .init_with_setup()
        .expect("initialize actual discrete GPU");
    let info = setup.adapter.get_info();
    assert_eq!(info.device_type, DeviceType::DiscreteGpu);
    eprintln!(
        "scientific parity adapter: {} ({:?})",
        info.name, info.backend
    );
    let cpu = Device::flex();
    pollster::block_on(async {
        for &kind in DemoKind::all() {
            let cpu_doc = demo::build(kind, &cpu, TraceLimits::default())
                .await
                .unwrap();
            let gpu_doc = demo::build(kind, &gpu, TraceLimits::default())
                .await
                .unwrap();
            cpu_doc.validate().unwrap();
            gpu_doc.validate().unwrap();
            assert_ne!(cpu_doc.trace.backend, gpu_doc.trace.backend);
            let mut cpu_value = serde_json::to_value(&cpu_doc).unwrap();
            let mut gpu_value = serde_json::to_value(&gpu_doc).unwrap();
            // Compute-device provenance is intentionally different; every
            // scientific value, label, binding and cost still participates.
            cpu_value["trace"]["backend"] = Value::Null;
            gpu_value["trace"]["backend"] = Value::Null;
            let mut max_error = 0.0;
            compare(&cpu_value, &gpu_value, kind.id(), &mut max_error);
            eprintln!(
                "{}: {} frames, maximum absolute numeric error {max_error:e}",
                kind.id(),
                gpu_doc.trace.frames.len()
            );
        }
    });
}
