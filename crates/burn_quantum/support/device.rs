//! Shared executable/test device selection, intentionally outside the library API.
#![allow(dead_code)]
use burn::tensor::Device;
use std::sync::OnceLock;

/// Runtime chosen once per process, so tests share one initialized GPU client.
#[derive(Clone, Debug)]
pub struct Execution {
    pub device: Device,
    pub backend: &'static str,
    pub adapter: String,
}

/// Select CPU (default), WGPU hardware, or CUDA explicitly; never fall back.
pub fn execution() -> Result<Execution, String> {
    static EXECUTION: OnceLock<Result<Execution, String>> = OnceLock::new();
    EXECUTION
        .get_or_init(|| {
            let backend = match std::env::var("BURN_QUANTUM_DEVICE") {
                Ok(value) => value,
                Err(std::env::VarError::NotPresent) => "cpu".into(),
                Err(error) => return Err(error.to_string()),
            };
            let index = std::env::var("BURN_QUANTUM_DEVICE_INDEX")
                .unwrap_or_else(|_| "0".into())
                .parse::<usize>()
                .map_err(|_| "BURN_QUANTUM_DEVICE_INDEX must be a nonnegative integer")?;
            let selected = match backend.as_str() {
                "cpu" => Execution {
                    device: Device::flex(),
                    backend: "cpu",
                    adapter: "Flex CPU".into(),
                },
                "wgpu" | "wgpu-integrated" | "wgpu-cpu" => wgpu(&backend, index)?,
                "cuda" => cuda(index)?,
                _ => return Err(format!("unknown BURN_QUANTUM_DEVICE: {backend}")),
            };
            eprintln!(
                "Backend: {}; adapter: {}",
                selected.backend, selected.adapter
            );
            Ok(selected)
        })
        .clone()
}

#[cfg(feature = "wgpu")]
fn wgpu(backend: &str, index: usize) -> Result<Execution, String> {
    use burn::tensor::{DeviceKind, wgpu::wgpu::DeviceType};
    let (kind, expected, label) = match backend {
        "wgpu" => (
            DeviceKind::DiscreteGpu(index),
            DeviceType::DiscreteGpu,
            "wgpu",
        ),
        "wgpu-integrated" => (
            DeviceKind::IntegratedGpu(index),
            DeviceType::IntegratedGpu,
            "wgpu-integrated",
        ),
        _ => (DeviceKind::Cpu, DeviceType::Cpu, "wgpu-cpu"),
    };
    let (device, setup) = Device::wgpu_options()
        .device_kind(kind)
        .init_with_setup()
        .map_err(|e| format!("{label} initialization failed: {e}"))?;
    let info = setup.adapter.get_info();
    if info.device_type != expected {
        return Err(format!(
            "requested {expected:?}, received {:?}",
            info.device_type
        ));
    }
    Ok(Execution {
        device,
        backend: label,
        adapter: format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend),
    })
}

#[cfg(not(feature = "wgpu"))]
fn wgpu(_: &str, _: usize) -> Result<Execution, String> {
    Err("WGPU requested but not compiled; enable --features wgpu".into())
}

#[cfg(feature = "cuda")]
fn cuda(index: usize) -> Result<Execution, String> {
    let device = Device::cuda(index);
    let identity = device
        .identity()
        .ok_or("CUDA device identity unavailable")?;
    Ok(Execution {
        device,
        backend: "cuda",
        adapter: identity.name,
    })
}

#[cfg(not(feature = "cuda"))]
fn cuda(_: usize) -> Result<Execution, String> {
    Err("CUDA requested but not compiled; enable --features cuda".into())
}
