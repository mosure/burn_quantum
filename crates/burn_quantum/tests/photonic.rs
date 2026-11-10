//! Independent physical/compilation contracts on CPU or the explicitly selected GPU.
mod common;
#[path = "../support/device.rs"]
mod runtime;
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{photonic::*, *};
use common::{close, complex_close, normalized, values};
use num_complex::Complex64;

fn device() -> Device {
    runtime::execution()
        .expect("requested backend unavailable")
        .device
}
fn node(kind: NodeKind, storage: f64) -> FabricNode {
    FabricNode {
        kind,
        storage: RailTransmission::uniform(storage).unwrap(),
    }
}
fn link(from: usize, to: usize, ticks: u64, transmission: f64) -> FabricLink {
    FabricLink {
        from: NodeId(from),
        to: NodeId(to),
        ticks,
        transmission: RailTransmission::uniform(transmission).unwrap(),
    }
}
fn fabric() -> Fabric {
    Fabric::new(
        vec![
            node(NodeKind::Source, 1.),
            node(NodeKind::Source, 1.),
            node(NodeKind::Switch, 1.),
            node(NodeKind::Fusion, 1.),
        ],
        vec![link(0, 2, 1, 0.9), link(1, 2, 1, 0.8), link(2, 3, 2, 1.)],
    )
    .unwrap()
}
fn swap_program() -> FusionProgram {
    let mut program = FusionProgram::new();
    let a = program
        .add_resource(ResourceState::bell_pair(), NodeId(0))
        .unwrap();
    let b = program
        .add_resource(ResourceState::bell_pair(), NodeId(1))
        .unwrap();
    program.fuse(a[1], b[0], NodeId(3)).unwrap();
    program
}
fn density(data: Vec<Complex64>, sites: usize, batch: usize, dev: &Device) -> DensityMatrix {
    let register = DualRail::register(sites).unwrap();
    let shape = [batch, register.size()];
    let state = StateVector::from_amplitudes(
        register,
        ComplexTensor::new(
            Tensor::from_data(
                TensorData::new(data.iter().map(|z| z.re as f32).collect(), shape),
                dev,
            ),
            Tensor::from_data(
                TensorData::new(data.iter().map(|z| z.im as f32).collect(), shape),
                dev,
            ),
        )
        .unwrap(),
        ResourceLimits::default(),
    )
    .unwrap();
    DensityMatrix::from_pure(&state, ResourceLimits::default()).unwrap()
}
#[path = "photonic/encoding.rs"]
mod encoding;
#[path = "photonic/fusion.rs"]
mod fusion;
#[path = "photonic/packets.rs"]
mod packets;
#[path = "photonic/program.rs"]
mod program;
#[path = "photonic/fabric.rs"]
mod routing;
