//! Optional real-device WGPU forward and backward numerical smoke test.
use burn::tensor::{Device, DeviceKind, Tensor, wgpu::wgpu::DeviceType};
use burn_quantum::{Angle, Circuit, DenseSimulator, Observable, Register, Rotation, gate::Axis};
fn main() -> burn_quantum::Result<()> {
    let (device, setup) = Device::wgpu_options()
        .device_kind(DeviceKind::DiscreteGpu(0))
        .init_with_setup()
        .map_err(|e| burn_quantum::Error::Unsupported(e.to_string()))?;
    let adapter = setup.adapter.get_info();
    assert_eq!(adapter.device_type, DeviceType::DiscreteGpu);
    println!("Adapter: {} ({:?})", adapter.name, adapter.backend);
    let device = device.autodiff();
    let mut circuit = Circuit::new(Register::qubits(1)?, 1, 0);
    circuit.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))?;
    let theta = Tensor::<1>::from_floats([0.7], &device).require_grad();
    let engine = DenseSimulator::default();
    let state = circuit.run(&engine, Some(theta.clone()), None, 1, &device)?;
    let z = Observable::z(0).expectation(&engine, &state)?;
    let value = z.clone().into_scalar::<f32>();
    let gradients = z.backward();
    let derivative = theta.grad(&gradients).unwrap().into_scalar::<f32>();
    assert!((value - 0.7f32.cos()).abs() < 1e-4);
    assert!((derivative + 0.7f32.sin()).abs() < 1e-4);
    println!("WGPU: <Z>={value}, derivative={derivative}");
    Ok(())
}
