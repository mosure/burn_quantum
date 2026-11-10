//! Checked native-buffer interoperability between Burn and Bevy's WGPU renderer.
//!
//! Both libraries use WGPU 30. Registration shares the renderer's real device
//! and queue. Export does not read tensor values back or copy them: it accepts
//! only existing contiguous, untiled F32 storage. The allocation remains pinned
//! through the caller's GPU submission. A renderer must bind it read-only.
//!
//! This optional path is separate from the default gallery's CPU simulation.
//! It does not imply that host tableau/decoder work runs on a GPU, or that
//! uploading a CPU-generated visualization trace becomes zero-copy.
use bevy::render::renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue};
use burn::tensor::{DType, Device, Tensor};
use burn_cubecl::{CubeBackend, CubeDevice, cubecl};
use burn_wgpu::{CubeTensor, RuntimeOptions, WgpuResource, WgpuSetup, wgpu};
use std::{any::Any, num::NonZeroU64};

/// Recoverable interoperability validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgeError(pub String);
impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BridgeError {}

/// One registered Burn runtime sharing Bevy's exact device and queue.
/// Register once per renderer, then retain or clone this value; registration
/// does not create a new GPU device. No adapter is selected implicitly.
#[derive(Clone)]
pub struct SharedGpuBridge {
    device: Device,
    cube_device: CubeDevice,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
}
impl SharedGpuBridge {
    /// Import the renderer's coherent WGPU handles. Access `RenderInstance`
    /// in the render world, where Bevy installs it. Do not register this setup
    /// separately through another Burn initialization path.
    pub fn from_renderer(
        instance: &RenderInstance,
        adapter: &RenderAdapter,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> Result<Self, BridgeError> {
        Self::from_setup(WgpuSetup {
            instance: instance.clone().into_inner(),
            adapter: adapter.clone().into_inner(),
            device: device.wgpu_device().clone(),
            queue: queue.clone().into_inner(),
            backend: adapter.get_info().backend,
        })
    }
    /// Import an already coherent setup shared with a renderer. Device/adapter
    /// capabilities are checked by CubeCL; all handles must belong together.
    pub fn from_setup(setup: WgpuSetup) -> Result<Self, BridgeError> {
        let limits = setup.device.limits();
        let queue = setup.queue.clone();
        let imported = burn_wgpu::try_init_device(setup, RuntimeOptions::default())
            .map_err(|e| BridgeError(e.to_string()))?;
        let cube_device = CubeDevice::Wgpu(imported);
        Ok(Self {
            device: Device::new(cube_device.clone()),
            cube_device,
            queue,
            limits,
        })
    }
    /// Device on which inputs and GPU quantum computations must be created.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Export an existing F32 tensor as read-only storage without materializing
    /// a host vector. Shape must match exactly and be nonempty. Strided/tiled,
    /// quantized, autodiff, fusion handles and other devices are rejected.
    /// The caller can explicitly materialize an appropriate tensor beforehand.
    pub fn export<const D: usize>(
        &self,
        tensor: Tensor<D>,
        expected_shape: [usize; D],
    ) -> Result<TensorBufferLease, BridgeError> {
        if tensor.dims() != expected_shape || expected_shape.contains(&0) {
            return Err(BridgeError("nonempty exact tensor shape required".into()));
        }
        let elements = expected_shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| BridgeError("tensor element count overflow".into()))?;
        let bytes = elements
            .checked_mul(4)
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| BridgeError("tensor byte count overflow".into()))?;
        let primitive: CubeTensor = tensor
            .try_into_primitive::<CubeBackend>()
            .map_err(|e| BridgeError(format!("an eager CubeCL tensor is required: {e:?}")))?;
        if primitive.device != self.cube_device {
            return Err(BridgeError(
                "tensor belongs to a different GPU runtime".into(),
            ));
        }
        if primitive.dtype != DType::F32 || primitive.qparams.is_some() {
            return Err(BridgeError(
                "only ordinary F32 tensors may be exported".into(),
            ));
        }
        if primitive.meta.is_tiled() || !primitive.is_contiguous() {
            return Err(BridgeError("tensor must be contiguous and untiled".into()));
        }
        let managed = primitive
            .client
            .get_resource::<cubecl::wgpu::WgpuServer<burn_wgpu::AutoCompiler>>(
                primitive.handle.clone(),
            )
            .map_err(|e| BridgeError(e.to_string()))?;
        primitive
            .client
            .flush()
            .map_err(|e| BridgeError(e.to_string()))?;
        let resource = managed.resource();
        validate_binding(resource, bytes, &self.limits)?;
        Ok(TensorBufferLease {
            resource: resource.clone(),
            bytes,
            shape: expected_shape.to_vec(),
            queue: self.queue.clone(),
            _allocation: Box::new(managed),
            _tensor: primitive,
        })
    }
}
fn validate_binding(
    resource: &WgpuResource,
    bytes: u64,
    limits: &wgpu::Limits,
) -> Result<(), BridgeError> {
    let end = resource
        .offset
        .checked_add(bytes)
        .ok_or_else(|| BridgeError("buffer range overflow".into()))?;
    if bytes == 0 || bytes > resource.size || end > resource.buffer.size() {
        return Err(BridgeError(
            "tensor bytes exceed its owned buffer range".into(),
        ));
    }
    if !resource
        .offset
        .is_multiple_of(u64::from(limits.min_storage_buffer_offset_alignment))
        || !bytes.is_multiple_of(4)
        || bytes > limits.max_storage_buffer_binding_size
    {
        return Err(BridgeError(
            "storage binding alignment or size limit".into(),
        ));
    }
    if !resource
        .buffer
        .usage()
        .contains(wgpu::BufferUsages::STORAGE)
    {
        return Err(BridgeError("tensor buffer lacks STORAGE usage".into()));
    }
    Ok(())
}

/// Read-only native storage view retaining both the tensor and allocator lease.
/// Use `submit_read` to keep the allocation alive until submitted GPU work is
/// complete. Merely cloning the raw WGPU buffer does not retain Burn's allocator
/// ownership. Never write this resource or submit it on another device's queue.
pub struct TensorBufferLease {
    resource: WgpuResource,
    bytes: u64,
    shape: Vec<usize>,
    queue: wgpu::Queue,
    _allocation: Box<dyn Any + Send + Sync>,
    _tensor: CubeTensor,
}
impl TensorBufferLease {
    /// Logical row-major tensor dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Exact logical bytes; allocator padding is excluded.
    pub fn byte_len(&self) -> u64 {
        self.bytes
    }
    /// WGPU binding for a `var<storage, read>` shader declaration.
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.resource.buffer,
            offset: self.resource.offset,
            size: NonZeroU64::new(self.bytes),
        })
    }
    /// Raw shared buffer and byte offset for a GPU copy/read-only pipeline.
    /// The allocation lease must outlive GPU execution, not only command encoding.
    pub fn buffer_range(&self) -> (&wgpu::Buffer, u64, u64) {
        (&self.resource.buffer, self.resource.offset, self.bytes)
    }
    /// Submit read-only commands on the shared queue and retain ownership until
    /// WGPU reports completion. Native applications must keep polling the device
    /// (Bevy does this); browsers drive progress through their event loop.
    pub fn submit_read(
        self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        let queue = self.queue.clone();
        let index = queue.submit(commands);
        queue.on_submitted_work_done(move || drop(self));
        index
    }
}
