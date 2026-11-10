//! A real GPU render proof: Burn computes the color; a fragment shader reads its
//! exact native allocation. Only the finished RGBA image is read back for testing.
#![cfg(all(feature = "gpu-bridge", not(target_arch = "wasm32")))]
use bevy::render::renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue};
use burn::tensor::{DType, Tensor};
use burn_quantum_visualization::bridge::SharedGpuBridge;
use burn_wgpu::wgpu;

#[test]
#[ignore = "requires a real native Vulkan GPU; run explicitly with --ignored"]
fn burn_buffer_is_sampled_by_shared_renderer_without_host_value_transfer() {
    pollster::block_on(run());
}
async fn run() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Burn-Bevy shared proof"),
            required_features: adapter.features() & wgpu::Features::SHADER_F16,
            ..Default::default()
        })
        .await
        .unwrap();
    eprintln!("bridge adapter: {:?}", adapter.get_info());
    let renderer = RenderDevice::new(device.clone());
    let bridge = SharedGpuBridge::from_renderer(
        &RenderInstance::new(instance),
        &RenderAdapter::new(adapter),
        &renderer,
        &RenderQueue::new(queue),
    )
    .unwrap();
    let color = Tensor::<1>::from_floats([0.1, 0.2, 0.3, 0.5], bridge.device()) * 2.0;
    assert!(bridge.export(color.clone(), [3]).is_err());
    assert!(bridge.export(color.clone().cast(DType::F16), [4]).is_err());
    let strided = Tensor::<2>::from_floats([[0., 1.], [2., 3.]], bridge.device()).transpose();
    assert!(bridge.export(strided, [2, 2]).is_err());
    let lease = bridge.export(color.clone(), [4]).unwrap();
    assert_eq!(lease.shape(), &[4]);
    assert_eq!(lease.byte_len(), 16);
    // An alias is transformed after export. The retained tensor and allocation
    // lease must prevent this owned operation from overwriting the exported data.
    let alias = color + 20.0;
    let alias_lease = bridge.export(alias, [4]).unwrap();
    drop(alias_lease);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shared Burn color"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<storage,read> color:array<f32>;
@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
    let p=array<vec2<f32>,3>(vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));
    return vec4(p[i],0.,1.);
}
@fragment fn fragment()->@location(0) vec4<f32> {
    return vec4(color[0],color[1],color[2],color[3]);
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("native tensor storage render"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pinned Burn allocation"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: lease.binding(),
        }],
    });
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bridge proof image"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("finished image oracle"),
        size: 1024,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = device.create_command_encoder(&Default::default());
    {
        let mut pass = commands.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("draw Burn color"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    commands.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    lease.submit_read([commands.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let data = readback.slice(..).get_mapped_range().unwrap();
    for y in 0..4 {
        for x in 0..4 {
            assert_eq!(
                &data[y * 256 + x * 4..y * 256 + x * 4 + 4],
                &[51, 102, 153, 255]
            );
        }
    }
}
