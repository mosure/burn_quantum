//! Two shared analytic surface textures, uploaded once and reused by every Bloch panel.
//! A shaded orthographic sphere needs no per-qubit render target, mesh or camera.
use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
#[derive(Resource)]
pub(crate) struct BlochSurfaces {
    pub sphere: Handle<Image>,
    pub disk: Handle<Image>,
}
pub(crate) const SIZE: f32 = 256.;
impl FromWorld for BlochSurfaces {
    fn from_world(world: &mut World) -> Self {
        let mut images = world.resource_mut::<Assets<Image>>();
        Self {
            sphere: images.add(texture(true)),
            disk: images.add(texture(false)),
        }
    }
}
fn texture(spatial: bool) -> Image {
    let mut bytes = Vec::with_capacity((SIZE * SIZE * 4.) as usize);
    let light = Vec3::new(-0.45, 0.65, 0.85).normalize();
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let px = (x as f32 + 0.5) / SIZE * 2. - 1.;
            let py = 1. - (y as f32 + 0.5) / SIZE * 2.;
            let r2 = px * px + py * py;
            let alpha = ((1. - r2) * SIZE * 0.5).clamp(0., 1.);
            let z = (1. - r2).max(0.).sqrt();
            let shade = if spatial {
                0.23 + 0.35 * Vec3::new(px, py, z).dot(light).max(0.) + 0.09 * (1. - z).powi(3)
            } else {
                1. // Flat area encoding: no shading that could imply a radial probability gradient.
            };
            let value = (shade * 255.) as u8;
            bytes.extend([value, value, value, (alpha * 255.) as u8]);
        }
    }
    Image::new(
        Extent3d {
            width: SIZE as u32,
            height: SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}
