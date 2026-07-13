//! Surface sampling for rigid body coupling.
//!
//! Samples particles on the surfaces of rigid body colliders for two-way
//! MPM-rigid body coupling. In 2D, samples polyline edges; in 3D, samples
//! triangle mesh surfaces.

#[cfg(feature = "rapier")]
mod rapier;
#[cfg(feature = "rapier")]
pub use rapier::*;

use encase::ShaderType;

#[cfg(feature = "dim2")]
#[derive(Copy, Clone, Debug, ShaderType)]
#[repr(C)]
pub struct GpuSampleIds {
    pub segment: glam::UVec2,
    pub collider: u32,
}

#[cfg(feature = "dim3")]
#[derive(Copy, Clone, Debug, ShaderType)]
#[repr(C)]
pub struct GpuSampleIds {
    pub triangle: glam::UVec3,
    pub collider: u32,
}
