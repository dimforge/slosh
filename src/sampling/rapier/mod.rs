#[cfg(feature = "dim2")]
mod sample_polyline;
#[cfg(feature = "dim3")]
mod sample_trimesh;

#[cfg(feature = "dim2")]
pub use sample_polyline::*;
#[cfg(feature = "dim3")]
pub use sample_trimesh::*;

#[derive(Copy, Clone, Debug)]
pub struct SamplingParams {
    pub base_vid: u32,
    pub collider_id: u32,
    pub sampling_step: f32,
}
