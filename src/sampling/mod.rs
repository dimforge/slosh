#[cfg(feature = "dim2")]
pub use sample_polyline::*;
#[cfg(feature = "dim3")]
pub use sample_trimesh::*;

#[cfg(feature = "dim2")]
mod sample_polyline;
#[cfg(feature = "dim3")]
mod sample_trimesh;
