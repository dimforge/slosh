use crate::math::Vector;
use bytemuck::{Pod, Zeroable};
use slang_hal::{BufferUsages, backend::Backend};
use stensor::tensor::{GpuScalar, GpuTensor};

/// Global simulation parameters applied to all particles.
///
/// These parameters control the time integration and external forces acting on
/// the simulation. They're uploaded to GPU memory once and accessed by
/// multiple kernels during each timestep.
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct SimulationParams {
    /// Gravitational acceleration vector (m/s²).
    pub gravity: Vector,
    /// Padding for GPU alignment (2D only).
    #[cfg(feature = "dim2")]
    pub padding: f32,
    /// Simulation timestep duration (seconds).
    pub dt: f32,
}

/// GPU-resident simulation parameters.
///
/// Wraps [`SimulationParams`] in a GPU uniform buffer for efficient access
/// across compute shaders.
pub struct GpuSimulationParams<B: Backend> {
    /// Uniform buffer containing simulation parameters.
    pub params: GpuScalar<SimulationParams, B>,
}

impl<B: Backend> GpuSimulationParams<B> {
    /// Uploads simulation parameters to GPU memory.
    ///
    /// Creates a uniform buffer that can be bound to compute shaders.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for buffer allocation
    /// * `params` - Simulation parameters to upload
    pub fn new(backend: &B, params: SimulationParams) -> Result<Self, B::Error> {
        Ok(Self {
            params: GpuTensor::scalar(
                backend,
                params,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
        })
    }
}
