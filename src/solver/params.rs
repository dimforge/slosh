use bytemuck::{Pod, Zeroable};
use slang_hal::backend::Backend;
use stensor::tensor::{GpuScalar, GpuTensor};
use wgpu::BufferUsages;

#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct SimulationParams {
    #[cfg(feature = "dim2")]
    pub gravity: nalgebra::Vector2<f32>,
    #[cfg(feature = "dim2")]
    pub padding: f32,
    #[cfg(feature = "dim3")]
    pub gravity: nalgebra::Vector3<f32>,
    pub dt: f32,
}

pub struct GpuSimulationParams<B: Backend> {
    pub params: GpuScalar<SimulationParams, B>,
}

impl<B: Backend> GpuSimulationParams<B> {
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
