use nexus::dynamics::GpuBodySet;
use slang_hal::backend::{Backend, Encoder};
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuTensor};
use crate::grid::grid::{GpuGrid, GpuGridMetadata};
use crate::solver::{GpuParticleModelData, GpuParticles, GpuSimulationParams, ParticleDynamics};

#[derive(Copy, Clone, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuTimestepBounds {
    compute_max_dt_as_uint: u32,
}

impl GpuTimestepBounds {
    // NOTE: this **MUST** match the constant in the GPU slang shader.
    const FLOAT_TO_INT: f32 = 1.0e12;
    pub fn new() -> GpuTimestepBounds {
        Self {
            compute_max_dt_as_uint: 0,
        }
    }

    pub fn computed_dt(&self) -> f32 {
        self.compute_max_dt_as_uint as f32 / Self::FLOAT_TO_INT
    }
}

#[derive(Shader)]
#[shader(
    module = "slosh::solver::timestep_bound",
    specialize = ["slosh::models::specializations"]
)]
pub struct WgTimestepBounds<B: Backend> {
    pub reset_timestep_bound: GpuFunction<B>,
    pub estimate_timestep_bound: GpuFunction<B>,
}

#[derive(ShaderArgs)]
pub struct TimestepBoundsArgs<'a, B: Backend, GpuModel: GpuParticleModelData> {
    grid: &'a GpuTensor<GpuGridMetadata, B>,
    particles_model: &'a GpuTensor<GpuModel, B>,
    particles_dyn: &'a GpuTensor<ParticleDynamics, B>,
    particles_len: &'a GpuScalar<u32, B>,
    result: &'a GpuScalar<GpuTimestepBounds, B>,
}

impl<B: Backend> WgTimestepBounds<B> {
    pub async fn compute_bounds<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        bounds: &GpuScalar<GpuTimestepBounds, B>,
        bounds_staging: &mut GpuScalar<GpuTimestepBounds, B>,
    ) -> Result<f32, B::Error> {
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        self.launch(backend, &mut pass, grid, particles, bounds)?;
        drop(pass);
        bounds_staging.copy_from_view(&mut encoder, bounds)?;
        backend.submit(encoder)?;

        let mut result = [GpuTimestepBounds::new()];
        backend.read_buffer(bounds_staging.buffer(), &mut result).await?;
        Ok(result[0].computed_dt())
    }

    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        bounds: &GpuScalar<GpuTimestepBounds, B>,
    ) -> Result<(), B::Error> {
        let args = TimestepBoundsArgs {
            grid: &grid.meta,
            particles_model: particles.models(),
            particles_dyn: particles.dynamics(),
            particles_len: particles.gpu_len(),
            result: bounds,
        };
        self.reset_timestep_bound.launch(backend, pass, &args, [1; 3])?;
        self.estimate_timestep_bound
            .launch(backend, pass, &args, [particles.len() as u32, 1, 1])
    }
}