use bytemuck::{Pod, Zeroable};
use nalgebra::Vector4;
use slang_hal::backend::{Backend, Encoder};
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use slosh::grid::grid::{GpuGrid, GpuGridMetadata};
use slosh::solver::{GpuParticleModelData, GpuParticles, GpuSimulationParams, ParticleDynamics, ParticlePosition, SimulationParams};
use stensor::tensor::GpuTensor;
use wgpu::BufferUsages;

#[cfg(feature = "dim2")]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ReadbackData {
    pub color: Vector4<f32>,
    pub deformation: nalgebra::Matrix2<f32>,
    pub position: nalgebra::Vector2<f32>,
    // NOTE: for now we are using explicit padding since
    //       gpu buffer read based on Pod/bytemuck is much
    //       faster (about 20x) than with ShaderType/encase.
    pub pad: [f32; 2],
}

#[cfg(feature = "dim3")]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ReadbackData {
    pub color: Vector4<f32>,
    // NOTE: for now we are using explicit padding since
    //       gpu buffer read based on Pod/bytemuck is much
    //       faster (about 20x) than with ShaderType/encase.
    pub deformation: nalgebra::Matrix4x3<f32>,
    pub position: Vector4<f32>,
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RenderConfig {
    mode: u32,
}

#[allow(dead_code)]
impl RenderConfig {
    pub const DEFAULT: Self = Self { mode: 0 };
    pub const VOLUME: Self = Self { mode: 1 };
    pub const VELOCITY: Self = Self { mode: 2 };
    pub const CDF_NORMALS: Self = Self { mode: 3 };
    pub const CDF_DISTANCES: Self = Self { mode: 4 };
    pub const CDF_SIGNS: Self = Self { mode: 5 };
}

#[derive(Shader)]
#[cfg_attr(feature = "dim2", shader(module = "slosh_testbed::prep_readback2"))]
#[cfg_attr(feature = "dim3", shader(module = "slosh_testbed::prep_readback3"))]
pub struct PrepReadback<B: Backend> {
    pub prep_readback: GpuFunction<B>,
}

pub struct GpuReadbackData<B: Backend> {
    pub mode: GpuTensor<RenderConfig, B>,
    pub base_colors: GpuTensor<Vector4<f32>, B>,
    pub instances: GpuTensor<ReadbackData, B>,
    pub instances_staging: GpuTensor<ReadbackData, B>,
}

impl<B: Backend> GpuReadbackData<B> {
    pub fn new(backend: &B, num_particles: usize) -> Result<Self, B::Error> {
        let config = RenderConfig::DEFAULT; // VELOCITY;

        let palette = [
            [124.0 / 255.0, 144.0 / 255.0, 1.0, 1.0],
            [8.0 / 255.0, 144.0 / 255.0, 1.0, 1.0],
            [124.0 / 255.0, 7.0 / 255.0, 1.0, 1.0],
            [124.0 / 255.0, 144.0 / 255.0, 7.0 / 255.0, 1.0],
            [200.0 / 255.0, 37.0 / 255.0, 1.0, 1.0],
            [124.0 / 255.0, 230.0 / 255.0, 25.0 / 255.0, 1.0],
        ];

        let instances: Vec<_> = (0..num_particles)
            .map(|_| ReadbackData::default())
            .collect();
        let base_colors: Vec<_> = (0..num_particles)
            .map(|i| palette[i % palette.len()].into())
            .collect();

        Ok(Self {
            mode: GpuTensor::scalar(backend, config, BufferUsages::STORAGE)?,
            base_colors: GpuTensor::vector(backend, base_colors, BufferUsages::STORAGE)?,
            instances: GpuTensor::vector(
                backend,
                instances,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            instances_staging: GpuTensor::vector_uninit(
                backend,
                num_particles as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
        })
    }
}

#[derive(ShaderArgs)]
struct PrepReadbackArgs<'a, B: Backend> {
    instances: &'a GpuTensor<ReadbackData, B>,
    base_colors: &'a GpuTensor<Vector4<f32>, B>,
    particles_pos: &'a GpuTensor<ParticlePosition, B>,
    particles_dyn: &'a GpuTensor<ParticleDynamics, B>,
    grid: &'a GpuTensor<GpuGridMetadata, B>,
    params: &'a GpuTensor<SimulationParams, B>,
    config: &'a GpuTensor<RenderConfig, B>,
}

impl<B: Backend> PrepReadback<B> {
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        encoder: &mut B::Encoder,
        data: &mut GpuReadbackData<B>,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
    ) -> Result<(), B::Error> {
        let args = PrepReadbackArgs {
            particles_pos: particles.positions(),
            particles_dyn: particles.dynamics(),
            grid: &grid.meta,
            params: &sim_params.params,
            config: &data.mode,
            instances: &data.instances,
            base_colors: &data.base_colors,
        };
        let mut pass = encoder.begin_pass();
        self.prep_readback
            .launch(backend, &mut pass, &args, [particles.len() as u32, 1, 1])?;
        drop(pass);

        data.instances_staging
            .copy_from_view(encoder, &data.instances)?;
        Ok(())
    }
}
