use crate::grid::grid::{GpuGrid, GpuGridMetadata};
use crate::models::{DruckerPrager, DruckerPragerPlasticState, ElasticCoefficients};
use crate::solver::params::GpuSimulationParams;
use crate::solver::particle_model::{GpuParticleModel, GpuParticleModelData};
use crate::solver::{
    GpuParticles, ParticleDynamics, ParticlePhase, ParticlePosition, SimulationParams,
};
use nexus::dynamics::GpuBodySet;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuTensor};

#[derive(Shader)]
#[shader(
    module = "slosh::solver::particle_update",
    specialize = ["slosh::models::specializations"]
)]
pub struct WgParticleUpdate<B: Backend> {
    pub particle_update: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct ParticleUpdateArgs<'a, B: Backend, GpuModel: GpuParticleModelData> {
    params: &'a GpuTensor<SimulationParams, B>,
    grid: &'a GpuTensor<GpuGridMetadata, B>,
    particles_model: &'a GpuTensor<GpuModel, B>,
    particles_pos: &'a GpuTensor<ParticlePosition, B>,
    particles_dyn: &'a GpuTensor<ParticleDynamics, B>,
    particles_len: &'a GpuScalar<u32, B>,
}

impl<B: Backend> WgParticleUpdate<B> {
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        _bodies: &GpuBodySet<B>,
    ) -> Result<(), B::Error> {
        let args = ParticleUpdateArgs {
            params: &sim_params.params,
            grid: &grid.meta,
            particles_model: &particles.models(),
            particles_pos: particles.positions(),
            particles_dyn: particles.dynamics(),
            particles_len: particles.gpu_len(),
        };
        self.particle_update
            .launch(backend, pass, &args, [particles.len() as u32, 1, 1])
    }
}
