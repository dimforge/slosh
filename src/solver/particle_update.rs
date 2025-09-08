use crate::grid::grid::{GpuGrid, GpuGridMetadata};
use crate::models::{DruckerPrager, DruckerPragerPlasticState, ElasticCoefficients, GpuModels};
use crate::solver::params::GpuSimulationParams;
use crate::solver::{
    GpuParticles, ParticleDynamics, ParticlePhase, ParticlePosition, SimulationParams,
};
use nexus::dynamics::GpuBodySet;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::GpuTensor;

#[derive(Shader)]
#[shader(module = "slosh::solver::particle_update")]
pub struct WgParticleUpdate<B: Backend> {
    pub particle_update: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct ParticleUpdateArgs<'a, B: Backend> {
    params: &'a GpuTensor<SimulationParams, B>,
    grid: &'a GpuTensor<GpuGridMetadata, B>,
    plasticity: &'a GpuTensor<DruckerPrager, B>,
    constitutive_model: &'a GpuTensor<ElasticCoefficients, B>,
    plastic_state: &'a GpuTensor<DruckerPragerPlasticState, B>,
    particles_pos: &'a GpuTensor<ParticlePosition, B>,
    particles_dyn: &'a GpuTensor<ParticleDynamics, B>,
    phases: &'a GpuTensor<ParticlePhase, B>,
}

impl<B: Backend> WgParticleUpdate<B> {
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B>,
        models: &GpuModels<B>,
        _bodies: &GpuBodySet<B>,
    ) -> Result<(), B::Error> {
        let args = ParticleUpdateArgs {
            params: &sim_params.params,
            grid: &grid.meta,
            plasticity: &models.drucker_prager_plasticity,
            constitutive_model: &models.linear_elasticity,
            plastic_state: &models.drucker_prager_plastic_state,
            particles_pos: &particles.positions,
            particles_dyn: &particles.dynamics,
            phases: &models.phases,
        };
        self.particle_update.launch(
            backend,
            pass,
            &args,
            [particles.positions.len() as u32, 1, 1],
        )
    }
}
