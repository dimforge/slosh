//! Particle state update kernel.
//!
//! Updates particle positions, deformation gradients, and material state after
//! grid velocities have been transferred back to particles.

use crate::grid::grid::{GpuGrid, GpuGridMetadata};
use crate::math::Matrix;
use crate::rbd::dynamics::GpuBodySet;
use crate::solver::params::GpuSimulationParams;
use crate::solver::particle_model::GpuParticleModelData;
use crate::solver::{
    Cdf, GpuParticles, Kinematics, ParticlePosition, ParticleProperties, SimulationParams,
};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuTensor};

/// GPU compute kernel for updating particle state.
///
/// Integrates particle positions using updated velocities, updates deformation
/// gradients, and applies constitutive models (elasticity, plasticity). Uses
/// shader specialization for material-specific code paths.
#[derive(Shader)]
#[shader(
    module = "slosh::solver::particle_update",
    specialize = ["slosh::models::specializations"]
)]
pub struct WgParticleUpdate<B: Backend> {
    /// Compiled particle update compute shader.
    pub particle_update: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct ParticleUpdateArgs<'a, B: Backend, GpuModel: GpuParticleModelData> {
    params: &'a GpuTensor<SimulationParams, B>,
    grid: &'a GpuTensor<GpuGridMetadata, B>,
    particles_model: &'a GpuTensor<GpuModel, B>,
    particles_pos: &'a GpuTensor<ParticlePosition, B>,
    particles_kin: &'a GpuTensor<Kinematics, B>,
    particles_cdf: &'a GpuTensor<Cdf, B>,
    particles_def_grad: &'a GpuTensor<Matrix<f32>, B>,
    particles_props: &'a GpuTensor<ParticleProperties, B>,
    particles_len: &'a GpuScalar<u32, B>,
}

impl<B: Backend> WgParticleUpdate<B> {
    /// Launches the particle update kernel.
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
            particles_model: particles.models(),
            particles_pos: particles.positions(),
            particles_kin: &particles.kinematics,
            particles_cdf: &particles.cdf,
            particles_def_grad: &particles.def_grad,
            particles_props: &particles.properties,
            particles_len: particles.gpu_len(),
        };
        self.particle_update
            .launch(backend, pass, &args, [particles.len() as u32, 1, 1])
    }
}
