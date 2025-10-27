//! Grid-to-Particle transfer with Collision Detection Field updates.

use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
};
use crate::solver::{
    GpuParticleModelData, GpuParticles, GpuSimulationParams, ParticleDynamics, ParticlePosition,
    SimulationParams,
};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

/// GPU kernel for G2P transfer with CDF updates for rigid body coupling.
///
/// Updates particle CDF (Collision Detection Field) data based on proximity
/// to rigid bodies during the G2P phase.
#[derive(Shader)]
#[shader(module = "slosh::solver::g2p_cdf")]
pub struct WgG2PCdf<B: Backend> {
    /// Compiled G2P-CDF compute shader.
    pub g2p_cdf: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct G2PCdfArgs<'a, B: Backend> {
    params: &'a GpuScalar<SimulationParams, B>,
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
    sorted_particle_ids: &'a GpuVector<u32, B>,
    particles_pos: &'a GpuVector<ParticlePosition, B>,
    particles_dyn: &'a GpuVector<ParticleDynamics, B>,
}

impl<B: Backend> WgG2PCdf<B> {
    /// Launches G2P with CDF updates for MPM particles.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `sim_params` - Simulation parameters
    /// * `grid` - Source grid
    /// * `particles` - Target particles to update
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
    ) -> Result<(), B::Error> {
        let args = G2PCdfArgs {
            params: &sim_params.params,
            grid: &grid.meta,
            hmap_entries: &grid.hmap_entries,
            active_blocks: &grid.active_blocks,
            nodes: &grid.nodes,
            sorted_particle_ids: particles.sorted_ids(),
            particles_pos: particles.positions(),
            particles_dyn: particles.dynamics(),
        };
        self.g2p_cdf.launch_indirect(
            backend,
            pass,
            &args,
            grid.indirect_n_g2p_p2g_groups.buffer(),
        )
    }
}
