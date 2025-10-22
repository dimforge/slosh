use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
};
use crate::solver::{GpuParticleModelData, GpuParticles, GpuSimulationParams, ParticleDynamics, ParticlePosition, SimulationParams};
use nexus::dynamics::{GpuBodySet, GpuMassProperties, GpuVelocity};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

#[derive(Shader)]
#[shader(module = "slosh::solver::g2p")]
pub struct WgG2P<B: Backend> {
    pub g2p: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct G2PArgs<'a, B: Backend> {
    params: &'a GpuScalar<SimulationParams, B>,
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
    sorted_particle_ids: &'a GpuVector<u32, B>,
    particles_pos: &'a GpuVector<ParticlePosition, B>,
    particles_dyn: &'a GpuVector<ParticleDynamics, B>,
    body_vels: &'a GpuVector<GpuVelocity, B>,
    body_mprops: &'a GpuVector<GpuMassProperties, B>,
}

impl<B: Backend> WgG2P<B> {
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        bodies: &GpuBodySet<B>,
    ) -> Result<(), B::Error> {
        let args = G2PArgs {
            params: &sim_params.params,
            grid: &grid.meta,
            hmap_entries: &grid.hmap_entries,
            active_blocks: &grid.active_blocks,
            nodes: &grid.nodes,
            sorted_particle_ids: particles.sorted_ids(),
            particles_pos: particles.positions(),
            particles_dyn: particles.dynamics(),
            body_vels: bodies.vels(),
            body_mprops: bodies.mprops(),
        };
        self.g2p.launch_indirect(
            backend,
            pass,
            &args,
            grid.indirect_n_g2p_p2g_groups.buffer(),
        )?;
        Ok(())
    }
}
