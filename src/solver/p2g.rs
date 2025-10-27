use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
};
use crate::solver::{
    GpuImpulses, GpuParticleModelData, GpuParticles, ParticleDynamics, ParticlePosition,
    RigidImpulse,
};
use nexus::dynamics::{GpuBodySet, GpuVelocity};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

#[derive(Shader)]
#[shader(module = "slosh::solver::p2g")]
pub struct WgP2G<B: Backend> {
    pub p2g: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct P2GArgs<'a, B: Backend> {
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    nodes_linked_lists: &'a GpuVector<[u32; 2], B>,
    particle_node_linked_lists: &'a GpuVector<u32, B>,
    particles_pos: &'a GpuVector<ParticlePosition, B>,
    particles_dyn: &'a GpuVector<ParticleDynamics, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
    body_vels: &'a GpuVector<GpuVelocity, B>,
    body_impulses: &'a GpuVector<RigidImpulse, B>,
}

impl<B: Backend> WgP2G<B> {
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        impulses: &GpuImpulses<B>,
        bodies: &GpuBodySet<B>,
    ) -> Result<(), B::Error> {
        let args = P2GArgs {
            grid: &grid.meta,
            hmap_entries: &grid.hmap_entries,
            active_blocks: &grid.active_blocks,
            nodes: &grid.nodes,
            nodes_linked_lists: &grid.nodes_linked_lists,
            particles_pos: particles.positions(),
            particles_dyn: particles.dynamics(),
            particle_node_linked_lists: particles.node_linked_lists(),
            body_vels: bodies.vels(),
            body_impulses: &impulses.incremental_impulses,
        };
        self.p2g.launch_indirect(
            backend,
            pass,
            &args,
            grid.indirect_n_g2p_p2g_groups.buffer(),
        )
    }
}
