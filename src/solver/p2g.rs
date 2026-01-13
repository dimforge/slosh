//! Particle-to-Grid (P2G) transfer kernel.
//!
//! Transfers particle mass, momentum, and forces to nearby grid nodes using
//! interpolation weights. This is the first major step of each MPM timestep.

use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
};
use crate::solver::{GpuBoundaryCondition, GpuImpulses, GpuMaterials, GpuParticleModelData, GpuParticles, ParticleDynamics, ParticlePosition, RigidImpulse};
use nexus::dynamics::{GpuBodySet, GpuVelocity};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

/// GPU compute kernel for Particle-to-Grid (P2G) momentum transfer.
///
/// Rasterizes particle mass and momentum onto the background grid using quadratic
/// B-spline interpolation. Also handles impulse accumulation for rigid body coupling.
#[derive(Shader)]
#[shader(module = "slosh::solver::p2g")]
pub struct WgP2G<B: Backend> {
    /// Compiled P2G compute shader.
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
    body_materials: &'a GpuVector<GpuBoundaryCondition, B>,
}

impl<B: Backend> WgP2G<B> {
    /// Launches the P2G kernel to transfer particle data to grid nodes.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for command recording
    /// * `pass` - Compute pass to record commands into
    /// * `grid` - Target grid to write momentum into
    /// * `particles` - Source particles to read from
    /// * `impulses` - Impulse buffers for rigid body coupling
    /// * `bodies` - Rigid bodies for coupling
    pub fn launch<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        impulses: &GpuImpulses<B>,
        bodies: &GpuBodySet<B>,
        body_materials: &GpuMaterials<B>,
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
            body_materials: &body_materials.materials,
        };
        self.p2g.launch_indirect(
            backend,
            pass,
            &args,
            grid.indirect_n_g2p_p2g_groups.buffer(),
        )
    }
}
