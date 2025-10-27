//! Particle-to-Grid transfer with Collision Detection Field for rigid bodies.

use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
};
use crate::sampling::GpuSampleIds;
use crate::solver::GpuRigidParticles;
use nexus::dynamics::GpuBodySet;
use nexus::math::Point;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

/// GPU kernel for P2G transfer from rigid body particles.
///
/// Transfers momentum from rigid body surface particles to grid nodes,
/// enabling two-way coupling between MPM and rigid bodies.
#[derive(Shader)]
#[shader(module = "slosh::solver::p2g_cdf")]
pub struct WgP2GCdf<B: Backend> {
    /// Compiled P2G-CDF compute shader.
    pub p2g_cdf: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct P2GCdfArgs<'a, B: Backend> {
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    rigid_nodes_linked_lists: &'a GpuVector<[u32; 2], B>,
    particle_node_linked_lists: &'a GpuVector<u32, B>,
    collider_vertices: &'a GpuVector<Point<f32>, B>,
    rigid_particle_indices: &'a GpuVector<GpuSampleIds, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
}

impl<B: Backend> WgP2GCdf<B> {
    /// Launches P2G transfer from rigid body particles to grid.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `grid` - Target grid
    /// * `rigid_particles` - Source rigid body particles
    /// * `bodies` - Rigid body set for vertex data
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        grid: &GpuGrid<B>,
        rigid_particles: &GpuRigidParticles<B>,
        bodies: &GpuBodySet<B>,
    ) -> Result<(), B::Error> {
        if rigid_particles.sample_points.is_empty() {
            return Ok(());
        }

        let args = P2GCdfArgs {
            grid: &grid.meta,
            hmap_entries: &grid.hmap_entries,
            active_blocks: &grid.active_blocks,
            nodes: &grid.nodes,
            rigid_nodes_linked_lists: &grid.rigid_nodes_linked_lists,
            particle_node_linked_lists: &rigid_particles.node_linked_lists,
            rigid_particle_indices: &rigid_particles.sample_ids,
            collider_vertices: bodies.shapes_vertex_buffers(),
        };
        self.p2g_cdf.launch_indirect(
            backend,
            pass,
            &args,
            grid.indirect_n_g2p_p2g_groups.buffer(),
        )
    }
}
