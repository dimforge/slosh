//! Grid node update kernel.
//!
//! Updates grid node velocities by applying forces (gravity, boundary conditions)
//! and solving momentum equations on the grid, then applies the rigid-body
//! collision boundary condition. The collision-detection results are cached per
//! node and reused across steps for blocks that already existed (colliders are
//! assumed to never move).

use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
    GpuNodeCollision,
};
use crate::math::{GpuSim, Vector};
use crate::rbd::dynamics::GpuBodySet;
use crate::rbd::shapes::GpuShape;
use crate::solver::params::GpuSimulationParams;
use crate::solver::{GpuBoundaryCondition, GpuMaterials, SimulationParams};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuTensor, GpuVector};

/// GPU compute kernel for updating grid node velocities.
///
/// Applies external forces (gravity), boundary conditions (sticky/slip walls),
/// and solves momentum equations on grid nodes, then applies the rigid-body
/// collision boundary condition. Runs between P2G and G2P stages.
#[derive(Shader)]
#[shader(module = "slosh::solver::grid_update")]
pub struct WgGridUpdate<B: Backend> {
    /// Compiled grid update compute shader.
    pub grid_update: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct GridUpdateArgs<'a, B: Backend> {
    sim_params: &'a GpuScalar<SimulationParams, B>,
    prev_grid: &'a GpuScalar<GpuGridMetadata, B>,
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    prev_hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    body_materials: &'a GpuVector<GpuBoundaryCondition, B>,
    collision_shapes: &'a GpuTensor<GpuShape, B>,
    collision_shape_poses: &'a GpuTensor<GpuSim, B>,
    collision_shape_vtx: &'a GpuTensor<Vector, B>,
    collision_shape_idx: &'a GpuTensor<u32, B>,
    prev_node_collisions: &'a GpuVector<GpuNodeCollision, B>,
    node_collisions: &'a GpuVector<GpuNodeCollision, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
}

impl<B: Backend> WgGridUpdate<B> {
    /// Launches the grid update kernel.
    ///
    /// Applies gravity, velocity clamping, and the rigid-body collision boundary
    /// condition to every active grid node. The collision-detection buffers are
    /// always bound (empty body sets keep at least one harmless dummy shape so
    /// the binding stays valid), so this kernel always runs even when there are
    /// no colliders.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for command recording
    /// * `pass` - Compute pass to record commands into
    /// * `sim_params` - Simulation parameters (gravity, timestep)
    /// * `grid` - Grid with nodes to update
    /// * `bodies` - Rigid bodies providing collision geometry
    /// * `body_materials` - Per-collider boundary conditions
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        bodies: &GpuBodySet<B>,
        body_materials: &GpuMaterials<B>,
    ) -> Result<(), B::Error> {
        let args = GridUpdateArgs {
            sim_params: &sim_params.params,
            prev_grid: &grid.prev_meta,
            grid: &grid.meta,
            prev_hmap_entries: &grid.prev_hmap_entries,
            active_blocks: &grid.active_blocks,
            body_materials: &body_materials.materials,
            collision_shapes: bodies.shapes(),
            collision_shape_poses: bodies.poses(),
            collision_shape_vtx: bodies.shapes_collision_vertices(),
            collision_shape_idx: bodies.shapes_collision_indices(),
            prev_node_collisions: &grid.prev_node_collisions,
            node_collisions: &grid.node_collisions,
            nodes: &grid.nodes,
        };

        self.grid_update.launch_indirect(
            backend,
            pass,
            &args,
            grid.indirect_n_g2p_p2g_groups.buffer(),
        )
    }
}
