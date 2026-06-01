//! Grid CDF (Collision Detection Field) update for rigid body coupling.

use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
    GpuNodeCollision,
};
use crate::math::{GpuSim, Point};
use crate::rbd::dynamics::GpuBodySet;
use crate::rbd::shapes::GpuShape;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{BufferUsages, Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuTensor, GpuVector};
use crate::solver::{GpuBoundaryCondition, GpuMaterials, GpuSimulationParams, SimulationParams};

/// GPU kernel for updating grid node CDF data from rigid bodies.
///
/// Computes signed distance fields and closest points on rigid body surfaces
/// for each active grid node.
#[derive(Shader)]
#[shader(module = "slosh::solver::grid_update_collide")]
pub struct WgGridUpdateCollide<B: Backend> {
    /// Compiled grid CDF update shader.
    pub grid_update: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct GridUpdateCollideArgs<'a, B: Backend> {
    params: &'a GpuScalar<SimulationParams, B>,
    prev_grid: &'a GpuScalar<GpuGridMetadata, B>,
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    prev_hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    body_materials: &'a GpuVector<GpuBoundaryCondition, B>,
    collision_shapes: &'a GpuTensor<GpuShape, B>,
    collision_shape_poses: &'a GpuTensor<GpuSim, B>,
    collision_shape_vtx: &'a GpuTensor<Point<f32>, B>,
    collision_shape_idx: &'a GpuTensor<u32, B>,
    prev_node_collisions: &'a GpuVector<GpuNodeCollision, B>,
    node_collisions: &'a GpuVector<GpuNodeCollision, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
}

impl<B: Backend> WgGridUpdateCollide<B> {
    /// Launches grid CDF update from rigid body geometries.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `grid` - Grid to update with CDF data
    /// * `bodies` - Rigid bodies providing collision geometry
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &mut GpuGrid<B>,
        bodies: &GpuBodySet<B>,
        body_materials: &GpuMaterials<B>,
    ) -> Result<(), B::Error> {
        if bodies.is_empty() {
            return Ok(());
        }

        // Lazily allocate the per-node collision caches and keep them sized to
        // the grid nodes. The collision results computed this step are reused on
        // the next step for blocks that still exist (static colliders).
        if grid.node_collisions.len() < grid.nodes.len() {
            let data = vec![GpuNodeCollision::default(); grid.nodes.len() as usize];
            grid.node_collisions = GpuVector::vector_encased(backend, &data, BufferUsages::STORAGE)?;

            if grid.prev_node_collisions.is_empty() {
                grid.prev_node_collisions =
                    GpuVector::vector_encased(backend, &data, BufferUsages::STORAGE)?;
            }
        }

        let args = GridUpdateCollideArgs {
            params: &sim_params.params,
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
