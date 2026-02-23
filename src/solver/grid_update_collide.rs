//! Grid CDF (Collision Detection Field) update for rigid body coupling.

use crate::grid::grid::{GpuActiveBlockHeader, GpuGrid, GpuGridMetadata, GpuGridNode};
use crate::math::{GpuSim, Point};
use crate::rbd::dynamics::GpuBodySet;
use crate::rbd::shapes::GpuShape;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuTensor, GpuVector};
use crate::solver::{GpuBoundaryCondition, GpuMaterials};

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
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    body_materials: &'a GpuVector<GpuBoundaryCondition, B>,
    collision_shapes: &'a GpuTensor<GpuShape, B>,
    collision_shape_poses: &'a GpuTensor<GpuSim, B>,
    collision_shape_vtx: &'a GpuTensor<Point<f32>, B>,
    collision_shape_idx: &'a GpuTensor<u32, B>,
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
        grid: &GpuGrid<B>,
        bodies: &GpuBodySet<B>,
        body_materials: &GpuMaterials<B>,
    ) -> Result<(), B::Error> {
        if bodies.is_empty() {
            return Ok(());
        }

        let args = GridUpdateCollideArgs {
            grid: &grid.meta,
            active_blocks: &grid.active_blocks,
            body_materials: &body_materials.materials,
            collision_shapes: bodies.shapes(),
            collision_shape_poses: bodies.poses(),
            collision_shape_vtx: bodies.shapes_collision_vertices(),
            collision_shape_idx: bodies.shapes_collision_indices(),
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
