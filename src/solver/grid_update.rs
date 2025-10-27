//! Grid node update kernel.
//!
//! Updates grid node velocities by applying forces (gravity, boundary conditions)
//! and solving momentum equations on the grid.

use crate::grid::grid::{GpuActiveBlockHeader, GpuGrid, GpuGridMetadata, GpuGridNode};
use crate::solver::SimulationParams;
use crate::solver::params::GpuSimulationParams;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

/// GPU compute kernel for updating grid node velocities.
///
/// Applies external forces (gravity), boundary conditions (sticky/slip walls),
/// and solves momentum equations on grid nodes. Runs between P2G and G2P stages.
#[derive(Shader)]
#[shader(module = "slosh::solver::grid_update")]
pub struct WgGridUpdate<B: Backend> {
    /// Compiled grid update compute shader.
    pub grid_update: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct GridUpdateArgs<'a, B: Backend> {
    sim_params: &'a GpuVector<SimulationParams, B>,
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
}

impl<B: Backend> WgGridUpdate<B> {
    /// Launches the grid update kernel.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for command recording
    /// * `pass` - Compute pass to record commands into
    /// * `sim_params` - Simulation parameters (gravity, timestep)
    /// * `grid` - Grid with nodes to update
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
    ) -> Result<(), B::Error> {
        let args = GridUpdateArgs {
            sim_params: &sim_params.params,
            grid: &grid.meta,
            active_blocks: &grid.active_blocks,
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
