use crate::grid::grid::{GpuActiveBlockHeader, GpuGrid, GpuGridMetadata, GpuGridNode};
use crate::solver::params::GpuSimulationParams;
use crate::solver::SimulationParams;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

#[derive(Shader)]
#[shader(module = "slosh::solver::grid_update")]
pub struct WgGridUpdate<B: Backend> {
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
