use crate::grid::grid::{GpuGrid, GpuGridHashMapEntry, GpuGridMetadata};
use crate::solver::GpuRigidParticles;
use nexus::math::Point;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::GpuScalar;

#[derive(Shader)]
#[shader(module = "slosh::grid::sort")]
pub struct WgSort<B: Backend> {
    pub(crate) touch_particle_blocks: GpuFunction<B>,
    pub(crate) touch_rigid_particle_blocks: GpuFunction<B>,
    pub(crate) mark_rigid_particles_needing_block: GpuFunction<B>,
    pub(crate) update_block_particle_count: GpuFunction<B>,
    pub(crate) copy_particles_len_to_scan_value: GpuFunction<B>,
    pub(crate) copy_scan_values_to_first_particles: GpuFunction<B>,
    pub(crate) finalize_particles_sort: GpuFunction<B>,
    pub(crate) sort_rigid_particles: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct SortArgs<'a, B: Backend> {
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuScalar<GpuGridHashMapEntry, B>,
    rigid_nodes_linked_lists: &'a GpuScalar<[u32; 2], B>,
    rigid_particles_pos: &'a GpuScalar<Point<f32>, B>,
    rigid_particle_node_linked_lists: &'a GpuScalar<u32, B>,
}

impl<B: Backend> WgSort<B> {
    pub fn launch_sort_rigid_particles(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        rigid_particles: &GpuRigidParticles<B>,
        grid: &GpuGrid<B>,
    ) -> Result<(), B::Error> {
        let args = SortArgs {
            grid: &grid.meta,
            hmap_entries: &grid.hmap_entries,
            rigid_nodes_linked_lists: &grid.rigid_nodes_linked_lists,
            rigid_particles_pos: &rigid_particles.sample_points,
            rigid_particle_node_linked_lists: &rigid_particles.node_linked_lists,
        };

        if !rigid_particles.is_empty() {
            self.sort_rigid_particles.launch(
                backend,
                pass,
                &args,
                [rigid_particles.len() as u32, 1, 1],
            )
        } else {
            Ok(())
        }
    }
}
