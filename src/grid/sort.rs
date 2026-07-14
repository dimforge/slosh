//! Particle sorting kernels for spatial acceleration.

use crate::grid::grid::{GpuGrid, GpuGridHashMapEntry, GpuGridMetadata};
use crate::math::Vector;
use crate::solver::GpuRigidParticles;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::GpuScalar;

/// GPU compute kernels for sorting particles into grid cells.
///
/// Implements spatial hashing and sorting to group particles by grid block
/// for efficient neighbor queries during P2G/G2P.
#[derive(Shader)]
#[shader(module = "slosh::grid::sort")]
pub struct WgSort<B: Backend> {
    // Legacy single-pass block activation, superseded by the two-pass
    // touch_primary_blocks / touch_neighbor_blocks. Kept bound for reference/fallback.
    #[allow(dead_code)]
    pub(crate) touch_particle_blocks: GpuFunction<B>,
    pub(crate) touch_primary_blocks: GpuFunction<B>,
    pub(crate) touch_neighbor_blocks: GpuFunction<B>,
    pub(crate) update_nbh_block_ids: GpuFunction<B>,
    // Bound to GPU kernels; currently only used by commented-out rigid-particle code paths.
    #[allow(dead_code)]
    pub(crate) touch_rigid_particle_blocks: GpuFunction<B>,
    #[allow(dead_code)]
    pub(crate) mark_rigid_particles_needing_block: GpuFunction<B>,
    pub(crate) update_block_particle_count: GpuFunction<B>,
    pub(crate) copy_particles_len_to_scan_value: GpuFunction<B>,
    pub(crate) copy_scan_values_to_first_particles_and_prepare_for_finalize: GpuFunction<B>,
    pub(crate) finalize_particles_sort: GpuFunction<B>,
    pub(crate) sort_rigid_particles: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct SortArgs<'a, B: Backend> {
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuScalar<GpuGridHashMapEntry, B>,
    #[cfg(feature = "node_particle_lists")]
    rigid_nodes_linked_lists: &'a GpuScalar<[u32; 2], B>,
    rigid_particles_pos: &'a GpuScalar<Vector, B>,
    #[cfg(feature = "node_particle_lists")]
    rigid_particle_node_linked_lists: &'a GpuScalar<u32, B>,
}

impl<B: Backend> WgSort<B> {
    /// Sorts rigid body particles into grid cells.
    ///
    /// Builds spatial linked lists for efficient rigid-MPM particle interactions.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `rigid_particles` - Rigid body surface particles to sort
    /// * `grid` - Target grid structure
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
            #[cfg(feature = "node_particle_lists")]
            rigid_nodes_linked_lists: &grid.rigid_nodes_linked_lists,
            rigid_particles_pos: &rigid_particles.sample_points,
            #[cfg(feature = "node_particle_lists")]
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
