//! Grid data structures and GPU kernels for sparse grid management.

use crate::grid::prefix_sum::{PrefixSumWorkspace, WgPrefixSum};
use crate::grid::sort::WgSort;
use crate::solver::{GpuParticleModelData, GpuParticles, GpuRigidParticles, ParticlePosition};
use bytemuck::{Pod, Zeroable};
use encase::ShaderType;
use nexus::math::Point;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs, BufferUsages};
use std::sync::Arc;
use stensor::tensor::{GpuScalar, GpuVector};

/// GPU kernels for grid initialization and management.
///
/// Handles sparse grid allocation, reset, and indirect dispatch setup.
#[derive(Shader)]
#[shader(module = "slosh::grid::grid")]
pub struct WgGrid<B: Backend> {
    reset_hmap: GpuFunction<B>,
    reset: GpuFunction<B>,
    init_indirect_workgroups: GpuFunction<B>,
}

// TODO: should we have all the kernel launches just use
//       the same ShaderArgs to avoid duplication?
//       Or maybe implement ShaderArgs for `GpuGrid`, `GpuParticles`, etc.
#[derive(ShaderArgs)]
struct GridArgs<'a, B: Backend> {
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    n_block_groups: &'a GpuVector<[u32; 3], B>,
    n_g2p_p2g_groups: &'a GpuVector<[u32; 3], B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
    nodes_linked_lists: &'a GpuVector<[u32; 2], B>,
    rigid_nodes_linked_lists: &'a GpuVector<[u32; 2], B>,
    scan_values: &'a GpuVector<u32, B>,
    // From particles
    particles_pos: &'a GpuVector<ParticlePosition, B>,
    particles_len: &'a GpuScalar<u32, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    rigid_particles_pos: &'a GpuVector<Point<f32>, B>,
    rigid_particle_needs_block: &'a GpuVector<u32, B>,
    sorted_particle_ids: &'a GpuVector<u32, B>,
    particle_node_linked_lists: &'a GpuVector<u32, B>,
}

impl<B: Backend> WgGrid<B> {
    /// Sorts particles into grid cells and allocates sparse grid blocks.
    ///
    /// This orchestrates the entire particle sorting process including:
    /// 1. Resetting the grid hashmap
    /// 2. Touching blocks where particles exist
    /// 3. Computing per-block particle counts
    /// 4. Running prefix sums for particle indexing
    /// 5. Finalizing sorted particle IDs
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `particles` - MPM particles to sort
    /// * `rigid_particles` - Rigid body particles to consider
    /// * `grid` - Target grid
    /// * `prefix_sum` - Workspace for prefix sum operations
    /// * `sort_module` - Sorting compute kernels
    /// * `prefix_sum_module` - Prefix sum kernel
    pub fn launch_sort<'a, GpuModel: GpuParticleModelData>(
        &'a self,
        backend: &B,
        pass: &mut B::Pass,
        particles: &GpuParticles<B, GpuModel>,
        rigid_particles: &GpuRigidParticles<B>,
        grid: &GpuGrid<B>,
        prefix_sum: &mut PrefixSumWorkspace<B>,
        sort_module: &'a WgSort<B>,
        prefix_sum_module: &'a WgPrefixSum<B>,
    ) -> Result<(), B::Error> {
        let args = GridArgs {
            grid: &grid.meta,
            hmap_entries: &grid.hmap_entries,
            n_block_groups: &grid.indirect_n_blocks_groups,
            n_g2p_p2g_groups: &grid.indirect_n_g2p_p2g_groups,
            nodes: &grid.nodes,
            nodes_linked_lists: &grid.nodes_linked_lists,
            rigid_nodes_linked_lists: &grid.rigid_nodes_linked_lists,
            scan_values: &grid.scan_values,
            particles_pos: particles.positions(),
            particles_len: particles.gpu_len(),
            active_blocks: &grid.active_blocks,
            rigid_particles_pos: &rigid_particles.sample_points,
            rigid_particle_needs_block: &rigid_particles.rigid_particle_needs_block,
            sorted_particle_ids: particles.sorted_ids(),
            particle_node_linked_lists: particles.node_linked_lists(),
        };

        // Retry until we allocated enough room on the sparse grid for all the blocks.
        let mut sparse_grid_has_the_correct_size = false;
        while !sparse_grid_has_the_correct_size {
            // - Reset next grid’s hashmap.
            // - Reset grid.num_active_blocks to 0.
            // - Run touch_particle_blocks on the next grid.
            // - Readback num_active_blocks.
            // - Update the hashmap & grid buffer sizes if its occupancy is too high.

            // NOTE: num_active_blocks := 0 is set in reset_hmap.
            self.reset_hmap
                .launch(backend, pass, &args, [grid.cpu_meta.hmap_capacity, 1, 1])?;

            sort_module.touch_particle_blocks.launch(
                backend,
                pass,
                &args,
                [particles.len() as u32, 1, 1],
            )?;

            // Ensure blocks exist wherever we have rigid particles that might affect
            // other blocks. This is done in two passes:
            // 1. Mark all rigid particles that need to ensure it’s associated block exists
            // 2. Touch the blocks with marked rigid particles.
            if !rigid_particles.is_empty() {
                sort_module.mark_rigid_particles_needing_block.launch(
                    backend,
                    pass,
                    &args,
                    [rigid_particles.len() as u32, 1, 1],
                )?;

                sort_module.touch_rigid_particle_blocks.launch(
                    backend,
                    pass,
                    &args,
                    [rigid_particles.len() as u32, 1, 1],
                )?;
            }

            // TODO: handle grid buffer resizing
            sparse_grid_has_the_correct_size = true;
        }

        // - Launch update_block_particle_count
        // - Launch copy_particle_len_to_scan_value
        // - Launch cumulated sum.
        // - Launch copy_scan_values_to_first_particles
        // - Launch finalize_particles_sort
        // - Launch write_blocks_multiplicity_to_scan_value
        // - Launch cumulated sum

        // Prepare workgroups for indirect dispatches based on the number of active blocks.
        self.init_indirect_workgroups
            .launch_grid(backend, pass, &args, [1, 1, 1])?;

        sort_module.update_block_particle_count.launch(
            backend,
            pass,
            &args,
            [particles.len() as u32, 1, 1],
        )?;

        sort_module
            .copy_particles_len_to_scan_value
            .launch_indirect(backend, pass, &args, grid.indirect_n_blocks_groups.buffer())?;
        prefix_sum_module.launch(backend, pass, prefix_sum, &grid.scan_values)?;

        sort_module
            .copy_scan_values_to_first_particles
            .launch_indirect(backend, pass, &args, grid.indirect_n_blocks_groups.buffer())?;

        // Reset here so the linked list heads get reset before `finalize_particles_sort` which
        // also setups the per-node linked list.
        self.reset.launch_indirect(
            backend,
            pass,
            &args,
            grid.indirect_n_g2p_p2g_groups.buffer(),
        )?;
        sort_module.finalize_particles_sort.launch(
            backend,
            pass,
            &args,
            [particles.len() as u32, 1, 1],
        )?;

        Ok(())
    }
}

/// Grid metadata stored on GPU.
///
/// Contains information about the sparse grid structure and capacity.
#[derive(Copy, Clone, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuGridMetadata {
    num_active_blocks: u32,
    cell_width: f32,
    hmap_capacity: u32,
    capacity: u32,
}

/// A single grid node storing momentum and collision detection data.
///
/// Each active grid cell has associated nodes that accumulate particle
/// contributions during the P2G phase.
#[derive(Copy, Clone, PartialEq, ShaderType)]
#[repr(C)]
pub struct GpuGridNode {
    momentum_velocity_mass: nalgebra::Vector4<f32>,
    momentum_velocity_mass_incompatible: nalgebra::Vector4<f32>,
    cdf: GpuGridNodeCdf,
}

/// Virtual block identifier in sparse grid space.
///
/// Uniquely identifies a block in the infinite virtual grid before mapping
/// to physical storage via the hashmap.
#[derive(Copy, Clone, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct BlockVirtualId {
    #[cfg(feature = "dim2")]
    id: nalgebra::Vector2<i32>,
    #[cfg(feature = "dim3")]
    id: nalgebra::Vector4<i32>, // Vector3 with padding.
}

/// Hash map entry mapping virtual block IDs to physical storage indices.
///
/// The sparse grid uses a hash table to map infinite virtual coordinates
/// to bounded physical memory.
#[derive(Copy, Clone, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuGridHashMapEntry {
    state: u32,
    #[cfg(feature = "dim2")]
    pad0: u32,
    #[cfg(feature = "dim3")]
    pad0: nalgebra::Vector3<u32>,
    key: BlockVirtualId,
    value: u32,
    #[cfg(feature = "dim2")]
    pad1: u32,
    #[cfg(feature = "dim3")]
    pad1: nalgebra::Vector3<u32>,
}

impl Default for GpuGridHashMapEntry {
    fn default() -> Self {
        Self {
            state: u32::MAX,
            pad0: Default::default(),
            key: BlockVirtualId::zeroed(),
            value: 0,
            pad1: Default::default(),
        }
    }
}

/// Header for an active grid block containing particles.
///
/// Tracks which particles belong to this block for efficient iteration.
#[derive(Copy, Clone, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuActiveBlockHeader {
    virtual_id: BlockVirtualId,
    first_particle: u32,
    num_particles: u32,
}

/// Collision detection field data for a grid node.
///
/// Stores signed distance and affinity information for MPM-rigid body coupling.
#[derive(Copy, Clone, PartialEq, Default, Debug, ShaderType)]
#[repr(C)]
pub struct GpuGridNodeCdf {
    /// Signed distance to nearest rigid body surface.
    pub distance: f32,
    /// Bitmask of rigid body affinities for this node.
    pub affinities: u32,
    /// ID of the closest rigid body.
    pub closest_id: u32,
}

/// GPU-resident sparse grid structure.
///
/// The MPM grid uses a sparse representation with a hashmap to efficiently
/// store only active blocks (blocks containing particles). This dramatically
/// reduces memory usage for spatially localized simulations.
pub struct GpuGrid<B: Backend> {
    /// CPU copy of grid metadata for readback.
    pub cpu_meta: GpuGridMetadata,
    /// GPU buffer containing grid metadata.
    pub meta: GpuScalar<GpuGridMetadata, B>,
    /// Pong buffer for grid metadata.
    pub prev_meta: GpuScalar<GpuGridMetadata, B>,
    /// Hash map entries for virtual-to-physical block mapping.
    pub hmap_entries: GpuVector<GpuGridHashMapEntry, B>,
    /// Pong buffer for hmap entries
    pub prev_hmap_entries: GpuVector<GpuGridHashMapEntry, B>,
    /// Grid node data (momentum, mass, CDF).
    pub nodes: GpuVector<GpuGridNode, B>,
    /// Active block headers tracking particle ranges.
    pub active_blocks: GpuVector<GpuActiveBlockHeader, B>,
    /// Workspace for prefix sum operations.
    pub scan_values: GpuVector<u32, B>,
    /// Per-node linked lists for MPM particles.
    pub nodes_linked_lists: GpuVector<[u32; 2], B>,
    /// Per-node linked lists for rigid body particles.
    pub rigid_nodes_linked_lists: GpuVector<[u32; 2], B>,
    /// Indirect dispatch arguments for block-parallel kernels.
    pub indirect_n_blocks_groups: Arc<GpuScalar<[u32; 3], B>>,
    /// Indirect dispatch arguments for node-parallel kernels.
    pub indirect_n_g2p_p2g_groups: Arc<GpuScalar<[u32; 3], B>>,
    /// Debug buffer for GPU-side diagnostics.
    pub debug: GpuVector<u32, B>,
}

impl<B: Backend> GpuGrid<B> {
    /// Creates a new sparse grid with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for buffer allocation
    /// * `capacity` - Maximum number of grid blocks (rounded up to power of 2)
    /// * `cell_width` - Width of each grid cell in meters
    pub fn with_capacity(backend: &B, capacity: u32, cell_width: f32) -> Result<Self, B::Error> {
        const NODES_PER_BLOCK: u32 = 64; // 8 * 8 in 2D and 4 * 4 * 4 in 3D.
        let capacity = capacity.next_power_of_two();
        let cpu_meta = GpuGridMetadata {
            num_active_blocks: 0,
            cell_width,
            hmap_capacity: capacity,
            capacity,
        };
        let meta = GpuScalar::scalar(
            backend,
            cpu_meta,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let prev_meta = GpuScalar::scalar(
            backend,
            cpu_meta,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let default_entries = vec![GpuGridHashMapEntry::default(); capacity as usize];
        let prev_hmap_entries = GpuVector::vector(backend, &default_entries, BufferUsages::STORAGE)?;
        let hmap_entries = GpuVector::vector(backend, &default_entries, BufferUsages::STORAGE)?;
        let nodes = GpuVector::vector_uninit_encased(
            backend,
            capacity * NODES_PER_BLOCK,
            BufferUsages::STORAGE,
        )?;
        let nodes_linked_lists =
            GpuVector::vector_uninit(backend, capacity * NODES_PER_BLOCK, BufferUsages::STORAGE)?;
        let rigid_nodes_linked_lists =
            GpuVector::vector_uninit(backend, capacity * NODES_PER_BLOCK, BufferUsages::STORAGE)?;
        let active_blocks = GpuVector::vector_uninit(backend, capacity, BufferUsages::STORAGE)?;
        let scan_values = GpuVector::vector_uninit(backend, capacity, BufferUsages::STORAGE)?;
        let indirect_n_blocks_groups = Arc::new(GpuVector::scalar_uninit(
            backend,
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
        )?);
        let indirect_n_g2p_p2g_groups = Arc::new(GpuVector::scalar_uninit(
            backend,
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
        )?);
        let debug = GpuVector::vector(backend, [0, 0], BufferUsages::STORAGE)?;

        Ok(Self {
            cpu_meta,
            meta,
            prev_meta,
            hmap_entries,
            prev_hmap_entries,
            nodes,
            active_blocks,
            scan_values,
            indirect_n_blocks_groups,
            indirect_n_g2p_p2g_groups,
            nodes_linked_lists,
            rigid_nodes_linked_lists,
            debug,
        })
    }

    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.meta, &mut self.prev_meta);
        std::mem::swap(&mut self.prev_hmap_entries, &mut self.hmap_entries);
    }
}

/*
#[cfg(test)]
#[cfg(feature = "dim3")]
mod test {
    use super::{GpuGrid, PrefixSumWorkspace, WgGrid, WgPrefixSum};
    use crate::grid::sort::WgSort;
    use crate::models::ElasticCoefficients;
    use crate::solver::{GpuParticles, GpuRigidParticles, Particle, ParticleDynamics};
    use nalgebra::vector;
    use slang_hal::gpu::GpuInstance;
    use slang_hal::kernel::KernelInvocationQueue;
    use slang_hal::Shader;
    use wgpu::Maintain;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_grid_sort() {
        let gpu = GpuInstance::new().await.unwrap();
        let prefix_sum_module = WgPrefixSum::from_device(gpu.device()).unwrap();
        let grid_module = WgGrid::from_device(gpu.device()).unwrap();
        let sort_module = WgSort::from_device(gpu.device()).unwrap();

        let cell_width = 1.0;
        let mut cpu_particles = vec![];
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let position = vector![i as f32, j as f32, k as f32] / cell_width / 2.0;
                    cpu_particles.push(Particle {
                        position,
                        dynamics: ParticleDynamics::with_density(cell_width / 4.0, 1.0),
                        model: ElasticCoefficients::from_young_modulus(100_000.0, 0.33),
                        plasticity: None,
                        phase: None,
                    });
                }
            }
        }

        let particles = GpuParticles::from_particles(gpu.device(), &cpu_particles);
        let grid = GpuGrid::with_capacity(gpu.device(), 100_000, cell_width);
        let mut prefix_sum = PrefixSumWorkspace::with_capacity(gpu.device(), 100_000);
        let mut queue = KernelInvocationQueue::new(gpu.device());
        #[cfg(target_os = "macos")]
        let touch_particle_blocks =
            crate::grid::sort::TouchParticleBlocks::from_device(gpu.device());
        let rigid_particles = GpuRigidParticles::new(gpu.device());

        grid_module.dispatch_sort(
            &particles,
            &rigid_particles,
            &grid,
            &mut prefix_sum,
            &sort_module,
            #[cfg(target_os = "macos")]
            &touch_particle_blocks,
            &prefix_sum_module,
            &mut queue,
        );

        // NOTE: run multiple times, the first execution is much slower.
        for _ in 0..3 {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            queue.encode(&mut encoder, None);
            let t0 = std::time::Instant::now();
            gpu.queue().submit(Some(encoder.finish()));
            gpu.device().poll(Maintain::Wait);
            println!("Grid sort gpu time: {}", t0.elapsed().as_secs_f32());
        }
    }
}
 */
