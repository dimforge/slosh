module p2g;

import slosh.solver.params;
import slosh.solver.particle;
import slosh.solver.boundary_condition;
import slosh.grid.kernel;
import slosh.grid.grid;
import slosh.solver.rigid_impulses;
import slosh.rbd.dynamics.body;
import slosh.aliases;

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
#endif
static const uint WORKGROUP_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;

// Staging buffers for one workgroup-sized chunk of particles. The chunk is loaded
// cooperatively (one particle per thread) and then read by every cell-thread, so
// each particle is fetched from global memory exactly once per block.
#if DIM == 2
groupshared float2 shared_pos[WORKGROUP_SIZE];
groupshared float2 shared_momentum[WORKGROUP_SIZE];
groupshared float2x2 shared_affine[WORKGROUP_SIZE];
#else
groupshared float3 shared_pos[WORKGROUP_SIZE];
groupshared float3 shared_momentum[WORKGROUP_SIZE];
groupshared float3x3 shared_affine[WORKGROUP_SIZE];
#endif
groupshared float shared_mass[WORKGROUP_SIZE];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func p2g_scatter_style(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<Kinematics> particles_kin,
    RWStructuredBuffer<Node> nodes,
    StructuredBuffer<uint> sorted_particle_ids,
) {
    let cell_width = grid[0].cell_width;
    let inv_cell_width = 1.0 / cell_width;
    let bid = block_id.x;

    let first_particle = active_blocks[bid].first_particle;
    let num_particles = active_blocks[bid].num_particles_with_extras;
    let last_particle = first_particle + num_particles;
    let block_vid = active_blocks[bid].virtual_id.id;

    // Each thread owns one cell (grid node) of this block and accumulates the
    // contribution of every particle assigned to the block into a register.
    // This avoids both global atomics and the per-cell workgroup reduction that
    // the previous implementation used (which serialized the whole workgroup with
    // ~7 barriers per cell, i.e. hundreds of barriers per particle batch).
#if DIM == 2
    let local_cell = int2(int(tid % 8u), int(tid / 8u));
    let cell_pos = float2(block_vid * 8 + local_cell) * cell_width;
    var acc = float3(0.0);
#else
    let local_cell = int3(int(tid % 4u), int((tid / 4u) % 4u), int(tid / 16u));
    let cell_pos = float3(block_vid * 4 + local_cell) * cell_width;
    var acc = float4(0.0);
#endif

    // Stream the block's particles through shared memory one chunk at a time.
    for (var chunk_base = first_particle; chunk_base < last_particle; chunk_base += WORKGROUP_SIZE) {
        // Wait for the previous chunk's readers before overwriting shared memory.
        GroupMemoryBarrierWithGroupSync();

        let load_idx = chunk_base + tid;
        if (load_idx < last_particle) {
            let pid = sorted_particle_ids[load_idx];
            let kin = particles_kin[pid];
            shared_pos[tid] = particles_pos[pid].pt;
            shared_mass[tid] = kin.mass;
            shared_affine[tid] = kin.affine;
            shared_momentum[tid] = kin.velocity * kin.mass + kin.force_dt;
        }

        GroupMemoryBarrierWithGroupSync();

        // `chunk_len` is uniform across the workgroup, so the barriers above stay in
        // uniform control flow regardless of the total particle count.
        let chunk_len = min(WORKGROUP_SIZE, last_particle - chunk_base);
        for (var p = 0u; p < chunk_len; p += 1u) {
            let dpt = cell_pos - shared_pos[p];
#if DIM == 2
            let weight = QuadraticKernel::eval(dpt.x * inv_cell_width)
                       * QuadraticKernel::eval(dpt.y * inv_cell_width);
#else
            let weight = QuadraticKernel::eval(dpt.x * inv_cell_width)
                       * QuadraticKernel::eval(dpt.y * inv_cell_width)
                       * QuadraticKernel::eval(dpt.z * inv_cell_width);
#endif
            // The quadratic kernel is exactly zero outside the 3-node support, which
            // is the common case for the dense cell x particle cross product. Skipping
            // the affine matrix-vector product there is the bulk of the saved work.
            if (weight != 0.0) {
                let momentum = mul(dpt, shared_affine[p]) + shared_momentum[p];
                acc += vector<float, DIM + 1>(momentum, shared_mass[p]) * weight;
            }
        }
    }

    // Write the accumulated node state to global memory. Every cell is written once
    // (no atomics, no inter-block races), zeroing cells that received no contribution.
    let global_chunk_id = block_header_id_to_physical_id(BlockHeaderId(bid));
    nodes[global_chunk_id.id + tid].momentum_velocity_mass = acc;
}
