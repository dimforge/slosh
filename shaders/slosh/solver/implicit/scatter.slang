//! Scatter leg of the implicit solver: the node side of the residual and of the operator.
//!
//! One workgroup per active block and one thread per node, streaming the block's particles
//! (primaries and extras) through shared memory like the scatter-style P2G. Each node
//! accumulates `acc = sum_p w_ip S_p (x_i - x_p)`, where `S_p` is `dt V0 inv_d` times the stress
//! (residual mode) or the stress differential (operator mode) stored by the gather leg. The node
//! then writes either the Newton residual `g = P (m v - rhs + acc)` or the operator value
//! `ap = P (m p + acc)`, with `P` the grid-level boundary projection, and the workgroup reduces
//! `|g|^2` or `p . ap` into one partial per block.

module scatter;

#include "slosh/features.h"

import slosh.solver.particle;
import slosh.solver.implicit.types;
import slosh.grid.kernel;
import slosh.grid.grid;
import slosh.aliases;

static const uint WORKGROUP_SIZE = 64;

#if DIM == 2
groupshared float2 shared_pos[WORKGROUP_SIZE];
groupshared float2x2 shared_stress[WORKGROUP_SIZE];
#else
groupshared float3 shared_pos[WORKGROUP_SIZE];
groupshared float3x3 shared_stress[WORKGROUP_SIZE];
#endif
groupshared int shared_zkey[WORKGROUP_SIZE];
groupshared float shared_dot[WORKGROUP_SIZE];

/// Sums `value` across the 64-thread workgroup through shared memory. Only thread 0 gets the
/// result.
public func workgroup_sum_64(tid: uint, value: float) -> float {
    shared_dot[tid] = value;
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        GroupMemoryBarrierWithGroupSync();
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
        }
    }
    return shared_dot[0];
}

func scatter_impl<let RESIDUAL: int>(
    block_id: uint3,
    tid: uint,
    grid: StructuredBuffer<Grid>,
    active_blocks: StructuredBuffer<ActiveBlockHeader>,
    sorted_particle_ids: StructuredBuffer<uint>,
    particles_pos: StructuredBuffer<Position>,
    implicit_particles: StructuredBuffer<ImplicitParticle>,
    cg_nodes: RWStructuredBuffer<CgNode>,
    partials: RWStructuredBuffer<float>,
) {
    let cell_width = grid[0].cell_width;
    let inv_cell_width = 1.0 / cell_width;
    let bid = block_id.x;

    let first_particle = active_blocks[bid].first_particle;
    let num_particles = active_blocks[bid].num_particles_with_extras;
    let last_particle = first_particle + num_particles;
    let primaries_end = first_particle + active_blocks[bid].num_particles;
    let block_vid = active_blocks[bid].virtual_id.id;

#if DIM == 2
    let local_cell = int2(int(tid % 8u), int(tid / 8u));
    let cell_pos = float2(block_vid * 8 + local_cell) * cell_width;
    let node_slab = local_cell.y;
#else
    let local_cell = int3(int(tid % 4u), int((tid / 4u) % 4u), int(tid / 16u));
    let cell_pos = float3(block_vid * 4 + local_cell) * cell_width;
    let node_slab = local_cell.z;
#endif
    var acc = Vect<float>(0.0);

    for (var chunk_base = first_particle; chunk_base < last_particle; chunk_base += WORKGROUP_SIZE) {
        // Wait for the previous chunk's readers before overwriting shared memory.
        GroupMemoryBarrierWithGroupSync();

        let load_idx = chunk_base + tid;
        if (load_idx < last_particle) {
            let pid = sorted_particle_ids[load_idx];
            let pos = particles_pos[pid].pt;
            shared_pos[tid] = pos;
            // Disabled particles carry a zero stress, so no enabled check is needed.
            shared_stress[tid] = implicit_particles[pid].stress;
            // Slab key along the sort axis, relative to this block (see the scatter-style P2G).
            let assoc_cell = round(pos / cell_width) - 1.0;
#if DIM == 2
            shared_zkey[tid] = max(int(assoc_cell.y) - block_vid.y * 8, -2);
#else
            shared_zkey[tid] = max(int(assoc_cell.z) - block_vid.z * 4, -2);
#endif
        }

        GroupMemoryBarrierWithGroupSync();

        let chunk_len = min(WORKGROUP_SIZE, last_particle - chunk_base);
        var zmin = shared_zkey[0];
        var zmax = shared_zkey[chunk_len - 1u];
        if (primaries_end > chunk_base && primaries_end < chunk_base + chunk_len) {
            let b = primaries_end - chunk_base;
            zmin = min(zmin, shared_zkey[b]);
            zmax = max(zmax, shared_zkey[b - 1u]);
        }

        let culled_len = (node_slab >= zmin && node_slab <= zmax + 2) ? chunk_len : 0u;
        for (var p = 0u; p < culled_len; p += 1u) {
            let dpt = cell_pos - shared_pos[p];
#if DIM == 2
            let weight = QuadraticKernel::eval(dpt.x * inv_cell_width)
                       * QuadraticKernel::eval(dpt.y * inv_cell_width);
#else
            let weight = QuadraticKernel::eval(dpt.x * inv_cell_width)
                       * QuadraticKernel::eval(dpt.y * inv_cell_width)
                       * QuadraticKernel::eval(dpt.z * inv_cell_width);
#endif
            if (weight != 0.0) {
                acc += mul(dpt, shared_stress[p]) * weight;
            }
        }
    }

    // Massless nodes are not unknowns.
    let gid = block_header_id_to_physical_id(BlockHeaderId(bid)).id + tid;
    let node = cg_nodes[gid];
    let mass = node.mass;
    var value = 0.0;
    if (RESIDUAL != 0) {
        // Newton residual: gradient of the incremental potential at `v`.
        let g = mass > 0.0 ? node.project(node.v * mass - node.rhs + acc) : Vect<float>(0.0);
        cg_nodes[gid].g = g;
        value = dot(g, g);
    } else {
        // A p = (M + dt^2 K + dt D) p.
        let ap = mass > 0.0 ? node.project(node.p * mass + acc) : Vect<float>(0.0);
        cg_nodes[gid].ap = ap;
        value = dot(node.p, ap);
    }

    let sum = workgroup_sum_64(tid, value);
    if (tid == 0) {
        partials[bid] = sum;
    }
}

/// Scatter leg of the operator.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func scatter_operator(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<uint> sorted_particle_ids,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<ImplicitParticle> implicit_particles,
    RWStructuredBuffer<CgNode> cg_nodes,
    RWStructuredBuffer<float> partials,
) {
    scatter_impl<0>(block_id, tid, grid, active_blocks, sorted_particle_ids, particles_pos,
                    implicit_particles, cg_nodes, partials);
}

/// Scatter leg of the residual: Newton residual at the trial velocity.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func scatter_residual(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<uint> sorted_particle_ids,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<ImplicitParticle> implicit_particles,
    RWStructuredBuffer<CgNode> cg_nodes,
    RWStructuredBuffer<float> partials,
) {
    scatter_impl<1>(block_id, tid, grid, active_blocks, sorted_particle_ids, particles_pos,
                    implicit_particles, cg_nodes, partials);
}
