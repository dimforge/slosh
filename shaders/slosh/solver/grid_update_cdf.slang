module grid_update;

import nexus.shapes.shape;
import nexus.aliases;
import slosh.grid.grid;
import slosh.collision.collide;

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
#endif

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func grid_update(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    uint tid_flat: SV_GroupIndex,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<Shape> collision_shapes,
    StructuredBuffer<Pose> collision_shape_poses,
    RWStructuredBuffer<Node> nodes,
) {
    let bid = block_id.x;
    let vid = active_blocks[bid].virtual_id;

    let global_chunk_id = block_header_id_to_physical_id(BlockHeaderId(bid));
#if DIM == 2
    let global_node_id = node_id(global_chunk_id, tid.xy);
    let cell_pos = float2(vid.id * 8 + int2(tid.xy)) * grid[0].cell_width;
#else
    let global_node_id = node_id(global_chunk_id, tid);
    let cell_pos = float3(vid.id * 4 + int3(tid)) * grid[0].cell_width;
#endif

    // PERF: store the list of blocks with a collision so we can skip the ones without
    //       collisions in teh other `_cdf` kernels.
    //       Or maybe just use some sort of flag and skip the block at the start of the kernel?
    let global_id = global_node_id.id;
    nodes[global_id].cdf = collide(collision_shapes, collision_shape_poses, grid[0].cell_width, cell_pos);
}
