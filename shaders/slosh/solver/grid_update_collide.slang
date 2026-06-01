module grid_update_collide;

import slosh.rbd.shapes.shape;
import slosh.aliases;
import slosh.grid.grid;
import slosh.solver.params;
import slosh.collision.collide;
import slosh.solver.boundary_condition;

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
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<Grid> prev_grid,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> prev_hmap_entries,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<BoundaryCondition> body_materials,
    StructuredBuffer<Shape> collision_shapes,
    StructuredBuffer<Pose> collision_shape_poses,
    #if DIM == 3
    StructuredBuffer<Vect> collision_shape_vtx,
    StructuredBuffer<uint> collision_shape_idx,
    #endif
    StructuredBuffer<NodeCollision> prev_node_collisions,
    RWStructuredBuffer<NodeCollision> node_collisions,
    RWStructuredBuffer<Node> nodes,
) {
    let bid = block_id.x;
    // HACK: force copying the virtual ID otherwise there is a naga bug when
    //       translating from WGSL to SpirV (works fine for Metal and HLSL…)
    let vid = BlockVirtualId(active_blocks[bid].virtual_id.id);

    let global_chunk_id = block_header_id_to_physical_id(BlockHeaderId(bid));
#if DIM == 2
    let global_node_id = node_id(global_chunk_id, tid.xy);
    let cell_pos = float2(vid.id * 8 + int2(tid.xy)) * grid[0].cell_width;
#else
    let global_node_id = node_id(global_chunk_id, tid);
    let cell_pos = float3(vid.id * 4 + int3(tid)) * grid[0].cell_width;
#endif

    let global_id = global_node_id.id;

    /*
     * Retrieve or compute the collision for this node.
     *
     * Collision objects are assumed to never move, so a node’s collision only
     * depends on its (fixed) world-space position. If this block already existed
     * in the previous step we reuse the cached result instead of running the
     * expensive collision-detection again.
     */
    var collision = NodeCollision(Vect<float>(0.0), Vect<float>(0.0), 0u, 0u);
    let prev_hid = find_block_header_id(prev_grid, prev_hmap_entries, vid);
    if (prev_hid.id != NONE) {
        let prev_pid = block_header_id_to_physical_id(prev_hid);
    #if DIM == 2
        let prev_global_id = node_id(prev_pid, tid.xy).id;
    #else
        let prev_global_id = node_id(prev_pid, tid).id;
    #endif
        collision = prev_node_collisions[prev_global_id];
    } else {
        collision = collide(
            collision_shapes,
            collision_shape_poses,
            #if DIM == 3
            collision_shape_vtx,
            collision_shape_idx,
            #endif
            grid[0].cell_width,
            cell_pos
        ).to_node();
    }

    node_collisions[global_id] = collision;

    if (collision.valid != 0u) {
        // Apply boundary condition.
        let material = body_materials[collision.collider_id];
        let dist_with_margin = dot(cell_pos - collision.point, collision.normal) - grid[0].cell_width;
        // TODO: take rigid-body velocity into account too.
        #if DIM == 2
            let node_velocity = nodes[global_node_id.id].momentum_velocity_mass.xy;
            let node_mass = nodes[global_node_id.id].momentum_velocity_mass.z;
            let projected = material.project_velocity(node_velocity, collision.normal, dist_with_margin, params.dt);
            nodes[global_node_id.id].momentum_velocity_mass = float3(projected, node_mass);
        #else
            let node_velocity = nodes[global_node_id.id].momentum_velocity_mass.xyz;
            let node_mass = nodes[global_node_id.id].momentum_velocity_mass.w;
            let projected = material.project_velocity(node_velocity, collision.normal, dist_with_margin, params.dt);
            nodes[global_node_id.id].momentum_velocity_mass = float4(projected, node_mass);
        #endif
    }
}
