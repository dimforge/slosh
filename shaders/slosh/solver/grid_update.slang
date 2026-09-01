module grid_update;

import slosh.rbd.shapes.shape;
import slosh.aliases;
import slosh.grid.grid;
import slosh.solver.params;
import slosh.collision.collide;
import slosh.solver.boundary_condition;

#include "slosh/features.h"

#if SLOSH_PML
typealias NodeMass = Vect<float>;
#else
typealias NodeMass = float;
#endif

// Reciprocal node mass, zero where the node has none.
func inv_mass_of(mass: NodeMass) -> NodeMass {
    return select(mass > NodeMass(0.0), NodeMass(1.0) / mass, NodeMass(0.0));
}

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
#endif

// Combined grid-node update kernel.
//
// This performs, in a single pass over the active grid nodes:
//   1. the momentum -> velocity update (gravity integration + velocity clamping), and
//   2. the rigid-body collision boundary condition.
//
// These used to be two separate kernels (`grid_update` and `grid_update_collide`).
// They iterate over the exact same nodes with the same indirect dispatch, so
// merging them avoids a redundant pass (and a redundant read/write of the node
// data). The collision-detection results are cached per node and reused across
// steps for blocks that already existed in the previous step (colliders are
// assumed to never move).
[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func grid_update(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    uint tid_flat: SV_GroupIndex,
    ConstantBuffer<SimulationParams> sim_params,
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
    let cell_width = grid[0].cell_width;

    let global_chunk_id = block_header_id_to_physical_id(BlockHeaderId(bid));

#if DIM == 2
    let global_node_id = node_id(global_chunk_id, tid.xy);
    let cell_pos = float2(vid.id * 8 + int2(tid.xy)) * cell_width;
#else
    let global_node_id = node_id(global_chunk_id, tid);
    let cell_pos = float3(vid.id * 4 + int3(tid)) * cell_width;
#endif

    let global_id = global_node_id.id;

    /*
     * Step 1: momentum -> velocity update (gravity + clamping).
     */
    let momentum_velocity_mass = nodes[global_id].momentum_velocity_mass;
#if SLOSH_PML
    let inv_mass = inv_mass_of(nodes[global_id].directional_mass);
#else
    let inv_mass = inv_mass_of(momentum_velocity_mass[DIM]);
#endif
    var velocity_mass = update_single_cell(sim_params, cell_width, cell_pos, momentum_velocity_mass, inv_mass);
#if SLOSH_CPIC
    // The incompatible lane carries no directional mass of its own: CPIC is not meant to overlap
    // with the materials that need one.
    let momentum_velocity_mass_incompatible = nodes[global_id].momentum_velocity_mass_incompatible;
    let inv_mass_incompatible = inv_mass_of(NodeMass(momentum_velocity_mass_incompatible[DIM]));
    let velocity_mass_incompatible = update_single_cell(sim_params, cell_width, cell_pos, momentum_velocity_mass_incompatible, inv_mass_incompatible);
    nodes[global_id].momentum_velocity_mass_incompatible = velocity_mass_incompatible;
#endif

    /*
     * Step 2: rigid-body collision boundary condition.
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
            cell_width,
            cell_pos
        ).to_node();
    }

    node_collisions[global_id] = collision;

    if (collision.valid != 0u) {
        // Apply boundary condition.
        let material = body_materials[collision.collider_id];
        let dist_with_margin = dot(cell_pos - collision.point, collision.normal) - cell_width;
        // TODO: take rigid-body velocity into account too.
        #if DIM == 2
            let node_velocity = velocity_mass.xy;
            let node_mass = velocity_mass.z;
            let projected = material.project_velocity(node_velocity, collision.normal, dist_with_margin, sim_params.dt, cell_width);
            velocity_mass = float3(projected, node_mass);
        #else
            let node_velocity = velocity_mass.xyz;
            let node_mass = velocity_mass.w;
            let projected = material.project_velocity(node_velocity, collision.normal, dist_with_margin, sim_params.dt, cell_width);
            velocity_mass = float4(projected, node_mass);
        #endif
    }

    nodes[global_id].momentum_velocity_mass = velocity_mass;
}

#if DIM == 2
func update_single_cell(sim_params: SimulationParams, cell_width: float, cell_pos: float2, momentum_velocity_mass: float3, inv_mass: NodeMass) -> float3 {
    // Gravity stays proportional to the real mass, so a material that rescales its own inertia
    // (the PML) falls at `g / scale` instead of carrying that much weight.
    let mass = momentum_velocity_mass.z;
    var velocity = (momentum_velocity_mass.xy + mass * sim_params.gravity * sim_params.dt) * inv_mass;
    // Clamp the velocity so it doesn't exceed 1 grid cell in one step.
    let vel_limit = float2(cell_width / sim_params.dt);
    velocity = clamp(velocity, -vel_limit, vel_limit);
    return float3(velocity, mass);
}
#else
func update_single_cell(sim_params: SimulationParams, cell_width: float, cell_pos: float3, momentum_velocity_mass: float4, inv_mass: NodeMass) -> float4 {
    // See the 2D overload above.
    let mass = momentum_velocity_mass.w;
    var velocity = (momentum_velocity_mass.xyz + mass * sim_params.gravity * sim_params.dt) * inv_mass;

    // Clamp the velocity so it doesn't exceed 1 grid cell in one step.
    let vel_limit = float3(cell_width / sim_params.dt);
    velocity = clamp(velocity, -vel_limit, vel_limit);
    return float4(velocity, mass);
}
#endif
