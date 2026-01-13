module grid_update;

import slosh.grid.grid;
import slosh.solver.params;

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
#endif

// NOTE: the only reason why this is its own kernel is because this makes us
//       exceed the 10 storage bindings on web platforms (because of the
//       collision-detection buffers).
//       If we ever end up moving the collision-detection to particles only,
//       we should consider doing the cell update in the p2g kernel.
[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func grid_update(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    uint tid_flat: SV_GroupIndex,
    ConstantBuffer<SimulationParams> sim_params,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    RWStructuredBuffer<Node> nodes,
) {
    let bid = block_id.x;
    let vid = active_blocks[bid].virtual_id;
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
    let momentum_velocity_mass = nodes[global_id].momentum_velocity_mass;
    let new_grid_velocity_mass = update_single_cell(sim_params, cell_width, cell_pos, momentum_velocity_mass);
    nodes[global_id].momentum_velocity_mass = new_grid_velocity_mass;
}

#if DIM == 2
func update_single_cell(sim_params: SimulationParams, cell_width: float, cell_pos: float2, momentum_velocity_mass: float3) -> float3 {
    let mass = momentum_velocity_mass.z;
    let inv_mass = select(mass > 0.0, 1.0 / mass, 0.0);
    var velocity = (momentum_velocity_mass.xy + mass * sim_params.gravity * sim_params.dt) * inv_mass;
    // Clamp the velocity so it doesn't exceed 1 grid cell in one step.
    let vel_limit = float2(cell_width / sim_params.dt);
    velocity = clamp(velocity, -vel_limit, vel_limit);
    return float3(velocity, mass);
}
#else
func update_single_cell(sim_params: SimulationParams, cell_width: float, cell_pos: float3, momentum_velocity_mass: float4) -> float4 {
    let mass = momentum_velocity_mass.w;
    let inv_mass = select(mass > 0.0, 1.0 / mass, 0.0);
    var velocity = (momentum_velocity_mass.xyz + mass * sim_params.gravity * sim_params.dt) * inv_mass;

    // Clamp the velocity so it doesn't exceed 1 grid cell in one step.
    let vel_limit = float3(cell_width / sim_params.dt);
    velocity = clamp(velocity, -vel_limit, vel_limit);
    return float4(velocity, mass);
}
#endif
