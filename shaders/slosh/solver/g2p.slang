module g2p;

import slosh.solver.params;
import slosh.solver.particle;
import slosh.grid.kernel;
import slosh.grid.grid;
import slosh.models.linear_elasticity;
import slosh.models.drucker_prager;
import nexus.dynamics.body;

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
static const uint NUM_SHARED_CELLS = 10 * 10; // block-size plus 2 from adjacent blocks: (8 + 2)^2

groupshared float3 shared_nodes_vel_mass[NUM_SHARED_CELLS];
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
static const uint NUM_SHARED_CELLS = 6 * 6 * 6; // block-size plus 2 from adjacent blocks: (4 + 2)^3

groupshared float4 shared_nodes_vel_mass[NUM_SHARED_CELLS];
#endif

groupshared NodeCdf shared_nodes_cdf[NUM_SHARED_CELLS]; // PERF: we don’t need the distance field from the cdf

static const uint WORKGROUP_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func g2p(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    uint tid_flat: SV_GroupIndex,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<Node> nodes,
    StructuredBuffer<uint> sorted_particle_ids,
    StructuredBuffer<Position> particles_pos,
    RWStructuredBuffer<Dynamics> particles_dyn,
    StructuredBuffer<BodyVelocity> body_vels,
    StructuredBuffer<BodyMassProperties> body_mprops,
) {
    let bid = block_id.x;

    // Block -> shared memory transfer.
    global_shared_memory_transfers(grid, hmap_entries, nodes, tid, active_blocks[bid].virtual_id);

    // Sync after shared memory initialization.
    GroupMemoryBarrierWithGroupSync();

    // Particle update. Runs g2p on shared memory only.
    let max_particle_id = active_blocks[bid].first_particle + active_blocks[bid].num_particles;

    // PERF: could we rely on the particle linked lists instead of the sorted ids?
    //       That could eliminate the cost of particle sorting.
    for (var sorted_particle_id = active_blocks[bid].first_particle + tid_flat;
         sorted_particle_id < max_particle_id;
         sorted_particle_id += WORKGROUP_SIZE) {
        let particle_id = sorted_particle_ids[sorted_particle_id];
        particle_g2p(body_vels, body_mprops, particles_pos, particles_dyn, particle_id, grid[0].cell_width, params.dt);
    }
}

func global_shared_memory_transfers(
    grid: StructuredBuffer<Grid>,
    hmap_entries: StructuredBuffer<GridHashMapEntry>,
    nodes: StructuredBuffer<Node>,
    tid: uint3,
    active_block_vid: BlockVirtualId
) {
    let base_block_pos_int = active_block_vid.id;

#if DIM == 2
    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            if ((i == 1 && tid.x > 1) || (j == 1 && tid.y > 1)) {
                // This shared node doesn’t exist.
                continue;
            }

            let octant = uint2(i, j);
            let octant_hid = find_block_header_id(grid, hmap_entries, BlockVirtualId(base_block_pos_int + int2(octant)));
            let shared_index = octant * 8 + tid.xy;
            let flat_shared_index = flatten_shared_index(shared_index.x, shared_index.y);

            if (octant_hid.id != NONE) {
                let global_chunk_id = block_header_id_to_physical_id(octant_hid);
                let global_node_id = node_id(global_chunk_id, tid.xy);
                shared_nodes_vel_mass[flat_shared_index] = nodes[global_node_id.id].momentum_velocity_mass;
                shared_nodes_cdf[flat_shared_index] = nodes[global_node_id.id].cdf;
            } else {
                // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                // NOTE: we don’t need to init global_id since it’s only read for the
                //       current chunk that is guaranteed to exist, not the 2x2 adjacent ones.
                shared_nodes_vel_mass[flat_shared_index] = float3(0.0);
                shared_nodes_cdf[flat_shared_index] = NodeCdf(0.0, 0, NONE);
            }
        }
    }
#else
    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= 1u; k++) {
                if ((i == 1 && tid.x > 1) || (j == 1 && tid.y > 1) || (k == 1 && tid.z > 1)) {
                    // This shared node doesn’t exist.
                    continue;
                }

                let octant = uint3(i, j, k);
                let octant_hid = find_block_header_id(grid, hmap_entries, BlockVirtualId(base_block_pos_int + int3(octant)));
                let shared_index = octant * 4 + tid;
                let flat_shared_index = flatten_shared_index(shared_index.x, shared_index.y, shared_index.z);

                if (octant_hid.id != NONE) {
                    let global_chunk_id = block_header_id_to_physical_id(octant_hid);
                    let global_node_id = node_id(global_chunk_id, tid);
                    shared_nodes_vel_mass[flat_shared_index] = nodes[global_node_id.id].momentum_velocity_mass;
                    shared_nodes_cdf[flat_shared_index] = nodes[global_node_id.id].cdf;
                } else {
                    // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                    // NOTE: we don’t need to init global_id since it’s only read for the
                    //       current chunk that is guaranteed to exist, not the 2x2x2 adjacent ones.
                    shared_nodes_vel_mass[flat_shared_index] = float4(0.0);
                    shared_nodes_cdf[flat_shared_index] = NodeCdf(0.0, 0, NONE);
                }
            }
        }
    }
#endif
}

func particle_g2p(
    body_vels: StructuredBuffer<BodyVelocity>,
    body_mprops: StructuredBuffer<BodyMassProperties>,
    particles_pos: StructuredBuffer<Position>,
    particles_dyn: RWStructuredBuffer<Dynamics>,
    particle_id: uint,
    cell_width: float,
    dt: float
) {
    // NOTE: having these into a var is needed so we can index [i] them.
    //       Does this have any impact on performances?
    var NBH_SHIFTS = NBH_SHIFTS;
    var NBH_SHIFTS_SHARED = NBH_SHIFTS_SHARED;

#if DIM == 2
    var rigid_vel = float2(0.0);
    var momentum_velocity_mass = float3(0.0);
    var velocity_gradient = float2x2(float2(0.0), float2(0.0));
#else
    var rigid_vel = float3(0.0);
    var momentum_velocity_mass = float4(0.0);
    var velocity_gradient = float3x3(float3(0.0), float3(0.0), float3(0.0));
#endif

    // G2P
    {
        let particle_pos = particles_pos[particle_id];
        let particle_vel = particles_dyn[particle_id].velocity;
        let particle_cdf = particles_dyn[particle_id].cdf;

        let inv_d = QuadraticKernel::inv_d(cell_width);
        let ref_elt_pos_minus_particle_pos = dir_to_associated_grid_node(particle_pos, cell_width);
        let w = QuadraticKernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

        let assoc_cell_before_integration = round(particle_pos.pt / cell_width);
        let assoc_cell_index_in_block = associated_cell_index_in_block_off_by_one(particle_pos, cell_width);
        let packed_cell_index_in_block = flatten_shared_index(
            assoc_cell_index_in_block.x,
            assoc_cell_index_in_block.y,
#if DIM == 3
            assoc_cell_index_in_block.z
#endif
        );

        [[ForceUnroll]]
        for (var i = 0u; i < NBH_LEN; i += 1u) {
            let shift = NBH_SHIFTS[i];
            let packed_shift = NBH_SHIFTS_SHARED[i];
            let shared_id = packed_cell_index_in_block + packed_shift;
            let cell_data = shared_nodes_vel_mass[shared_id];
            let cell_cdf = shared_nodes_cdf[shared_id];
            let is_compatible = affinities_are_compatible(particle_cdf.affinity, cell_cdf.affinities);

#if DIM == 2
            let dpt = ref_elt_pos_minus_particle_pos + float2(shift) * cell_width;
#else
            let dpt = ref_elt_pos_minus_particle_pos + float3(shift) * cell_width;
#endif

            var cpic_cell_data = cell_data;

            if (!is_compatible) {
                if (cell_cdf.closest_id != NONE) {
                    let body_vel = body_vels[cell_cdf.closest_id]; // TODO: invalid if there is no body.
                    let body_com = body_mprops[cell_cdf.closest_id].com;
                    let cell_center = dpt + particle_pos.pt;
                    let body_pt_vel =  body_velocity_at_point(body_com, body_vel, cell_center);
                    let particle_ghost_vel = body_pt_vel + project_velocity(particle_vel - body_pt_vel, particle_cdf.normal);

#if DIM == 2
                    cpic_cell_data = float3(particle_ghost_vel, cell_data.z);
#else
                    cpic_cell_data = float4(particle_ghost_vel, cell_data.w);
#endif
                } else {
                    // If there is no adjacent collider, the ghost vel is the particle vel.
#if DIM == 2
                    cpic_cell_data = float3(particle_vel, cell_data.z);
#else
                    cpic_cell_data = float4(particle_vel, cell_data.w);
#endif
                }
            }

#if DIM == 2
            let weight = w[0][shift.x] * w[1][shift.y];
            momentum_velocity_mass += cpic_cell_data * weight;
            velocity_gradient += (weight * inv_d) * outer_product(cpic_cell_data.xy, dpt);
#else
            let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];
            momentum_velocity_mass += cpic_cell_data * weight;
            velocity_gradient += (weight * inv_d) * outer_product(cpic_cell_data.xyz, dpt);
#endif
        }

        for (var i = 0u; i < 16u; i++) {
            if (affinity_bit(i, particle_cdf.affinity)) {
                let body_vel = body_vels[i];
                let body_com = body_mprops[i].com;
                rigid_vel += body_velocity_at_point(body_com, body_vel, particle_pos.pt);
            }
        }
    }

    particles_dyn[particle_id].cdf.rigid_vel = rigid_vel;
    // Set the particle velocity, and store the velocity gradient into the affine matrix.
    // The rest will be dealt with in the particle update kernel(s).
    particles_dyn[particle_id].affine = velocity_gradient;
#if DIM == 2
    particles_dyn[particle_id].velocity = momentum_velocity_mass.xy;
#else
    particles_dyn[particle_id].velocity = momentum_velocity_mass.xyz;
#endif
}

// TODO: upstream to wgebra?
#if DIM == 2
func outer_product(a: float2, b: float2) -> float2x2 {
    return float2x2(
        a * b.x,
        a * b.y,
    );
}

// Note that this is different from p2g. We don’t need to shift the index since the truncated
// blocks (the neighbor blocks) are in the quadrants with larger indices.
func flatten_shared_index(x: uint, y: uint) -> uint {
    return x + y * 10;
}
#else
func outer_product(a: float3, b: float3) -> float3x3 {
    return float3x3(
        a * b.x,
        a * b.y,
        a * b.z,
    );
}


// Note that this is different from p2g. We don’t need to shift the index since the truncated
// blocks (the neighbor blocks) are in the octants with larger indices.
func flatten_shared_index(x: uint, y: uint, z: uint) -> uint {
    return x + y * 6 + z * 6 * 6;
}
#endif
