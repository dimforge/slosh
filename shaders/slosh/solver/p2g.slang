module p2g;

import slosh.solver.params;
import slosh.solver.particle;
import slosh.grid.kernel;
import slosh.grid.grid;
import slosh.solver.rigid_impulses;
import nexus.dynamics.body;
import nexus.aliases;

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
static const uint NUM_SHARED_CELLS = 10 * 10; // block-size plus 2 from adjacent blocks: (8 + 2)^2
groupshared float3 shared_vel_mass[NUM_SHARED_CELLS];
groupshared float2x2 shared_affine[NUM_SHARED_CELLS];
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
static const uint NUM_SHARED_CELLS = 6 * 6 * 6; // block-size plus 2 from adjacent blocks: (4 + 2)^3
groupshared float4 shared_vel_mass[NUM_SHARED_CELLS];
groupshared float3x3 shared_affine[NUM_SHARED_CELLS];
#endif
groupshared SharedNode shared_nodes[NUM_SHARED_CELLS];
groupshared Position shared_pos[NUM_SHARED_CELLS];
groupshared uint shared_affinities[NUM_SHARED_CELLS];
groupshared Vect shared_normals[NUM_SHARED_CELLS];
// TODO: is computing the max with an atomic faster than doing a reduction?
groupshared Atomic<uint> max_linked_list_length;
// NOTE: workgroupUniformLoad doesn’t work on atomics, so we need that additional variable
//       to write `max_linked_list_length` into and then read with workgroupUniformLoad.
groupshared uint max_linked_list_length_uniform;

struct SharedNode {
    uint particle_id;
    uint global_id;
}

struct P2GStepResult {
#if DIM == 2
    float3 new_momentum_velocity_mass;
    float2 impulse;
    float ang_impulse;
#else
    float4 new_momentum_velocity_mass;
    float3 impulse;
    float3 ang_impulse;
#endif
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func p2g(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    uint tid_flat: SV_GroupIndex,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<NodeLinkedList> nodes_linked_lists,
    StructuredBuffer<uint> particle_node_linked_lists,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<Dynamics> particles_dyn,
    RWStructuredBuffer<Node> nodes,
    StructuredBuffer<BodyVelocity> body_vels,
    RWStructuredBuffer<IntegerImpulseAtomic> body_impulses,

) {
    let bid = block_id.x;
    let vid = active_blocks[bid].virtual_id;

    // Figure out how many time we’ll have to iterate through
    // the particle linked-list to traverse them all.
    if (tid_flat == 0) {
        // Technically not needed because the WebGpu spec. requires zeroing shared memory, but wgpu has an option to disable that.
        max_linked_list_length.store(0u);
    }

    GroupMemoryBarrierWithGroupSync();
    fetch_max_linked_lists_length(grid, hmap_entries, nodes_linked_lists, tid, vid, bid);
    GroupMemoryBarrierWithGroupSync();

    max_linked_list_length_uniform = max_linked_list_length.load();

    // Block -> shared memory transfer.
    fetch_nodes(grid, hmap_entries, nodes_linked_lists, tid, vid, bid);

    /* Run p2g. Note that we have one thread per cell we want to gather data into. */
    // NOTE: we shift by (8, 8) or (4, 4, 4) so we are in the top-most octant. This is the octant we
    //       got enough information for a full gather.
#if DIM == 2
    var new_momentum_velocity_mass = float3(0.0);
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 8u, tid.y + 8u);
#else
    var new_momentum_velocity_mass = float4(0.0);
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 4u, tid.y + 4u, tid.z + 4u);
#endif

    // TODO: we store the global_id in shared memory for convenience. Should we just recompute it instead?
    let global_id = shared_nodes[packed_cell_index_in_block].global_id;
    let node_affinities = nodes[global_id].cdf.affinities;
    let collider_id = nodes[global_id].cdf.closest_id;
    var total_result = P2GStepResult();

    // NOTE: read the linked list with workgroupUniformLoad so that is is considered
    //       part of a uniform execution flow (for the barriers to be valid).
    let len = workgroupUniformLoad(max_linked_list_length_uniform);
    for (var i = 0u; i < len; i += 1u) {
        GroupMemoryBarrierWithGroupSync();
        fetch_next_particle(particles_pos, particles_dyn, particle_node_linked_lists, tid);
        GroupMemoryBarrierWithGroupSync();

        let partial_result = p2g_step(body_vels, body_impulses, packed_cell_index_in_block, grid[0].cell_width, node_affinities, collider_id);
        total_result.new_momentum_velocity_mass += partial_result.new_momentum_velocity_mass;
        total_result.impulse += partial_result.impulse;
        total_result.ang_impulse += partial_result.ang_impulse;
    }

    // Grid update.
#if DIM == 2
    let cell_pos = float2(vid.id * 8 + int2(tid.xy)) * grid[0].cell_width;
#else
    let cell_pos = float3(vid.id * 4 + int3(tid)) * grid[0].cell_width;
#endif

// NOTE: the only reason why we don’t do the grid update is because this makes us
//       exceed the 10 storage bindings on web platforms (because of the
//       collision-detection buffers).
//       If we ever end up moving the collision-detection to particles only,
//       we should consider doing the cell update in the p2g kernel.

    // Write the node state to global memory.
    nodes[global_id].momentum_velocity_mass = total_result.new_momentum_velocity_mass;
    // Apply the impulse to the closest body.
    // PERF: we should probably run a reduction here to get per-collider accumulated impulses
    //       before adding to global memory. Because it is very likely that every single thread
    //       here targets the same body.

    if (collider_id != NONE) {
#if DIM == 2
        body_impulses[collider_id].linear_x.add(flt2int(total_result.impulse.x));
        body_impulses[collider_id].linear_y.add(flt2int(total_result.impulse.y));
        body_impulses[collider_id].angular.add(flt2int(total_result.ang_impulse));
#else
        body_impulses[collider_id].linear_x.add(flt2int(total_result.impulse.x));
        body_impulses[collider_id].linear_y.add(flt2int(total_result.impulse.y));
        body_impulses[collider_id].linear_z.add(flt2int(total_result.impulse.z));
        body_impulses[collider_id].angular_x.add(flt2int(total_result.ang_impulse.x));
        body_impulses[collider_id].angular_y.add(flt2int(total_result.ang_impulse.y));
        body_impulses[collider_id].angular_z.add(flt2int(total_result.ang_impulse.z));
#endif
    }
}

func p2g_step(
    body_vels: StructuredBuffer<BodyVelocity>,
    // NOTE: could have been StructuredBuffer but the kernel itself needs RW.
    body_impulses: RWStructuredBuffer<IntegerImpulseAtomic>,
    packed_cell_index_in_block: uint,
    cell_width: float,
    node_affinity: uint,
    collider_id: uint
) -> P2GStepResult {
    // NOTE: having these into a var is needed so we can index [i] them.
    //       Does this have any impact on performances?
    var NBH_SHIFTS = NBH_SHIFTS;
    var NBH_SHIFTS_SHARED = NBH_SHIFTS_SHARED;

    // Shift to reach the first node with particles contributing to the current cell’s data.
#if DIM == 2
    let bottommost_contributing_node = flatten_shared_shift(2u, 2u);
    var new_momentum_velocity_mass = float3(0.0);
#else
    let bottommost_contributing_node = flatten_shared_shift(2u, 2u, 2u);
    var new_momentum_velocity_mass = float4(0.0);
#endif
    var impulse = Vect<float>(0.0);
    var ang_impulse = AngVect(0.0);

    [[ForceUnroll]]
    for (var i = 0u; i < NBH_LEN; i += 1u) {
        let packed_shift = NBH_SHIFTS_SHARED[i];
        let nbh_shared_index = packed_cell_index_in_block - bottommost_contributing_node + packed_shift;
        let particle_pos = shared_pos[nbh_shared_index];
        let particle_vel_mass = shared_vel_mass[nbh_shared_index];
        let particle_affine = shared_affine[nbh_shared_index];
        let ref_elt_pos_minus_particle_pos = dir_to_associated_grid_node(particle_pos, cell_width);
        // TODO: only compute the one weight we need.
        let w = QuadraticKernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

#if DIM == 2
        let particle_vel = particle_vel_mass.xy;
        let particle_mass = particle_vel_mass.z;
        let shift = uint2(2u, 2) - NBH_SHIFTS[i];
        let momentum = particle_vel * particle_mass;
        let dpt = ref_elt_pos_minus_particle_pos + float2(shift) * cell_width; // cell_pos - particle_pos
        let weight = w[0][shift.x] * w[1][shift.y];
#else
        let particle_vel = particle_vel_mass.xyz;
        let particle_mass = particle_vel_mass.w;
        let shift = uint3(2u, 2, 2) - NBH_SHIFTS[i];
        let momentum = particle_vel * particle_mass;
        let dpt = ref_elt_pos_minus_particle_pos + float3(shift) * cell_width; // cell_pos - particle_pos
        let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];
#endif

        let particle_affinity = shared_affinities[nbh_shared_index];
        if (!affinities_are_compatible(node_affinity, particle_affinity)) {
            if (collider_id != NONE) {
                let particle_normal = shared_normals[nbh_shared_index];
                let body_vel = body_vels[collider_id];
                let body_com = body_impulses[collider_id].com;
                let cell_center = dpt + particle_pos.pt;
                let body_pt_vel = body_velocity_at_point(body_com, body_vel, cell_center);
                let particle_ghost_vel = body_pt_vel + project_velocity(particle_vel - body_pt_vel, particle_normal);
                let delta_impulse = (particle_vel - particle_ghost_vel) * (weight * particle_mass);

                // TODO: we could do the ang impulse calcs only once after all the `p2g_step` executions.
                let lever_arm = body_com - cell_center;

    #if DIM == 2
                let delta_ang_impulse = dot(delta_impulse, float2(lever_arm.y, -lever_arm.x));
    #else
                let delta_ang_impulse = cross(delta_impulse, lever_arm);
    #endif

                ang_impulse += delta_ang_impulse;
                impulse += delta_impulse;

                continue;
            }
        } else {
            new_momentum_velocity_mass += vector<float, DIM + 1>(mul(dpt, particle_affine) + momentum, particle_mass) * weight;
        }
    }

    return P2GStepResult(new_momentum_velocity_mass, impulse, ang_impulse);
}

#if DIM == 2
    static const uint K_RANGE = 0;
#else
    static const uint K_RANGE = 1;
#endif


func fetch_max_linked_lists_length(
    grid: StructuredBuffer<Grid>,
    hmap_entries: StructuredBuffer<GridHashMapEntry>,
    nodes_linked_lists: StructuredBuffer<NodeLinkedList>,
    tid: uint3,
    active_block_vid: BlockVirtualId,
    bid: uint,
) {
#if DIM == 2
    let base_block_pos_int = active_block_vid.id - int2(1, 1);
#else
    let base_block_pos_int = active_block_vid.id - int3(1, 1, 1);
#endif

    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= K_RANGE; k++) {
#if DIM == 2
                if ((i == 0 && tid.x < 6) || (j == 0 && tid.y < 6)) {
                    // This thread is targeting a non-existent cell in shared memory.
                    continue;
                }

                let octant = uint2(i, j);
                let octant_hid = find_block_header_id(grid, hmap_entries, BlockVirtualId(base_block_pos_int + int2(octant)));
#else
                if ((i == 0 && tid.x < 2) || (j == 0 && tid.y < 2) || (k == 0 && tid.z < 2)) {
                    // This thread is targeting a non-existent cell in shared memory.
                    continue;
                }

                let octant = uint3(i, j, k);
                let octant_hid = find_block_header_id(grid, hmap_entries, BlockVirtualId(base_block_pos_int + int3(octant)));
#endif
                if (octant_hid.id != NONE) {
                    let global_chunk_id = block_header_id_to_physical_id(octant_hid);
#if DIM == 2
                    let global_node_id = node_id(global_chunk_id, tid.xy);
#else
                    let global_node_id = node_id(global_chunk_id, tid);
#endif
                    let len = nodes_linked_lists[global_node_id.id].len;
                    max_linked_list_length.max(len);
                }
            }
        }
    }
}

func fetch_nodes(
    grid: StructuredBuffer<Grid>,
    hmap_entries: StructuredBuffer<GridHashMapEntry>,
    nodes_linked_lists: StructuredBuffer<NodeLinkedList>,
    tid: uint3,
    active_block_vid: BlockVirtualId,
    bid: uint
) {
#if DIM == 2
    let base_block_pos_int = active_block_vid.id - int2(1, 1);
#else
    let base_block_pos_int = active_block_vid.id - int3(1, 1, 1);
#endif

    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= K_RANGE; k++) {
#if DIM == 2
                if ((i == 0 && tid.x < 6) || (j == 0 && tid.y < 6)) {
                    // This thread is targeting a non-existent cell in shared memory.
                    continue;
                }

                let octant = uint2(i, j);
                let octant_hid = find_block_header_id(grid, hmap_entries, BlockVirtualId(base_block_pos_int + int2(octant)));
                let shared_index = octant * 8 + tid.xy;
                let shared_node_index = flatten_shared_index(shared_index.x, shared_index.y);
#else
                if ((i == 0 && tid.x < 2) || (j == 0 && tid.y < 2) || (k == 0 && tid.z < 2)) {
                    // This thread is targeting a non-existent cell in shared memory.
                    continue;
                }

                let octant = uint3(i, j, k);
                let octant_hid = find_block_header_id(grid, hmap_entries, BlockVirtualId(base_block_pos_int + int3(octant)));
                let shared_index = octant * 4 + tid;
                let shared_node_index = flatten_shared_index(shared_index.x, shared_index.y, shared_index.z);
#endif
                if (octant_hid.id != NONE) {
                    let global_chunk_id = block_header_id_to_physical_id(octant_hid);
#if DIM == 2
                    let global_node_id = node_id(global_chunk_id, tid.xy);
#else
                    let global_node_id = node_id(global_chunk_id, tid);
#endif
                    let particle_id = nodes_linked_lists[global_node_id.id].head;
                    shared_nodes[shared_node_index].particle_id = particle_id;
                    shared_nodes[shared_node_index].global_id = global_node_id.id;
                } else {
                    // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                    // NOTE: we don’t need to init global_id since it’s only read for the
                    //       current chunk that is guaranteed to exist, not the 2x2x2 adjacent ones.
                    shared_nodes[shared_node_index].particle_id = NONE;
                }
            }
        }
    }
}

func fetch_next_particle(
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<Dynamics> particles_dyn,
    StructuredBuffer<uint> particle_node_linked_lists,
    tid: uint3
) {
    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= K_RANGE; k++) {
#if DIM == 2
                if ((i == 0 && tid.x < 6) || (j == 0 && tid.y < 6)) {
                    continue;
                }
                let octant = uint2(i, j);
                let shared_index = octant * 8 + tid.xy;
                let shared_flat_index = flatten_shared_index(shared_index.x, shared_index.y);
#else
                if ((i == 0 && tid.x < 2) || (j == 0 && tid.y < 2) || (k == 0 && tid.z < 2)) {
                    continue;
                }
                let octant = uint3(i, j, k);
                let shared_index = octant * 4 + tid;
                let shared_flat_index = flatten_shared_index(shared_index.x, shared_index.y, shared_index.z);
#endif
                let curr_particle_id = shared_nodes[shared_flat_index].particle_id;

                if (curr_particle_id != NONE) {
                    shared_affinities[shared_flat_index] = particles_dyn[curr_particle_id].cdf.affinity;
                    shared_normals[shared_flat_index] = particles_dyn[curr_particle_id].cdf.normal;
                    shared_pos[shared_flat_index] = particles_pos[curr_particle_id];
                    shared_affine[shared_flat_index] = particles_dyn[curr_particle_id].affine;

#if DIM == 2
                    shared_vel_mass[shared_flat_index] = float3(particles_dyn[curr_particle_id].velocity, particles_dyn[curr_particle_id].mass);
#else
                    shared_vel_mass[shared_flat_index] = float4(particles_dyn[curr_particle_id].velocity, particles_dyn[curr_particle_id].mass);
#endif

                    let next_particle_id = particle_node_linked_lists[curr_particle_id];
                    shared_nodes[shared_flat_index].particle_id = next_particle_id;
                } else {
                    // TODO: would it be worth skipping writing zeros if we already
                    //       did it at the previous step? (if we already reached the end
                    //       of the particle linked list)
                    shared_affinities[shared_flat_index] = 0u;
                    shared_normals[shared_flat_index] = Vect<float>(0.0);
#if DIM == 2
                    shared_pos[shared_flat_index].pt = float2(0.0);
                    shared_affine[shared_flat_index] = float2x2(float2(0.0), float2(0.0));
                    shared_vel_mass[shared_flat_index] = float3(0.0);
#else
                    shared_pos[shared_flat_index].pt = float3(0.0);
                    shared_affine[shared_flat_index] = float3x3(float3(0.0), float3(0.0), float3(0.0));
                    shared_vel_mass[shared_flat_index] = float4(0.0);
#endif
                }
            }
        }
    }
}

#if DIM == 2
func flatten_shared_index(x: uint, y: uint) -> uint {
    return (x - 6) + (y - 6) * 10;
}
func flatten_shared_shift(x: uint, y: uint) -> uint {
    return x + y * 10;
}
#else
func flatten_shared_index(x: uint, y: uint, z: uint) -> uint {
    return (x - 2) + (y - 2) * 6 + (z - 2) * 6 * 6;
}
func flatten_shared_shift(x: uint, y: uint, z: uint) -> uint {
    return x + y * 6 + z * 6 * 6;
}
#endif
