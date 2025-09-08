module p2g_cdf;

import slosh.solver.params;
import slosh.solver.particle;
import slosh.grid.kernel;
import slosh.grid.grid;
import nexus.dynamics.body;
import nexus.aliases;

#if DIM == 2
import nexus.shapes.segment;
#else
import nexus.shapes.triangle;
#endif

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
static const uint NUM_SHARED_CELLS = 10 * 10; // block-size plus 2 from adjacent blocks: (8 + 2)^2
groupshared Segment shared_primitives[NUM_SHARED_CELLS];
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
static const uint NUM_SHARED_CELLS = 6 * 6 * 6; // block-size plus 2 from adjacent blocks: (4 + 2)^3
groupshared Triangle shared_primitives[NUM_SHARED_CELLS];
#endif
groupshared SharedNode shared_nodes[NUM_SHARED_CELLS];
groupshared uint shared_collider_ids[NUM_SHARED_CELLS];
// TODO: is computing the max with an atomic faster than doing a reduction?
groupshared Atomic<uint> max_linked_list_length;
// NOTE: workgroupUniformLoad doesn’t work on atomics, so we need that additional variable
//       to write `max_linked_list_length` into and then read with workgroupUniformLoad.
groupshared uint max_linked_list_length_uniform;

struct SharedNode {
    uint particle_id;
    uint global_id;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func p2g_cdf(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    uint tid_flat: SV_GroupIndex,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<NodeLinkedList> rigid_nodes_linked_lists,
    StructuredBuffer<uint> particle_node_linked_lists,
    StructuredBuffer<Vect> collider_vertices,
    StructuredBuffer<RigidParticleIndices> rigid_particle_indices,
    RWStructuredBuffer<Node> nodes,
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
    fetch_max_linked_lists_length(grid, hmap_entries, rigid_nodes_linked_lists, tid, vid, bid);
    GroupMemoryBarrierWithGroupSync();

    max_linked_list_length_uniform = max_linked_list_length.load();

    // Block -> shared memory transfer.
    fetch_nodes(grid, hmap_entries, rigid_nodes_linked_lists, tid, vid, bid);

    /* Run p2g. Note that we have one thread per cell we want to gather data into. */
    // NOTE: we shift by (8, 8) or (4, 4, 4) so we are in the top-most octant. This is the octant we
    //       got enough information for a full gather.
#if DIM == 2
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 8u, tid.y + 8u);
    let cell_pos = float2(vid.id * 8 + int2(tid.xy)) * grid[0].cell_width;
#else
    let packed_cell_index_in_block = flatten_shared_index(tid.x + 4u, tid.y + 4u, tid.z + 4u);
    let cell_pos = float3(vid.id * 4 + int3(tid)) * grid[0].cell_width;
#endif

    // TODO: we store the global_id in shared memory for convenience. Should we just recompute it instead?
    let global_id = shared_nodes[packed_cell_index_in_block].global_id;
    var node_cdf = nodes[global_id].cdf;
    let node_affinities = node_cdf.affinities;

    // NOTE: read the linked list with workgroupUniformLoad so that is is considered
    //       part of a uniform execution flow (for the barriers to be valid).
    let len = workgroupUniformLoad(max_linked_list_length_uniform);
    for (var i = 0u; i < len; i += 1u) {
        GroupMemoryBarrierWithGroupSync();
        fetch_next_particle(particle_node_linked_lists, collider_vertices, rigid_particle_indices, tid);
        GroupMemoryBarrierWithGroupSync();

        let partial_result = p2g_step(packed_cell_index_in_block, grid[0].cell_width, cell_pos);

        if (partial_result.closest_id != NONE) {
            node_cdf.affinities |= partial_result.affinities;

            if (partial_result.distance < node_cdf.distance) {
                node_cdf.distance = partial_result.distance;
                node_cdf.closest_id = partial_result.closest_id;
            }
        }
    }

    // Write the node cdf to global memory.
    nodes[global_id].cdf = node_cdf;
}

func p2g_step(
    packed_cell_index_in_block: uint,
    cell_width: float,
    cell_pos: Vect
) -> NodeCdf {
    // NOTE: having these into a var is needed so we can index [i] them.
    //       Does this have any impact on performances?
    var NBH_SHIFTS_SHARED = NBH_SHIFTS_SHARED;

    // Shift to reach the first node with particles contributing to the current cell’s data.
#if DIM == 2
    let bottommost_contributing_node = flatten_shared_shift(2u, 2u);
#else
    let bottommost_contributing_node = flatten_shared_shift(2u, 2u, 2u);
#endif
    var result = NodeCdf(1.0e10, 0u, NONE);

    [[ForceUnroll]]
    for (var i = 0u; i < NBH_LEN; i += 1u) {
        let packed_shift = NBH_SHIFTS_SHARED[i];
        let nbh_shared_index = packed_cell_index_in_block - bottommost_contributing_node + packed_shift;

        let collider_id = shared_collider_ids[nbh_shared_index];

        if (collider_id == NONE) {
            continue;
        }

        let primitive = shared_primitives[nbh_shared_index];

#if DIM == 2
        /*
         * Project on Segment
         */
        let proj = primitive.project_local_point(cell_pos);
        if (any(proj != primitive.a) && any(proj != primitive.b)) {
            // This is a valid projection.
            let dpt = cell_pos - proj;
            let distance = length(dpt);
            let ab = primitive.b - primitive.a;
            let sign = dot(dpt, float2(-ab.y, ab.x)) < 0.0;
            result.affinities |= (1u << collider_id)
                | (uint(sign) << (collider_id + SIGN_BITS_SHIFT));

            if (distance < result.distance) {
                result.distance = min(result.distance, distance);
                result.closest_id = collider_id;
            }
        }
#else
        /*
         * Project on Triangle
         */
        let ap = cell_pos - primitive.a;
        let bp = cell_pos - primitive.b;
        let cp = cell_pos - primitive.c;
        let ab = primitive.b - primitive.a;
        let ac = primitive.c - primitive.a;
        let bc = primitive.c - primitive.b;
        let n = cross(ab, ac);
        let n_length = length(n);

        if (n_length != 0.0
            && dot(cross(ab, n), ap) <= 0.0
            && dot(cross(bc, n), bp) <= 0.0
            && dot(cross(ac, n), cp) >= 0.0) // Positive sign due to calculating with `ac` instead of `ca`.
         {
             // This is a valid projection on the face interior.
             let signed_dist = dot(n, ap) / n_length;
             let sign = signed_dist < 0.0;
             let distance = abs(signed_dist);
             result.affinities |= (1u << collider_id)
                 | (uint(sign) << (collider_id + SIGN_BITS_SHIFT));

             if (distance < result.distance) {
                 result.distance = min(result.distance, distance);
                 result.closest_id = collider_id;
             }
        }
#endif
    }

    return result;
}

#if DIM == 2
    static const uint K_RANGE = 0;
#else
    static const uint K_RANGE = 1;
#endif


// TODO: this could just reuse the same function from p2g.slang?
func fetch_max_linked_lists_length(
    grid: StructuredBuffer<Grid>,
    hmap_entries: StructuredBuffer<GridHashMapEntry>,
    rigid_nodes_linked_lists: StructuredBuffer<NodeLinkedList>,
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
                    let len = rigid_nodes_linked_lists[global_node_id.id].len;
                    max_linked_list_length.max(len);
                }
            }
        }
    }
}

func fetch_nodes(
    grid: StructuredBuffer<Grid>,
    hmap_entries: StructuredBuffer<GridHashMapEntry>,
    rigid_nodes_linked_lists: StructuredBuffer<NodeLinkedList>,
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
                    let particle_id = rigid_nodes_linked_lists[global_node_id.id].head;
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
    StructuredBuffer<uint> particle_node_linked_lists,
    StructuredBuffer<Vect> collider_vertices,
    StructuredBuffer<RigidParticleIndices> rigid_particle_indices,
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
                    let rigid_idx = rigid_particle_indices[curr_particle_id];
                    shared_collider_ids[shared_flat_index] = rigid_idx.collider;

#if DIM == 2
                    shared_primitives[shared_flat_index] = Segment(
                        collider_vertices[rigid_idx.segment.x],
                        collider_vertices[rigid_idx.segment.y]
                    );
#else
                    shared_primitives[shared_flat_index] = Triangle(
                        collider_vertices[rigid_idx.triangle.x],
                        collider_vertices[rigid_idx.triangle.y],
                        collider_vertices[rigid_idx.triangle.z]
                    );
#endif

                    let next_particle_id = particle_node_linked_lists[curr_particle_id];
                    shared_nodes[shared_flat_index].particle_id = next_particle_id;
                } else {
                    // TODO: would it be worth skipping writing zeros if we already
                    //       did it at the previous step? (if we already reached the end
                    //       of the particle linked list)
                    shared_collider_ids[shared_flat_index] = NONE;
#if DIM == 2
                    shared_primitives[shared_flat_index] = Segment(Vect<float>(0.0), Vect<float>(0.0));
#else
                    shared_primitives[shared_flat_index] = Triangle(Vect<float>(0.0), Vect<float>(0.0), Vect<float>(0.0));
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
