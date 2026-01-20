/*
 * TODO: is there a way to re-use the G2P pattern accross shaders?
 *       In particular, this is very similar to the `g2p.wgsl` file
 *       but with different quantities being transfered.
 */

module g2p_cdf;

import stensor.linalg.inv;
import slosh.solver.params;
import slosh.solver.particle;
import slosh.grid.kernel;
import slosh.grid.grid;

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
static const uint NUM_SHARED_CELLS = 10 * 10; // block-size plus 2 from adjacent blocks: (8 + 2)^2
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
static const uint NUM_SHARED_CELLS = 6 * 6 * 6; // block-size plus 2 from adjacent blocks: (4 + 2)^3
#endif

groupshared NodeCdf shared_nodes[NUM_SHARED_CELLS];

static const uint WORKGROUP_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;

[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func g2p_cdf(
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
) {
    let bid = block_id.x;

    // HACK: force copying the virtual ID otherwise there is a naga bug when
    //       translating from WGSL to SpirV (works fine for Metal and HLSL…)
    let vid_ = active_blocks[bid].virtual_id.id;
    let vid = BlockVirtualId(vid_);

    // Block -> shared memory transfer.
    global_shared_memory_transfers(grid, hmap_entries, nodes, tid, vid);

    // Sync after shared memory initialization.
    GroupMemoryBarrierWithGroupSync();

    // Particle update. Runs g2p on shared memory only.
    let max_particle_id = active_blocks[bid].first_particle + active_blocks[bid].num_particles;

    for (var sorted_particle_id = active_blocks[bid].first_particle + tid_flat;
         sorted_particle_id < max_particle_id;
         sorted_particle_id += WORKGROUP_SIZE) {
        let particle_id = sorted_particle_ids[sorted_particle_id];
        particle_g2p(particles_pos, particles_dyn, particle_id, grid[0].cell_width, params.dt);
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
            let flat_id = flatten_shared_index(shared_index.x, shared_index.y);

            if (octant_hid.id != NONE) {
                let global_chunk_id = block_header_id_to_physical_id(octant_hid);
                let global_node_id = node_id(global_chunk_id, tid.xy);
                shared_nodes[flat_id] = nodes[global_node_id.id].cdf;
            } else {
                // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                // NOTE: we don’t need to init global_id since it’s only read for the
                //       current chunk that is guaranteed to exist, not the 2x2 adjacent ones.
                shared_nodes[flat_id] = NodeCdf(0.0, 0, NONE);
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
                let shared_node_id = flatten_shared_index(shared_index.x, shared_index.y, shared_index.z);

                if (octant_hid.id != NONE) {
                    let global_chunk_id = block_header_id_to_physical_id(octant_hid);
                    let global_node_id = node_id(global_chunk_id, tid);
                    shared_nodes[shared_node_id] = nodes[global_node_id.id].cdf;
                } else {
                    // This octant doesn’t exist. Fill shared memory with zeros/NONE.
                    // NOTE: we don’t need to init global_id since it’s only read for the
                    //       current chunk that is guaranteed to exist, not the 2x2x2 adjacent ones.
                    shared_nodes[shared_node_id] = NodeCdf(0.0, 0, NONE);
                }
            }
        }
    }
#endif
}

func particle_g2p(
    particles_pos: StructuredBuffer<Position>,
    particles_dyn: RWStructuredBuffer<Dynamics>,
    particle_id: uint,
    cell_width: float,
    dt: float
) {
    var contact_dist = 0.0;
    var contact_normal = float3(0.0);
    var particle_affinity = 0u;
    // TODO: would using a mat4 be faster?
    var affinity_signs: Array<float, 16> = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    };

    let prev_affinity = particles_dyn[particle_id].cdf.affinity;
    let particle_pos = particles_pos[particle_id];
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

    // Eqn. 21 to determine the sign bits.
    // Also combines the affinity masks.
    [[ForceUnroll]]
    for (var i = 0u; i < NBH_LEN; i += 1u) {
        let shift = NBH_SHIFTS[i];
        let packed_shift = NBH_SHIFTS_SHARED[i];
        let cell_data = shared_nodes[packed_cell_index_in_block + packed_shift];
        particle_affinity |= cell_data.affinities & AFFINITY_BITS_MASK;

#if DIM == 2
        let weight = w[0][shift.x] * w[1][shift.y];
#else
        let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];
#endif

        [[ForceUnroll]] // NOTE: this unrolling is essential for performances (x3 factor).
        for (var i_collider = 0u; i_collider < 16u; i_collider += 1u) {
            let compatible = float(affinity_bit(i_collider, cell_data.affinities));
            let sign = select(sign_bit(i_collider, cell_data.affinities) && !shape_has_solid_interior(i_collider), -1.0, 1.0);
            affinity_signs[i_collider] += compatible * weight * sign * cell_data.distance;
        }
    }

    // Convert the affinity signs to bits.
    [[ForceUnroll]] // NOTE: this unrolling is essential for performances (x3 factor).
    for (var i_collider = 0u; i_collider < 16u; i_collider += 1u) {
        // Only change the sign bit matching affinities that didn’t exist before.
        let mask = 1u << (i_collider + SIGN_BITS_SHIFT);
        if ((prev_affinity & (1u << i_collider)) == 0) {
            let sgn_bit = select(affinity_signs[i_collider] < 0.0, mask, 0u);
            particle_affinity |= sgn_bit;
        } else {
            particle_affinity |= prev_affinity & mask;
        }
    }

    // At this state the `affinity` (+ sign) bitmask is filled.
    // Now, compute the contact distance/normal using MLS reconstruction (Eq. 4)
#if DIM == 2
    var qtq = float3x3(0.0); // Matrix M
    var qtu = float3(0.0);
#else
    var qtq = float4x4(0.0); // Matrix M
    var qtu = float4(0.0);
#endif

    [[ForceUnroll]]
    for (var i = 0u; i < NBH_LEN; i += 1u) {
        let shift = NBH_SHIFTS[i];
        let packed_shift = NBH_SHIFTS_SHARED[i];
        var cell_data = shared_nodes[packed_cell_index_in_block + packed_shift];
#if DIM == 2
        let dpt = ref_elt_pos_minus_particle_pos + float2(shift) * cell_width;
        let weight = w[0][shift.x] * w[1][shift.y];
#else
        let dpt = ref_elt_pos_minus_particle_pos + float3(shift) * cell_width;
        let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];
#endif
        let combined_affinity = cell_data.affinities & particle_affinity & AFFINITY_BITS_MASK;
        let sign_differences = ((cell_data.affinities >> SIGN_BITS_SHIFT)
            ^ (particle_affinity >> SIGN_BITS_SHIFT)) & combined_affinity;

#if DIM == 2
        let p = float3(dpt, 1.0);
#else
        let p = float4(dpt, 1.0);
#endif

        if (combined_affinity != 0u) {
            if (sign_differences == 0u) {
                // All signs match, positive distance.
                qtq += outer_product(p, p) * weight;
                qtu += p * weight * cell_data.distance;
            } else { // if (sign_differences & (sign_differences - 1u)) != 0u {
                // Exactly one sign difference, negative distance.
                qtq += outer_product(p, p) * weight;
                qtu += p * weight * -cell_data.distance;
            }
        }
    }

    if (determinant(qtq) > 1.0e-8) {
#if DIM == 2
        let result = mul(qtu, inv(qtq));
        let len = length(result.xy);
        let normal = select(len > 1.0e-6, result.xy / len, float2(0.0));
        // PERF: init the rigid-velocities here instead of in g2p?
        particles_dyn[particle_id].cdf = Cdf(normal, float2(0.0), result.z, particle_affinity);
#else
        let result = mul(qtu, inv(qtq));
        let normal = result.xyz / length(result.xyz);
        particles_dyn[particle_id].cdf = Cdf(normal, float3(0.0), result.w, particle_affinity);
#endif
    } else {
        // TODO: store the affinity in this case too?
        particles_dyn[particle_id].cdf = Cdf();
    }
}

func shape_has_solid_interior(i_collider: uint) -> bool {
    // TODO: needs to be false for unoriented trimeshes and polylines,
    //       true for geometric primitives.
    return false;
}

// TODO: upstream to stensor?
#if DIM == 2
func outer_product(a: float3, b: float3) -> float3x3 {
    return float3x3(
        a * b.x,
        a * b.y,
        a * b.z,
    );
}

// Note that this is different from p2g. We don’t need to shift the index since the truncated
// blocks (the neighbor blocks) are in the quadrants with larger indices.
func flatten_shared_index(x: uint, y: uint) -> uint {
    return x + y * 10;
}
#else
func outer_product(a: float4, b: float4) -> float4x4 {
    return float4x4(
        a * b.x,
        a * b.y,
        a * b.z,
        a * b.w
    );
}


// Note that this is different from p2g. We don’t need to shift the index since the truncated
// blocks (the neighbor blocks) are in the octants with larger indices.
func flatten_shared_index(x: uint, y: uint, z: uint) -> uint {
    return x + y * 6 + z * 6 * 6;
}
#endif