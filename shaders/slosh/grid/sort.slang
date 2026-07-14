module sort;

import slosh.grid.grid;
import slosh.solver.particle;

// SLOSH_NODE_PARTICLE_LISTS gates the per-node particle linked-list build (see grid.slang).
// Default on so a standalone compile keeps the full behavior.
#ifndef SLOSH_NODE_PARTICLE_LISTS
#define SLOSH_NODE_PARTICLE_LISTS 1
#endif

// Returns the within-block sort bucket for a particle counted/inserted into its primary
// block: one bucket per associated-cell slab along the slowest-varying node axis (y in 2D,
// z in 3D).
func primary_sort_bucket(assoc: vector<uint, DIM>) -> uint {
#if DIM == 2
    return assoc.y;
#else
    return assoc.z;
#endif
}

// Returns the within-block sort bucket for a particle counted/inserted as an "extra" into the
// neighbour block shifted by `bshift` from its primary block. The associated slab relative to
// the neighbour block can be negative; slabs below -2 cannot influence any node of that block
// (the quadratic stencil only covers slabs [assoc, assoc + 2]), so they are clamped at -2.
func extra_sort_bucket(assoc: vector<uint, DIM>, bshift: vector<int, DIM>) -> uint {
#if DIM == 2
    let local = int(assoc.y) - bshift.y * 8;
#else
    let local = int(assoc.z) - bshift.z * 4;
#endif
    return NUM_PRIMARY_SORT_BUCKETS + uint(max(local, -2) + 2);
}

// True if a particle whose local base-cell index is `assoc` spills its quadratic stencil into
// the +1 neighbour block reached by `bshift` (component 0 or 1 per axis).
func extra_spills(assoc: vector<uint, DIM>, bshift: vector<int, DIM>) -> bool {
    let mask = vector<uint, DIM>(assoc >= EXTRA_PARTICLE_MIN_SHIFT);
    let bshift_u = vector<uint, DIM>(bshift);
    return all((bshift_u * mask) == bshift_u);
}

// Marks only each particle's primary (base) block as active.
//
// First half of the two-pass block activation that replaces `touch_particle_blocks`: instead
// of every particle redundantly inserting all NUM_ASSOC_BLOCKS of its stencil (~NUM_ASSOC_BLOCKS×
// redundancy), each particle inserts only its base block here, and `touch_neighbor_blocks` then
// inserts the +1 neighbour blocks once per active base block.
[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func touch_primary_blocks(
    uint3 invocation_id: SV_DispatchThreadID,
    RWStructuredBuffer<AtomicGrid> grid,
    RWStructuredBuffer<AtomicGridHashMapEntry> hmap_entries,
    RWStructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<Position> particles_pos,
    ConstantBuffer<uint> particles_len,
) {
    let id = invocation_id.x;
    if (id < particles_len) {
        let cell_width = grid[0].cell_width;
        let particle = particles_pos[id];
        let block = block_associated_to_point(cell_width, particle.pt);
        mark_block_as_active(grid, hmap_entries, active_blocks, block);
    }
}

// Marks the +1 neighbour blocks of every already-active base block as active.
//
// Second half of the two-pass block activation. One thread per base block: reads the block's
// virtual id and inserts its NUM_NBH_BLOCKS forward neighbours into the hashmap. `num_base_blocks`
// is a snapshot of `num_active_blocks` taken before this pass runs, so the neighbour blocks
// appended during the pass (at indices >= num_base_blocks) are not themselves processed.
[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func touch_neighbor_blocks(
    uint3 invocation_id: SV_DispatchThreadID,
    RWStructuredBuffer<AtomicGrid> grid,
    RWStructuredBuffer<AtomicGridHashMapEntry> hmap_entries,
    RWStructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<uint> num_base_blocks,
) {
    let id = invocation_id.x;
    if (id < num_base_blocks[0]) {
        let vid = active_blocks[id].virtual_id;
        var blocks = blocks_associated_to_block(vid);
        for (var i = 1u; i < NUM_ASSOC_BLOCKS; i += 1u) {
            mark_block_as_active(grid, hmap_entries, active_blocks, blocks[i]);
        }
    }
}

// Precomputes, for each active block, the header IDs of its +1 neighbour blocks.
//
// Each particle contributes "extras" to the +1 neighbour blocks of its primary block. Rather
// than re-querying the hashmap for those neighbours once per particle in the count/finalize
// passes, this resolves them once per active block and caches them in `nbh_block_ids`. Inactive
// neighbours are stored as NONE. Must run after all blocks have been touched.
[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func update_nbh_block_ids(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    RWStructuredBuffer<ActiveBlockHeader> active_blocks,
) {
    let id = invocation_id.x;
    if (id < grid[0].num_active_blocks) {
        let vid = active_blocks[id].virtual_id;
        var assoc = blocks_associated_to_block(vid);
        for (var nbh = 0u; nbh < NUM_NBH_BLOCKS; nbh += 1u) {
            let nbh_hid = find_block_header_id(grid, hmap_entries, assoc[nbh + 1]);
            active_blocks[id].nbh_block_ids[nbh] = nbh_hid;
        }
    }
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func touch_particle_blocks(
    uint3 invocation_id: SV_DispatchThreadID,
    RWStructuredBuffer<AtomicGrid> grid,
    RWStructuredBuffer<AtomicGridHashMapEntry> hmap_entries,
    RWStructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<Position> particles_pos,
    ConstantBuffer<uint> particles_len,
) {
    let _gs_n = particles_len;
    let _gs_total = min((_gs_n + (GRID_WORKGROUP_SIZE - 1u)) / GRID_WORKGROUP_SIZE, 65535u) * GRID_WORKGROUP_SIZE;
    for (var id = invocation_id.x; id < _gs_n; id += _gs_total) {
        let cell_width = grid[0].cell_width;
        let particle = particles_pos[id];
        var blocks = blocks_associated_to_point(cell_width, particle.pt);
        for (var i = 0u; i < NUM_ASSOC_BLOCKS; i += 1u) {
            mark_block_as_active(grid, hmap_entries, active_blocks, blocks[i]);
        }
    }
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func touch_rigid_particle_blocks(
    uint3 invocation_id: SV_DispatchThreadID,
    RWStructuredBuffer<AtomicGrid> grid,
    RWStructuredBuffer<AtomicGridHashMapEntry> hmap_entries,
    RWStructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<Position> rigid_particles_pos,
    StructuredBuffer<uint> rigid_particle_needs_block,
) {
    let id = invocation_id.x;
    if (id < rigid_particles_pos.getCount()) {
        let cell_width = grid[0].cell_width;
        let entry_id = id / 32u;
        let entry_bit = 1u << (id % 32u);
        let needs_block = (rigid_particle_needs_block[entry_id] & entry_bit) != 0;

        if (needs_block) {
            let particle = rigid_particles_pos[id];
            var block = block_associated_to_point(cell_width, particle.pt);
            mark_block_as_active(grid, hmap_entries, active_blocks, block);
        }
    }
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func mark_rigid_particles_needing_block(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<Position> rigid_particles_pos,
    RWStructuredBuffer<Atomic<uint>> rigid_particle_needs_block,
) {
    let id = invocation_id.x;
    if (id < rigid_particles_pos.getCount()) {
        let cell_width = grid[0].cell_width;
        let particle = rigid_particles_pos[id];
        // PERF: we should look at the local cell coordinates of the point
        //       in the block and only touch adjacent blocks if the
        //       cell coordinate along the corresponding dimension is 1
        //       or 2. Because particles at cells 0 or 1 won’t contribute
        //       to neighbor cells.
        var blocks = blocks_associated_to_point(cell_width, particle.pt);
        var i = 0u;

        for (; i < NUM_ASSOC_BLOCKS; i += 1u) {
            if (find_block_header_id(grid, hmap_entries, blocks[i]).id != NONE) {
                break;
            }
        }


        let entry_id = id / 32u;
        let entry_bit = 1u << (id % 32u);

        // PERF: this should be a workgroup reduction instead of
        //       global-memory atomics.
        if (i > 0u && i < NUM_ASSOC_BLOCKS) {
            rigid_particle_needs_block[entry_id].or(entry_bit);
        } else {
            rigid_particle_needs_block[entry_id].and(~entry_bit);
        }
    }
}

// TODO: can this kernel be combined with touch_particle_blocks?
[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func update_block_particle_count(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<Position> particles_pos,
    ConstantBuffer<uint> particles_len,
    RWStructuredBuffer<AtomicActiveBlockHeader> active_blocks,
) {
    let _gs_n = particles_len;
    let _gs_total = min((_gs_n + (GRID_WORKGROUP_SIZE - 1u)) / GRID_WORKGROUP_SIZE, 65535u) * GRID_WORKGROUP_SIZE;
    for (var id = invocation_id.x; id < _gs_n; id += _gs_total) {
        let cell_width = grid[0].cell_width;
        let particle = particles_pos[id];

        var blocks = blocks_associated_to_point(cell_width, particle.pt);
        let assoc = associated_cell_index_in_block_off_by_one(particle, cell_width);

        // The particle's primary (base) block gets it as a regular particle. Only the per-slab
        // bucket counter is incremented: `num_particles` (sum of primary buckets) and
        // `num_particles_with_extras` (sum of all buckets) are derived per block in the copy pass,
        // so we avoid two extra — and heavily contended — atomics per particle here.
        let active_block_id_0 = find_block_header_id(grid, hmap_entries, blocks[0]);
        active_blocks[active_block_id_0.id].sort_bucket_cursors[primary_sort_bucket(assoc)].add(1u);

        // Each +1 neighbour block also receives the particle as an "extra" if its quadratic
        // stencil actually spills into it. The neighbour header IDs were precomputed by
        // `update_nbh_block_ids`, so we read them from the primary block instead of doing a
        // hashmap lookup per particle.
        for (var i = 1u; i < NUM_ASSOC_BLOCKS; i += 1u) {
            let bshift = blocks[i].id - blocks[0].id;
            if (extra_spills(assoc, bshift)) {
                let block_i = active_blocks[active_block_id_0.id].nbh_block_ids[i - 1];
                active_blocks[block_i.id].sort_bucket_cursors[extra_sort_bucket(assoc, bshift)].add(1u);
            }
        }
    }
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func copy_particles_len_to_scan_value(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    RWStructuredBuffer<uint> scan_values,
) {
    let id = invocation_id.x;
    if (id < grid[0].num_active_blocks) {
        // The sorted array reserves room for every particle a block touches, extras included.
        // `num_particles_with_extras` is the sum of all slab buckets (the count pass no longer
        // maintains it as a separate atomic).
        var total = 0u;
        for (var k = 0u; k < NUM_SORT_BUCKETS; k += 1u) {
            total += active_blocks[id].sort_bucket_cursors[k];
        }
        scan_values[id] = total;
    }
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func copy_scan_values_to_first_particles_and_prepare_for_finalize(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<uint> scan_values,
    RWStructuredBuffer<ActiveBlockHeader> active_blocks,
) {
    let id = invocation_id.x;
    if (id < grid[0].num_active_blocks) {
        let first = scan_values[id];
        active_blocks[id].first_particle = first;
        // Convert the per-bucket counts accumulated by the count pass into running insertion
        // cursors. The cursors are *absolute* offsets into the sorted array (i.e. `first_particle`
        // is baked in), so the finalize pass can scatter each particle with a single atomic.
        // Primary buckets come first, so primaries land in [first_particle, first_particle +
        // num_particles) as the scatter P2G expects, with the extras after them; both segments
        // end up ordered by slab key.
        //
        // The running total advanced past the primary buckets is `num_particles`; the grand total
        // is `num_particles_with_extras`. Both are derived here rather than maintained as
        // per-particle atomics in the count pass.
        var running = first;
        for (var k = 0u; k < NUM_SORT_BUCKETS; k += 1u) {
            if (k == NUM_PRIMARY_SORT_BUCKETS) {
                active_blocks[id].num_particles = running - first;
            }
            let count = active_blocks[id].sort_bucket_cursors[k];
            active_blocks[id].sort_bucket_cursors[k] = running;
            running += count;
        }
        active_blocks[id].num_particles_with_extras = running - first;
    }
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func finalize_particles_sort(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<Position> particles_pos,
    ConstantBuffer<uint> particles_len,
#if SLOSH_NODE_PARTICLE_LISTS
    RWStructuredBuffer<AtomicNodeLinkedList> nodes_linked_lists,
    RWStructuredBuffer<uint> particle_node_linked_lists,
#endif
    RWStructuredBuffer<uint> sorted_particle_ids,
    RWStructuredBuffer<AtomicActiveBlockHeader> active_blocks,

) {
    let _gs_n = particles_len;
    let _gs_total = min((_gs_n + (GRID_WORKGROUP_SIZE - 1u)) / GRID_WORKGROUP_SIZE, 65535u) * GRID_WORKGROUP_SIZE;
    for (var id = invocation_id.x; id < _gs_n; id += _gs_total) {
        let cell_width = grid[0].cell_width;
        let particle = particles_pos[id];

        var blocks = blocks_associated_to_point(cell_width, particle.pt);
        let assoc = associated_cell_index_in_block_off_by_one(particle, cell_width);

        // Place the particle in its primary block's range. The prepare pass turned the bucket
        // counts into absolute insertion cursors (first_particle baked in), so the atomically
        // claimed slot is already the final sorted index.
        let active_block_id_0 = find_block_header_id(grid, hmap_entries, blocks[0]);
        let slot0 = active_blocks[active_block_id_0.id].sort_bucket_cursors[primary_sort_bucket(assoc)].add(1u);
        sorted_particle_ids[slot0] = id;

        // Place the particle as an "extra" into each +1 neighbour block whose stencil it spills
        // into, using the extra slab bucket cursors (extras land after the primaries because
        // their buckets come last). Neighbour header IDs are reused from `update_nbh_block_ids`.
        for (var i = 1u; i < NUM_ASSOC_BLOCKS; i += 1u) {
            let bshift = blocks[i].id - blocks[0].id;
            if (extra_spills(assoc, bshift)) {
                let block_i = active_blocks[active_block_id_0.id].nbh_block_ids[i - 1];
                let slot_i = active_blocks[block_i.id].sort_bucket_cursors[extra_sort_bucket(assoc, bshift)].add(1u);
                sorted_particle_ids[slot_i] = id;
            }
        }

#if SLOSH_NODE_PARTICLE_LISTS
        // Setup the per-node particle linked-list (still consumed by the gather P2G and the
        // rigid/CDF paths).
        let node_global_id = node_id(block_header_id_to_physical_id(active_block_id_0), assoc);
        let prev_head = nodes_linked_lists[node_global_id.id].head.exchange(id);
        nodes_linked_lists[node_global_id.id].len.add(1u);
        particle_node_linked_lists[id] = prev_head;
#endif
    }
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func sort_rigid_particles(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<Position> rigid_particles_pos,
#if SLOSH_NODE_PARTICLE_LISTS
    RWStructuredBuffer<AtomicNodeLinkedList> rigid_nodes_linked_lists,
    RWStructuredBuffer<uint> rigid_particle_node_linked_lists,
#endif
) {
    let id = invocation_id.x;
    if (id < rigid_particles_pos.getCount()) {
        let cell_width = grid[0].cell_width;
        let particle = rigid_particles_pos[id];
        let block_id = block_associated_to_point(cell_width, particle.pt);

        // Place the particle to its sorted place.
        let active_block_id = find_block_header_id(grid, hmap_entries, block_id);

        // NOTE: if the rigid particle doesn’t map to any block, we can just ignore it
        //       is it won’t affect the simulation.
#if SLOSH_NODE_PARTICLE_LISTS
        if (active_block_id.id != NONE) {
            // Setup the per-node rigid particle linked-list.
            let node_local_id = associated_cell_index_in_block_off_by_one(particle, cell_width);
            let node_global_id = node_id(block_header_id_to_physical_id(active_block_id), node_local_id);
            let prev_head = rigid_nodes_linked_lists[node_global_id.id].head.exchange(id);
            rigid_nodes_linked_lists[node_global_id.id].len.add(1u);
            rigid_particle_node_linked_lists[id] = prev_head;
        }
#endif
    }
}