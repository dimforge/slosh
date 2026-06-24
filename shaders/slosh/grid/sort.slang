module sort;

import slosh.grid.grid;
import slosh.solver.particle;

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
        let active_block_id_0 = find_block_header_id(grid, hmap_entries, blocks[0]);
        active_blocks[active_block_id_0.id].num_particles.add(1u);
        active_blocks[active_block_id_0.id].num_particles_with_extras.add(1u);

        let assoc = associated_cell_index_in_block_off_by_one(particle, cell_width);
        let mask = uint3(assoc >= 2);

        for (var i = 1u; i < NUM_ASSOC_BLOCKS; i += 1u) {
            let bshift = blocks[i].id - blocks[0].id;
            if (all((bshift * mask) == bshift)) {
                let active_block_id_i = find_block_header_id(grid, hmap_entries, blocks[i]);
                active_blocks[active_block_id_i.id].num_particles_with_extras.add(1u);
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
        scan_values[id] = active_blocks[id].num_particles_with_extras;
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
        active_blocks[id].first_particle = scan_values[id];
        active_blocks[id].num_particles_with_extras = active_blocks[id].num_particles;
        active_blocks[id].num_particles = 0u;
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
    RWStructuredBuffer<AtomicNodeLinkedList> nodes_linked_lists,
    RWStructuredBuffer<uint> particle_node_linked_lists,
    RWStructuredBuffer<uint> sorted_particle_ids,
    RWStructuredBuffer<AtomicActiveBlockHeader> active_blocks,

) {
    let _gs_n = particles_len;
    let _gs_total = min((_gs_n + (GRID_WORKGROUP_SIZE - 1u)) / GRID_WORKGROUP_SIZE, 65535u) * GRID_WORKGROUP_SIZE;
    for (var id = invocation_id.x; id < _gs_n; id += _gs_total) {
        let cell_width = grid[0].cell_width;
        let particle = particles_pos[id];

        // Place the particle to its sorted place.
        var blocks = blocks_associated_to_point(cell_width, particle.pt);
        let active_block_id_0 = find_block_header_id(grid, hmap_entries, blocks[0]);
        let target_index = active_blocks[active_block_id_0.id].first_particle +
            active_blocks[active_block_id_0.id].num_particles.add(1u);
        sorted_particle_ids[target_index] = id;

        let assoc = associated_cell_index_in_block_off_by_one(particle, cell_width);
        let mask = uint3(assoc >= 2);

        for (var i = 1u; i < NUM_ASSOC_BLOCKS; i += 1u) {
            let bshift = blocks[i].id - blocks[0].id;
            if (all((bshift * mask) == bshift)) {
                let active_block_id_i = find_block_header_id(grid, hmap_entries, blocks[i]);
                let target_index = active_blocks[active_block_id_i.id].first_particle +
                    active_blocks[active_block_id_i.id].num_particles_with_extras.add(1u);
                sorted_particle_ids[target_index] = id;
            }
        }

        // Setup the per-node particle linked-list.
        let node_local_id = associated_cell_index_in_block_off_by_one(particle, cell_width);
        let node_global_id = node_id(block_header_id_to_physical_id(active_block_id_0), node_local_id);
        let prev_head = nodes_linked_lists[node_global_id.id].head.exchange(id);
        nodes_linked_lists[node_global_id.id].len.add(1u);
        particle_node_linked_lists[id] = prev_head;
    }
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func sort_rigid_particles(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<Position> rigid_particles_pos,
    RWStructuredBuffer<AtomicNodeLinkedList> rigid_nodes_linked_lists,
    RWStructuredBuffer<uint> rigid_particle_node_linked_lists,
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
        if (active_block_id.id != NONE) {
            // Setup the per-node rigid particle linked-list.
            let node_local_id = associated_cell_index_in_block_off_by_one(particle, cell_width);
            let node_global_id = node_id(block_header_id_to_physical_id(active_block_id), node_local_id);
            let prev_head = rigid_nodes_linked_lists[node_global_id.id].head.exchange(id);
            rigid_nodes_linked_lists[node_global_id.id].len.add(1u);
            rigid_particle_node_linked_lists[id] = prev_head;
        }
    }
}