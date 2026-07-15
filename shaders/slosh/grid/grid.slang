module grid;

import slosh.aliases;

// SLOSH_CPIC gates the CPIC parts of Node: the incompatible-momentum lane and the cdf collision
// field. register_shaders sets it from the cpic cargo feature so this layout matches the Rust
// GpuGridNode. Default on so a compile that never sets the macro still builds the full layout.
#ifndef SLOSH_CPIC
#define SLOSH_CPIC 1
#endif

// SLOSH_NODE_PARTICLE_LISTS gates the per-node particle linked lists (built in
// finalize_particles_sort, cleared in reset). Set from the node_particle_lists cargo feature; the
// bound buffers and the Rust GridArgs fields share it so everything stays in sync.
#ifndef SLOSH_NODE_PARTICLE_LISTS
#define SLOSH_NODE_PARTICLE_LISTS 1
#endif

// TODO: a lot if what exposed from this module should be methods of Grid (or other structs)
//       rather than free function. (For now they have just been converted from WGSL to Slang.)

public struct NodeLinkedListGeneric<MaybeAtomicUint> {
    public MaybeAtomicUint head;
    public MaybeAtomicUint len;
}

public typealias NodeLinkedList = NodeLinkedListGeneric<uint>;
public typealias AtomicNodeLinkedList = NodeLinkedListGeneric<Atomic<uint>>;

public static const uint G2P_P2G_WORKGROUP_SIZE = 64;
public static const uint NUM_CELL_PER_BLOCK = 64; // 8 * 8 in 2D and 4 * 4 * 4 in 3D.
// NOTE: this MUST match the NUM_CELL_PER_BLOCK. Some kernels (like reset) rely on it.
public static const uint GRID_WORKGROUP_SIZE = NUM_CELL_PER_BLOCK;

/*
 * Some index types.
 */
public struct BlockVirtualId {
    public vector<int, DIM> id;

    public __init(vector<int, DIM> i) {
        this.id = i;
    }
}

public struct BlockHeaderId {
    public uint id;

    public __init(uint i) {
        this.id = i;
    }
}

public struct BlockPhysicalId {
    public uint id;

    public __init(uint i) {
        this.id = i;
    }
}

public struct NodePhysicalId {
    public uint id;

    public __init(uint i) {
        this.id = i;
    }
}

/*
 *
 * HashMap for the grid.
 *
 */
public static const uint NONE = 0xffffffffu;

#if DIM == 2
func pack_key(key: BlockVirtualId) -> uint {
    return (bit_cast<uint, int>(key.id.x + 0x00007fff) & 0x0000ffffu) |
        ((bit_cast<uint, int>(key.id.y + 0x00007fff) & 0x0000ffffu) << 16);
}
#else
func pack_key(key: BlockVirtualId) -> uint {
    // NOTE: we give the X and Z axis one more bit than Y.
    //       This is assuming Y-up and the fact that we want
    //       more room on the X-Z plane rather than along the up axis.
    return (bit_cast<uint, int>(key.id.x + 0x000003ff) & 0x000007ffu) |
        ((bit_cast<uint, int>(key.id.y + 0x000001ff) & 0x000003ffu) << 11) |
        ((bit_cast<uint, int>(key.id.z + 0x000003ff) & 0x000007ffu) << 21);
}
#endif

func hash(packed_key: uint) -> uint {
    // Murmur3 hash function.
    var key = packed_key;
    key *= 0xcc9e2d51u;
    key = (key << 15) | (key >> 17);
    key *= 0x1b873593u;
    return key;
}

// IMPORTANT: if this struct is changed (including its layout), be sure to
//            modify the GpuGridHashMapEntry struct on the Rust side to ensure
//            it has the right size. Otherwise the hashmap will break.
public struct GridHashMapEntryGeneric<MaybeAtomicUint> {
    // Indicates if the entry is free or empty.
    public MaybeAtomicUint state;
    // The key stored on this entry.
    public BlockVirtualId key;
    // The associated value.
    public BlockHeaderId value;
}

public typealias GridHashMapEntry = GridHashMapEntryGeneric<uint>;
public typealias AtomicGridHashMapEntry = GridHashMapEntryGeneric<Atomic<uint>>;

// The hash map implementation is inspired from https://nosferalatu.com/SimpleGPUHashTable.html
func insertion_index(
    hmap_entries: RWStructuredBuffer<AtomicGridHashMapEntry>,
    capacity: uint,
    key: BlockVirtualId
) -> uint {
    let packed_key = pack_key(key);
    var slot = hash(packed_key) & (capacity - 1u);

    // NOTE: if there is no more room in the hashmap to store the data, we just do nothing.
    // It is up to the user to detect the high occupancy, resize the hashmap, and re-run
    // the failed insertion.
    for (var k = 0u; k < capacity; k++) {
        // TODO PERF: would it be more efficient to move the `state` into its own
        //       vector with only atomics?

        // TODO PERF: on platforms that support compare-exchange with stronger memory
        //            ordering, the infinite loop can be avoided.

        // FIXME: this will work on CUDA, but will likely fail spuriously on METAL/WebGPU
        //        because these targets only have compareExchangeWeak (which would have
        //        required an infinite loop like the one commented-out, but it doesn’t look
        //        like we have access to whether the operation succeeded).
        let old_value = hmap_entries[slot].state.compareExchange(NONE, packed_key);
        if (old_value == packed_key) {
            // The entry already exists.
            return NONE;
        } else if (old_value == NONE) {
            // We found a slot.
            hmap_entries[slot].key = key;
            return slot;
        } else {
            // Continue the search, this slot is already taken.
        }

        /*
        while (true) {
            let exch = hmap_entries[slot].state.compareExchange(NONE, packed_key);
            if (exch.exchanged) {
                // We found a slot.
                hmap_entries[slot].key = key;
                return slot;
            } else if (exch.old_value == packed_key) {
                return NONE;
            } else if (exch.old_value != NONE) {
                // The slot is already taken.
                break;
            }
            // Otherwise we need to loop since we hit a case where the exchange could
            // have happened but didn’t due to the weak nature of the operation.
        }
        */

        slot = (slot + 1u) % capacity & (capacity - 1u);
    }

    return NONE;
}

public func find_block_header_id(
    grid: StructuredBuffer<Grid>,
    hmap_entries: StructuredBuffer<GridHashMapEntry>,
    key: BlockVirtualId
) -> BlockHeaderId {
    let packed_key = pack_key(key);
    var slot = hash(packed_key) & (grid[0].hmap_capacity - 1);

    while (true) {
        let state = hmap_entries[slot].state;
        if (state == packed_key) {
            return hmap_entries[slot].value;
        } else if (state == NONE) {
            // Using break here instead of `return BlockHeaderId(NONE);` otherwise slang optimizes-out the
            // final return and naga complains about a missing return after the infinite loop.
            break;
        }

         slot = (slot + 1) & (grid[0].hmap_capacity - 1);
    }

    return BlockHeaderId(NONE);
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
void reset_hmap(
    uint3 invocation_id: SV_DispatchThreadID,
    RWStructuredBuffer<Grid> grid,
    RWStructuredBuffer<GridHashMapEntry> hmap_entries,
) {
    let id = invocation_id.x;
    if (id < grid[0].hmap_capacity) {
        // Only `state` needs clearing: insertion CAS-probes on `state` and the winner writes
        // `key`, and `find_block_header_id` reads `value` only after matching `state` (which is
        // always written before any lookup). Skipping key/value saves 44 of the 48 bytes/entry
        // on this clear, which sweeps the whole fixed-size hashmap every substep.
        hmap_entries[id].state = NONE;
    }
    if (id == 0) {
        grid[0].num_active_blocks = 0u;
    }
}

/*
 * Sparse grid definition.
 */
#if DIM == 2
public static const uint NUM_ASSOC_BLOCKS = 4;
#else
public static const uint NUM_ASSOC_BLOCKS = 8;
#endif
// Number of +1 neighbour blocks of a primary block (all associated blocks but the primary).
public static const uint NUM_NBH_BLOCKS = NUM_ASSOC_BLOCKS - 1;
public static const int OFF_BY_ONE = 1;

// Number of "slab" buckets for the within-block counting sort of regular particles whose
// primary block is the block being sorted. One bucket per associated-cell slab along the
// slowest-varying node axis (y in 2D, z in 3D), so the sorted order lets P2G derive tight
// per-chunk slab bounds for culling.
#if DIM == 2
public static const uint NUM_PRIMARY_SORT_BUCKETS = 8;
// Number of slab buckets for "extras" (particles spilling in from a neighbour block). Their
// associated slab relative to this block lies in [-2, block_width - 1], clamped at -2 (slabs
// below -2 cannot influence any node of the block anyway): [-2, 7] -> 10 buckets in 2D.
public static const uint NUM_EXTRA_SORT_BUCKETS = 10;
#else
public static const uint NUM_PRIMARY_SORT_BUCKETS = 4;
// [-2, 3] -> 6 buckets in 3D.
public static const uint NUM_EXTRA_SORT_BUCKETS = 6;
#endif
// Total number of within-block sort buckets. Primary buckets come first so primaries end up
// contiguous in [first_particle, first_particle + num_particles), which G2P relies on.
public static const uint NUM_SORT_BUCKETS = NUM_PRIMARY_SORT_BUCKETS + NUM_EXTRA_SORT_BUCKETS;

// Minimum local base-cell index along an axis for a particle's quadratic stencil to spill into
// the +1 neighbour block along that axis.
#if DIM == 2
public static const uint EXTRA_PARTICLE_MIN_SHIFT = 6;
#else
public static const uint EXTRA_PARTICLE_MIN_SHIFT = 2;
#endif

public struct ActiveBlockHeaderGeneric<MaybeAtomicUint> {
    public BlockVirtualId virtual_id; // Needed to compute the world-space position of a block.
    public uint first_particle;
    public MaybeAtomicUint num_particles_with_extras; // Total count of particles contributing to this block.
    public MaybeAtomicUint num_particles; // Count of particles assigned to this block exclusively.
    // Per-slab-bucket cursors for the within-block counting sort. The count pass accumulates
    // per-bucket counts here; the prepare pass converts them in place to absolute insertion
    // cursors (first_particle baked in); the finalize pass increments them while inserting.
    // The resulting segment is ordered by slab key, which P2G uses for per-chunk culling.
    public MaybeAtomicUint sort_bucket_cursors[NUM_SORT_BUCKETS];
    // Header IDs of the +1 neighbour blocks, precomputed once per block to avoid repeated
    // hashmap lookups in the count/finalize passes. Inactive neighbours are stored as NONE.
    public BlockHeaderId nbh_block_ids[NUM_NBH_BLOCKS];
}

public typealias ActiveBlockHeader = ActiveBlockHeaderGeneric<uint>;
public typealias AtomicActiveBlockHeader = ActiveBlockHeaderGeneric<Atomic<uint>>;

public struct GridGeneric<MaybeAtomic> {
    public MaybeAtomic num_active_blocks;
    public float cell_width;
    // NOTE: the hashmap capacity MUST be a power of 2.
    public uint hmap_capacity;
    public uint capacity;
}

public typealias Grid = GridGeneric<uint>;
public typealias AtomicGrid = GridGeneric<Atomic<uint>>;

public static const uint AFFINITY_BITS_MASK = 0x0000ffffu;
public static const uint SIGN_BITS_SHIFT = 16;

public struct NodeCdf {
    public float distance;
    // Two bits per collider.
    // The 16 first bits are for affinity, the 16 last are for signs.
    public uint affinities;
    // Index to the closest collider.
    public uint closest_id;

    public __init(float distance, uint affinities, uint closest_id) {
        this.distance = distance;
        this.affinities = affinities;
        this.closest_id = closest_id;
    }
}

public func affinity_bit(i_collider: uint, affinity: uint) -> bool {
    return (affinity & (1u << i_collider)) != 0;
}

public func sign_bit(i_collider: uint, affinity: uint) -> bool {
    return ((affinity >> SIGN_BITS_SHIFT) & (1u << i_collider)) != 0;
}

public func affinities_are_compatible(affinity1: uint, affinity2: uint) -> bool {
    let affinities_in_common = affinity1 & affinity2 & AFFINITY_BITS_MASK;
    let signs1 = (affinity1 >> SIGN_BITS_SHIFT) & affinities_in_common;
    let signs2 = (affinity2 >> SIGN_BITS_SHIFT) & affinities_in_common;
    return signs1 == signs2;
}

public struct Node {
    /// The first three components contains either the cell’s momentum or its velocity
    /// (depending on the context). The fourth component contains the cell’s mass.
    // TODO: maybe we don’t really need to pack ourself.
    public vector<float, DIM + 1> momentum_velocity_mass;
#if SLOSH_CPIC
    // If a particle is incompatible with this node (as pepr CPIC’s concept of compatibility
    // based on affinities), it will contribute to this field instead of `momentum_velocity_mass`.
    // That way the P2G/G2P transfers on incompatible nodes still work properly and we don’t
    // loose the contributions of other compatible particles influencing this incompatible node too.
    public vector<float, DIM + 1> momentum_velocity_mass_incompatible;
    public NodeCdf cdf;
#endif
}

#if DIM == 2
public func block_associated_to_point(cell_width: float, pt: float2) -> BlockVirtualId {
    let assoc_cell = round(pt / cell_width) - 1.0;
    let assoc_block = floor(assoc_cell / 8.0);
    return BlockVirtualId(int2(
        int(assoc_block.x),
        int(assoc_block.y),
    ));
}
#else
public func block_associated_to_point(cell_width: float, pt: float3) -> BlockVirtualId {
    let assoc_cell = round(pt / cell_width) - 1.0;
    let assoc_block = floor(assoc_cell / 4.0);
    return BlockVirtualId(int3(
        int(assoc_block.x),
        int(assoc_block.y),
        int(assoc_block.z),
    ));
}
#endif

public func blocks_associated_to_point(cell_width: float, pt: Vect) -> Array<BlockVirtualId, NUM_ASSOC_BLOCKS> {
    let main_block = block_associated_to_point(cell_width, pt);
    return blocks_associated_to_block(main_block);
}

public func blocks_associated_to_block(block: BlockVirtualId) -> Array<BlockVirtualId, NUM_ASSOC_BLOCKS> {
    #if DIM == 2
    return Array<BlockVirtualId, NUM_ASSOC_BLOCKS>(
        BlockVirtualId(block.id + int2(0, 0)),
        BlockVirtualId(block.id + int2(0, 1)),
        BlockVirtualId(block.id + int2(1, 0)),
        BlockVirtualId(block.id + int2(1, 1)),
    );
    #else
    return Array<BlockVirtualId, NUM_ASSOC_BLOCKS>(
        BlockVirtualId(block.id + int3(0, 0, 0)),
        BlockVirtualId(block.id + int3(0, 0, 1)),
        BlockVirtualId(block.id + int3(0, 1, 0)),
        BlockVirtualId(block.id + int3(0, 1, 1)),
        BlockVirtualId(block.id + int3(1, 0, 0)),
        BlockVirtualId(block.id + int3(1, 0, 1)),
        BlockVirtualId(block.id + int3(1, 1, 0)),
        BlockVirtualId(block.id + int3(1, 1, 1)),
    );
    #endif
}

public func mark_block_as_active(
    grid: RWStructuredBuffer<AtomicGrid>,
    hmap_entries: RWStructuredBuffer<AtomicGridHashMapEntry>,
    active_blocks: RWStructuredBuffer<ActiveBlockHeader>,
    block: BlockVirtualId
) {
    let slot = insertion_index(hmap_entries, grid[0].hmap_capacity, block);

    if (slot != NONE) {
        let block_header_id = grid[0].num_active_blocks.add(1u);
        active_blocks[block_header_id].virtual_id = block;
        active_blocks[block_header_id].first_particle = 0u;
        active_blocks[block_header_id].num_particles_with_extras = 0u;
        active_blocks[block_header_id].num_particles = 0u;
        for (var k = 0u; k < NUM_SORT_BUCKETS; k += 1u) {
            active_blocks[block_header_id].sort_bucket_cursors[k] = 0u;
        }
        hmap_entries[slot].value = BlockHeaderId(block_header_id);
    }
}

public func block_header_id_to_physical_id(hid: BlockHeaderId) -> BlockPhysicalId {
    return BlockPhysicalId(hid.id * NUM_CELL_PER_BLOCK);
}

#if DIM == 2
public func node_id(pid: BlockPhysicalId, shift_in_block: uint2) -> NodePhysicalId {
    return NodePhysicalId(pid.id + shift_in_block.x + shift_in_block.y * 8);
}
#else
public func node_id(pid: BlockPhysicalId, shift_in_block: uint3) -> NodePhysicalId {
    return NodePhysicalId(pid.id + shift_in_block.x + shift_in_block.y * 4 + shift_in_block.z * 4 * 4);
}
#endif

func div_ceil(x: uint, y: uint) -> uint {
    return (x + y - 1) / y;
}

// Snapshots the current `num_active_blocks` into a single-element buffer.
//
// Used by the two-pass block activation: after `touch_primary_blocks` has activated all
// particle base blocks, this captures their count so that `touch_neighbor_blocks` only
// iterates over base blocks (and not over the neighbour blocks it appends).
[shader("compute")]
[numthreads(1, 1, 1)]
func capture_num_active_blocks(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    RWStructuredBuffer<uint> num_base_blocks,
) {
    num_base_blocks[0] = grid[0].num_active_blocks;
}

[shader("compute")]
[numthreads(1, 1, 1)]
func init_indirect_workgroups(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    RWStructuredBuffer<uint> n_block_groups,
    RWStructuredBuffer<uint> n_g2p_p2g_groups
) {
    let num_active_blocks = grid[0].num_active_blocks;
    n_block_groups[0] = div_ceil(num_active_blocks, GRID_WORKGROUP_SIZE);
    n_block_groups[1] = 1;
    n_block_groups[2] = 1;
    n_g2p_p2g_groups[0] = num_active_blocks;
    n_g2p_p2g_groups[1] = 1;
    n_g2p_p2g_groups[2] = 1;
}

[shader("compute")]
[numthreads(GRID_WORKGROUP_SIZE, 1, 1)]
func reset(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    RWStructuredBuffer<Node> nodes,
#if SLOSH_NODE_PARTICLE_LISTS
    RWStructuredBuffer<NodeLinkedList> nodes_linked_lists,
    RWStructuredBuffer<NodeLinkedList> rigid_nodes_linked_lists,
#endif
) {
// TODO: slang doesn’t have a way to retrieve the number of workgroups?
//   let num_threads = num_workgroups.x * GRID_WORKGROUP_SIZE * num_workgroups.y * num_workgroups.z;
   let i = invocation_id.x;
   let num_nodes = grid[0].num_active_blocks * NUM_CELL_PER_BLOCK;
//   for (var i = invocation_id.x; i < num_nodes; i += num_threads) {
    if (i < num_nodes) {
       nodes[i].momentum_velocity_mass = vector<float, DIM + 1>(0.0);
#if SLOSH_CPIC
       nodes[i].momentum_velocity_mass_incompatible = vector<float, DIM + 1>(0.0);
       nodes[i].cdf = NodeCdf(0.0, 0, NONE);
#endif
#if SLOSH_NODE_PARTICLE_LISTS
       nodes_linked_lists[i].head = NONE;
       nodes_linked_lists[i].len = 0u;
       rigid_nodes_linked_lists[i].head = NONE;
       rigid_nodes_linked_lists[i].len = 0u;
#endif
   }
//   }
}
