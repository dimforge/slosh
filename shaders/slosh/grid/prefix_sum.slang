module prefix_sum;


static const uint WORKGROUP_SIZE = 256;
groupshared uint workspace[WORKGROUP_SIZE];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func prefix_sum(
    uint3 block_id: SV_GroupID,
    uint3 thread_id: SV_GroupThreadID,
    RWStructuredBuffer<uint> data,
    RWStructuredBuffer<uint> aux,
) {
    let bid = block_id.x;
    let tid = thread_id.x;
    let data_len = data.getCount();

    if (bid * WORKGROUP_SIZE >= data_len) {
        return;
    }

    let data_block_len = data_len - bid * WORKGROUP_SIZE;
    let shared_len = clamp(next_power_of_two(data_block_len), 1u, WORKGROUP_SIZE);
    let elt_id = tid + bid * WORKGROUP_SIZE;

    // Init the shared memory.
    if (elt_id < data_len) {
        workspace[tid] = data[elt_id];
    } else {
        workspace[tid] = 0u;
    }

    // Up-sweep.
    {
        var d = shared_len / 2;
        var offset = 1u;
        while (d > 0) {
            GroupMemoryBarrierWithGroupSync();
            if (tid < d) {
                let ia = tid * 2u * offset + offset - 1u;
                let ib = (tid * 2u + 1u) * offset + offset - 1u;

                let sum = workspace[ia] + workspace[ib];
                workspace[ib] = sum;
            }

            d /= 2u;
            offset *= 2u;
        }
    }

    if (tid == 0) {
        let total_sum = workspace[shared_len - 1];
        aux[bid] = total_sum;
        workspace[shared_len - 1] = 0u;
    }

    // Down-sweep.
    {
        var d = 1u;
        var offset = shared_len / 2u;
        while (d < shared_len) {
            GroupMemoryBarrierWithGroupSync();
            if (tid < d) {
                let ia = tid * 2u * offset + offset - 1u;
                let ib = (tid * 2u + 1u) * offset + offset - 1u;

                let a = workspace[ia];
                let b = workspace[ib];

                workspace[ia] = b;
                workspace[ib] = a + b;
            }

            d *= 2u;
            offset /= 2u;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (elt_id < data_len) {
        data[elt_id] = workspace[tid];
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func add_data_grp(
    uint3 block_id: SV_GroupID,
    uint3 thread_id: SV_DispatchThreadID,
    RWStructuredBuffer<uint> data,
    RWStructuredBuffer<uint> aux,
) {
    let tid = thread_id.x;
    let bid = block_id.x;
    if (tid < data.getCount()) {
        data[tid] += aux[bid];
    }
}

// See Bit Twiddling Hack: https://graphics.stanford.edu/%7Eseander/bithacks.html#RoundUpPowerOf2
func next_power_of_two(val: uint) -> uint {
    var v = val;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}