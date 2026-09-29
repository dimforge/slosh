//! Node-parallel vector kernels and reductions of the implicit solver: the conjugate gradient,
//! the Newton iteration and its backtracking line search.
//!
//! The solve runs entirely on the GPU. Every node-parallel kernel is dispatched indirectly from
//! one of three buffers (`newton_dispatch`, `cg_dispatch`, `ls_dispatch`) that the reduction
//! kernels arm or zero as the iteration proceeds, so the host encodes a fixed schedule and the
//! finished stages launch no work. Dot products and energies are reduced in two passes: one
//! partial per block (or per 64 particles), then a single workgroup sums the partials and updates
//! the scalars.

module cg;

#include "slosh/features.h"

import slosh.aliases;
import slosh.grid.grid;
import slosh.grid.kernel;
import slosh.collision.collide;
import slosh.solver.params;
import slosh.solver.particle;
import slosh.solver.boundary_condition;
import slosh.solver.implicit.types;
import stensor.utils.mat;

static const uint NODES_PER_BLOCK = 64;
static const uint PARTICLE_WORKGROUP_SIZE = 64;
static const uint REDUCE_WORKGROUP_SIZE = 256;

static const uint MODE_CG_INIT = 0;
static const uint MODE_CG_ALPHA = 1;
static const uint MODE_CG_BETA = 2;
static const uint MODE_NEWTON_CHECK = 3;
static const uint MODE_LS_PREPARE = 4;
static const uint MODE_LS_ENERGY = 5;
static const uint MODE_INIT_ENERGY = 6;
static const uint MODE_LS_RESIDUAL = 7;

groupshared float shared_dot[64];
groupshared float reduce_workspace[REDUCE_WORKGROUP_SIZE];

/// Sums `value` across a 64-thread workgroup through shared memory. Only thread 0 gets the result.
func workgroup_sum_64(tid: uint, value: float) -> float {
    shared_dot[tid] = value;
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        GroupMemoryBarrierWithGroupSync();
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
        }
    }
    return shared_dot[0];
}

/// Mass preconditioner: `v / m`, or zero on massless nodes.
func mass_precondition(v: Vect<float>, mass: float) -> Vect<float> {
    return mass > 0.0 ? v / mass : Vect<float>(0.0);
}

#if DIM == 2
func node_velocity(n: Node) -> Vect<float> {
    return n.momentum_velocity_mass.xy;
}
func node_mass(n: Node) -> float {
    return n.momentum_velocity_mass.z;
}
func with_velocity(n: Node, v: Vect<float>) -> Node {
    var result = n;
    result.momentum_velocity_mass = float3(v, n.momentum_velocity_mass.z);
    return result;
}
#else
func node_velocity(n: Node) -> Vect<float> {
    return n.momentum_velocity_mass.xyz;
}
func node_mass(n: Node) -> float {
    return n.momentum_velocity_mass.w;
}
func with_velocity(n: Node, v: Vect<float>) -> Node {
    var result = n;
    result.momentum_velocity_mass = float4(v, n.momentum_velocity_mass.w);
    return result;
}
#endif

/// Resets the scalars and arms the indirect dispatches.
[shader("compute")]
[numthreads(1, 1, 1)]
func cg_reset(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    RWStructuredBuffer<CgScalars> scalars,
    RWStructuredBuffer<uint> cg_dispatch,
    RWStructuredBuffer<uint> newton_dispatch,
    RWStructuredBuffer<uint> ls_dispatch,
    RWStructuredBuffer<uint> ls_particle_dispatch,
) {
    var sc = scalars[0];
    sc.rz = 0.0;
    sc.rz0 = 0.0;
    sc.p_ap = 0.0;
    sc.alpha = 0.0;
    sc.beta = 0.0;
    sc.cg_converged = 0;
    sc.cg_iters = 0;
    sc.g_sq = 0.0;
    sc.g0_sq = 0.0;
    sc.energy = 0.0;
    sc.energy_trial = 0.0;
    sc.gdx = 0.0;
    sc.ls_alpha = 0.0;
    sc.ls_state = LS_IDLE;
    sc.ls_trial = 0;
    sc.newton_iters = 0;
    sc.newton_done = 0;
    scalars[0] = sc;

    let n = grid[0].num_active_blocks;
    cg_dispatch[0] = n;
    cg_dispatch[1] = 1;
    cg_dispatch[2] = 1;
    newton_dispatch[0] = n;
    newton_dispatch[1] = 1;
    newton_dispatch[2] = 1;
    ls_dispatch[0] = 0;
    ls_dispatch[1] = 1;
    ls_dispatch[2] = 1;
    ls_particle_dispatch[0] = 0;
    ls_particle_dispatch[1] = 1;
    ls_particle_dispatch[2] = 1;
}

/// Per-particle coefficient `dt V0 inv_d` (zero when disabled) and reference volume.
[shader("compute")]
[numthreads(64, 1, 1)]
func prepare_particles(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<Kinematics> particles_kin,
    StructuredBuffer<ParticleProperties> particles_props,
    RWStructuredBuffer<ImplicitParticle> implicit_particles,
    ConstantBuffer<uint> particles_len,
) {
    let n = particles_len;
    let total = min((n + 63u) / 64u, 65535u) * 64u;
    for (var pid = invocation_id.x; pid < n; pid += total) {
        let enabled = particles_kin[pid].enabled;
        let init_volume = particles_props[pid].init_volume;
        let coeff = enabled != 0 ? params.dt * init_volume * QuadraticKernel::inv_d(grid[0].cell_width) : 0.0;
        implicit_particles[pid].coeff = coeff;
        implicit_particles[pid].init_volume = init_volume;
        implicit_particles[pid].energy = 0.0;
        implicit_particles[pid].stress = Mat<float>(0.0);
    }
}

/// Seeds the solve from the grid after the explicit grid update: with the stress left out of the
/// affine matrices, the nodes hold the force-free velocity `v_free` (gravity, clamp and boundary
/// conditions applied), so `rhs = m v_free` and the initial iterate is `v_free`. Nodes inside a
/// collider's margin with a stick or slip material get the matching projection; separating
/// contacts stay explicit corrections applied after the solve.
[shader("compute")]
[numthreads(4, 4, 4)]
func seed_from_grid(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<BoundaryCondition> body_materials,
    StructuredBuffer<NodeCollision> node_collisions,
    StructuredBuffer<Node> nodes,
    RWStructuredBuffer<CgNode> cg_nodes,
) {
    let bid = block_id.x;
    let vid = BlockVirtualId(active_blocks[bid].virtual_id.id);
    let cell_width = grid[0].cell_width;
    let global_chunk_id = block_header_id_to_physical_id(BlockHeaderId(bid));
#if DIM == 2
    let gid = node_id(global_chunk_id, tid.xy).id;
    let cell_pos = float2(vid.id * 8 + int2(tid.xy)) * cell_width;
#else
    let gid = node_id(global_chunk_id, tid).id;
    let cell_pos = float3(vid.id * 4 + int3(tid)) * cell_width;
#endif

    let node = nodes[gid];
    let mass = node_mass(node);
    let v = mass > 0.0 ? node_velocity(node) : Vect<float>(0.0);

    var bc = BC_NONE;
    var normal = Vect<float>(0.0);
    let collision = node_collisions[gid];
    if (collision.valid != 0u) {
        let dist_with_margin = dot(cell_pos - collision.point, collision.normal) - cell_width;
        if (dist_with_margin <= 0.0) {
            bc = body_materials[collision.collider_id].implicit_projection();
            normal = collision.normal;
        }
    }

    var cg = cg_nodes[gid];
    cg.rhs = v * mass;
    cg.x = v;
    cg.dx = Vect<float>(0.0);
    cg.v = v;
    cg.g = Vect<float>(0.0);
    cg.r = Vect<float>(0.0);
    cg.z = Vect<float>(0.0);
    cg.p = Vect<float>(0.0);
    cg.ap = Vect<float>(0.0);
    cg.normal = normal;
    cg.mass = mass;
    cg.bc = bc;
    cg_nodes[gid] = cg;
}

/// Sticks every node of a fixed particle's stencil: fixed particles are a Dirichlet condition on
/// the grid, where the explicit path only resets their velocity once per substep (a constraint
/// whose strength shrinks with the substep length). Stores are plain since every writer stores
/// the same value.
[shader("compute")]
[numthreads(64, 1, 1)]
func fixed_nodes(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<ParticleProperties> particles_props,
    StructuredBuffer<Kinematics> particles_kin,
    RWStructuredBuffer<CgNode> cg_nodes,
    ConstantBuffer<uint> particles_len,
) {
    var NBH_SHIFTS = NBH_SHIFTS;
    let n = particles_len;
    let total = min((n + 63u) / 64u, 65535u) * 64u;
    for (var pid = invocation_id.x; pid < n; pid += total) {
        if (particles_props[pid].fixed == 0 || particles_kin[pid].enabled == 0) {
            continue;
        }
        let cell_width = grid[0].cell_width;
        // Base node of the quadratic stencil, in cell coordinates.
        let assoc_cell = round(particles_pos[pid].pt / cell_width) - 1.0;

        [[ForceUnroll]]
        for (var i = 0u; i < NBH_LEN; i += 1u) {
            let shift = NBH_SHIFTS[i];
#if DIM == 2
            let cell = assoc_cell + float2(shift);
            let block = floor(cell / 8.0);
            let local = uint2(cell - block * 8.0);
            let vid = BlockVirtualId(int2(block));
#else
            let cell = assoc_cell + float3(shift);
            let block = floor(cell / 4.0);
            let local = uint3(cell - block * 4.0);
            let vid = BlockVirtualId(int3(block));
#endif
            let header = find_block_header_id(grid, hmap_entries, vid);
            if (header.id != NONE) {
                let gid = node_id(block_header_id_to_physical_id(header), local).id;
                cg_nodes[gid].bc = BC_STICK;
                cg_nodes[gid].normal = Vect<float>(0.0);
                // The seed kernel already put the (zero) fixed velocity in the iterate.
            }
        }
    }
}

/// Re-applies the cached rigid-collision boundary condition to the solved velocities: the
/// separating (contact) condition is nonlinear and stays an explicit correction, and the stick
/// and slip projections are idempotent on the solve's result.
[shader("compute")]
[numthreads(4, 4, 4)]
func apply_boundary_conditions(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<BoundaryCondition> body_materials,
    StructuredBuffer<NodeCollision> node_collisions,
    RWStructuredBuffer<Node> nodes,
) {
    let bid = block_id.x;
    let vid = BlockVirtualId(active_blocks[bid].virtual_id.id);
    let cell_width = grid[0].cell_width;
    let global_chunk_id = block_header_id_to_physical_id(BlockHeaderId(bid));
#if DIM == 2
    let gid = node_id(global_chunk_id, tid.xy).id;
    let cell_pos = float2(vid.id * 8 + int2(tid.xy)) * cell_width;
#else
    let gid = node_id(global_chunk_id, tid).id;
    let cell_pos = float3(vid.id * 4 + int3(tid)) * cell_width;
#endif

    let collision = node_collisions[gid];
    if (collision.valid != 0u) {
        let node = nodes[gid];
        let material = body_materials[collision.collider_id];
        let dist_with_margin = dot(cell_pos - collision.point, collision.normal) - cell_width;
        let projected = material.project_velocity(node_velocity(node), collision.normal, dist_with_margin, params.dt, cell_width);
        nodes[gid] = with_velocity(node, projected);
    }
}

/// Starts a CG solve of `A dx = -g` from `dx = 0`: `r = -g`, `z = M^-1 r`, `p = z`, plus the
/// block partial of `r . z`.
[shader("compute")]
[numthreads(64, 1, 1)]
func cg_init_residual(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    RWStructuredBuffer<CgNode> cg_nodes,
    RWStructuredBuffer<float> partials,
) {
    let gid = block_id.x * NODES_PER_BLOCK + tid;
    let node = cg_nodes[gid];
    let r = -node.g;
    let z = mass_precondition(r, node.mass);
    cg_nodes[gid].dx = Vect<float>(0.0);
    cg_nodes[gid].r = r;
    cg_nodes[gid].z = z;
    cg_nodes[gid].p = z;

    let sum = workgroup_sum_64(tid, dot(r, z));
    if (tid == 0) {
        partials[block_id.x] = sum;
    }
}

/// `dx += alpha p`, `r -= alpha A p`, `z = M^-1 r`, plus the block partial of the new `r . z`.
[shader("compute")]
[numthreads(64, 1, 1)]
func cg_update_solution(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    StructuredBuffer<CgScalars> scalars,
    RWStructuredBuffer<CgNode> cg_nodes,
    RWStructuredBuffer<float> partials,
) {
    let gid = block_id.x * NODES_PER_BLOCK + tid;
    let alpha = scalars[0].alpha;
    let node = cg_nodes[gid];
    let r = node.r - node.ap * alpha;
    let z = mass_precondition(r, node.mass);
    cg_nodes[gid].dx = node.dx + node.p * alpha;
    cg_nodes[gid].r = r;
    cg_nodes[gid].z = z;

    let sum = workgroup_sum_64(tid, dot(r, z));
    if (tid == 0) {
        partials[block_id.x] = sum;
    }
}

/// `p = z + beta p`.
[shader("compute")]
[numthreads(64, 1, 1)]
func cg_update_direction(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    StructuredBuffer<CgScalars> scalars,
    RWStructuredBuffer<CgNode> cg_nodes,
) {
    let gid = block_id.x * NODES_PER_BLOCK + tid;
    let beta = scalars[0].beta;
    let node = cg_nodes[gid];
    cg_nodes[gid].p = node.z + node.p * beta;
}

/// Opens a line search along the Newton direction: the current trial velocity becomes the base
/// `x`, and the block partial of `g . dx` gives the directional derivative.
[shader("compute")]
[numthreads(64, 1, 1)]
func ls_prepare(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    RWStructuredBuffer<CgNode> cg_nodes,
    RWStructuredBuffer<float> partials,
) {
    let gid = block_id.x * NODES_PER_BLOCK + tid;
    let node = cg_nodes[gid];
    cg_nodes[gid].x = node.v;

    let sum = workgroup_sum_64(tid, dot(node.g, node.dx));
    if (tid == 0) {
        partials[block_id.x] = sum;
    }
}

/// Trial velocity `v = x + alpha dx`, mirrored into the grid nodes, plus the block partial of the
/// kinetic term `1/2 m |v - v_free|^2` of the incremental potential.
[shader("compute")]
[numthreads(64, 1, 1)]
func ls_trial(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    StructuredBuffer<CgScalars> scalars,
    RWStructuredBuffer<Node> nodes,
    RWStructuredBuffer<CgNode> cg_nodes,
    RWStructuredBuffer<float> partials,
) {
    let gid = block_id.x * NODES_PER_BLOCK + tid;
    let alpha = scalars[0].ls_alpha;
    let node = cg_nodes[gid];
    let v = node.x + node.dx * alpha;
    cg_nodes[gid].v = v;
    nodes[gid] = with_velocity(nodes[gid], v);

    var kinetic = 0.0;
    if (node.mass > 0.0) {
        let dv = v - node.rhs / node.mass;
        kinetic = 0.5 * node.mass * dot(dv, dv);
    }
    let sum = workgroup_sum_64(tid, kinetic);
    if (tid == 0) {
        partials[block_id.x] = sum;
    }
}

/// After a failed line search, restores the base velocity so the grid holds the last accepted
/// iterate.
[shader("compute")]
[numthreads(64, 1, 1)]
func ls_finalize(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    StructuredBuffer<CgScalars> scalars,
    RWStructuredBuffer<Node> nodes,
    RWStructuredBuffer<CgNode> cg_nodes,
) {
    if (scalars[0].ls_state == LS_SEARCHING) {
        let gid = block_id.x * NODES_PER_BLOCK + tid;
        let x = cg_nodes[gid].x;
        cg_nodes[gid].v = x;
        nodes[gid] = with_velocity(nodes[gid], x);
    }
}

/// Block partials (per 64 particles) of the particle energies stored by the residual gather.
[shader("compute")]
[numthreads(64, 1, 1)]
func energy_particles(
    uint3 block_id: SV_GroupID,
    uint tid: SV_GroupIndex,
    StructuredBuffer<ImplicitParticle> implicit_particles,
    RWStructuredBuffer<float> partials_particles,
    ConstantBuffer<uint> particles_len,
) {
    let pid = block_id.x * PARTICLE_WORKGROUP_SIZE + tid;
    let energy = pid < particles_len ? implicit_particles[pid].energy : 0.0;
    let sum = workgroup_sum_64(tid, energy);
    if (tid == 0) {
        partials_particles[block_id.x] = sum;
    }
}

/// Sums `count` partials with the 256-thread workgroup; only thread 0 gets the result.
func reduce_partials(tid: uint, partials: StructuredBuffer<float>, count: uint) -> float {
    let num_chunks = (count + REDUCE_WORKGROUP_SIZE - 1) / REDUCE_WORKGROUP_SIZE;
    var acc = 0.0;
    for (var k = 0u; k < num_chunks; k++) {
        let idx = tid + k * REDUCE_WORKGROUP_SIZE;
        if (idx < count) {
            acc += partials[idx];
        }
    }
    reduce_workspace[tid] = acc;
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        GroupMemoryBarrierWithGroupSync();
        if (tid < stride) {
            reduce_workspace[tid] += reduce_workspace[tid + stride];
        }
    }
    let result = reduce_workspace[0];
    // Let every thread leave the workspace before a second reduction reuses it.
    GroupMemoryBarrierWithGroupSync();
    return result;
}

/// Sums the block partials in a single workgroup and advances the solve according to `MODE`. Runs
/// unconditionally; only the final write checks the state flags so the barriers stay in uniform
/// control flow.
func reduce_impl<let MODE: uint>(
    tid: uint,
    grid: StructuredBuffer<Grid>,
    cg_params: ConstantBuffer<CgParams>,
    particles_len: uint,
    partials: StructuredBuffer<float>,
    partials_particles: StructuredBuffer<float>,
    scalars: RWStructuredBuffer<CgScalars>,
    cg_dispatch: RWStructuredBuffer<uint>,
    newton_dispatch: RWStructuredBuffer<uint>,
    ls_dispatch: RWStructuredBuffer<uint>,
    ls_particle_dispatch: RWStructuredBuffer<uint>,
) {
    let num_blocks = grid[0].num_active_blocks;
    let num_particle_groups = (particles_len + PARTICLE_WORKGROUP_SIZE - 1) / PARTICLE_WORKGROUP_SIZE;
    let sum_nodes = reduce_partials(tid, partials, num_blocks);
    var sum_particles = 0.0;
    if (MODE == MODE_LS_ENERGY || MODE == MODE_INIT_ENERGY) {
        sum_particles = reduce_partials(tid, partials_particles, num_particle_groups);
    }

    if (tid == 0) {
        var sum = sum_nodes;
        var sc = scalars[0];

        // Keep every binding referenced in every mode (wgpu drops unused bindings from the
        // derived layout while the reflection still lists them). The condition never holds.
        if (sc.padding == 0xFFFFFFFFu) {
            sum += partials_particles[0] + cg_params.armijo + float(particles_len);
            cg_dispatch[0] = 0;
            newton_dispatch[0] = 0;
            ls_dispatch[0] = 0;
            ls_particle_dispatch[0] = 0;
        }

        if (MODE == MODE_INIT_ENERGY) {
            sc.energy = sum + sum_particles;
        } else if (MODE == MODE_NEWTON_CHECK) {
            if (sc.newton_done == 0) {
                sc.g_sq = sum;
                if (sc.newton_iters == 0) {
                    sc.g0_sq = sum;
                }
                // Relative tolerance on the residual norm (also stops on NaN or a zero residual).
                if (!(sum > cg_params.newton_rel_tol_sq * sc.g0_sq)) {
                    sc.newton_done = 1;
                    newton_dispatch[0] = 0;
                    cg_dispatch[0] = 0;
                    ls_dispatch[0] = 0;
                    ls_particle_dispatch[0] = 0;
                } else {
                    // Arm the CG solve of this iteration.
                    cg_dispatch[0] = num_blocks;
                    sc.cg_converged = 0;
                    sc.cg_iters = 0;
                    sc.rz = 0.0;
                }
                sc.ls_state = LS_IDLE;
            }
        } else if (MODE == MODE_CG_INIT) {
            if (sc.newton_done == 0 && sc.cg_converged == 0) {
                sc.rz0 = sum;
                sc.rz = sum;
                if (!(sum > 0.0)) {
                    sc.cg_converged = 1;
                    cg_dispatch[0] = 0;
                }
            }
        } else if (MODE == MODE_CG_ALPHA) {
            if (sc.newton_done == 0 && sc.cg_converged == 0) {
                sc.p_ap = sum;
                if (sum > 0.0 && sum <= 3.4028235e38) {
                    sc.alpha = sc.rz / sum;
                } else {
                    // Negative curvature or breakdown: keep the current iterate.
                    sc.alpha = 0.0;
                    sc.cg_converged = 1;
                    cg_dispatch[0] = 0;
                }
            }
        } else if (MODE == MODE_CG_BETA) {
            if (sc.newton_done == 0 && sc.cg_converged == 0) {
                let rz_new = sum;
                sc.beta = sc.rz > 0.0 ? rz_new / sc.rz : 0.0;
                sc.rz = rz_new;
                sc.cg_iters += 1;
                if (!(rz_new > cg_params.cg_rel_tol_sq * sc.rz0)) {
                    sc.cg_converged = 1;
                    cg_dispatch[0] = 0;
                }
            }
        } else if (MODE == MODE_LS_PREPARE) {
            if (sc.newton_done == 0) {
                sc.gdx = sum;
                if (!(sum < 0.0)) {
                    // Not a descent direction (or no direction at all): stop.
                    sc.newton_done = 1;
                    newton_dispatch[0] = 0;
                    cg_dispatch[0] = 0;
                    ls_dispatch[0] = 0;
                    ls_particle_dispatch[0] = 0;
                } else {
                    sc.ls_alpha = 1.0;
                    sc.ls_trial = 0;
                    sc.ls_state = LS_SEARCHING;
                    ls_dispatch[0] = num_blocks;
                    ls_particle_dispatch[0] = num_particle_groups;
                }
            }
        } else if (MODE == MODE_LS_ENERGY || MODE == MODE_LS_RESIDUAL) {
            if (sc.newton_done == 0 && sc.ls_state == LS_SEARCHING) {
                sc.ls_trial += 1;
                let max_trials = max(cg_params.line_search_steps, 1u);
                var accept = cg_params.line_search_steps == 0;
                if (MODE == MODE_LS_ENERGY) {
                    // Armijo sufficient decrease on the incremental potential (false on NaN).
                    let energy = sum_nodes + sum_particles;
                    sc.energy_trial = energy;
                    accept = accept || energy <= sc.energy + cg_params.armijo * sc.ls_alpha * sc.gdx;
                    if (accept) {
                        sc.energy = energy;
                    }
                } else {
                    // Decrease of the residual norm (false on NaN).
                    accept = accept || sum <= sc.g_sq * (1.0 - cg_params.armijo);
                    if (accept) {
                        sc.energy_trial = sum;
                    }
                }
                if (accept) {
                    sc.ls_state = LS_ACCEPTED;
                    sc.newton_iters += 1;
                    ls_dispatch[0] = 0;
                    ls_particle_dispatch[0] = 0;
                } else if (sc.ls_trial >= max_trials) {
                    // No acceptable step: the finalize kernel restores the base velocity and the
                    // Newton iteration stops.
                    sc.newton_done = 1;
                    newton_dispatch[0] = 0;
                    cg_dispatch[0] = 0;
                    ls_dispatch[0] = 0;
                    ls_particle_dispatch[0] = 0;
                } else {
                    sc.ls_alpha *= 0.5;
                }
            }
        }

        scalars[0] = sc;
    }
}

#define REDUCE_ENTRY_POINT(NAME, MODE) \
    [shader("compute")] \
    [numthreads(256, 1, 1)] \
    func NAME( \
        uint tid: SV_GroupIndex, \
        StructuredBuffer<Grid> grid, \
        ConstantBuffer<CgParams> cg_params, \
        ConstantBuffer<uint> particles_len, \
        StructuredBuffer<float> partials, \
        StructuredBuffer<float> partials_particles, \
        RWStructuredBuffer<CgScalars> scalars, \
        RWStructuredBuffer<uint> cg_dispatch, \
        RWStructuredBuffer<uint> newton_dispatch, \
        RWStructuredBuffer<uint> ls_dispatch, \
        RWStructuredBuffer<uint> ls_particle_dispatch, \
    ) { \
        reduce_impl<MODE>(tid, grid, cg_params, particles_len, partials, partials_particles, \
                          scalars, cg_dispatch, newton_dispatch, ls_dispatch, ls_particle_dispatch); \
    }

REDUCE_ENTRY_POINT(cg_reduce_init, MODE_CG_INIT)
REDUCE_ENTRY_POINT(cg_reduce_alpha, MODE_CG_ALPHA)
REDUCE_ENTRY_POINT(cg_reduce_beta, MODE_CG_BETA)
REDUCE_ENTRY_POINT(newton_check, MODE_NEWTON_CHECK)
REDUCE_ENTRY_POINT(ls_reduce_prepare, MODE_LS_PREPARE)
REDUCE_ENTRY_POINT(ls_reduce_energy, MODE_LS_ENERGY)
REDUCE_ENTRY_POINT(ls_reduce_residual, MODE_LS_RESIDUAL)
REDUCE_ENTRY_POINT(init_energy, MODE_INIT_ENERGY)
