//! Gather leg of the implicit solver: the particle side of the residual and of the operator.
//!
//! One workgroup per active block, mirroring G2P: a node vector (with its one-block halo) is
//! loaded into shared memory, then each primary particle gathers the corresponding velocity
//! gradient `sum_i w_i inv_d v_i (x_i - x_p)^T`.
//!
//! - In residual mode the vector is the trial velocity: the particle builds its trial deformation
//!   gradient `F (I + dt C)` (stored layout), evaluates its model there (projected for plastic
//!   models) and stores the Kirchhoff stress, the projected gradient and the incremental energy.
//! - In operator mode the vector is the CG search direction: the particle linearizes its model
//!   at the stored trial gradient and stores the stress differential.
//!
//! Both store `dt V0 inv_d` times the stress for the scatter leg.

module gather;

#include "slosh/features.h"

import slosh.solver.params;
import slosh.solver.particle;
import slosh.solver.implicit.types;
import slosh.grid.kernel;
import slosh.grid.grid;
import slosh.models.interfaces;
import slosh.models.implicit_interfaces;
import stensor.utils.mat;
import slosh.aliases;

#if COMPTIME
    import slosh.models.default;
    extern struct ImplicitParticleModel : IImplicitParticleModel = DefaultParticleModel;
    extern struct ParticleModel : IParticleModel = DefaultParticleModel;
#else
    extern struct ImplicitParticleModel : IImplicitParticleModel;
    extern struct ParticleModel : IParticleModel;
#endif

#if DIM == 2
static const uint WORKGROUP_SIZE_X = 8;
static const uint WORKGROUP_SIZE_Y = 8;
static const uint WORKGROUP_SIZE_Z = 1;
static const uint NUM_SHARED_CELLS = 10 * 10;
groupshared float2 shared_nodes_vel[NUM_SHARED_CELLS];
#else
static const uint WORKGROUP_SIZE_X = 4;
static const uint WORKGROUP_SIZE_Y = 4;
static const uint WORKGROUP_SIZE_Z = 4;
static const uint NUM_SHARED_CELLS = 6 * 6 * 6;
groupshared float3 shared_nodes_vel[NUM_SHARED_CELLS];
#endif

static const uint WORKGROUP_SIZE = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z;

/// Loads the node vector of the block and its halo into shared memory. Same octant walk as G2P.
func load_halo<let RESIDUAL: int>(
    grid: StructuredBuffer<Grid>,
    hmap_entries: StructuredBuffer<GridHashMapEntry>,
    cg_nodes: StructuredBuffer<CgNode>,
    tid: uint3,
    active_block_vid: BlockVirtualId
) {
    let base_block_pos_int = active_block_vid.id;

#if DIM == 2
    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            if ((i == 1 && tid.x > 1) || (j == 1 && tid.y > 1)) {
                continue;
            }
            let octant = uint2(i, j);
            let octant_hid = find_block_header_id(grid, hmap_entries, BlockVirtualId(base_block_pos_int + int2(octant)));
            let shared_index = octant * 8 + tid.xy;
            let flat_shared_index = flatten_shared_index(shared_index.x, shared_index.y);
            if (octant_hid.id != NONE) {
                let global_node_id = node_id(block_header_id_to_physical_id(octant_hid), tid.xy);
                let node = cg_nodes[global_node_id.id];
                shared_nodes_vel[flat_shared_index] = RESIDUAL != 0 ? node.v : node.p;
            } else {
                shared_nodes_vel[flat_shared_index] = float2(0.0);
            }
        }
    }
#else
    for (var i = 0u; i <= 1u; i++) {
        for (var j = 0u; j <= 1u; j++) {
            for (var k = 0u; k <= 1u; k++) {
                if ((i == 1 && tid.x > 1) || (j == 1 && tid.y > 1) || (k == 1 && tid.z > 1)) {
                    continue;
                }
                let octant = uint3(i, j, k);
                let octant_hid = find_block_header_id(grid, hmap_entries, BlockVirtualId(base_block_pos_int + int3(octant)));
                let shared_index = octant * 4 + tid;
                let flat_shared_index = flatten_shared_index(shared_index.x, shared_index.y, shared_index.z);
                if (octant_hid.id != NONE) {
                    let global_node_id = node_id(block_header_id_to_physical_id(octant_hid), tid);
                    let node = cg_nodes[global_node_id.id];
                    shared_nodes_vel[flat_shared_index] = RESIDUAL != 0 ? node.v : node.p;
                } else {
                    shared_nodes_vel[flat_shared_index] = float3(0.0);
                }
            }
        }
    }
#endif
}

/// Gathers the velocity gradient (residual mode) or its differential (operator mode) of one
/// particle and evaluates its model accordingly.
func particle_gather<let RESIDUAL: int>(
    particles_pos: StructuredBuffer<Position>,
    particles_props: StructuredBuffer<ParticleProperties>,
    particles_def_grad: StructuredBuffer<Mat<float>>,
    particles_model: RWByteAddressBuffer,
    implicit_particles: RWStructuredBuffer<ImplicitParticle>,
    particle_id: uint,
    cell_width: float,
    dt: float
) {
    var NBH_SHIFTS = NBH_SHIFTS;
    var NBH_SHIFTS_SHARED = NBH_SHIFTS_SHARED;

    let coeff = implicit_particles[particle_id].coeff;
    var stress = Mat<float>(0.0);
    var energy = 0.0;
    var trial_def_grad = particles_def_grad[particle_id];
    let props = particles_props[particle_id];

    // A zero coefficient marks a disabled particle. The gradient and volume checks also keep
    // both buffers referenced by the operator variant: wgpu drops unused bindings from the
    // derived layout while the reflection still lists them.
    let valid = coeff != 0.0 && determinant(trial_def_grad) > 0.0 && props.init_volume > 0.0;
    if (valid) {
        let particle_pos = particles_pos[particle_id];
        let inv_d = QuadraticKernel::inv_d(cell_width);
        let ref_elt_pos_minus_particle_pos = dir_to_associated_grid_node(particle_pos, cell_width);
        let w = QuadraticKernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);
        let assoc_cell_index_in_block = associated_cell_index_in_block_off_by_one(particle_pos, cell_width);
        let packed_cell_index_in_block = flatten_shared_index(
            assoc_cell_index_in_block.x,
            assoc_cell_index_in_block.y,
#if DIM == 3
            assoc_cell_index_in_block.z
#endif
        );

        var vel_grad = Mat<float>(0.0);

        [[ForceUnroll]]
        for (var i = 0u; i < NBH_LEN; i += 1u) {
            let shift = NBH_SHIFTS[i];
            let packed_shift = NBH_SHIFTS_SHARED[i];
            let shared_id = packed_cell_index_in_block + packed_shift;
            let cell_vel = shared_nodes_vel[shared_id];
#if DIM == 2
            let dpt = ref_elt_pos_minus_particle_pos + float2(shift) * cell_width;
            let weight = w[0][shift.x] * w[1][shift.y];
#else
            let dpt = ref_elt_pos_minus_particle_pos + float3(shift) * cell_width;
            let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];
#endif
            vel_grad += (weight * inv_d) * outer_product(cell_vel, dpt);
        }

        let init_volume = props.init_volume;
    #if SLOSH_PML
        let data = ParticleUpdateData(dt, cell_width, particle_id, vel_grad, init_volume, props.phase, props.stiffness_damping);
    #else
        let data = ParticleUpdateData(dt, cell_width, particle_id, vel_grad, init_volume, props.phase);
    #endif
        let implicit_model = ImplicitParticleModel();

        if (RESIDUAL != 0) {
            // Trial deformation gradient, mirroring the particle update: solids advance the full
            // tensor, fluids only their volume ratio (stored on the diagonal).
            let model = ParticleModel();
            let flags = model.model_flags(particles_model, particle_id);
            if ((flags & ModelFlags::FLUID) == 0) {
                trial_def_grad += mul(trial_def_grad, vel_grad * dt);
            } else {
                let d = trial_def_grad[0][0];
                trial_def_grad = diag(Vect<float>(d + d * trace(vel_grad) * dt));
            }
            let trial = implicit_model.trial_state(particles_model, data, trial_def_grad);
            stress = trial.kirchoff_stress * coeff;
            energy = trial.energy_density * init_volume;
        } else {
            let def_grad = implicit_particles[particle_id].trial_def_grad;
            let dtau = implicit_model.stress_differential(particles_model, data, def_grad, vel_grad);
            stress = dtau * coeff;
        }
    }

    implicit_particles[particle_id].stress = stress;
    if (RESIDUAL != 0) {
        implicit_particles[particle_id].trial_def_grad = trial_def_grad;
        implicit_particles[particle_id].energy = energy;
    }
}

func gather_impl<let RESIDUAL: int>(
    block_id: uint3,
    tid: uint3,
    tid_flat: uint,
    params: ConstantBuffer<SimulationParams>,
    grid: StructuredBuffer<Grid>,
    hmap_entries: StructuredBuffer<GridHashMapEntry>,
    active_blocks: StructuredBuffer<ActiveBlockHeader>,
    cg_nodes: StructuredBuffer<CgNode>,
    sorted_particle_ids: StructuredBuffer<uint>,
    particles_pos: StructuredBuffer<Position>,
    particles_props: StructuredBuffer<ParticleProperties>,
    particles_def_grad: StructuredBuffer<Mat<float>>,
    particles_model: RWByteAddressBuffer,
    implicit_particles: RWStructuredBuffer<ImplicitParticle>,
) {
    let bid = block_id.x;
    // HACK: force copying the virtual ID (naga bug, as in G2P).
    let vid_ = active_blocks[bid].virtual_id.id;
    let vid = BlockVirtualId(vid_);

    load_halo<RESIDUAL>(grid, hmap_entries, cg_nodes, tid, vid);
    GroupMemoryBarrierWithGroupSync();

    // Every particle is a primary of exactly one block, so walking the primaries covers all of
    // them.
    let max_particle_id = active_blocks[bid].first_particle + active_blocks[bid].num_particles;
    for (var sorted_particle_id = active_blocks[bid].first_particle + tid_flat;
         sorted_particle_id < max_particle_id;
         sorted_particle_id += WORKGROUP_SIZE) {
        let particle_id = sorted_particle_ids[sorted_particle_id];
        particle_gather<RESIDUAL>(particles_pos, particles_props, particles_def_grad, particles_model,
                                  implicit_particles, particle_id, grid[0].cell_width, params.dt);
    }
}

/// Gather leg of the operator: stress differentials along the CG search direction.
[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func gather_operator(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    uint tid_flat: SV_GroupIndex,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<CgNode> cg_nodes,
    StructuredBuffer<uint> sorted_particle_ids,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<ParticleProperties> particles_props,
    StructuredBuffer<Mat<float>> particles_def_grad,
    RWByteAddressBuffer particles_model,
    RWStructuredBuffer<ImplicitParticle> implicit_particles,
) {
    gather_impl<0>(block_id, tid, tid_flat, params, grid, hmap_entries, active_blocks, cg_nodes,
                   sorted_particle_ids, particles_pos, particles_props, particles_def_grad,
                   particles_model, implicit_particles);
}

/// Gather leg of the residual: trial state of every particle at the trial velocity.
[shader("compute")]
[numthreads(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)]
func gather_residual(
    uint3 block_id: SV_GroupID,
    uint3 tid: SV_GroupThreadID,
    uint tid_flat: SV_GroupIndex,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<GridHashMapEntry> hmap_entries,
    StructuredBuffer<ActiveBlockHeader> active_blocks,
    StructuredBuffer<CgNode> cg_nodes,
    StructuredBuffer<uint> sorted_particle_ids,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<ParticleProperties> particles_props,
    StructuredBuffer<Mat<float>> particles_def_grad,
    RWByteAddressBuffer particles_model,
    RWStructuredBuffer<ImplicitParticle> implicit_particles,
) {
    gather_impl<1>(block_id, tid, tid_flat, params, grid, hmap_entries, active_blocks, cg_nodes,
                   sorted_particle_ids, particles_pos, particles_props, particles_def_grad,
                   particles_model, implicit_particles);
}

#if DIM == 2
func outer_product(a: float2, b: float2) -> float2x2 {
    return float2x2(a * b.x, a * b.y);
}

func flatten_shared_index(x: uint, y: uint) -> uint {
    return x + y * 10;
}
#else
func outer_product(a: float3, b: float3) -> float3x3 {
    return float3x3(a * b.x, a * b.y, a * b.z);
}

func flatten_shared_index(x: uint, y: uint, z: uint) -> uint {
    return x + y * 6 + z * 6 * 6;
}
#endif
