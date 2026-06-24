module timestep_bound;

import slosh.aliases;
import slosh.grid.grid;
import slosh.models.utils;
import slosh.models.interfaces;
import slosh.solver.particle;

#if COMPTIME
    // Only include the default when checking compilation in build.rs which doesn't support specialization yet.
    import slosh.models.default;
    extern struct ParticleModel : IParticleModel = DefaultParticleModel;
#else
    extern struct ParticleModel : IParticleModel;
#endif


struct GpuTimestepBounds {
    Atomic<uint> computed_max_dt_as_uint;

    static const float FLOAT_TO_INT = 1.0e12;
    static func secs_to_int(secs: float) -> uint {
        return uint(floor(secs * FLOAT_TO_INT));
    }
}

static const uint WORKGROUP_SIZE = 64;

[shader("compute")]
[numthreads(1, 1, 1)]
func reset_timestep_bound(
    RWStructuredBuffer<GpuTimestepBounds> result,
) {
    result[0].computed_max_dt_as_uint = 4294967295;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func estimate_timestep_bound(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Grid> grid,
    ByteAddressBuffer particles_model,
    StructuredBuffer<Kinematics> particles_kin,
    StructuredBuffer<Mat<float>> particles_def_grad,
    StructuredBuffer<ParticleProperties> particles_props,
    ConstantBuffer<uint> particles_len,
    RWStructuredBuffer<GpuTimestepBounds> result,
) {
    let _gs_n = particles_len;
    let _gs_total = min((_gs_n + (WORKGROUP_SIZE - 1u)) / WORKGROUP_SIZE, 65535u) * WORKGROUP_SIZE;
    for (var particle_id = invocation_id.x; particle_id < _gs_n; particle_id += _gs_total) {

    if (particles_kin[particle_id].enabled == 0) {
        continue;
    }

    let cell_width = grid[0].cell_width;

    /*
     * Model-specific restrictions.
     * Will usually be based on sounds speed for the material (section 4.1).
     */
    let density0 = particles_kin[particle_id].mass / particles_props[particle_id].init_volume;
    let def_grad = particles_def_grad[particle_id];
    let velocity = particles_kin[particle_id].velocity;
    let affine = particles_kin[particle_id].affine;
    let mass = particles_kin[particle_id].mass;

    let model = ParticleModel();
    var dt = model.timestep_bound(particles_model, particle_id, density0, def_grad, velocity, cell_width);

    /*
     * Velocity-based restrictions (section 4.2)
     */
   var norm_affine_squared = 0.0;
   [[ForceUnroll]]
   for (var i = 0; i < DIM; i++) {
      [[ForceUnroll]]
      for (var j = 0; j < DIM; j++) {
        norm_affine_squared += affine[i][j] * affine[i][j];
      }
   }

   let d = (cell_width * cell_width) / 4.0;
   let norm_b = d * sqrt(norm_affine_squared) / mass;
   let apic_v = norm_b * 6.0 * sqrt(DIM) / cell_width;
    let v = length(velocity); // + apic_v;
    dt = min(dt, cell_width / v);

    let candidate = GpuTimestepBounds::secs_to_int(dt);
    result[0].computed_max_dt_as_uint.min(candidate);
    }
}
