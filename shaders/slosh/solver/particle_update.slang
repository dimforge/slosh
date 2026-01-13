module particle_update;

import slosh.solver.params;
import slosh.solver.particle;
import slosh.grid.kernel;
import slosh.grid.grid;
//import slosh.models.neo_hookean_elasticity;
import slosh.models.linear_elasticity;
import slosh.models.drucker_prager;
import slosh.models.interfaces;
import stensor.geometry.svd2;
import stensor.geometry.svd3;
import stensor.utils.mat;
import nexus.shapes.cuboid;
import nexus.aliases;

#if COMPTIME
    // Only include the default when checking compilation in build.rs which doesn’t support specialization yet.
    import slosh.models.default;
    extern struct ParticleModel : IParticleModel = DefaultParticleModel;
#else
    extern struct ParticleModel : IParticleModel;
#endif


public struct Phase {
    public float phase;
    public float max_stretch;
}

[shader("compute")]
[numthreads(64, 1, 1)]
func particle_update(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<Grid> grid,
    RWByteAddressBuffer particles_model,
    RWStructuredBuffer<Position> particles_pos,
    RWStructuredBuffer<Dynamics> particles_dyn,
    ConstantBuffer<uint> particles_len,
) {
    let particle_id = invocation_id.x;

    if (particle_id >= particles_len) {
        return;
    }

    let model = ParticleModel();
    let flags = model.model_flags(particles_model, particle_id);
    let dt = params.dt;
    let cell_width = grid[0].cell_width;
    var dynamics = particles_dyn[particle_id];
    let particle_pos = particles_pos[particle_id].pt;

    /*
     * Advection.
     */
     // TODO: double check that we never need the reprojection below.
     //       This reprojection isn’t part of the original MPM-MLS/CPIC paper but we added
     //       it at some point as it appearered that we’d still get some penetrating particles.
     //       However, that might have been caused by other bugs so it is unsure if we need to
     //       keep it now.
     //    if (dynamics.cdf.signed_distance < -0.05 * cell_width) {
     //        dynamics.velocity = dynamics.cdf.rigid_vel + project_velocity((dynamics.velocity - dynamics.cdf.rigid_vel), dynamics.cdf.normal);
     //    }

    // Clamp the max velocity a particle can get.
    // TODO: clamp the grid velocities instead?
    if (length(dynamics.velocity) > cell_width / dt) {
        dynamics.velocity = dynamics.velocity / length(dynamics.velocity) * cell_width / dt;
    }

    // Apply Rayleigh mass-proportional damping (implicit integration for stability).
    // v_new = v / (1 + damping * dt)
    dynamics.velocity = dynamics.velocity / (1.0 + dynamics.damping * dt);

    let new_particle_pos = particle_pos + dynamics.velocity * dt;

    /*
     * Penalty impulse.
     */
     // TODO: apply the penalty impulse as an extra force on the grid instead of on
     //       changing the particle velocity directly?
//     static const float PENALTY_COEFF = 1.0e3;
//     if (dynamics.cdf.signed_distance < -0.05 * cell_width) { // && dynamics.cdf.signed_distance > -0.3 * cell_width {
//         let corrected_dist = max(dynamics.cdf.signed_distance, -0.3 * cell_width);
//         let impulse = (dt * -corrected_dist * PENALTY_COEFF) * dynamics.cdf.normal;
//         dynamics.velocity += impulse; // / curr_particle_vol.mass;
//     }

    /*
     * Deformation gradient update.
     */
    if ((flags & ModelFlags::FLUID) == 0) {
        // NOTE: the velocity gradient was stored in the affine buffer.
        dynamics.def_grad += mul(dynamics.def_grad, dynamics.affine * dt);
    } else {
        let def_grad0 = dynamics.def_grad[0][0];
        var new_def_grad_diag_elt = def_grad0 + def_grad0 * dynamics.vel_grad_det * dt;
        dynamics.def_grad = diag(Vect<float>(new_def_grad_diag_elt));
    }

    /*
     * Constitutive model.
     */
    let update_data = ParticleUpdateData(dt, cell_width, particle_id);
    let update_result = model.update(particles_model, update_data, dynamics);

    /*
     * Affine matrix for APIC transfer.
     */
    let inv_d = QuadraticKernel::inv_d(cell_width);
    // NOTE: the velocity gradient was stored in the affine buffer.
    let affine = dynamics.affine * dynamics.mass - update_result.kirchoff_stress * (dynamics.init_volume * inv_d * dt);

    /*
     * Write back the new particle properties.
     */
    if (all(new_particle_pos == new_particle_pos) && determinant(dynamics.def_grad) > 0.0) { // Avoid NaNs and invalid deformation gradients
        particles_pos[particle_id].pt = new_particle_pos;
        dynamics.affine = affine;
    } else {
        dynamics.enabled = 0; // This particles diverged, disable it.
        dynamics.velocity = Vect<float>(0.0);
        dynamics.def_grad = diag(Vect<float>(1.0));
        dynamics.affine = Mat<float>(0.0);
        dynamics.mass = 0.0;
    }
    dynamics.force_dt = Vect<float>(0.0);

    particles_dyn[particle_id] = dynamics;
}
