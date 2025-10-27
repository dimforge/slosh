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

// Only include that when checking compilation in build.rs which doesn’t support specialization yet.
#if COMPILE_CHECK
    import slosh.models.specializations;
#endif

extern struct ParticleModel : IParticleModel;

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
//    RWStructuredBuffer<Phase> phases,
) {
    let particle_id = invocation_id.x;

    if (particle_id >= particles_len) {
        return;
    }

    let model = ParticleModel();
    let flags = model.model_flags(particles_model, particle_id);
    let dt = params.dt;
    let cell_width = grid[0].cell_width;
    let dynamics = particles_dyn[particle_id];
    let particle_pos = particles_pos[particle_id].pt;
    var new_particle_vel = dynamics.velocity;

    /*
     * Advection.
     */
    if (dynamics.cdf.signed_distance < -0.05 * cell_width) {
        new_particle_vel = dynamics.cdf.rigid_vel + project_velocity((new_particle_vel - dynamics.cdf.rigid_vel), dynamics.cdf.normal);
    }

    // Clamp the max velocity a particle can get.
    // TODO: clamp the grid velocities instead?
    if (length(new_particle_vel) > cell_width / dt) {
        new_particle_vel = new_particle_vel / length(new_particle_vel) * cell_width / dt;
    }

    let new_particle_pos = particle_pos + new_particle_vel * dt;

    /*
     * Penalty impulse.
     */
     static const float PENALTY_COEFF = 1.0e3;
     if (dynamics.cdf.signed_distance < -0.05 * cell_width) { // && dynamics.cdf.signed_distance > -0.3 * cell_width {
         let corrected_dist = max(dynamics.cdf.signed_distance, -0.3 * cell_width);
         let impulse = (dt * -corrected_dist * PENALTY_COEFF) * dynamics.cdf.normal;
         new_particle_vel += impulse; // / curr_particle_vol.mass;
     }

    /*
     * Deformation gradient update.
     */
    var new_deformation_gradient: Mat<float>;

    if ((flags & ModelFlags::FLUID) == 0) {
        // NOTE: the velocity gradient was stored in the affine buffer.
        new_deformation_gradient = dynamics.def_grad +
           mul(dynamics.def_grad, dynamics.affine * dt);
    } else {
        let def_grad0 = dynamics.def_grad[0][0];
        var new_def_grad_diag_elt = def_grad0 + def_grad0 * dynamics.vel_grad_det * dt;
        new_deformation_gradient = diag(Vect<float>(new_def_grad_diag_elt));
    }

    /*
     * Constitutive model.
     */
    let update_data = ParticleUpdateData(dt, cell_width, particle_id, dynamics.init_volume);
    let update_result = model.update(particles_model, update_data, dynamics.affine, new_deformation_gradient);

//    var phase = phases[particle_id].phase;
//
//    // Update Phase.
//    // TODO: should be stress based instead.
//    let max_stretch = phases[particle_id].max_stretch;
//    if (phase > 0.0 && max_stretch > 0.0) {
//    #if DIM == 2
//        let svd = Svd2(new_deformation_gradient);
//        if (svd.S.x > max_stretch || svd.S.y > max_stretch) {
//            phases[particle_id].phase = 0.0;
//            phase = 0.0;
//        }
//    #else
//        let svd = Svd3(new_deformation_gradient);
//        if (svd.S.x > max_stretch || svd.S.y > max_stretch || svd.S.z > max_stretch) {
//            phases[particle_id].phase = 0.0;
//            phase = 0.0;
//        }
//    #endif
//    }
//
//    // Plasticity.
//    if (phase == 0.0) {
//        let projection = plasticity[particle_id].project(plastic_state[particle_id], new_deformation_gradient);
//        plastic_state[particle_id] = projection.state;
//        new_deformation_gradient = projection.deformation_gradient;
//    }
//
//    // Elasticity.
//    let stress = constitutive_model[particle_id].kirchoff_stress(new_deformation_gradient);

    /*
     * Affine matrix for APIC transfer.
     */
    let inv_d = QuadraticKernel::inv_d(cell_width);
    // NOTE: the velocity gradient was stored in the affine buffer.
    let affine = dynamics.affine * dynamics.mass - update_result.kirchoff_stress * (dynamics.init_volume * inv_d * dt);

    /*
     * Write back the new particle properties.
     */
    if (all(new_particle_pos == new_particle_pos) && determinant(new_deformation_gradient) > 0.0) { // Avoid NaNs and invalid deformation gradients
        particles_pos[particle_id].pt = new_particle_pos;
        particles_dyn[particle_id].velocity = new_particle_vel;
        particles_dyn[particle_id].def_grad = new_deformation_gradient;
        particles_dyn[particle_id].affine = affine;
    } else {
        particles_dyn[particle_id].enabled = 0; // This particles diverged, disable it.
        particles_dyn[particle_id].velocity = Vect<float>(0.0);
        particles_dyn[particle_id].def_grad = diag(Vect<float>(1.0));
        particles_dyn[particle_id].affine = Mat<float>(0.0);
        particles_dyn[particle_id].mass = 0.0;
    }
}
