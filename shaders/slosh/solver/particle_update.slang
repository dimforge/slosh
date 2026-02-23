module particle_update;

import slosh.solver.params;
import slosh.solver.particle;
import slosh.solver.boundary_condition;
import slosh.grid.kernel;
import slosh.grid.grid;
//import slosh.models.neo_hookean_elasticity;
import slosh.models.linear_elasticity;
import slosh.models.drucker_prager;
import slosh.models.interfaces;
import stensor.geometry.svd2;
import stensor.geometry.svd3;
import stensor.utils.mat;
import slosh.rbd.shapes.cuboid;
import slosh.aliases;

#if COMPTIME
    // Only include the default when checking compilation in build.rs which doesn't support specialization yet.
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
    RWStructuredBuffer<Kinematics> particles_kin,
    StructuredBuffer<Cdf> particles_cdf,
    RWStructuredBuffer<Mat<float>> particles_def_grad,
    RWStructuredBuffer<ParticleProperties> particles_props,
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
    var kin = particles_kin[particle_id];
    let cdf = particles_cdf[particle_id];
    var def_grad = particles_def_grad[particle_id];
    let props = particles_props[particle_id];
    let particle_pos = particles_pos[particle_id].pt;


    /*
     * Advection.
     */
     // TODO: double check that we never need the reprojection below.
     //       This reprojection isn't part of the original MPM-MLS/CPIC paper but we added
     //       it at some point as it appeared that we'd still get some penetrating particles.
     //       However, that might have been caused by other bugs so it is unsure if we need to
     //       keep it now.
     if (cdf.signed_distance < -0.05 * cell_width) {
         let slip = BoundaryCondition(BoundaryConditionType::Slip, 0.0);
         kin.velocity = cdf.rigid_vel + slip.project_velocity((kin.velocity - cdf.rigid_vel), cdf.normal);
     }

    // Clamp the max velocity a particle can get.
    // TODO: clamp the grid velocities instead?
    if (length(kin.velocity) > cell_width / dt) {
        kin.velocity = kin.velocity / length(kin.velocity) * cell_width / dt;
    }

    // Apply Rayleigh mass-proportional damping (implicit integration for stability).
    // v_new = v / (1 + damping * dt)
    kin.velocity = kin.velocity / (1.0 + props.damping * dt);


    // If the particle is fixed, clear its velocity and its affine matrix.
    // This isn't ideal (this should typically be handled on the grid) but we sometimes
    // need sub-grid-sized fixed particles.
    if (props.fixed != 0) {
        kin.velocity = Vect<float>(0.0);
    }

    let new_particle_pos = particle_pos + kin.velocity * dt;

    /*
     * Penalty impulse.
     */
     // TODO: apply the penalty impulse as an extra force on the grid instead of on
     //       changing the particle velocity directly?
     static const float PENALTY_COEFF = 1.0e3;
     if (cdf.signed_distance < -0.05 * cell_width) { // && cdf.signed_distance > -0.3 * cell_width {
         let corrected_dist = max(cdf.signed_distance, -0.3 * cell_width);
         let impulse = (dt * -corrected_dist * PENALTY_COEFF) * cdf.normal;
         kin.velocity += impulse; // / curr_particle_vol.mass;
     }

    /*
     * Deformation gradient update.
     */
    if ((flags & ModelFlags::FLUID) == 0) {
        // NOTE: the velocity gradient was stored in the affine buffer.
        def_grad += mul(def_grad, kin.affine * dt);
    } else {
        let def_grad0 = def_grad[0][0];
        var new_def_grad_diag_elt = def_grad0 + def_grad0 * kin.vel_grad_det * dt;
        def_grad = diag(Vect<float>(new_def_grad_diag_elt));
    }

    // NOTE: we only reset the velocity gradient to zero for fixed particles **after** the deformation
    //       gradient update so we keep its elastic response. It acts as a cube of elastic material which
    //       center doesn’t move but can deform.
    if (props.fixed != 0) {
        kin.affine = Mat<float>(0.0);
    }

    /*
     * Constitutive model.
     */

    let update_data = ParticleUpdateData(dt, cell_width, particle_id, kin.affine, props.init_volume, props.phase);
    let update_result = model.update(particles_model, update_data, def_grad);

    // Write back init_volume if the model modified it (e.g., swell).
    let effective_init_volume = select(update_result.init_volume >= 0.0, update_result.init_volume, props.init_volume);
    if (update_result.init_volume >= 0.0) {
        particles_props[particle_id].init_volume = update_result.init_volume;
    }

    /*
     * Affine matrix for APIC transfer.
     */
    let inv_d = QuadraticKernel::inv_d(cell_width);
    // NOTE: the velocity gradient was stored in the affine buffer.
    let affine = kin.affine * kin.mass - update_result.kirchoff_stress * (effective_init_volume * inv_d * dt);

    /*
     * Write back the new particle properties.
     */
    if (all(new_particle_pos == new_particle_pos) && determinant(def_grad) > 0.0) { // Avoid NaNs and invalid deformation gradients
        particles_pos[particle_id].pt = new_particle_pos;
        kin.affine = affine;
    } else {
        kin.enabled = 0; // This particles diverged, disable it.
        kin.velocity = Vect<float>(0.0);
        def_grad = diag(Vect<float>(1.0));
        kin.affine = Mat<float>(0.0);
        kin.mass = 0.0;
    }
    kin.force_dt = Vect<float>(0.0);

    particles_kin[particle_id] = kin;
    particles_def_grad[particle_id] = def_grad;
}
