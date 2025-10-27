module timestep_bound;

import slosh.models.utils;
import nexus.aliases;

public struct ElasticitySoundSpeedTimestepBound {
    public float alpha;
    public float bulk_modulus;
    public float shear_modulus;

    public __init(float alpha, float young_modulus, float poisson_ratio) {
        this.alpha = alpha;
        this.bulk_modulus = hook_to_bulk_modulus(young_modulus, poisson_ratio);
        this.shear_modulus = hook_to_shear_modulus(young_modulus, poisson_ratio);
    }

    public static func from_elasticity(float alpha, float young_modulus, float poisson_ratio) -> ElasticitySoundSpeedTimestepBound {
        return ElasticitySoundSpeedTimestepBound(
            alpha,
            hook_to_bulk_modulus(young_modulus, poisson_ratio),
            hook_to_shear_modulus(young_modulus, poisson_ratio),
        );
    }

    public func timestep_bound(
        density0: float,
        velocity: Vect<float>,
        cell_width: float,
    ) -> float {
        let c_dir = sqrt((this.bulk_modulus + 4.0 / 3.0 * this.shear_modulus)
            * 1.0 // TODO: particle.deformation_gradient.determinant()
            / density0);
        let max_denom = max(length(velocity), c_dir);
        return this.alpha * cell_width / max_denom;
    }
}


/*
static const uint WORKGROUP_SIZE = 64;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func estimate_timestep_bound(
) {
unsafe fn timestep_length_for_particle(
    min_dt: Real,
    max_dt: Real,
    particle_id: u32,
    particle_status: &ParticleStatus,
    particle_volume: &ParticleVolume,
    particle_vel: &ParticleVelocity,
    cell_width: Real,
    result: &mut GpuTimestepLength,
    updater: impl ParticleUpdater,
) {
    let particle_id = invocation_id.x;

    if (particle_id >= particle_len) {
        return;
    }

//    if particle_status.failed {
//        return;
//    }
//

    var dt = max_dt;
    let mut dt: Real = max_dt;

    let (active_timestep_bounds, candidate_dt) = updater.estimate_particle_timestep_length(
        cell_width,
        particle_id,
        particle_status,
        particle_volume,
        particle_vel,
    );
    dt = dt.min(candidate_dt);

    if active_timestep_bounds.contains(ActiveTimestepBounds::PARTICLE_VELOCITY_BOUND) {
        // Velocity-based restriction.
        // let norm_b = d * velocity_gradient.norm();
        // let apic_v = norm_b * 6.0 * (DIM as Real).sqrt() / cell_width;
        let v = particle_vel.vector.norm(); //  + apic_v;

        dt = dt.min(cell_width / v);
    }

    if dt < min_dt && max_dt > min_dt {
        dt = min_dt;
    }

    let candidate = GpuTimestepLength::from_sec(dt);
    result.0.global_red_min(candidate.0);
}

}
*/