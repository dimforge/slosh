module boundary_condition;

import slosh.aliases;
import slosh.collision.collide;

// Depth, in cells, of the `NonReflecting` absorbing band, graded linearly from full strength at
// the collider surface. A deep, gentle band absorbs much better than a shallow, aggressive one,
// and grading makes absorption nearly insensitive to how the surface aligns with the node lattice.
//
// Must not exceed `COLLISION_REPORT_CELLS`, past which no collision is reported at all.
static const float ABSORBING_LAYERS = 6.0;

public enum BoundaryConditionType {
    // Stick condition.
    Stick,
    // Tangential-only motion.
    Slip,
    // One-sided non-penetration condition with friction.
    Separate,
    // Same as Separate, but only apply friction if the collision normal has a positive z component.
    FrictionZUp,
    // Lysmer-Kuhlemeyer dashpots
    // This boundary condition simulates a boundary made of an infinite amount of
    // material with the same elastic coefficients as the colliding material.
    // See https://docs.itascacg.com/flac3d700/3dec/docproject/source/options/dynamic/dynamic_boudaryconditions.html
    NonReflecting,
    Disabled,
}

public struct BoundaryCondition {
    BoundaryConditionType type;
    float friction;
    // Pressure and shear wave speeds of the material in contact with the boundary.
    // Only read by the NonReflecting boundary condition.
    float wave_speed_p;
    float wave_speed_s;

    public __init(BoundaryConditionType type, float friction) {
        this.type = type;
        this.friction = friction;
        this.wave_speed_p = 0.0;
        this.wave_speed_s = 0.0;
    }

    public __init(BoundaryConditionType type, float friction, float wave_speed_p, float wave_speed_s) {
        this.type = type;
        this.friction = friction;
        this.wave_speed_p = wave_speed_p;
        this.wave_speed_s = wave_speed_s;
    }

    public func project_velocity(vel: Vect, n: Vect, float dist, float dt, float cell_width) -> Vect {
        switch (this.type) {
            case BoundaryConditionType::Stick: {
                if (dist <= 0.0) {
                    return Vect<float>(0.0);
                } else {
                    return vel;
                }
            }
            case BoundaryConditionType::Slip: {
                // TODO: add predictive constraint?
                if (dist <= 0.0) {
                    let normal_vel = dot(vel, n);
                    let tangent_vel = vel - n * normal_vel;
                    return tangent_vel;
                } else {
                    return vel;
                }
            }
            case BoundaryConditionType::FrictionZUp:
            case BoundaryConditionType::Separate: {
                let normal_vel = dot(vel, n);

                if (normal_vel < 0.0) {
                    if (dist <= 0.0) {
                        #if DIM == 3
                        let apply_friction = this.type == BoundaryConditionType::Separate ||
                            (this.type == BoundaryConditionType::FrictionZUp && n.z >= 0.0);
                        #else
                        let apply_friction = this.type == BoundaryConditionType::Separate ||
                            (this.type == BoundaryConditionType::FrictionZUp && n.y >= 0.0);
                        #endif
                        var friction = this.friction;
                        if (!apply_friction) {
                            friction = 0.0;
                        }
                        let tangent_vel = vel - n * normal_vel;
                        let tangent_vel_len = length(tangent_vel);
                        let tangent_vel_dir = select(tangent_vel_len > 1.0e-8, tangent_vel / tangent_vel_len, Vect<float>(0.0));
                        return tangent_vel_dir * max(0.0, tangent_vel_len + this.friction * normal_vel);
                    } else if (dist <= cell_width * 0.5 && -normal_vel * dt > dist) {
                        // Predictive. The `dist` bound keeps this to half a cell, so the deeper
                        // reporting band `NonReflecting` needs does not extend it.
                       return vel - (dist / dt + normal_vel) * n;
                   } else {
                       return vel;
                   }
                } else {
                    return vel;
                }
            }
            case BoundaryConditionType::NonReflecting: {
                // `dist` is the distance to the surface minus one cell. Full strength on and
                // inside the surface, fading out over `ABSORBING_LAYERS` cells.
                let depth = dist + cell_width;
                let weight = clamp(1.0 - depth / (ABSORBING_LAYERS * cell_width), 0.0, 1.0);

                if (weight <= 0.0) {
                    return vel;
                }

                // Lysmer-Kuhlemeyer viscous dashpots: the boundary applies the traction a
                // semi-infinite continuation of the material would, `t_n = -rho c_p v_n` and
                // `t_t = -rho c_s v_t`.
                //
                // The node mass cancels out of the resulting velocity change, leaving the wave's
                // CFL number `c * dt / h` as the damping factor; the clamp only guards against
                // wave speeds inconsistent with the current timestep.
                let normal_vel = dot(vel, n);
                let tangent_vel = vel - n * normal_vel;
                let damping_p = clamp(weight * this.wave_speed_p * dt / cell_width, 0.0, 1.0);
                let damping_s = clamp(weight * this.wave_speed_s * dt / cell_width, 0.0, 1.0);
                return n * (normal_vel * (1.0 - damping_p)) + tangent_vel * (1.0 - damping_s);
            }
            case BoundaryConditionType::Disabled:
            default: {
                return vel;
            }
        }
    }
}



