module boundary_condition;

import slosh.aliases;

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

    public __init(BoundaryConditionType type, float friction) {
        this.type = type;
        this.friction = friction;
    }

    public func project_velocity(vel: Vect, n: Vect, float dist, float dt) -> Vect {
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
                        let apply_friction = this.type == BoundaryConditionType::Separate ||
                            (this.type == BoundaryConditionType::FrictionZUp && n.z >= 0.0);
                        var friction = this.friction;
                        if (!apply_friction) {
                            friction = 0.0;
                        }
                        let tangent_vel = vel - n * normal_vel;
                        let tangent_vel_len = length(tangent_vel);
                        let tangent_vel_dir = select(tangent_vel_len > 1.0e-8, tangent_vel / tangent_vel_len, Vect<float>(0.0));
                        return tangent_vel_dir * max(0.0, tangent_vel_len + this.friction * normal_vel);
                    } else if (-normal_vel * dt > dist) {
                        // Predictive
                       return vel - (dist / dt + normal_vel) * n;
                   } else {
                       return vel;
                   }
                } else {
                    return vel;
                }
            }
            case BoundaryConditionType::NonReflecting: {
                return vel;
            }
            case BoundaryConditionType::Disabled:
            default: {
                return vel;
            }
        }
    }
}



