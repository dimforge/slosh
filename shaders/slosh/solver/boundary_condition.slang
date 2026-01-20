module boundary_condition;

import nexus.aliases;

public enum BoundaryConditionType {
    // Stick condition.
    Stick,
    // Tangential-only motion.
    Slip,
    // One-sided non-penetration condition with friction.
    Separate,
    // Lysmer-Kuhlemeyer dashpots
    // This boundary condition simulates a boundary made of an infinite amount of
    // material with the same elastic coefficients as the colliding material.
    // See https://docs.itascacg.com/flac3d700/3dec/docproject/source/options/dynamic/dynamic_boudaryconditions.html
    NonReflecting,
}

public struct BoundaryCondition {
    BoundaryConditionType type;
    float friction;

    public __init(BoundaryConditionType type, float friction) {
        this.type = type;
        this.friction = friction;
    }

    public func project_velocity(vel: Vect, n: Vect) -> Vect {
        switch (this.type) {
            case BoundaryConditionType::Stick: {
                return Vect<float>(0.0);
            }
            case BoundaryConditionType::Slip: {
                let normal_vel = dot(vel, n);
                let tangent_vel = vel - n * normal_vel;
                return tangent_vel;
            }
            case BoundaryConditionType::Separate: {
                let normal_vel = dot(vel, n);

                if (normal_vel < 0.0) {
                    let tangent_vel = vel - n * normal_vel;
                    let tangent_vel_len = length(tangent_vel);
                    let tangent_vel_dir = select(tangent_vel_len > 1.0e-8, tangent_vel / tangent_vel_len, Vect<float>(0.0));
                    return tangent_vel_dir * max(0.0, tangent_vel_len + this.friction * normal_vel);
                } else {
                    return vel;
                }
            }
            case BoundaryConditionType::NonReflecting: {
                return vel;
            }
            default: {
                return vel;
            }
        }
    }
}



