module rigid_impulse;

import slosh.solver.params;
import slosh.grid.grid;
import nexus.dynamics.body;
import nexus.aliases;

public struct IntegerImpulse {
    // HACK: we store the center of mass here to reduce the number
    //       of bindings needed for p2g.
    public Vect com;
    public Vect<int> linear;

#if DIM == 2
    public int angular;
#else
    public Vect<int> angular;
#endif
}

public struct IntegerImpulseAtomic {
    // HACK: we store the center of mass here to reduce the number
    //       of bindings needed for p2g.
#if DIM == 2
    public Vect com;
    public Atomic<int> linear_x;
    public Atomic<int> linear_y;
    public Atomic<int> angular;
    int padding;
#else
    public Vect com;
    public Atomic<int> linear_x;
    public Atomic<int> linear_y;
    public Atomic<int> linear_z;
    int padding_a;
    public Atomic<int> angular_x;
    public Atomic<int> angular_y;
    public Atomic<int> angular_z;
    int padding_b;
#endif
}


static const float FLOAT_TO_INT_FACTOR = 1e5;

public func flt2int(flt: float) -> int {
    return int(flt * FLOAT_TO_INT_FACTOR);
}

public func int2flt(i: int) -> float {
    return float(i) / FLOAT_TO_INT_FACTOR;
}

public func int_impulse_to_float(imp: IntegerImpulse) -> Impulse {
#if DIM == 2
    return Impulse(
        float2(int2flt(imp.linear.x), int2flt(imp.linear.y)),
        int2flt(imp.angular)
    );
#else
    return Impulse(
        float3(int2flt(imp.linear.x), int2flt(imp.linear.y), int2flt(imp.linear.z)),
        float3(int2flt(imp.angular.x), int2flt(imp.angular.y), int2flt(imp.angular.z))
    );
#endif
}

// NOTE: this is set to 16 exactly because we are currently limited to 16 bodies
//       due to the CPIC affinity bitmask size.
[shader("compute")]
[numthreads(16, 1, 1)]
func update(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<BodyMassProperties> local_mprops,
    RWStructuredBuffer<Pose> poses,
    RWStructuredBuffer<BodyVelocity> vels,
    RWStructuredBuffer<BodyMassProperties> mprops,
    RWStructuredBuffer<IntegerImpulse> incremental_impulses,
) {
    let id = invocation_id.x;

    if (id < vels.getCount()) {
        var inc_impulse = int_impulse_to_float(incremental_impulses[id]);

        // Reset the incremental impulse to zero for the next substep.
#if DIM == 2
        incremental_impulses[id] = IntegerImpulse(float2(0.0), int2(0), 0);
#else
        incremental_impulses[id] = IntegerImpulse(float3(0.0), int3(0), int3(0));
#endif

        // Apply impulse and integrate
        var new_vel = apply_impulse(mprops[id], vels[id], inc_impulse);

        // Cap the velocities to not move more than a fraction of a cell-width in a given substep.
        let linvel_norm = length(new_vel.linear);
        let angvel_norm = length(new_vel.angular);
        let lin_limit = 0.1 * grid[0].cell_width / params.dt;
        let ang_limit = 1.0; // TODO: what’s a good angular limit?

        if ((length(inc_impulse.linear) != 0.0 || length(inc_impulse.angular) != 0.0)) {
            if (linvel_norm > lin_limit) {
                new_vel.linear = new_vel.linear * (lin_limit / linvel_norm);
            }
            if (angvel_norm > ang_limit) {
                new_vel.angular = new_vel.angular * (ang_limit / angvel_norm);
            }
        }

        var new_pose = integrate_velocity(poses[id], new_vel, local_mprops[id].com, params.dt);

        // Apply gravity.
        let mass_mask = Vect<float>(mprops[id].inv_mass != Vect<float>(0.0));
        new_vel.linear += params.gravity * mass_mask * params.dt;

        vels[id] = new_vel;
        poses[id] = new_pose;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
func update_world_mass_properties(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<Pose> poses,
    StructuredBuffer<BodyMassProperties> local_mprops,
    RWStructuredBuffer<BodyMassProperties> mprops,
    RWStructuredBuffer<IntegerImpulse> incremental_impulses,
) {
    let id = invocation_id.x;

    if (id < mprops.getCount()) {
        let new_mprops = update_mprops(poses[id], local_mprops[id]);
        incremental_impulses[id].com = new_mprops.com;
        mprops[id] = new_mprops;
    }
}