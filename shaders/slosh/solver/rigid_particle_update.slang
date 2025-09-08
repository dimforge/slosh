module rigid_particle_update;

import nexus.dynamics.body;
import nexus.aliases;
import slosh.solver.particle;

static const uint WORKGROUP_SIZE = 64;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func transform_sample_points(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<RigidParticleIndices> rigid_particle_indices,
    StructuredBuffer<Pose> poses,
    StructuredBuffer<Vect> local_pts,
    RWStructuredBuffer<Vect> world_pts,
) {
    let id = invocation_id.x;

    if (id < local_pts.getCount()) {
        let collider_id = rigid_particle_indices[id].collider;
        let pose = poses[collider_id];
        world_pts[id] = pose.mul_pt(local_pts[id]);
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func transform_shape_points(
    uint3 invocation_id: SV_DispatchThreadID,
    StructuredBuffer<uint> vertex_collider_ids,
    StructuredBuffer<Pose> poses,
    StructuredBuffer<Vect> local_pts,
    RWStructuredBuffer<Vect> world_pts,
) {
    let id = invocation_id.x;

    if (id < local_pts.getCount()) {
        let collider_id = vertex_collider_ids[id];
        let pose = poses[collider_id];
        world_pts[id] = pose.mul_pt(local_pts[id]);
    }
}