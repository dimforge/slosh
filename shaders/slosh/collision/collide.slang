module collide;

import nexus.shapes.shape;
import nexus.aliases;
import slosh.grid.grid;

public func collide_cdf(
    StructuredBuffer<Shape> collision_shapes,
    StructuredBuffer<Pose> collision_shape_poses,
    float cell_width,
    Vect point,
) -> NodeCdf {
    static const float MAX_FLT = 1.0e10; // Is the f32::MAX constant defined somewhere in WGSL?
    var cdf = NodeCdf(MAX_FLT, 0u, NONE);

    let dist_cap = Vect<float>(cell_width * 1.5);

    // TODO: don’t  rely on the array length, e.g., if the user wants to
    //       preallocate the array to add more dynamically.
    for (var i = 0u; i < collision_shapes.getCount(); i++) {
        // FIXME: figure out a way to support more than 16 colliders.
        let shape = collision_shapes[i];
        let shape_pose = collision_shape_poses[i];
        let shape_type = shape.shape_type();
        if (shape_type != Shape::SHAPE_TYPE_POLYLINE && shape_type != Shape::SHAPE_TYPE_TRIMESH) {
            let proj = shape.project_point_on_boundary(shape_pose, point);
            let dpt = proj.point - point;

            if (proj.is_inside || all(abs(dpt) <= dist_cap)) {
                let dist = length(dpt);
                // TODO: take is_inside into account to select the deepest
                //       penetration as the closest collider?
                cdf.closest_id = select(dist < cdf.distance, i, cdf.closest_id);
                cdf.distance = min(cdf.distance, dist);
                cdf.affinities |= select(proj.is_inside, 0x00010001u, 0x00000001u) << i;
            }
        }
    }

    return cdf;
}

public struct GridCollision {
    public Vect normal;
    public uint collider_id;
    public bool valid;

    public __init(Vect normal, uint collider_id, bool valid) {
        this.normal = normal;
        this.collider_id = collider_id;
        this.valid = valid;
    }
}


public func collide(
    StructuredBuffer<Shape> collision_shapes,
    StructuredBuffer<Pose> collision_shape_poses,
    float cell_width,
    Vect point,
) -> GridCollision {
    static const float MAX_FLT = 1.0e10; // Is the f32::MAX constant defined somewhere in WGSL?
    var collision = GridCollision(Vect<float>(0.0), 0, false);

    let dist_cap = Vect<float>(cell_width * 1.5);
    var best_dist = MAX_FLT;

    // TODO: don’t  rely on the array length, e.g., if the user wants to
    //       preallocate the array to add more dynamically.
    [[ForceUnroll]]
    for (var i = 0u; i < 16; i++) {
        if (i >= collision_shapes.getCount()) {
            return collision;
        }

        let shape = collision_shapes[i];
        let shape_pose = collision_shape_poses[i];
        let shape_type = shape.shape_type();
        if (shape_type != Shape::SHAPE_TYPE_POLYLINE && shape_type != Shape::SHAPE_TYPE_TRIMESH) {
            let proj = shape.project_point_on_boundary(shape_pose, point);
            let dpt = proj.point - point;

            if (proj.is_inside || all(abs(dpt) <= dist_cap)) {
                let sign = (proj.is_inside) ? -1.0 : 1.0;
                let dist = length(dpt);
                if (sign * dist < best_dist && dist != 0.0) {
                    collision.normal = dpt / (dist * -sign);
                    collision.collider_id = i;
                    collision.valid = true;
                    best_dist = dist;
                }
            }
        }

        if (shape_type == Shape::SHAPE_TYPE_TRIMESH) {
            // TODO: project point on trimesh with pseudo-normals
        }
    }

    return collision;
}