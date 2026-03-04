module collide;

import slosh.rbd.shapes.shape;
import slosh.aliases;
import slosh.grid.grid;
import slosh.collision.trimesh;
import slosh.rbd.queries.projection;

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
    public Vect point;
    public uint collider_id;
    public bool valid;

    public __init(Vect normal, Vect point, uint collider_id, bool valid) {
        this.normal = normal;
        this.point = point;
        this.collider_id = collider_id;
        this.valid = valid;
    }
}


public func collide(
    StructuredBuffer<Shape> collision_shapes,
    StructuredBuffer<Pose> collision_shape_poses,
    StructuredBuffer<Vect> collision_shape_vtx,
    StructuredBuffer<uint> collision_shape_idx,
    float cell_width,
    Vect point,
) -> GridCollision {
    static const float MAX_FLT = 1.0e10; // Is the f32::MAX constant defined somewhere in WGSL?
    var collision = GridCollision(Vect<float>(0.0), Vect<float>(0.0), 0, false);

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

        var proj = ProjectionResult(point, false);
        bool handled = false;

        if (shape_type != Shape::SHAPE_TYPE_POLYLINE && shape_type != Shape::SHAPE_TYPE_TRIMESH) {
            proj = shape.project_point_on_boundary(shape_pose, point);
            handled = true;
        }

    #if DIM == 3
        if (shape_type == Shape::SHAPE_TYPE_TRIMESH) {
            let trimesh = shape.to_trimesh();
            let local_pt = shape_pose.inv_mul_pt(point);
            let local_proj = trimesh.project_local_point(collision_shape_idx, collision_shape_vtx, local_pt, dist_cap.x);

            if (local_proj._1) {
                proj = ProjectionResult(shape_pose.mul_pt(local_proj._0.point), local_proj._0.is_inside);
                handled = true;
            }
        }
    #endif

        if (handled) {
            let dpt = proj.point - point;

            if (proj.is_inside || all(abs(dpt) <= dist_cap)) {
                let sign = (proj.is_inside) ? -1.0 : 1.0;
                let dist = length(dpt);
                if (sign * dist < best_dist && dist != 0.0) {
                    collision.normal = dpt / (dist * -sign);
                    collision.point = proj.point;
                    collision.collider_id = i;
                    collision.valid = true;
                    best_dist = dist;
                }
            }
        }
    }

    return collision;
}