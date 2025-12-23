module trimesh;

import nexus.shapes.triangle;
import nexus.queries.projection;
import nexus.aliases;

// A triangle mesh.
//
// The triangle mesh is composed of multiple buffers packed into a Vec<uint> (the "index buffer") and a Vec<Vect> (the "vertex buffer").
// The "index buffer" contains: [BVH topology, triangle indices]
// The "vertex buffer" contains: [BVH AABBs, triangle vertices, vertex pseudo-normals, edges pseudo-normals].
// The `vertex pseudo-normal` contains one Vect per vertex, in the same order as the vertices in `triangle vertices`.
// The `edges pseudo-normal` contains three Vect per triangle (one for each edge), in the same order as `triangle indices`.
public struct TriMesh {
    /// Index of the root AABB in the vertex buffer.
    uint bvh_vtx_root_id;
    /// The root AABB left-child index.
    uint bvh_idx_root_id;
    // The number of BVH nodes. Triangle indices are stored after the last bvh node.
    uint bvh_node_len;
    // The total number of triangles in the mesh.
    uint num_triangles;
    // The total number of vertices in the mesh.
    uint num_vertices;
//    // The mesh’s AABB.
//    Aabb root_aabb;

//    /// Simply return the root aabb.
//    public func aabb() -> Aabb {
//        return this.root_aabb;
//    }

    public func bvh_node_aabb(
        vtx: StructuredBuffer<Vect>,
        node_id: uint
    ) -> Aabb {
        let vid = this.bvh_vtx_root_id + node_id * 2u; // Multiply by 2 since there are two values per AABB (min/max).
        return Aabb(vtx[vid], vtx[vid + 1]);
    }

    public func bvh_node_idx(
        idx: StructuredBuffer<uint>,
        node_id: uint
    ) -> BvhIdx {
        let base_id = this.bvh_idx_root_id + node_id * 3u; // 3 integers per node for the tree topolgy.
        return BvhIdx(idx[base_id], idx[base_id + 1], idx[base_id + 2]);
    }

    public func triangle_vids(
        idx: StructuredBuffer<uint>,
        tri_id: uint
    ) -> uint3 {
        let base_id = this.bvh_idx_root_id + this.bvh_node_len * 3u + tri_id * 3u;
        let base_vid = this.bvh_vtx_root_id + this.bvh_node_len * 2u;
        let a = base_vid + idx[base_id];
        let b = base_vid + idx[base_id + 1];
        let c = base_vid + idx[base_id + 2];
        return uint3(a, b, c);
    }

    public func triangle(
        idx: StructuredBuffer<uint>,
        vtx: StructuredBuffer<Vect>,
        tri_id: uint
    ) -> Triangle {
        let vids = this.triangle_vids(idx, tri_id);
        let a = vtx[vids.x];
        let b = vtx[vids.y];
        let c = vtx[vids.z];
        return Triangle(a, b, c);
    }

    public func pseudo_normal(
        idx: StructuredBuffer<uint>,
        vtx: StructuredBuffer<Vect>,
        tri_id: uint,
        feat_type: uint,
        feat_id: uint
    ) -> Vect {
        switch (feat_type) {
            case FEATURE_VERTEX: {
                let vids = this.triangle_vids(idx, tri_id);
                return vtx[vids[feat_id] + this.num_vertices];
            }
            case FEATURE_EDGE: {
                let base_vid = this.bvh_vtx_root_id
                    // Two points per BVH node.
                    + this.bvh_node_len * 2u
                    // Triangle vertices.
                    + this.num_vertices
                    // One pseudo-normal per vertex.
                    + this.num_vertices
                    // Three pseudo-normal per triangle (one per edge).
                    + tri_id * 3u;
                return vtx[base_vid + feat_id];
            }
            #if DIM == 3
            case FEATURE_SOLID:
            case FEATURE_FACE: {
                let tri = this.triangle(idx, vtx, tri_id);
                let ab = tri.b - tri.a;
                let ac = tri.c - tri.a;
                return cross(ab, ac);
            }
            #endif
            default: { return Vect<float>(0.0); }
        }
    }

    public func project_local_point(
        // Contains BVH topology, triangle indices
        StructuredBuffer<uint> idx,
        // Contains AABBs, vertices, and pseudo-normals.
        StructuredBuffer<Vect> vtx,
        Vect pt,
    ) -> ProjectionResult {
        var curr = 0u;
        var best = 1.0e10;
        var best_proj = ProjectionWithLocation.solid(pt);
        var best_tri_id = 0;

        while (curr < this.bvh_node_len) {
            let node_idx = this.bvh_node_idx(idx, curr);
            if (node_idx.entry_index == 0xffffffffu) {
                // This is a leaf.
                let tri = this.triangle(idx, vtx, node_idx.shape_index);
                let proj = triangle_project_local_point_and_get_location(tri, pt);
                let dist = length(proj.point - pt);
                if (dist < best) {
                    best = dist;
                    best_proj = proj;
                    best_tri_id = node_idx.shape_index;
                }

                // Continue traversal.
                curr = node_idx.exit_index;
            } else {
                let aabb = this.bvh_node_aabb(vtx, curr);
                let proj = aabb.project_local_point(pt);
                if (length(proj - pt) < best) {
                    curr = node_idx.entry_index;
                } else {
                    curr = node_idx.exit_index;
                }
            }
        }

        let pn = this.pseudo_normal(idx, vtx, best_tri_id, best_proj.feature_type, best_proj.id);
        let is_inside = dot(pn, pt - best_proj.point) <= 0.0;
        return ProjectionResult(best_proj.point, is_inside);
    }
}

public struct BvhIdx {
    uint entry_index;
    uint exit_index;
    uint shape_index;

    public __init(uint entry_index, uint exit_index, uint shape_index) {
        this.entry_index = entry_index;
        this.exit_index = exit_index;
        this.shape_index = shape_index;
    }
}

/*
 *
 * AABB
 *
 */
public struct Aabb {
    Vect mins;
    Vect maxs;

    public __init(Vect mins, Vect maxs) {
        this.mins = mins;
        this.maxs = maxs;
    }

    public func project_local_point(pt: Vect) -> Vect {
        let mins_pt = this.mins - pt;
        let pt_maxs = pt - this.maxs;
        let shift = max(mins_pt, Vect<float>(0.0)) - max(pt_maxs, Vect<float>(0.0));
        return pt + shift;
    }
}


/*
 *
 * Point projection on triangles.
 *
 */
static const uint AB = 0;
static const uint AC = 1;
static const uint BC = 2;
static const uint FACE_CW = 3;
static const uint FACE_CCW = 4;


static const uint FEATURE_VERTEX = 0;
static const uint FEATURE_EDGE = 1;
static const uint FEATURE_FACE = 2;
static const uint FEATURE_SOLID = 3;

struct ProjectionWithLocation {
    Vect point;
    float3 bcoords;
    uint feature_type;
    uint id;
    bool inside;

    __init(Vect point, float3 bcoords, uint feature_type, uint id, bool inside) {
        this.point = point;
        this.bcoords = bcoords;
        this.feature_type = feature_type;
        this.id = id;
        this.inside = inside;
    }

    static func vertex(pt: Vect, id: uint, inside: bool) -> ProjectionWithLocation {
        return ProjectionWithLocation(pt, float3(0.0), FEATURE_VERTEX, id, inside);
    }

    static func edge(pt: Vect, bcoords: float2, id: uint, inside: bool) -> ProjectionWithLocation {
        return ProjectionWithLocation(pt, float3(bcoords, 0.0), FEATURE_EDGE, id, inside);
    }

    static func face(pt: Vect, bcoords: float3, id: uint, inside: bool) -> ProjectionWithLocation {
        return ProjectionWithLocation(pt, bcoords, FEATURE_FACE, id, inside);
    }

    static func solid(pt: Vect) -> ProjectionWithLocation {
        return ProjectionWithLocation(pt, float3(0.0), FEATURE_SOLID, 0, true);
    }
}

struct ProjectionInfo {
    uint feature;
    float3 params;
}

// Checks on which edge voronoï region the point is.
// For 2D and 3D, it uses explicit cross/perp products that are
// more numerically stable.
func stable_check_edges_voronoi(
    ab: Vect,
    ac: Vect,
    bc: Vect,
    ap: Vect,
    bp: Vect,
    cp: Vect,
    ab_ap: float,
    ab_bp: float,
    ac_ap: float,
    ac_cp: float,
    ac_bp: float,
    ab_cp: float,
) -> ProjectionInfo {
#if DIM == 3
    let n = cross(ab, ac);
    let vc = dot(n, cross(ab, ap));
    if (vc < 0.0 && ab_ap >= 0.0 && ab_bp <= 0.0) {
        return ProjectionInfo(AB, float3(0.0));
    }

    let vb = -dot(n, cross(ac, cp));
    if (vb < 0.0 && ac_ap >= 0.0 && ac_cp <= 0.0) {
        return ProjectionInfo(AC, float3(0.0));
    }

    let va = dot(n, cross(bc, bp));
    if (va < 0.0 && ac_bp - ab_bp >= 0.0 && ab_cp - ac_cp >= 0.0) {
        return ProjectionInfo(BC, float3(0.0));
    }

    if (dot(n, ap) >= 0.0) {
        return ProjectionInfo(FACE_CW, float3(va, vb, vc));
    } else {
        return ProjectionInfo(FACE_CCW, float3(va, vb, vc));
    }
#else
    // TODO
    return ProjectionInfo(FACE_CW, float3(0.0));
#endif
}

func triangle_project_local_point_and_get_location(shape: Triangle, pt: Vect) -> ProjectionWithLocation {
    // To understand the ideas, consider reading the slides below
    // https://box2d.org/files/ErinCatto_GJK_GDC2010.pdf
    let a = shape.a;
    let b = shape.b;
    let c = shape.c;

    let ab = b - a;
    let ac = c - a;
    let ap = pt - a;

    let ab_ap = dot(ab, ap);
    let ac_ap = dot(ac, ap);

    if (ab_ap <= 0.0 && ac_ap <= 0.0) {
        // Voronoï region of `a`.
        return ProjectionWithLocation.vertex(a, 0, false);
    }

    let bp = pt - b;
    let ab_bp = dot(ab, bp);
    let ac_bp = dot(ac, bp);

    if (ab_bp >= 0.0 && ac_bp <= ab_bp) {
        // Voronoï region of `b`.
        return ProjectionWithLocation.vertex(b, 1, false);
    }

    let cp = pt - c;
    let ab_cp = dot(ab, cp);
    let ac_cp = dot(ac, cp);

    if (ac_cp >= 0.0 && ab_cp <= ac_cp) {
        // Voronoï region of `c`.
        return ProjectionWithLocation.vertex(c, 2, false);
    }

    let bc = c - b;
    let proj = stable_check_edges_voronoi(
        ab, ac, bc, ap, bp, cp, ab_ap, ab_bp, ac_ap, ac_cp, ac_bp, ab_cp,
    );

    switch (proj.feature) {
       case AB: {
           // Voronoï region of `ab`.
           let v = ab_ap / dot(ab, ab);
           let bcoords = float2(1.0 - v, v);
           let res = a + ab * v;
           return ProjectionWithLocation.edge(res, bcoords, 0, false);
       }
       case AC: {
           // Voronoï region of `ac`.
           let w = ac_ap / dot(ac, ac);
           let bcoords = float2(1.0 - w, w);
           let res = a + ac * w;
           return ProjectionWithLocation.edge(res, bcoords, 2, false);
       }
       case BC: {
           // Voronoï region of `bc`.
           let w = dot(bc, bp) / dot(bc, bc);
           let bcoords = float2(1.0 - w, w);
           let res = b + bc * w;
           return ProjectionWithLocation.edge(res, bcoords, 1, false);
       }
       case FACE_CW:
       case FACE_CCW: {
           // Voronoï region of the face.
           // NOTE: in some cases, numerical instability
           // may result in the denominator being zero
           // when the triangle is nearly degenerate.
           if (proj.params.x + proj.params.y + proj.params.z != 0.0) {
               let denom = 1.0 / (proj.params.x + proj.params.y + proj.params.z);
               let v = proj.params.y * denom;
               let w = proj.params.z * denom;
               let bcoords = float3(1.0 - v - w, v, w);
               let res = a + ab * v + ac * w;
               return ProjectionWithLocation.face(res, bcoords, proj.feature, false);
           }
       }
       default: { /* FACE_INTERIOR, 2D only (implemented below) */ }
    }

    // Special treatment if we work in 2D because in this case we really are inside of the object.
    // NOTE: this should never be reached in 3D.
    return ProjectionWithLocation.solid(pt);
}
