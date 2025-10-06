use std::collections::HashSet;
use encase::ShaderType;
use nalgebra::{vector, Point3, Vector3};
use rapier::geometry::{Segment, TriMesh, Triangle};

// Epsilon used as a length threshold in various steps of the sampling. In particular, this avoids
// degenerate geometries from generating invalid samples.
const EPS: f32 = 1.0e-5;

pub struct TriangleSample {
    pub triangle_id: u32,
    pub point: Point3<f32>,
}

#[derive(Copy, Clone, Debug, ShaderType)]
#[repr(C)]
pub struct GpuSampleIds {
    pub triangle: Vector3<u32>,
    pub collider: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct SamplingParams {
    pub base_vid: u32,
    pub collider_id: u32,
    pub sampling_step: f32,
}

#[derive(Default, Clone)]
pub struct SamplingBuffers {
    pub local_samples: Vec<Point3<f32>>,
    pub samples: Vec<Point3<f32>>,
    pub samples_ids: Vec<GpuSampleIds>,
}


pub fn sample_trimesh(trimesh: &TriMesh, params: &SamplingParams, buffers: &mut SamplingBuffers) {
    let samples = sample_mesh(trimesh.vertices(), trimesh.indices(), params.sampling_step);

    for sample in samples {
        let tri_idx = trimesh.indices()[sample.triangle_id as usize];
        let sample_id = GpuSampleIds {
            triangle: vector![
                    params.base_vid + tri_idx[0],
                    params.base_vid + tri_idx[1],
                    params.base_vid + tri_idx[2]
                ],
            collider: params.collider_id,
        };
        buffers.local_samples.push(sample.point);
        buffers.samples.push(sample.point);
        buffers.samples_ids.push(sample_id);
    }

    println!(
        "Num rigid particles: {}, num triangles: {}",
        buffers.samples.len(),
        trimesh.indices().len()
    );
}

/// Samples a triangle mesh with a set of points such that at least one point is generated
/// inside each cell on a grid on the x-y plane with cells sized by `xy_spacing`.
pub fn sample_mesh(
    vertices: &[Point3<f32>],
    indices: &[[u32; 3]],
    xy_spacing: f32,
) -> Vec<TriangleSample> {
    let mut samples = vec![];
    // TODO: switch to a matrix of boolean to avoid hashing if
    //       this proves to be a perf bottleneck.
    let mut visited_segs = HashSet::new();

    let mut seg_needs_sampling = |mut ia: u32, mut ib: u32| {
        if ib > ia {
            std::mem::swap(&mut ia, &mut ib);
        }

        visited_segs.insert([ia, ib])
    };

    for (tri_id, idx) in indices.iter().enumerate() {
        let tri = Triangle::new(
            vertices[idx[0] as usize],
            vertices[idx[1] as usize],
            vertices[idx[2] as usize],
        );
        sample_triangle(tri, &mut samples, xy_spacing, tri_id as u32);

        if seg_needs_sampling(idx[0], idx[1]) {
            let seg = Segment::new(vertices[idx[0] as usize], vertices[idx[1] as usize]);
            sample_edge(seg, &mut samples, xy_spacing, tri_id as u32);
        }

        if seg_needs_sampling(idx[1], idx[2]) {
            let seg = Segment::new(vertices[idx[1] as usize], vertices[idx[2] as usize]);
            sample_edge(seg, &mut samples, xy_spacing, tri_id as u32);
        }

        if seg_needs_sampling(idx[2], idx[0]) {
            let seg = Segment::new(vertices[idx[2] as usize], vertices[idx[0] as usize]);
            sample_edge(seg, &mut samples, xy_spacing, tri_id as u32);
        }
    }

    samples
}

/// Samples a triangle edge with a set of points such that at least one point is generated
/// inside each cell on a grid on the x-y plane with cells sized by `xy_spacing`.
///
/// The returned samples will not contain `edge.a`. It might contain `edge.b` (but it is unlikely)
/// if it aligns exactly with the internal sampling spacing.
pub fn sample_edge(
    edge: Segment,
    samples: &mut Vec<TriangleSample>,
    xy_spacing: f32,
    triangle_id: u32,
) {
    let ab = edge.b - edge.a;
    let edge_length = ab.norm();

    if edge_length > EPS {
        let edge_dir = ab / edge_length;
        let spacing = xy_spacing / 2.0f32.sqrt();
        let nsteps = (edge_length / spacing).ceil() as usize;

        // Start at one so we don’t push edge.a.
        for i in 1..nsteps {
            let point = edge.a + edge_dir * (spacing * i as f32);
            samples.push(TriangleSample { point, triangle_id })
        }
    }
}

/// Samples a triangle with a set of points such that at least one point is generated
/// inside each cell on a grid on the x-y plane with cells sized by `xy_spacing`.
///
/// Tha sampling has the following characteristics:
/// 1. Guarantees at least one sample per cell in the "ambient" XY grid.
/// 2. The sampling grid is oriented along the base (longest edge) and height (orthogonal to the
///    base) of the triangle.
/// 3. Samples are selected strictly from the domain of the triangle (up to rounding error).
/// 4. No sample is placed on the base or any of the triangle vertices. Samples will generally not
///    be on any of the two other edges either (but may due so some fortuitous alignment
///    with the internal stepping length along the height of the triangle).
///
/// Because this does not attempt to sample the edges of the triangles, small or thin triangles
/// might not result in any samples. Edges should be sampled separately with [`sample_edge`].
pub fn sample_triangle(
    triangle: Triangle,
    samples: &mut Vec<TriangleSample>,
    xy_spacing: f32,
    triangle_id: u32,
) {
    // select the longest edge as the base
    let distance_ab = nalgebra::distance(&triangle.b, &triangle.a);
    let distance_bc = nalgebra::distance(&triangle.c, &triangle.b);
    let distance_ca = nalgebra::distance(&triangle.a, &triangle.c);
    let max = distance_ab.max(distance_bc).max(distance_ca);

    let triangle = if max == distance_bc {
        Triangle {
            a: triangle.b,
            b: triangle.c,
            c: triangle.a,
        }
    } else if max == distance_ca {
        Triangle {
            a: triangle.c,
            b: triangle.a,
            c: triangle.b,
        }
    } else {
        triangle
    };

    let ac = triangle.c - triangle.a;
    let base = triangle.b - triangle.a;
    let base_length = base.norm();
    let base_dir = base / base_length;

    // Adjust the spacing so it matches the required spacing on the x-y plane.
    // For simplicity, we just divide by sqrt(2) so that the spacing in any direction is guaranteed
    // to be smaller or equal to the inner-circle diameter of any cell from the implicit grid with
    // spacing `xy_spacing`.
    // We could use a more fine-grained adjustment that depends on the angle between the base-dir
    // and the world x-y axes. But this doesn’t make a significant difference in point count or
    // computation times. However, the sampling looks worse (less uniform in practice). So we stick
    // to the simple sqrt(2) approach.
    let spacing = xy_spacing / 2.0f32.sqrt();

    // Calculate the step increment on the base.
    let base_step_count = (base_length / spacing).ceil();
    let base_step = base_dir * spacing;

    // Project C on the base AB.
    let ac_offset_length = ac.dot(&base_dir);
    let bc_offset_length = base_length - ac_offset_length;

    if ac_offset_length < EPS || bc_offset_length < EPS || base_length < EPS {
        return;
    }

    // Compute the triangle’s height vector.
    let height = ac - base_dir * ac_offset_length;
    let height_length = height.norm();
    let height_dir = height / height_length;
    // Calculate the tangents.
    let tan_alpha = height_length / ac_offset_length;
    let tan_beta = height_length / bc_offset_length;

    // Start at 1 so we don’t sample the perpendicular edge if it’s at a right angle
    // with `triangle.a`.
    for i in 1..base_step_count as u32 {
        let base_position = triangle.a + (i as f32) * base_step;

        // Compute the height at the current base_position. The point at the
        // end of that height is either in the line (AC) or (BC), whichever is closer.
        let height_ac = tan_alpha * nalgebra::distance(&triangle.a, &base_position);
        let height_bc = tan_beta * nalgebra::distance(&triangle.b, &base_position);
        let height_length = height_ac.min(height_bc);

        // Calculate the step increment on the height.
        let height_step_count = (height_length / spacing).ceil();
        let height_step = height_dir * spacing;

        // Start at 1 so we don’t sample the basis edge.
        for j in 1..height_step_count as u32 {
            let particle_position = base_position + (j as f32) * height_step;

            if particle_position.iter().any(|e| !e.is_finite()) {
                continue;
            }

            samples.push(TriangleSample {
                point: particle_position,
                triangle_id,
            });
        }
    }
}