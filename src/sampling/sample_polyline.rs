use encase::ShaderType;
use nalgebra::{Point2, Vector2, vector};
use rapier::geometry::{Polyline, Segment};

#[derive(Copy, Clone, Debug, ShaderType)]
#[repr(C)]
pub struct GpuSampleIds {
    pub segment: Vector2<u32>,
    pub collider: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SamplingParams {
    pub base_vid: u32,
    pub collider_id: u32,
    pub sampling_step: f32,
}

#[derive(Default, Clone)]
pub struct SamplingBuffers {
    pub samples: Vec<Point2<f32>>,
    pub samples_ids: Vec<GpuSampleIds>,
}

pub fn sample_polyline(
    polyline: &Polyline,
    params: &SamplingParams,
    buffers: &mut SamplingBuffers,
) {
    for seg_idx in polyline.indices() {
        let seg = Segment::new(
            polyline.vertices()[seg_idx[0] as usize],
            polyline.vertices()[seg_idx[1] as usize],
        );
        let sample_id = GpuSampleIds {
            segment: vector![params.base_vid + seg_idx[0], params.base_vid + seg_idx[1]],
            collider: params.collider_id,
        };
        buffers.samples.push(seg.a);
        buffers.samples_ids.push(sample_id);

        if let Some(dir) = seg.direction() {
            for i in 0.. {
                let shift = (i as f32) * params.sampling_step;
                if shift > seg.length() {
                    break;
                }

                buffers.samples.push(seg.a + *dir * shift);
                buffers.samples_ids.push(sample_id);
            }

            buffers.samples.push(seg.b);
            buffers.samples_ids.push(sample_id);
        }
    }
}
