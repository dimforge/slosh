use crate::{
    math::{Vector, vector},
    sampling::{GpuSampleIds, SamplingParams},
};

use glam::UVec2;
use rapier::geometry::Polyline;

#[derive(Default, Clone)]
pub struct SamplingBuffers {
    pub samples: Vec<Vector>,
    pub samples_ids: Vec<GpuSampleIds>,
}

pub fn sample_polyline(
    polyline: &Polyline,
    params: &SamplingParams,
    buffers: &mut SamplingBuffers,
) {
    for seg_idx in polyline.indices() {
        let a = vector(polyline.vertices()[seg_idx[0] as usize]);
        let b = vector(polyline.vertices()[seg_idx[1] as usize]);
        let sample_id = GpuSampleIds {
            segment: UVec2::new(params.base_vid + seg_idx[0], params.base_vid + seg_idx[1]),
            collider: params.collider_id,
        };
        buffers.samples.push(a);
        buffers.samples_ids.push(sample_id);

        let ab = b - a;
        let length = ab.length();
        if length > 0.0 {
            let dir = ab / length;
            for i in 0.. {
                let shift = (i as f32) * params.sampling_step;
                if shift > length {
                    break;
                }

                buffers.samples.push(a + dir * shift);
                buffers.samples_ids.push(sample_id);
            }

            buffers.samples.push(b);
            buffers.samples_ids.push(sample_id);
        }
    }
}
