use crate::models::{DruckerPrager, ElasticCoefficients};
use bytemuck::{Pod, Zeroable};
use encase::ShaderType;
use nexus::dynamics::body::BodyCouplingEntry;
use nalgebra::{vector, Matrix2, Point2, Vector2};
use rapier::geometry::{ColliderSet, Polyline, Segment};
use stensor::tensor::GpuTensor;
// use nexus::shapes::ShapeBuffers;
use slang_hal::backend::Backend;
use wgpu::BufferUsages;
// use nexus::dynamics::body::BodyCouplingEntry;
use nexus::dynamics::GpuBodySet;
use nexus::shapes::ShapeBuffers;

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
#[repr(C)]
pub struct ParticleDynamics {
    pub velocity: Vector2<f32>,
    pub def_grad: Matrix2<f32>,
    pub affine: Matrix2<f32>,
    pub cdf: Cdf,
    pub init_volume: f32,
    pub init_radius: f32,
    pub mass: f32,
}

impl ParticleDynamics {
    pub fn with_density(radius: f32, density: f32) -> Self {
        let exponent = if cfg!(feature = "dim2") { 2 } else { 3 };
        let init_volume = (radius * 2.0).powi(exponent); // NOTE: the particles are square-ish.
        Self {
            velocity: Vector2::zeros(),
            def_grad: Matrix2::identity(),
            affine: Matrix2::zeros(),
            init_volume,
            init_radius: radius,
            mass: init_volume * density,
            cdf: Cdf::default(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ParticlePhase {
    pub phase: f32,
    pub max_stretch: f32,
}

#[derive(Copy, Clone, PartialEq, Debug, Default, ShaderType)]
#[repr(C)]
pub struct Cdf {
    pub normal: Vector2<f32>,
    pub rigid_vel: Vector2<f32>,
    pub signed_distance: f32,
    pub affinity: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct Particle {
    pub position: Vector2<f32>,
    pub dynamics: ParticleDynamics,
    pub model: ElasticCoefficients,
    pub plasticity: Option<DruckerPrager>,
    pub phase: Option<ParticlePhase>,
}

pub struct GpuRigidParticles<B: Backend> {
    pub local_sample_points: GpuTensor<Point2<f32>, B>,
    pub sample_points: GpuTensor<Point2<f32>, B>,
    pub rigid_particle_needs_block: GpuTensor<u32, B>,
    pub node_linked_lists: GpuTensor<u32, B>,
    pub sample_ids: GpuTensor<GpuSampleIds, B>,
}

impl<B: Backend> GpuRigidParticles<B> {
    pub fn new(backend: &B) -> Result<Self, B::Error> {
        Self::from_rapier(
            backend,
            &ColliderSet::default(),
            &GpuBodySet::new(backend, &[], &[], &ShapeBuffers::default())?,
            &[],
            1.0,
        )
    }

    pub fn from_rapier(
        backend: &B,
        colliders: &ColliderSet,
        gpu_bodies: &GpuBodySet<B>,
        coupling: &[BodyCouplingEntry],
        sampling_step: f32,
    ) -> Result<Self, B::Error> {
        let mut sampling_buffers = SamplingBuffers::default();

        for (collider_id, (coupling, gpu_data)) in coupling
            .iter()
            .zip(gpu_bodies.shapes_data().iter())
            .enumerate()
        {
            let collider = &colliders[coupling.collider];
            if let Some(polyline) = collider.shape().as_polyline() {
                let rngs = gpu_data.polyline_rngs();
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: rngs[0],
                    sampling_step,
                };
                sample_polyline(polyline, &sampling_params, &mut sampling_buffers)
            }
        }

        Ok(Self {
            local_sample_points: GpuTensor::vector(
                backend,
                &sampling_buffers.samples,
                BufferUsages::STORAGE,
            )?,
            sample_points: GpuTensor::vector(
                backend,
                &sampling_buffers.samples,
                BufferUsages::STORAGE,
            )?,
            node_linked_lists: unsafe {
                GpuTensor::vector_uninit(
                    backend,
                    sampling_buffers.samples.len() as u32,
                    BufferUsages::STORAGE,
                )?
            },
            sample_ids: GpuTensor::vector_encased(
                backend,
                &sampling_buffers.samples_ids,
                BufferUsages::STORAGE,
            )?,
            // NOTE: this is a packed bitmask so each u32 contains
            //       the flag for 32 particles.
            rigid_particle_needs_block: unsafe {
                GpuTensor::vector_uninit(
                    backend,
                    sampling_buffers.samples.len().div_ceil(32) as u32,
                    BufferUsages::STORAGE,
                )?
            },
        })
    }

    pub fn len(&self) -> u64 {
        self.sample_points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub type ParticlePosition = Vector2<f32>;

pub struct GpuParticles<B: Backend> {
    pub positions: GpuTensor<ParticlePosition, B>,
    pub dynamics: GpuTensor<ParticleDynamics, B>,
    pub sorted_ids: GpuTensor<u32, B>,
    pub node_linked_lists: GpuTensor<u32, B>,
}

impl<B: Backend> GpuParticles<B> {
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn len(&self) -> usize {
        self.positions.len() as usize
    }

    pub fn from_particles(backend: &B, particles: &[Particle]) -> Result<Self, B::Error> {
        let positions: Vec<_> = particles.iter().map(|p| p.position).collect();
        let dynamics: Vec<_> = particles.iter().map(|p| p.dynamics).collect();

        Ok(Self {
            positions: GpuTensor::vector(
                backend,
                &positions,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            dynamics: GpuTensor::vector_encased(backend, &dynamics, BufferUsages::STORAGE)?,
            sorted_ids: unsafe {
                GpuTensor::vector_uninit(backend, particles.len() as u32, BufferUsages::STORAGE)?
            },
            node_linked_lists: unsafe {
                GpuTensor::vector_uninit(backend, particles.len() as u32, BufferUsages::STORAGE)?
            },
        })
    }
}

#[derive(Copy, Clone, Debug, ShaderType)]
#[repr(C)]
pub struct GpuSampleIds {
    pub segment: Vector2<u32>,
    pub collider: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct SamplingParams {
    base_vid: u32,
    collider_id: u32,
    sampling_step: f32,
}

#[derive(Default, Clone)]
struct SamplingBuffers {
    samples: Vec<Point2<f32>>,
    samples_ids: Vec<GpuSampleIds>,
}

// TODO: move this elsewhere?
fn sample_polyline(polyline: &Polyline, params: &SamplingParams, buffers: &mut SamplingBuffers) {
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
