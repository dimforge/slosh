use crate::models::{DruckerPrager, ElasticCoefficients};
use bytemuck::{Pod, Zeroable};
use encase::ShaderType;
use nalgebra::{vector, Point4};
use nexus::dynamics::body::BodyCouplingEntry;
use rapier::geometry::{ColliderSet, Polyline, Segment, TriMesh};
use stensor::tensor::GpuTensor;
// use nexus::shapes::ShapeBuffers;
use slang_hal::backend::Backend;
use wgpu::BufferUsages;
// use nexus::dynamics::body::BodyCouplingEntry;
use nexus::dynamics::GpuBodySet;
use nexus::math::{Matrix, Point, Vector};
use nexus::shapes::ShapeBuffers;
use crate::sampling;
use crate::sampling::{GpuSampleIds, SamplingBuffers, SamplingParams};

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
#[repr(C)]
pub struct ParticleDynamics {
    pub velocity: Vector<f32>,
    pub def_grad: Matrix<f32>,
    pub affine: Matrix<f32>,
    pub cdf: Cdf,
    pub init_volume: f32,
    pub init_radius: f32,
    pub mass: f32,
}

impl ParticleDynamics {
    pub fn new(radius: f32, density: f32) -> Self {
        let exponent = if cfg!(feature = "dim2") { 2 } else { 3 };
        let init_volume = (radius * 2.0).powi(exponent); // NOTE: the particles are square-ish.
        Self {
            velocity: Vector::zeros(),
            def_grad: Matrix::identity(),
            affine: Matrix::zeros(),
            init_volume,
            init_radius: radius,
            mass: init_volume * density,
            cdf: Cdf::default(),
        }
    }

    pub fn set_density(&mut self, density: f32) {
        self.mass = self.init_volume * density;
    }
}

#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ParticlePhase {
    pub phase: f32,
    pub max_stretch: f32,
}

impl Default for ParticlePhase {
    fn default() -> Self {
        Self {
            phase: 1.0,
            max_stretch: f32::MAX,
        }
    }
}

impl ParticlePhase {
    pub fn broken() -> Self {
        Self {
            phase: 0.0,
            max_stretch: -1.0,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug, Default, ShaderType)]
#[repr(C)]
pub struct Cdf {
    pub normal: Vector<f32>,
    pub rigid_vel: Vector<f32>,
    pub signed_distance: f32,
    pub affinity: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct Particle {
    pub position: Point<f32>,
    pub dynamics: ParticleDynamics,
    pub model: ElasticCoefficients,
    pub plasticity: Option<DruckerPrager>,
    pub phase: ParticlePhase,
}

pub struct ParticleBuilder(Particle);

impl From<ParticleBuilder> for Particle {
    fn from(value: ParticleBuilder) -> Self {
        value.0
    }
}

impl ParticleBuilder {
    pub fn new(position: Point<f32>, radius: f32, density: f32) -> Self {
        const DEFAULT_YOUNG_MODULUS: f32 = 1_000.0;
        const DEFAULT_POISSON_RATIO: f32 = 0.2;

        Self(Particle {
            position,
            dynamics: ParticleDynamics::new(radius, density),
            model: ElasticCoefficients::from_young_modulus(
                DEFAULT_YOUNG_MODULUS,
                DEFAULT_POISSON_RATIO,
            ),
            plasticity: None,
            phase: ParticlePhase::default(),
        })
    }

    pub fn elastic(mut self, young_modulus: f32, poisson_ratio: f32) -> Self {
        self.0.model = ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio);
        self.0.plasticity = None;
        self
    }

    pub fn sand(mut self, young_modulus: f32, poisson_ratio: f32) -> Self {
        self.0.model = ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio);
        self.0.plasticity = Some(DruckerPrager::new(young_modulus, poisson_ratio));
        self.0.phase = ParticlePhase::broken();
        self
    }

    pub fn build(self) -> Particle {
        self.0
    }
}

pub struct GpuRigidParticles<B: Backend> {
    pub local_sample_points: GpuTensor<Point<f32>, B>,
    pub sample_points: GpuTensor<Point<f32>, B>,
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

            #[cfg(feature = "dim2")]
            if let Some(polyline) = collider.shape().as_polyline() {
                let rngs = gpu_data.polyline_rngs();
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: rngs[0],
                    sampling_step,
                };
                sampling::sample_polyline(polyline, &sampling_params, &mut sampling_buffers)
            }

            #[cfg(feature = "dim3")]
            if let Some(trimesh) = collider.shape().as_trimesh() {
                let rngs = gpu_data.trimesh_rngs();
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: rngs[0],
                    sampling_step,
                };
                sampling::sample_trimesh(trimesh, &sampling_params, &mut sampling_buffers)
            } else if let Some(heightfield) = collider.shape().as_heightfield() {
                let (vtx, idx) = heightfield.to_trimesh();
                let trimesh = TriMesh::new(vtx, idx).unwrap();
                let rngs = gpu_data.trimesh_rngs();
                let sampling_params = SamplingParams {
                    collider_id: collider_id as u32,
                    base_vid: rngs[0],
                    sampling_step,
                };
                sampling::sample_trimesh(&trimesh, &sampling_params, &mut sampling_buffers)
            }
        }

        Ok(Self {
            local_sample_points: GpuTensor::vector_encased(
                backend,
                &sampling_buffers.samples,
                BufferUsages::STORAGE,
            )?,
            sample_points: GpuTensor::vector_encased(
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

#[cfg(feature = "dim2")]
pub type ParticlePosition = Point<f32>;
#[cfg(feature = "dim3")]
pub type ParticlePosition = Point4<f32>;


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
        #[cfg(feature = "dim2")]
        let positions: Vec<_> = particles.iter().map(|p| p.position).collect();
        #[cfg(feature = "dim3")]
        let positions: Vec<_> = particles.iter().map(|p| p.position.coords.push(0.0).into()).collect();
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
