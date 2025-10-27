use bytemuck::{Pod, Zeroable};
use encase::ShaderType;
use nexus::dynamics::body::BodyCouplingEntry;
use rapier::geometry::ColliderSet;
use std::ops::RangeBounds;
use stensor::tensor::{GpuScalar, GpuTensor};
// use nexus::shapes::ShapeBuffers;
use slang_hal::backend::Backend;
use wgpu::BufferUsages;
// use nexus::dynamics::body::BodyCouplingEntry;
use crate::sampling;
use crate::sampling::{GpuSampleIds, SamplingBuffers, SamplingParams};
use crate::solver::particle_model::GpuParticleModelData;
use nexus::dynamics::GpuBodySet;
use nexus::math::{Matrix, Point, Vector};
use nexus::shapes::ShapeBuffers;

#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
#[repr(C)]
pub struct ParticleDynamics {
    pub velocity: Vector<f32>,
    pub def_grad: Matrix<f32>,
    pub affine: Matrix<f32>,
    pub vel_grad_det: f32,
    pub cdf: Cdf,
    pub init_volume: f32,
    pub init_radius: f32,
    pub mass: f32,
    pub enabled: u32,
}

impl ParticleDynamics {
    pub fn new(radius: f32, density: f32) -> Self {
        let exponent = if cfg!(feature = "dim2") { 2 } else { 3 };
        let init_volume = (radius * 2.0).powi(exponent); // NOTE: the particles are square-ish.
        Self {
            velocity: Vector::zeros(),
            def_grad: Matrix::identity(),
            affine: Matrix::zeros(),
            vel_grad_det: 0.0,
            init_volume,
            init_radius: radius,
            mass: init_volume * density,
            cdf: Cdf::default(),
            enabled: 1,
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
pub struct Particle<Model> {
    pub position: Point<f32>,
    pub dynamics: ParticleDynamics,
    pub model: Model,
}

impl<Model> Particle<Model> {
    pub fn new(position: Point<f32>, radius: f32, density: f32, model: Model) -> Self {
        Particle {
            position,
            dynamics: ParticleDynamics::new(radius, density),
            model,
        }
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
                let trimesh = rapier::geometry::TriMesh::new(vtx, idx).unwrap();
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
            node_linked_lists: GpuTensor::vector_uninit(
                backend,
                sampling_buffers.samples.len() as u32,
                BufferUsages::STORAGE,
            )?,
            sample_ids: GpuTensor::vector_encased(
                backend,
                &sampling_buffers.samples_ids,
                BufferUsages::STORAGE,
            )?,
            // NOTE: this is a packed bitmask so each u32 contains
            //       the flag for 32 particles.
            rigid_particle_needs_block: GpuTensor::vector_uninit(
                backend,
                sampling_buffers.samples.len().div_ceil(32) as u32,
                BufferUsages::STORAGE,
            )?,
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
pub type ParticlePosition = nalgebra::Point4<f32>;

struct SoAParticles<GpuModel: GpuParticleModelData> {
    positions: Vec<ParticlePosition>,
    dynamics: Vec<ParticleDynamics>,
    models: Vec<GpuModel>,
}

impl<GpuModel: GpuParticleModelData> SoAParticles<GpuModel> {
    pub fn new(particles: &[Particle<GpuModel::Model>]) -> Self {
        #[cfg(feature = "dim2")]
        let positions: Vec<_> = particles.iter().map(|p| p.position).collect();
        #[cfg(feature = "dim3")]
        let positions: Vec<_> = particles
            .iter()
            .map(|p| p.position.coords.push(0.0).into())
            .collect();
        let dynamics: Vec<_> = particles.iter().map(|p| p.dynamics).collect();
        let models: Vec<_> = particles
            .iter()
            .map(|p| GpuModel::from_model(p.model))
            .collect();

        Self {
            positions,
            dynamics,
            models,
        }
    }
}

pub struct GpuParticles<B: Backend, GpuModel: GpuParticleModelData> {
    len: usize,
    gpu_len: GpuScalar<u32, B>,
    positions: GpuTensor<ParticlePosition, B>,
    dynamics: GpuTensor<ParticleDynamics, B>,
    models: GpuTensor<GpuModel, B>,
    sorted_ids: GpuTensor<u32, B>,
    node_linked_lists: GpuTensor<u32, B>,
}

impl<B: Backend, GpuModel: GpuParticleModelData> GpuParticles<B, GpuModel> {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn gpu_len(&self) -> &GpuScalar<u32, B> {
        &self.gpu_len
    }

    pub fn from_particles(
        backend: &B,
        particles: &[Particle<GpuModel::Model>],
    ) -> Result<Self, B::Error> {
        let data = SoAParticles::new(particles);
        let resizeable = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        Ok(Self {
            len: particles.len(),
            gpu_len: GpuTensor::scalar(
                backend,
                particles.len() as u32,
                BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
            positions: GpuTensor::vector(backend, &data.positions, resizeable)?,
            dynamics: GpuTensor::vector_encased(backend, &data.dynamics, resizeable)?,
            models: GpuTensor::vector(backend, &data.models, resizeable)?,
            sorted_ids: GpuTensor::vector_uninit(backend, particles.len() as u32, resizeable)?,
            node_linked_lists: GpuTensor::vector_uninit(
                backend,
                particles.len() as u32,
                resizeable,
            )?,
        })
    }

    /// Removes a range of data from this buffer, shifting elements to fill the gap.
    ///
    /// If the operation succeeded, returns the number of removed elements.
    pub fn shift_remove(
        &mut self,
        backend: &B,
        range: impl RangeBounds<usize> + Clone,
    ) -> Result<usize, B::Error> {
        // If new vectors are added to the struct, this code needs to be updated to adjust
        // the buffers accordingly after range removal.
        let Self {
            len,
            gpu_len,
            positions,
            dynamics,
            models,
            sorted_ids: _,
            node_linked_lists: _,
        } = self;

        let removed = positions.shift_remove_encased(backend, range.clone())?;
        dynamics.shift_remove_encased(backend, range.clone())?;
        models.shift_remove(backend, range.clone())?;

        *len -= removed;
        backend.write_buffer(gpu_len.buffer_mut(), 0, &[*len as u32])?;
        Ok(removed)
    }

    /// Appends particles at the end of this buffer.
    pub fn append(
        &mut self,
        backend: &B,
        particles: &[Particle<GpuModel::Model>],
    ) -> Result<(), B::Error> {
        // If new vectors are added to the struct, this code needs to be updated to adjust
        // the buffers accordingly for appending particles.
        let Self {
            len,
            gpu_len,
            positions,
            dynamics,
            models,
            sorted_ids,
            node_linked_lists,
        } = self;

        let data = SoAParticles::new(particles);

        let zeros = vec![0; particles.len()];

        dynamics.append_encased(backend, &data.dynamics)?;
        positions.append_encased(backend, &data.positions)?;
        models.append(backend, &data.models)?;
        sorted_ids.append(backend, &zeros)?;
        node_linked_lists.append(backend, &zeros)?;

        *len += particles.len();
        backend.write_buffer(gpu_len.buffer_mut(), 0, &[*len as u32])?;
        println!("New len: {}", *len);

        Ok(())
    }

    pub fn models(&self) -> &GpuTensor<GpuModel, B> {
        &self.models
    }

    pub fn positions(&self) -> &GpuTensor<ParticlePosition, B> {
        &self.positions
    }

    pub fn dynamics(&self) -> &GpuTensor<ParticleDynamics, B> {
        &self.dynamics
    }

    pub fn sorted_ids(&self) -> &GpuTensor<u32, B> {
        &self.sorted_ids
    }

    pub fn node_linked_lists(&self) -> &GpuTensor<u32, B> {
        &self.node_linked_lists
    }
}
