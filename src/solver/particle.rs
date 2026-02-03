use bytemuck::{Pod, Zeroable};
use encase::ShaderType;
use nexus::dynamics::body::BodyCouplingEntry;
use rapier::geometry::ColliderSet;
use std::ops::RangeBounds;
use stensor::tensor::{GpuScalar, GpuTensor};
// use nexus::shapes::ShapeBuffers;
use slang_hal::{BufferUsages, backend::Backend};
// use nexus::dynamics::body::BodyCouplingEntry;
use crate::sampling;
use crate::sampling::{GpuSampleIds, SamplingBuffers, SamplingParams};
use crate::solver::particle_model::GpuParticleModelData;
use nexus::dynamics::GpuBodySet;
use nexus::math::{Matrix, Point, Vector};
use nexus::shapes::ShapeBuffers;

/// Physical state of a single MPM particle.
///
/// Contains all the dynamic properties that evolve during simulation:
/// velocity, deformation gradient, mass, volume, and collision detection
/// information for rigid body coupling.
///
/// This struct is GPU-compatible and directly uploaded to compute shaders.
#[derive(Copy, Clone, PartialEq, Debug, ShaderType)]
#[repr(C)]
pub struct ParticleDynamics {
    /// Current velocity (m/s).
    pub velocity: Vector<f32>,
    /// Deformation gradient tracking how the particle has deformed from its initial state.
    pub def_grad: Matrix<f32>,
    /// APIC affine velocity matrix for improved momentum conservation.
    pub affine: Matrix<f32>,
    /// Additional force applied indirectly to the particle.
    /// Resets automatically at the `particle_update` stage of the
    /// MPM pipeline.
    pub force_dt: Vector<f32>,
    /// Determinant of velocity gradient (for volume change tracking).
    pub vel_grad_det: f32,
    /// Collision detection field data for rigid body coupling.
    pub cdf: Cdf,
    /// Initial particle volume (m² in 2D, m³ in 3D).
    pub init_volume: f32,
    /// Initial particle radius (m).
    pub init_radius: f32,
    /// Particle mass (kg).
    pub mass: f32,
    /// Rayleigh mass-proportional damping coefficient (1/s).
    ///
    /// Applies a damping force proportional to velocity: F_damp = -damping * m * v.
    /// Typical values: 0.0 (no damping) to 10.0 (heavy damping).
    pub damping: f32,
    /// The particle phase (used by materials that can break).
    pub phase: f32,
    /// Whether this particle is active (1) or disabled (0).
    pub enabled: u32,
    /// Whether this particle is fixed (1) or dynamic (0).
    pub fixed: u32,
}

impl ParticleDynamics {
    /// Creates new particle dynamics from radius and material density.
    ///
    /// Initializes the particle at rest with identity deformation gradient.
    ///
    /// # Arguments
    ///
    /// * `radius` - Particle radius (m), used to compute initial volume
    /// * `density` - Material density (kg/m³ or kg/m² in 2D)
    pub fn new(radius: f32, density: f32) -> Self {
        let exponent = if cfg!(feature = "dim2") { 2 } else { 3 };
        let init_volume = (radius * 2.0).powi(exponent); // NOTE: the particles are square-ish.
        Self {
            velocity: Vector::zeros(),
            def_grad: Matrix::identity(),
            affine: Matrix::zeros(),
            force_dt: Vector::zeros(),
            vel_grad_det: 0.0,
            init_volume,
            init_radius: radius,
            mass: init_volume * density,
            damping: 0.0,
            cdf: Cdf::default(),
            phase: 1.0,
            enabled: 1,
            fixed: 0,
        }
    }

    /// Sets whether this particle is fixed (true) or dynamic (false).
    pub fn set_fixed(&mut self, fixed: bool) {
        self.fixed = fixed as u32;
    }

    /// Sets the damping coefficient for this particle.
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping;
    }

    /// Updates the particle mass based on a new density.
    ///
    /// Keeps the initial volume constant and recomputes mass = volume × density.
    pub fn set_density(&mut self, density: f32) {
        self.mass = self.init_volume * density;
    }
}

/// Phase field data for fracture mechanics (experimental).
///
/// Tracks material damage and maximum stretch for particle-based fracture.
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ParticlePhase {
    /// Phase field value (1.0 = intact, 0.0 = broken).
    pub phase: f32,
    /// Maximum allowable stretch before fracture.
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
    /// Creates a phase field value representing a fully broken particle.
    pub fn broken() -> Self {
        Self {
            phase: 0.0,
            max_stretch: -1.0,
        }
    }
}

/// Collision Detection Field (CDF) data for MPM-rigid body coupling.
///
/// Stores signed distance and contact information between a particle and
/// the nearest rigid body surface. Used to handle two-way coupling.
#[derive(Copy, Clone, PartialEq, Debug, Default, ShaderType)]
#[repr(C)]
pub struct Cdf {
    /// Surface normal of the closest rigid body surface.
    pub normal: Vector<f32>,
    /// Velocity of the rigid body at the contact point.
    pub rigid_vel: Vector<f32>,
    /// Signed distance to the nearest rigid body surface (negative = penetration).
    pub signed_distance: f32,
    /// Index of the rigid body this particle is coupled with.
    pub affinity: u32,
}

/// A single MPM particle with position, dynamics, and material model.
///
/// This is the main CPU-side representation of a particle. It combines
/// geometric position, physical state, and material properties.
///
/// # Type Parameters
///
/// * `Model` - Material model type (e.g., [`ParticleModel`](crate::solver::ParticleModel))
#[derive(Copy, Clone, Debug)]
pub struct Particle<Model> {
    /// Spatial position (m).
    pub position: Point<f32>,
    /// Physical state (velocity, deformation, mass, etc.).
    pub dynamics: ParticleDynamics,
    /// Material model defining constitutive behavior.
    pub model: Model,
}

impl<Model> Particle<Model> {
    /// Creates a new particle with the given properties.
    ///
    /// # Arguments
    ///
    /// * `position` - Initial position in world space
    /// * `radius` - Particle radius (used to compute mass and volume)
    /// * `density` - Material density
    /// * `model` - Material model instance
    pub fn new(position: Point<f32>, radius: f32, density: f32, model: Model) -> Self {
        Particle {
            position,
            dynamics: ParticleDynamics::new(radius, density),
            model,
        }
    }
}

/// GPU buffers for particles sampled from rigid body surfaces.
///
/// For two-way coupling between MPM and rigid bodies, collider surfaces are
/// sampled with particles. These particles move with the rigid bodies and
/// interact with MPM particles through the CDF (Collision Detection Field).
pub struct GpuRigidParticles<B: Backend> {
    /// Sample points in local (body-relative) coordinates.
    pub local_sample_points: GpuTensor<Point<f32>, B>,
    /// Sample points transformed to world coordinates.
    pub sample_points: GpuTensor<Point<f32>, B>,
    /// Bitmask indicating which rigid particles need grid cell blocking.
    pub rigid_particle_needs_block: GpuTensor<u32, B>,
    /// Linked list for spatially sorting rigid particles into grid cells.
    pub node_linked_lists: GpuTensor<u32, B>,
    /// Metadata associating each sample with its source collider and body.
    pub sample_ids: GpuTensor<GpuSampleIds, B>,
}

impl<B: Backend> GpuRigidParticles<B> {
    /// Creates an empty set of rigid particles.
    pub fn new(backend: &B) -> Result<Self, B::Error> {
        Self::from_rapier(
            backend,
            &ColliderSet::default(),
            &GpuBodySet::new(backend, &[], &[], &ShapeBuffers::default())?,
            &[],
            1.0,
        )
    }

    /// Samples particles from Rapier collider surfaces for MPM coupling.
    ///
    /// Samples points on the surfaces of specified colliders at regular intervals
    /// determined by `sampling_step`. These particles will track rigid body motion
    /// and interact with MPM particles.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for buffer allocation
    /// * `colliders` - Rapier collider set
    /// * `gpu_bodies` - GPU representation of rigid bodies
    /// * `coupling` - List of colliders to sample (with coupling mode)
    /// * `sampling_step` - Distance between sample points on surfaces
    ///
    /// # Returns
    ///
    /// GPU buffers containing sampled particles, or an error if buffer allocation fails.
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

    /// Returns the number of rigid body particles.
    pub fn len(&self) -> u64 {
        self.sample_points.len()
    }

    /// Returns true if there are no rigid body particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Particle position type (2D: Point2, 3D: Point4 for alignment).
#[cfg(feature = "dim2")]
pub type ParticlePosition = Point<f32>;
/// Particle position type (2D: Point2, 3D: Point4 for alignment).
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

/// GPU buffers storing all MPM particle data in Structure-of-Arrays layout.
///
/// Separates particle data into individual buffers (positions, dynamics, models)
/// for efficient GPU access patterns.
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
    /// Returns true if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of particles.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns reference to GPU buffer containing particle count.
    pub fn gpu_len(&self) -> &GpuScalar<u32, B> {
        &self.gpu_len
    }

    /// Uploads CPU-side particles to GPU buffers.
    ///
    /// Converts from Array-of-Structures to Structure-of-Arrays layout.
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

    /// Returns reference to material model buffer.
    pub fn models(&self) -> &GpuTensor<GpuModel, B> {
        &self.models
    }

    /// Returns mutable reference to material model buffer.
    pub fn models_mut(&mut self) -> &mut GpuTensor<GpuModel, B> {
        &mut self.models
    }

    /// Returns reference to position buffer.
    pub fn positions(&self) -> &GpuTensor<ParticlePosition, B> {
        &self.positions
    }

    /// Returns reference to position buffer.
    pub fn positions_mut(&mut self) -> &mut GpuTensor<ParticlePosition, B> {
        &mut self.positions
    }

    /// Returns reference to dynamics buffer (velocity, deformation, mass).
    pub fn dynamics(&self) -> &GpuTensor<ParticleDynamics, B> {
        &self.dynamics
    }

    /// Returns mutable reference to dynamics buffer (velocity, deformation, mass).
    pub fn dynamics_mut(&mut self) -> &mut GpuTensor<ParticleDynamics, B> {
        &mut self.dynamics
    }

    /// Returns reference to sorted particle ID buffer.
    pub fn sorted_ids(&self) -> &GpuTensor<u32, B> {
        &self.sorted_ids
    }

    /// Returns reference to per-particle linked list buffer for grid neighbor queries.
    pub fn node_linked_lists(&self) -> &GpuTensor<u32, B> {
        &self.node_linked_lists
    }
}
