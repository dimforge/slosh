//! High-level MPM simulation pipeline orchestration.
//!
//! This module provides the main entry point for running MPM simulations. The pipeline
//! coordinates the execution of all MPM algorithm stages on the GPU.

use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::prefix_sum::{PrefixSumWorkspace, WgPrefixSum};
use crate::grid::sort::WgSort;
use crate::solver::{GpuImpulses, GpuParticleModelData, GpuParticles, GpuRigidParticles, GpuSimulationParams, GpuTimestepBounds, Particle, SimulationParams, WgG2P, WgG2PCdf, WgGridUpdate, WgGridUpdateCdf, WgP2G, WgP2GCdf, WgParticleUpdate, WgRigidImpulses, WgRigidParticleUpdate, WgTimestepBounds};
use nexus::dynamics::GpuBodySet;
use nexus::dynamics::body::{BodyCoupling, BodyCouplingEntry};
use nexus::math::GpuSim;
use rapier::dynamics::RigidBodySet;
use rapier::geometry::ColliderSet;
use slang_hal::Shader;
use slang_hal::backend::{Backend, Encoder};
use slang_hal::re_exports::minislang::SlangCompiler;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use stensor::tensor::{GpuScalar, GpuTensor, GpuVector};
use wgpu::BufferUsages;

/// GPU compute pipeline for Material Point Method simulation.
///
/// This struct holds all the compiled compute shaders needed to execute a complete
/// MPM simulation step. It orchestrates the following stages:
/// 1. Update rigid body particles from coupled colliders
/// 2. Sort particles into grid cells
/// 3. Transfer data from particles to grid (P2G)
/// 4. Update grid velocities with forces and boundary conditions
/// 5. Transfer data from grid back to particles (G2P)
/// 6. Update particle positions and deformation gradients
/// 7. Apply impulses to coupled rigid bodies
///
/// # Type Parameters
///
/// * `B` - Backend type implementing GPU operations
/// * `GpuModel` - Particle material model data layout (must match shader expectations)
pub struct MpmPipeline<B: Backend, GpuModel: GpuParticleModelData> {
    grid: WgGrid<B>,
    prefix_sum: WgPrefixSum<B>,
    sort: WgSort<B>,
    p2g: WgP2G<B>,
    p2g_cdf: WgP2GCdf<B>,
    grid_update_cdf: WgGridUpdateCdf<B>,
    grid_update: WgGridUpdate<B>,
    particles_update: WgParticleUpdate<B>,
    g2p: WgG2P<B>,
    g2p_cdf: WgG2PCdf<B>,
    rigid_particles_update: WgRigidParticleUpdate<B>,
    pub timestep_bounds: WgTimestepBounds<B>,
    /// Rigid body impulse computation kernel (publicly accessible for external use).
    pub impulses: WgRigidImpulses<B>,
    _phantom: PhantomData<GpuModel>,
}

/// GPU-resident simulation state for MPM.
///
/// Contains all the data needed to execute an MPM simulation step, including
/// particles, grid, rigid body coupling information, and simulation parameters.
/// All data lives in GPU memory for efficient computation.
///
/// # Type Parameters
///
/// * `B` - Backend type implementing GPU operations
/// * `GpuModel` - Particle material model data layout
pub struct MpmData<B: Backend, GpuModel: GpuParticleModelData> {
    /// The simulation timestep.
    pub base_dt: f32,
    /// Global simulation parameters (gravity, timestep).
    pub sim_params: GpuSimulationParams<B>,
    /// Spatial grid for momentum transfer.
    pub grid: GpuGrid<B>,
    /// MPM particles (positions, velocities, masses, material properties).
    pub particles: GpuParticles<B, GpuModel>, // TODO: keep private?
    /// Particles sampled from rigid body collider surfaces for two-way coupling.
    pub rigid_particles: GpuRigidParticles<B>,
    /// Rigid bodies coupled with the MPM simulation.
    pub bodies: GpuBodySet<B>,
    /// Accumulated impulses to apply to rigid bodies from MPM interactions.
    pub impulses: GpuImpulses<B>,
    /// Staging buffer for reading rigid body poses back to CPU.
    pub poses_staging: GpuVector<GpuSim, B>,
    pub timestep_bounds: GpuScalar<GpuTimestepBounds, B>,
    pub timestep_bounds_staging: GpuScalar<GpuTimestepBounds, B>,
    prefix_sum: PrefixSumWorkspace<B>,
    coupling: Vec<BodyCouplingEntry>,
}

/// Shader specialization configuration for the MPM pipeline.
///
/// Defines module paths for specializing parts of the MPM pipeline using Slang's
/// link-time specialization feature. This allows compiling different material models
/// without code duplication.
pub struct MpmSpecializations {
    /// Module paths defining particle material model implementations.
    pub particle_model: Vec<String>,
}

impl<B: Backend, GpuModel: GpuParticleModelData> MpmData<B, GpuModel> {
    /// Creates new MPM simulation data with default two-way coupling for all colliders.
    ///
    /// Automatically configures one-way coupling (MPM affects rigid bodies, but not vice versa)
    /// for all colliders attached to rigid bodies. For custom coupling configuration,
    /// use [`with_select_coupling`](Self::with_select_coupling).
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for buffer allocation
    /// * `params` - Global simulation parameters (gravity, timestep)
    /// * `particles` - Initial CPU-side particle data to upload
    /// * `bodies` - Rigid bodies from Rapier physics engine
    /// * `colliders` - Colliders from Rapier (used for MPM-rigid body coupling)
    /// * `cell_width` - Spatial width of each grid cell
    /// * `grid_capacity` - Maximum number of active grid cells
    ///
    /// # Returns
    ///
    /// GPU-resident simulation state ready for stepping.
    pub fn new(
        backend: &B,
        params: SimulationParams,
        particles: &[Particle<GpuModel::Model>],
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, B::Error> {
        let coupling: Vec<_> = colliders
            .iter()
            .filter_map(|(co_handle, co)| {
                let rb_handle = co.parent()?;
                Some(BodyCouplingEntry {
                    body: rb_handle,
                    collider: co_handle,
                    mode: BodyCoupling::OneWay,
                })
            })
            .collect();
        Self::with_select_coupling(
            backend,
            params,
            particles,
            bodies,
            colliders,
            coupling,
            cell_width,
            grid_capacity,
        )
    }

    /// Creates new MPM simulation data with custom rigid body coupling configuration.
    ///
    /// Allows fine-grained control over which colliders participate in MPM-rigid body
    /// coupling and the coupling mode (one-way vs. two-way).
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for buffer allocation
    /// * `params` - Global simulation parameters (gravity, timestep)
    /// * `particles` - Initial CPU-side particle data to upload
    /// * `bodies` - Rigid bodies from Rapier physics engine
    /// * `colliders` - Colliders from Rapier
    /// * `coupling` - Explicit list of collider-body pairs to couple with MPM
    /// * `cell_width` - Spatial width of each grid cell
    /// * `grid_capacity` - Maximum number of active grid cells
    ///
    /// # Returns
    ///
    /// GPU-resident simulation state ready for stepping.
    pub fn with_select_coupling(
        backend: &B,
        params: SimulationParams,
        particles: &[Particle<GpuModel::Model>],
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        coupling: Vec<BodyCouplingEntry>,
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, B::Error> {
        let sampling_step = cell_width; // TODO: * 1.5 ?
        let bodies = GpuBodySet::from_rapier(backend, bodies, colliders, &coupling)?;
        let sim_params = GpuSimulationParams::new(backend, params)?;
        let particles = GpuParticles::from_particles(backend, particles)?;
        let rigid_particles =
            GpuRigidParticles::from_rapier(backend, colliders, &bodies, &coupling, sampling_step)?;
        let grid = GpuGrid::with_capacity(backend, grid_capacity, cell_width)?;
        let prefix_sum = PrefixSumWorkspace::with_capacity(backend, grid_capacity)?;
        let impulses = GpuImpulses::new(backend)?;
        let poses_staging = GpuVector::vector_uninit(
            backend,
            bodies.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;
        let bounds = GpuTimestepBounds::new();
        let timestep_bounds = GpuTensor::scalar(backend, bounds, BufferUsages::STORAGE | BufferUsages::COPY_SRC)?;
        let timestep_bounds_staging = GpuTensor::scalar(backend, bounds, BufferUsages::COPY_DST | BufferUsages::MAP_READ)?;

        Ok(Self {
            sim_params,
            particles,
            rigid_particles,
            bodies,
            impulses,
            grid,
            prefix_sum,
            poses_staging,
            coupling,
            timestep_bounds,
            timestep_bounds_staging,
            base_dt: params.dt,
        })
    }

    /// Returns the list of rigid body coupling entries.
    ///
    /// Each entry specifies a collider-body pair that participates in MPM-rigid body
    /// interaction and the coupling mode.
    pub fn coupling(&self) -> &[BodyCouplingEntry] {
        &self.coupling
    }
}

impl<B: Backend, GpuModel: GpuParticleModelData> MpmPipeline<B, GpuModel> {
    /// Creates a new MPM compute pipeline by compiling all necessary shaders.
    ///
    /// This compiles and prepares all GPU compute kernels needed for the MPM algorithm.
    /// Shader compilation happens once at initialization; the resulting pipeline can
    /// execute many simulation steps efficiently.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for shader compilation
    /// * `compiler` - Slang compiler with registered shader modules (see [`crate::register_shaders`])
    ///
    /// # Returns
    ///
    /// A ready-to-use MPM pipeline, or an error if shader compilation fails.
    pub fn new(backend: &B, compiler: &SlangCompiler) -> Result<Self, B::Error> {
        Ok(Self {
            grid: WgGrid::from_backend(backend, compiler)?,
            prefix_sum: WgPrefixSum::from_backend(backend, compiler)?,
            sort: WgSort::from_backend(backend, compiler)?,
            p2g: WgP2G::from_backend(backend, compiler)?,
            p2g_cdf: WgP2GCdf::from_backend(backend, compiler)?,
            grid_update: WgGridUpdate::from_backend(backend, compiler)?,
            grid_update_cdf: WgGridUpdateCdf::from_backend(backend, compiler)?,
            particles_update: WgParticleUpdate::with_specializations(
                backend,
                compiler,
                &GpuModel::specialization_modules(),
            )?,
            rigid_particles_update: WgRigidParticleUpdate::from_backend(backend, compiler)?,
            g2p: WgG2P::from_backend(backend, compiler)?,
            g2p_cdf: WgG2PCdf::from_backend(backend, compiler)?,
            impulses: WgRigidImpulses::from_backend(backend, compiler)?,
            timestep_bounds: WgTimestepBounds::with_specializations(backend, compiler, &GpuModel::specialization_modules())?,
            _phantom: PhantomData,
        })
    }

    /// Executes one complete MPM simulation timestep.
    ///
    /// Advances the simulation forward by the timestep defined in `data.sim_params.dt`.
    /// This method orchestrates all stages of the MPM algorithm:
    ///
    /// 1. **Rigid particle update**: Update particles sampled from rigid body surfaces
    /// 2. **Grid sort**: Sort particles into grid cells for efficient neighbor queries
    /// 3. **P2G transfers**: Transfer particle mass/momentum to grid (both MPM and rigid particles)
    /// 4. **Grid update**: Apply forces and solve momentum equations on grid
    /// 5. **G2P transfers**: Interpolate grid velocities back to particles
    /// 6. **Particle update**: Integrate particle positions and update deformation gradients
    /// 7. **Impulse application**: Apply accumulated forces back to rigid bodies
    ///
    /// All operations execute as GPU compute passes. The encoder records commands but
    /// does not submit them; call `backend.queue().submit()` after this returns.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for command recording
    /// * `encoder` - Command encoder to record GPU operations into
    /// * `data` - Mutable simulation state (particles, grid, etc.)
    ///
    /// # Returns
    ///
    /// `Ok(())` if all GPU commands were recorded successfully, or an error if
    /// any kernel launch fails.
    pub async fn launch_step(
        &self,
        backend: &B,
        encoder: &mut B::Encoder,
        data: &mut MpmData<B, GpuModel>,
        // mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), B::Error> {
        {
            let mut pass = encoder.begin_pass(); // "update rigid particles", timestamps.as_deref_mut());
            self.impulses.launch_update_world_mass_properties(
                backend,
                &mut pass,
                &data.impulses,
                &data.bodies,
            )?;
            self.rigid_particles_update.launch(
                backend,
                &mut pass,
                &data.bodies,
                &data.rigid_particles,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("grid sort", timestamps.as_deref_mut());
            self.grid.launch_sort(
                backend,
                &mut pass,
                &data.particles,
                &data.rigid_particles,
                &data.grid,
                &mut data.prefix_sum,
                &self.sort,
                &self.prefix_sum,
            )?;
            self.sort.launch_sort_rigid_particles(
                backend,
                &mut pass,
                &data.rigid_particles,
                &data.grid,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("grid_update_cdf", timestamps.as_deref_mut());
            self.grid_update_cdf
                .launch(backend, &mut pass, &data.grid, &data.bodies)?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("p2g_cdf", timestamps.as_deref_mut());
            self.p2g_cdf.launch(
                backend,
                &mut pass,
                &data.grid,
                &data.rigid_particles,
                &data.bodies,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("g2p_cdf", timestamps.as_deref_mut());
            self.g2p_cdf.launch(
                backend,
                &mut pass,
                &data.sim_params,
                &data.grid,
                &data.particles,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("p2g", timestamps.as_deref_mut());
            self.p2g.launch(
                backend,
                &mut pass,
                &data.grid,
                &data.particles,
                &data.impulses,
                &data.bodies,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("grid_update", timestamps.as_deref_mut());
            self.grid_update
                .launch(backend, &mut pass, &data.sim_params, &data.grid)?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("g2p", timestamps.as_deref_mut());
            self.g2p.launch(
                backend,
                &mut pass,
                &data.sim_params,
                &data.grid,
                &data.particles,
                &data.bodies,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("particles_update", timestamps.as_deref_mut());
            self.particles_update.launch(
                backend,
                &mut pass,
                &data.sim_params,
                &data.grid,
                &data.particles,
                &data.bodies,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("integrate_bodies", timestamps.as_deref_mut());
            // TODO: should this be in a separate pipeline? Within impulse probably?
            self.impulses.launch(
                backend,
                &mut pass,
                &data.grid,
                &data.sim_params,
                &data.impulses,
                &data.bodies,
            )?;
        }

        Ok(())
    }
}

/*
#[cfg(test)]
#[cfg(feature = "dim3")]
mod test {
    use crate::models::ElasticCoefficients;
    use crate::pipeline::{MpmData, MpmPipeline};
    use crate::solver::{Particle, ParticleDynamics, SimulationParams};
    use nalgebra::vector;
    use rapier::prelude::{ColliderSet, RigidBodySet};
    use slang_hal::gpu::GpuInstance;
    use slang_hal::kernel::KernelInvocationQueue;
    use wgpu::Maintain;

    #[futures_test::test]
    #[serial_test::serial]
    async fn pipeline_queue_step() {
        let gpu = GpuInstance::new().await.unwrap();
        let pipeline = MpmPipeline::new(gpu.backend()).unwrap();

        let cell_width = 1.0;
        let mut cpu_particles = vec![];
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let position = vector![i as f32, j as f32, k as f32] / cell_width / 2.0;
                    cpu_particles.push(Particle {
                        position,
                        dynamics: ParticleDynamics::with_density(cell_width / 4.0, 1.0),
                        model: ElasticCoefficients::from_young_modulus(100_000.0, 0.33),
                        plasticity: None,
                        phase: None,
                    });
                }
            }
        }

        let params = SimulationParams {
            gravity: vector![0.0, -9.81, 0.0],
            dt: (1.0 / 60.0) / 10.0,
        };
        let mut data = MpmData::new(
            gpu.backend(),
            params,
            &cpu_particles,
            &RigidBodySet::default(),
            &ColliderSet::default(),
            cell_width,
            100_000,
        );
        let mut queue = KernelInvocationQueue::new(gpu.backend());
        pipeline.queue_step(&mut data, &mut queue, false);

        for _ in 0..3 {
            let mut encoder = gpu.backend().create_command_encoder(&Default::default());
            queue.encode(&mut encoder, None);
            let t0 = std::time::Instant::now();
            gpu.queue().submit(Some(encoder.finish()));
            gpu.backend().poll(Maintain::Wait);
            println!("Sim step time: {}", t0.elapsed().as_secs_f32());
        }
    }
}
 */
