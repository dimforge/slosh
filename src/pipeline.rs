//! High-level MPM simulation pipeline orchestration.
//!
//! This module provides the main entry point for running MPM simulations. The pipeline
//! coordinates the execution of all MPM algorithm stages on the GPU.

use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::prefix_sum::{PrefixSumWorkspace, WgPrefixSum};
use crate::grid::sort::WgSort;
use crate::math::{GpuSim, Vector};
use crate::rbd::dynamics::GpuBodySet;
use crate::rbd::dynamics::body::{BodyCoupling, BodyCouplingEntry};
use crate::solver::{
    GpuBoundaryCondition, GpuImpulses, GpuMaterials, GpuParticleModelData, GpuParticles,
    GpuRigidParticles, GpuSimulationParams, GpuTimestepBounds, Particle, SimulationParams, WgG2P,
    WgGridUpdate, WgP2G, WgP2GScatterStyle, WgParticleUpdate, WgRigidImpulses,
    WgRigidParticleUpdate, WgTimestepBounds,
};
// The CDF kernel wrappers read the gated `Node.cdf`, so they only exist under the `cpic` feature.
#[cfg(feature = "cpic")]
use crate::solver::{WgG2PCdf, WgGridUpdateCdf, WgP2GCdf};
use rapier::dynamics::RigidBodySet;
use rapier::geometry::{ColliderHandle, ColliderSet};
use slang_hal::backend::{Backend, Encoder, GpuTimestamps};
use slang_hal::{BufferUsages, Shader, SlangCompiler};
use std::any::Any;
use std::marker::PhantomData;
use stensor::tensor::{GpuScalar, GpuTensor, GpuVector};

/// Selects which optional kernel families `MpmPipeline` compiles.
///
/// A pipeline that never dispatches some kernels (say hooks replace the built-in P2G/G2P and the
/// CDF/rigid paths go unused) can skip compiling them. Skipping is mandatory once the `Node`
/// struct is slimmed: the CDF kernels read `Node.cdf`, so they cannot be compiled after `cpic`
/// gates that field out.
///
/// The gather P2G and the CDF/rigid paths read the per-node linked lists, which only exist with
/// the `node_particle_lists` feature (on by default). Keep it enabled if you use
/// `builtin_transfers` (gather), `cdf`, or `rigid_particles`.
#[derive(Copy, Clone, Debug)]
pub struct MpmPipelineKernels {
    /// Built-in P2G/G2P transfer kernels (gather + scatter). Skip only if hooks handle
    /// `run_p2g`/`run_g2p` on every step.
    pub builtin_transfers: bool,
    /// CDF (rigid-body collision field) kernels. Read `Node.cdf`, so compiled only under the
    /// `cpic` feature. Compile-only for now: `launch_step` doesn't dispatch the CDF passes yet.
    pub cdf: bool,
    /// Rigid-particle update kernel. Compile-only for now: `launch_step` doesn't dispatch the
    /// rigid-particle pass yet.
    pub rigid_particles: bool,
    /// Run the per-node `reset` pass during the sort. Its only unconditional job is zeroing
    /// `Node.momentum_velocity_mass`, which every P2G variant overwrites on every active node
    /// anyway, so set `false` to skip it when the built-in P2G (or a `run_p2g` hook) is known to
    /// write every active node.
    ///
    /// Only honored with both `cpic` and `node_particle_lists` off. With either on, `reset` also
    /// does init those features rely on (the cpic incompatible/cdf lanes, the linked-list
    /// heads/len), so `launch_sort` forces it regardless of this flag.
    pub node_reset: bool,
    /// Built-in grid_update pass (momentum to velocity, gravity, velocity clamp, rigid-collision
    /// boundary condition). Disable only when a `run_p2g` hook folds this into its own kernel;
    /// the grid must hold velocities by the time G2P runs.
    pub grid_update: bool,
    /// Built-in particles_update pass (advection, deformation-gradient update, constitutive
    /// model, APIC affine). Disable only when a `run_g2p` hook folds this into its own kernel;
    /// particle state must be fully advanced before `after_particles_update` hooks run.
    pub particles_update: bool,
}

impl Default for MpmPipelineKernels {
    fn default() -> Self {
        Self {
            builtin_transfers: true,
            cdf: true,
            rigid_particles: true,
            node_reset: true,
            grid_update: true,
            particles_update: true,
        }
    }
}

impl MpmPipelineKernels {
    /// Minimal set for pipelines whose hooks replace the built-in transfers and that don't use
    /// the CDF or rigid-particle paths.
    pub fn hooks_only() -> Self {
        Self {
            builtin_transfers: false,
            cdf: false,
            rigid_particles: false,
            ..Self::default()
        }
    }
}

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
    // Kept for the alternative/CDF code paths that are currently commented out in `step`.
    #[allow(dead_code)]
    p2g: Option<WgP2G<B>>,
    p2g_scatter_style: Option<WgP2GScatterStyle<B>>,
    // The CDF kernels read `Node.cdf`, so gate them behind `cpic`: a build without the feature
    // (slim 16 B node) must not construct a kernel referencing the removed field.
    #[cfg(feature = "cpic")]
    #[allow(dead_code)]
    p2g_cdf: Option<WgP2GCdf<B>>,
    #[cfg(feature = "cpic")]
    #[allow(dead_code)]
    grid_update_cdf: Option<WgGridUpdateCdf<B>>,
    grid_update: Option<WgGridUpdate<B>>,
    particles_update: Option<WgParticleUpdate<B>>,
    g2p: Option<WgG2P<B>>,
    #[cfg(feature = "cpic")]
    #[allow(dead_code)]
    g2p_cdf: Option<WgG2PCdf<B>>,
    #[allow(dead_code)]
    rigid_particles_update: Option<WgRigidParticleUpdate<B>>,
    /// Maximum timestep bound calculation.
    pub timestep_bounds: WgTimestepBounds<B>,
    /// Rigid body impulse computation kernel (publicly accessible for external use).
    pub impulses: WgRigidImpulses<B>,
    /// Kernel families this pipeline was built with; also gates which built-in dispatches
    /// `launch_step` runs.
    kernels: MpmPipelineKernels,
    _phantom: PhantomData<GpuModel>,
}

/// Callbacks for adding custom steps to the MPM pipeline.
pub trait MpmPipelineHooks<B: Backend, GpuModel: GpuParticleModelData> {
    /// Custom operation run after particles are sorted and attached to the grid.
    fn after_particle_sort(
        &mut self,
        _backend: &B,
        _encoder: &mut B::Encoder,
        _data: &mut MpmData<B, GpuModel>,
        _state: &mut dyn Any,
        _timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    /// Replace the built-in Particle-To-Grid transfer.
    ///
    /// Return `Ok(true)` if the transfer was handled by this hook; the pipeline then
    /// skips its built-in P2G. Return `Ok(false)` (the default) to keep slosh's P2G.
    /// The hook is responsible for beginning its own compute pass.
    fn run_p2g(
        &mut self,
        _backend: &B,
        _encoder: &mut B::Encoder,
        _data: &mut MpmData<B, GpuModel>,
        _state: &mut dyn Any,
        _timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<bool, B::Error> {
        Ok(false)
    }

    /// Replace the built-in Grid-To-Particle transfer.
    ///
    /// Return `Ok(true)` if the transfer was handled by this hook; the pipeline then
    /// skips its built-in G2P. Return `Ok(false)` (the default) to keep slosh's G2P.
    /// The hook is responsible for beginning its own compute pass.
    fn run_g2p(
        &mut self,
        _backend: &B,
        _encoder: &mut B::Encoder,
        _data: &mut MpmData<B, GpuModel>,
        _state: &mut dyn Any,
        _timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<bool, B::Error> {
        Ok(false)
    }

    /// Custom operation run after the main Particle-To-Grid transfer.
    fn after_p2g(
        &mut self,
        _backend: &B,
        _encoder: &mut B::Encoder,
        _data: &mut MpmData<B, GpuModel>,
        _state: &mut dyn Any,
        _timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    /// Custom operation run after updating the grid.
    fn after_grid_update(
        &mut self,
        _backend: &B,
        _encoder: &mut B::Encoder,
        _data: &mut MpmData<B, GpuModel>,
        _state: &mut dyn Any,
        _timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    /// Custom operation run after the Grid-To-Particle transfer.
    fn after_g2p(
        &mut self,
        _backend: &B,
        _encoder: &mut B::Encoder,
        _data: &mut MpmData<B, GpuModel>,
        _state: &mut dyn Any,
        _timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    /// Custom operation run after updating particles.
    fn after_particles_update(
        &mut self,
        _backend: &B,
        _encoder: &mut B::Encoder,
        _data: &mut MpmData<B, GpuModel>,
        _state: &mut dyn Any,
        _timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), B::Error> {
        Ok(())
    }
}

impl<B: Backend, GpuModel: GpuParticleModelData> MpmPipelineHooks<B, GpuModel> for () {}

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
    /// Gravitational acceleration vector (m/s²).
    pub gravity: Vector,
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
    /// MPM materials associated to each rigid-body.
    pub body_materials: GpuMaterials<B>,
    /// Accumulated impulses to apply to rigid bodies from MPM interactions.
    pub impulses: GpuImpulses<B>,
    /// Staging buffer for reading rigid body poses back to CPU.
    pub poses_staging: GpuVector<GpuSim, B>,
    /// The timestep estimate computed from particles and their models.
    pub timestep_bounds: GpuScalar<GpuTimestepBounds, B>,
    /// Staging buffer for reading the timestep bound estimate.
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
        materials: &[(ColliderHandle, GpuBoundaryCondition)],
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
        let materials: Vec<_> = coupling
            .iter()
            .map(|c| {
                materials
                    .iter()
                    .find(|e| e.0 == c.collider)
                    .map(|e| e.1)
                    .unwrap_or_default()
            })
            .collect();
        Self::with_select_coupling(
            backend,
            params,
            particles,
            bodies,
            colliders,
            coupling,
            &materials,
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
        materials: &[GpuBoundaryCondition], // Must have the same size as `coupling`.
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, B::Error> {
        assert_eq!(coupling.len(), materials.len());

        let sampling_step = cell_width; // TODO: * 1.5 ?
        let bodies = GpuBodySet::from_rapier(backend, bodies, colliders, &coupling)?;
        let body_materials = GpuMaterials::new(backend, materials)?;
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
        let timestep_bounds = GpuTensor::scalar(
            backend,
            bounds,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )?;
        let timestep_bounds_staging = GpuTensor::scalar(
            backend,
            bounds,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;

        Ok(Self {
            sim_params,
            particles,
            gravity: params.gravity,
            rigid_particles,
            bodies,
            body_materials,
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
        Self::new_with_kernels(backend, compiler, MpmPipelineKernels::default())
    }

    /// Like [`Self::new`], but only compiles the kernel families selected by `kernels`. Any
    /// skipped kernel must never be dispatched.
    pub fn new_with_kernels(
        backend: &B,
        compiler: &SlangCompiler,
        kernels: MpmPipelineKernels,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            grid: WgGrid::from_backend(backend, compiler)?,
            prefix_sum: WgPrefixSum::from_backend(backend, compiler)?,
            sort: WgSort::from_backend(backend, compiler)?,
            p2g: kernels
                .builtin_transfers
                .then(|| WgP2G::from_backend(backend, compiler))
                .transpose()?,
            p2g_scatter_style: kernels
                .builtin_transfers
                .then(|| WgP2GScatterStyle::from_backend(backend, compiler))
                .transpose()?,
            #[cfg(feature = "cpic")]
            p2g_cdf: kernels
                .cdf
                .then(|| WgP2GCdf::from_backend(backend, compiler))
                .transpose()?,
            grid_update: kernels
                .grid_update
                .then(|| WgGridUpdate::from_backend(backend, compiler))
                .transpose()?,
            #[cfg(feature = "cpic")]
            grid_update_cdf: kernels
                .cdf
                .then(|| WgGridUpdateCdf::from_backend(backend, compiler))
                .transpose()?,
            #[cfg(feature = "comptime")]
            particles_update: kernels
                .particles_update
                .then(|| WgParticleUpdate::from_backend(backend, compiler))
                .transpose()?,
            #[cfg(feature = "runtime")]
            particles_update: kernels
                .particles_update
                .then(|| {
                    WgParticleUpdate::with_specializations(
                        backend,
                        compiler,
                        &GpuModel::specialization_modules(),
                    )
                })
                .transpose()?,
            rigid_particles_update: kernels
                .rigid_particles
                .then(|| WgRigidParticleUpdate::from_backend(backend, compiler))
                .transpose()?,
            g2p: kernels
                .builtin_transfers
                .then(|| WgG2P::from_backend(backend, compiler))
                .transpose()?,
            #[cfg(feature = "cpic")]
            g2p_cdf: kernels
                .cdf
                .then(|| WgG2PCdf::from_backend(backend, compiler))
                .transpose()?,
            impulses: WgRigidImpulses::from_backend(backend, compiler)?,
            #[cfg(feature = "comptime")]
            timestep_bounds: WgTimestepBounds::from_backend(backend, compiler)?,
            #[cfg(feature = "runtime")]
            timestep_bounds: WgTimestepBounds::with_specializations(
                backend,
                compiler,
                &GpuModel::specialization_modules(),
            )?,
            kernels,
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
        hooks: &mut dyn MpmPipelineHooks<B, GpuModel>,
        hooks_state: &mut dyn Any,
        mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), B::Error> {
        // {
        //     let mut pass = encoder.begin_pass("update_rigid_particles", timestamps.as_deref_mut());
        //     self.impulses.launch_update_world_mass_properties(
        //         backend,
        //         &mut pass,
        //         &data.impulses,
        //         &data.bodies,
        //     )?;
        //     self.rigid_particles_update.launch(
        //         backend,
        //         &mut pass,
        //         &data.bodies,
        //         &data.rigid_particles,
        //     )?;
        // }

        {
            let mut pass = encoder.begin_pass("grid_sort", timestamps.as_deref_mut());
            data.grid.swap_buffers();
            self.grid.launch_sort(
                backend,
                &mut pass,
                &data.particles,
                &data.rigid_particles,
                &data.grid,
                &mut data.prefix_sum,
                &self.sort,
                &self.prefix_sum,
                self.kernels.node_reset,
            )?;
            // self.sort.launch_sort_rigid_particles(
            //     backend,
            //     &mut pass,
            //     &data.rigid_particles,
            //     &data.grid,
            // )?;
        }

        hooks.after_particle_sort(
            backend,
            encoder,
            data,
            hooks_state,
            timestamps.as_deref_mut(),
        )?;

        // CDF passes, not yet wired. The cdf kernels are now `#[cfg(feature = "cpic")]
        // Option<..>`, so re-enabling this means gating the block on `cpic`, unwrapping with
        // `.as_ref().expect(..)`, and setting `MpmPipelineKernels::cdf`.
        // {
        //     let mut pass = encoder.begin_pass("grid_update_cdf", timestamps.as_deref_mut());
        //     self.grid_update_cdf
        //         .launch(backend, &mut pass, &data.grid, &data.bodies)?;
        // }
        //
        // {
        //     let mut pass = encoder.begin_pass("p2g_cdf", timestamps.as_deref_mut());
        //     self.p2g_cdf.launch(
        //         backend,
        //         &mut pass,
        //         &data.grid,
        //         &data.rigid_particles,
        //         &data.bodies,
        //     )?;
        // }
        //
        // {
        //     let mut pass = encoder.begin_pass("g2p_cdf", timestamps.as_deref_mut());
        //     self.g2p_cdf.launch(
        //         backend,
        //         &mut pass,
        //         &data.sim_params,
        //         &data.grid,
        //         &data.particles,
        //     )?;
        // }

        // Let a hook replace the built-in P2G transfer (e.g. to fuse extra fields into the
        // same sweep). When no hook handles it, run slosh's default scatter-style P2G.
        if !hooks.run_p2g(
            backend,
            encoder,
            data,
            hooks_state,
            timestamps.as_deref_mut(),
        )? {
            let mut pass = encoder.begin_pass("p2g", timestamps.as_deref_mut());
            self.p2g_scatter_style
                .as_ref()
                .expect(
                    "pipeline built without builtin transfer kernels; a hook must handle run_p2g",
                )
                .launch(
                    backend,
                    &mut pass,
                    &data.grid,
                    &data.particles,
                    &data.impulses,
                    &data.bodies,
                    &data.body_materials,
                )?;
        }

        hooks.after_p2g(
            backend,
            encoder,
            data,
            hooks_state,
            timestamps.as_deref_mut(),
        )?;

        if let Some(grid_update) = &self.grid_update {
            let mut pass = encoder.begin_pass("grid_update", timestamps.as_deref_mut());
            grid_update.launch(
                backend,
                &mut pass,
                &data.sim_params,
                &data.grid,
                &data.bodies,
                &data.body_materials,
            )?;
        }

        hooks.after_grid_update(
            backend,
            encoder,
            data,
            hooks_state,
            timestamps.as_deref_mut(),
        )?;

        // Let a hook replace the built-in G2P transfer (e.g. to fuse extra fields into the
        // same sweep). When no hook handles it, run slosh's default G2P.
        if !hooks.run_g2p(
            backend,
            encoder,
            data,
            hooks_state,
            timestamps.as_deref_mut(),
        )? {
            let mut pass = encoder.begin_pass("g2p", timestamps.as_deref_mut());
            self.g2p
                .as_ref()
                .expect(
                    "pipeline built without builtin transfer kernels; a hook must handle run_g2p",
                )
                .launch(
                    backend,
                    &mut pass,
                    &data.sim_params,
                    &data.grid,
                    &data.particles,
                    &data.bodies,
                    &data.body_materials,
                )?;
        }

        hooks.after_g2p(
            backend,
            encoder,
            data,
            hooks_state,
            timestamps.as_deref_mut(),
        )?;

        if let Some(particles_update) = &self.particles_update {
            let mut pass = encoder.begin_pass("particles_update", timestamps.as_deref_mut());
            particles_update.launch(
                backend,
                &mut pass,
                &data.sim_params,
                &data.grid,
                &data.particles,
                &data.bodies,
            )?;
        }

        hooks.after_particles_update(
            backend,
            encoder,
            data,
            hooks_state,
            timestamps.as_deref_mut(),
        )?;

        {
            let mut pass = encoder.begin_pass("integrate_bodies", timestamps);
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
