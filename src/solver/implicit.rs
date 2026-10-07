//! Implicit grid velocity solve: backward Euler with a Newton iteration on the incremental
//! potential.
//!
//! Each substep minimizes
//! `E(v) = 1/2 (v - v_free)^T M (v - v_free) + sum_p V0 psi(F_p(v)) + dt R(v)`
//! over the grid velocities, where `v_free` is the force-free velocity the explicit grid update
//! produces (gravity, clamp and boundary conditions applied) when the particle update leaves the
//! stress out of the affine matrices, `F_p(v)` is the trial deformation gradient (projected for
//! plastic models) and `R` the dissipation potential of the rate-dependent terms. Its gradient is
//! the backward Euler residual `M (v - v_free) - dt f(x^n + dt v, v)`, so a minimizer is a fully
//! implicit step. Newton's method solves it: at every iterate a residual pass evaluates the trial
//! state of each particle, a matrix-free conjugate gradient solves `(M + dt^2 K + dt D) dx = -g`
//! with the (definiteness-fixed) Hessian at that state, and a backtracking line search picks the
//! step, on the incremental potential or on the residual norm. Grid-level stick/slip boundary
//! conditions and fixed particles are projections inside the operator; the separating (contact)
//! condition stays an explicit correction re-applied after the solve. The advection limit of one
//! cell per step still holds: what the implicit solve removes is the sound-speed restriction on
//! the timestep.
//!
//! Everything runs on the GPU: the host encodes a fixed schedule of dispatches whose workgroup
//! counts live in indirect buffers that the reduction kernels arm or zero as the Newton
//! iteration, the CG solve and the line search progress. A one-iteration, no-line-search
//! configuration is the classic semi-implicit scheme (one linearization per substep).
//!
//! The constitutive models take part through the Slang interface `IImplicitParticleModel` (see
//! `shaders/slosh/models/implicit_interfaces.slang`), which the default models implement and
//! which a custom model specialization must export as `ImplicitParticleModel`.

use crate::grid::grid::{
    GpuActiveBlockHeader, GpuGrid, GpuGridHashMapEntry, GpuGridMetadata, GpuGridNode,
    GpuNodeCollision,
};
use crate::math::{Matrix, Vector};
use crate::rbd::dynamics::GpuBodySet;
use crate::solver::{
    GpuBoundaryCondition, GpuMaterials, GpuParticleModelData, GpuParticles, GpuSimulationParams,
    Kinematics, ParticlePosition, ParticleProperties, SimulationParams,
};
use bytemuck::{Pod, Zeroable};
use encase::ShaderType;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{BufferUsages, Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuTensor, GpuVector};

/// Line search criterion of the Newton iteration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum LineSearchCriterion {
    /// Armijo sufficient decrease of the incremental potential. Needs the models' energy
    /// densities (the default models provide them) and costs one particle reduction per trial.
    #[default]
    Energy,
    /// Decrease of the residual norm. Only needs the stresses, so it works for models without a
    /// potential (rate- or history-dependent ones), at the cost of one scatter per trial.
    Residual,
}

/// Parameters of the implicit grid solve.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ImplicitSolverParams {
    /// Newton iterations per substep. One iteration without line search is the semi-implicit
    /// scheme; a few iterations with line search converge to the fully implicit step.
    pub max_newton_iters: u32,
    /// Relative tolerance on the Newton residual norm: the iteration stops once `|g|` drops below
    /// `newton_rel_tol` times its initial value.
    pub newton_rel_tol: f32,
    /// Conjugate gradient iterations per Newton iteration. All the iterations are encoded up
    /// front; once the solve converges the remaining ones launch no work.
    pub max_cg_iters: u32,
    /// Relative CG tolerance on the preconditioned residual norm: the solve stops once `r . z`
    /// drops below `cg_rel_tol^2` times its initial value.
    pub cg_rel_tol: f32,
    /// Backtracking steps of the line search (step lengths `1, 1/2, 1/4, ...`). Zero disables the
    /// line search and always takes the full Newton step.
    pub line_search_steps: u32,
    /// What the line search measures.
    pub line_search: LineSearchCriterion,
}

impl Default for ImplicitSolverParams {
    fn default() -> Self {
        Self {
            max_newton_iters: 3,
            newton_rel_tol: 1.0e-2,
            max_cg_iters: 30,
            cg_rel_tol: 1.0e-3,
            line_search_steps: 4,
            line_search: LineSearchCriterion::Energy,
        }
    }
}

impl ImplicitSolverParams {
    /// The semi-implicit configuration: one linearization per substep, no line search.
    pub fn semi_implicit() -> Self {
        Self {
            max_newton_iters: 1,
            line_search_steps: 0,
            ..Self::default()
        }
    }
}

/// Time integration scheme of the MPM grid update.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum MpmIntegrator {
    /// Symplectic Euler: the timestep is bounded by the sound speed of the stiffest material.
    #[default]
    Explicit,
    /// Backward Euler solved by Newton's method on the grid. Stable at timesteps far beyond the
    /// sound-speed limit, at the price of some numerical damping. Requires a pipeline built with
    /// [`MpmPipelineKernels::implicit`](crate::pipeline::MpmPipelineKernels::implicit).
    Implicit(ImplicitSolverParams),
}

/// Armijo sufficient-decrease constant of the line search.
const ARMIJO: f32 = 1.0e-4;

/// Per-node state of the solve. Mirror of `CgNode` in `shaders/slosh/solver/implicit/types.slang`,
/// with the 16-byte vector padding of the structured-buffer layout spelled out so the struct is
/// plain data (readable and writable from the host without encase).
#[derive(Copy, Clone, Debug, PartialEq, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuCgNode {
    pub rhs: Vector,
    #[cfg(feature = "dim3")]
    pub _pad0: f32,
    pub x: Vector,
    #[cfg(feature = "dim3")]
    pub _pad1: f32,
    pub dx: Vector,
    #[cfg(feature = "dim3")]
    pub _pad2: f32,
    pub v: Vector,
    #[cfg(feature = "dim3")]
    pub _pad3: f32,
    pub g: Vector,
    #[cfg(feature = "dim3")]
    pub _pad4: f32,
    pub r: Vector,
    #[cfg(feature = "dim3")]
    pub _pad5: f32,
    pub z: Vector,
    #[cfg(feature = "dim3")]
    pub _pad6: f32,
    pub p: Vector,
    #[cfg(feature = "dim3")]
    pub _pad7: f32,
    pub ap: Vector,
    #[cfg(feature = "dim3")]
    pub _pad8: f32,
    // WGSL packs a scalar right after a `vec3` (size 12, align 16), so `mass` follows `normal`
    // without padding and the struct is padded at its end instead.
    pub normal: Vector,
    pub mass: f32,
    pub bc: u32,
    #[cfg(feature = "dim3")]
    pub _pad9: [u32; 3],
}

#[cfg(feature = "dim3")]
static_assertions::assert_eq_size!(GpuCgNode, [u8; 176]);
#[cfg(feature = "dim2")]
static_assertions::assert_eq_size!(GpuCgNode, [u8; 88]);

/// Per-particle scratch of the solve. Mirror of `ImplicitParticle` in the shader.
#[derive(Copy, Clone, Debug, PartialEq, Default, ShaderType)]
#[repr(C)]
pub struct GpuImplicitParticle {
    pub stress: Matrix,
    pub trial_def_grad: Matrix,
    pub coeff: f32,
    pub init_volume: f32,
    pub energy: f32,
    pub padding: u32,
}

/// Scalars of the Newton / conjugate gradient iteration. Mirror of `CgScalars` in the shader.
#[derive(Copy, Clone, Debug, PartialEq, Default, Pod, Zeroable)]
#[repr(C)]
pub struct CgScalars {
    /// Current CG `r . z`.
    pub rz: f32,
    /// Initial CG `r . z` of the current solve.
    pub rz0: f32,
    /// Last `p . A p`.
    pub p_ap: f32,
    /// Last CG step length.
    pub alpha: f32,
    /// Last CG direction coefficient.
    pub beta: f32,
    /// Non-zero once the current CG solve stopped (converged, or breakdown).
    pub cg_converged: u32,
    /// CG iterations of the last solve.
    pub cg_iters: u32,
    /// Squared norm of the Newton residual at the current iterate.
    pub g_sq: f32,
    /// Squared norm of the initial Newton residual.
    pub g0_sq: f32,
    /// Incremental potential at the current iterate (energy criterion).
    pub energy: f32,
    /// Incremental potential (or residual norm) at the last line search trial.
    pub energy_trial: f32,
    /// Directional derivative `g . dx` of the current line search.
    pub gdx: f32,
    /// Step length of the current line search trial.
    pub ls_alpha: f32,
    /// Line search state.
    pub ls_state: u32,
    /// Trials evaluated by the current line search.
    pub ls_trial: u32,
    /// Completed Newton iterations.
    pub newton_iters: u32,
    /// Non-zero once the Newton iteration stopped.
    pub newton_done: u32,
    pub padding: u32,
}

/// Uniform parameters of the solve. Mirror of `CgParams` in the shader.
#[derive(Copy, Clone, Debug, PartialEq, Default, Pod, Zeroable)]
#[repr(C)]
pub struct CgParams {
    pub cg_rel_tol_sq: f32,
    pub newton_rel_tol_sq: f32,
    pub armijo: f32,
    pub line_search_steps: u32,
    pub criterion: u32,
    pub padding0: u32,
    pub padding1: u32,
    pub padding2: u32,
}

impl CgParams {
    fn from_params(params: &ImplicitSolverParams) -> Self {
        Self {
            cg_rel_tol_sq: params.cg_rel_tol * params.cg_rel_tol,
            newton_rel_tol_sq: params.newton_rel_tol * params.newton_rel_tol,
            armijo: ARMIJO,
            line_search_steps: params.line_search_steps,
            criterion: match params.line_search {
                LineSearchCriterion::Energy => 0,
                LineSearchCriterion::Residual => 1,
            },
            padding0: 0,
            padding1: 0,
            padding2: 0,
        }
    }
}

/// GPU buffers of the implicit solve. Grown lazily to the grid and particle capacities.
pub struct ImplicitWorkspace<B: Backend> {
    /// Per-node Newton / CG state.
    pub cg_nodes: GpuVector<GpuCgNode, B>,
    /// Per-particle scratch (trial state, linearized stress, energy).
    pub particles: GpuVector<GpuImplicitParticle, B>,
    /// One dot-product / energy partial per active block.
    pub partials: GpuVector<f32, B>,
    /// One energy partial per 64 particles.
    pub partials_particles: GpuVector<f32, B>,
    /// Solver scalars, kept on the GPU.
    pub scalars: GpuScalar<CgScalars, B>,
    scalars_staging: GpuScalar<CgScalars, B>,
    /// Uniform solve parameters.
    pub cg_params: GpuScalar<CgParams, B>,
    /// Indirect dispatch of the per-Newton-iteration node kernels.
    pub newton_dispatch: GpuScalar<[u32; 3], B>,
    /// Indirect dispatch of the CG kernels, zeroed on convergence.
    pub cg_dispatch: GpuScalar<[u32; 3], B>,
    /// Indirect dispatch of the line search node kernels, zeroed on acceptance.
    pub ls_dispatch: GpuScalar<[u32; 3], B>,
    /// Indirect dispatch of the line search particle kernel.
    pub ls_particle_dispatch: GpuScalar<[u32; 3], B>,
    uploaded_params: CgParams,
}

impl<B: Backend> ImplicitWorkspace<B> {
    /// The CG node and partial buffers are also readable and writable from the host, for
    /// diagnostics and tests.
    const NODE_USAGES: BufferUsages = BufferUsages::STORAGE
        .union(BufferUsages::COPY_SRC)
        .union(BufferUsages::COPY_DST);

    /// Allocates an empty workspace.
    pub fn new(backend: &B) -> Result<Self, B::Error> {
        let uploaded_params = CgParams::from_params(&ImplicitSolverParams::default());
        let dispatch = |backend: &B| {
            GpuTensor::scalar(
                backend,
                [0u32, 1, 1],
                BufferUsages::STORAGE | BufferUsages::INDIRECT,
            )
        };
        Ok(Self {
            // Never bound empty: WebGPU rejects zero-sized bindings.
            cg_nodes: GpuVector::vector_uninit(backend, 1, Self::NODE_USAGES)?,
            particles: GpuVector::vector_uninit_encased(backend, 1, BufferUsages::STORAGE)?,
            partials: GpuVector::vector_uninit(backend, 1, Self::NODE_USAGES)?,
            partials_particles: GpuVector::vector_uninit(backend, 1, Self::NODE_USAGES)?,
            scalars: GpuTensor::scalar(
                backend,
                CgScalars::default(),
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            scalars_staging: GpuTensor::scalar(
                backend,
                CgScalars::default(),
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )?,
            cg_params: GpuTensor::scalar(
                backend,
                uploaded_params,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
            newton_dispatch: dispatch(backend)?,
            cg_dispatch: dispatch(backend)?,
            ls_dispatch: dispatch(backend)?,
            ls_particle_dispatch: dispatch(backend)?,
            uploaded_params,
        })
    }

    /// Grows the buffers to the current grid and particle sizes.
    pub fn ensure_capacity<GpuModel: GpuParticleModelData>(
        &mut self,
        backend: &B,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
    ) -> Result<(), B::Error> {
        let num_nodes = grid.nodes.len();
        if self.cg_nodes.len() < num_nodes {
            self.cg_nodes = GpuVector::vector_uninit(backend, num_nodes as u32, Self::NODE_USAGES)?;
        }
        let num_blocks = grid.active_blocks.len();
        if self.partials.len() < num_blocks {
            self.partials =
                GpuVector::vector_uninit(backend, num_blocks as u32, Self::NODE_USAGES)?;
        }
        let num_particles = particles.len().max(1) as u64;
        if self.particles.len() < num_particles {
            self.particles = GpuVector::vector_uninit_encased(
                backend,
                num_particles as u32,
                BufferUsages::STORAGE,
            )?;
        }
        let num_groups = num_particles.div_ceil(64);
        if self.partials_particles.len() < num_groups {
            self.partials_particles =
                GpuVector::vector_uninit(backend, num_groups as u32, Self::NODE_USAGES)?;
        }
        Ok(())
    }

    /// Uploads the solve parameters if they changed.
    fn set_params(&mut self, backend: &B, params: &ImplicitSolverParams) -> Result<(), B::Error> {
        let cg_params = CgParams::from_params(params);
        if cg_params != self.uploaded_params {
            self.uploaded_params = cg_params;
            backend.write_buffer(self.cg_params.buffer_mut(), 0, &[cg_params])?;
        }
        Ok(())
    }

    /// Reads back the solver scalars of the last solve (blocking). Diagnostics only: this waits
    /// for every queued submission.
    pub async fn read_scalars(&mut self, backend: &B) -> Result<CgScalars, B::Error> {
        let mut encoder = backend.begin_encoding();
        self.scalars_staging
            .copy_from_view(&mut encoder, &self.scalars)?;
        backend.submit(encoder)?;
        let mut result = [CgScalars::default()];
        backend
            .read_buffer(self.scalars_staging.buffer(), &mut result)
            .await?;
        Ok(result[0])
    }
}

/// Gather leg of the operator and of the residual (particle side).
#[derive(Shader)]
#[shader(
    module = "slosh::solver::implicit::gather",
    specialize = ["slosh::models::specializations"]
)]
pub struct WgImplicitGather<B: Backend> {
    gather_operator: GpuFunction<B>,
    gather_residual: GpuFunction<B>,
}

/// Scatter leg of the operator and of the residual (node side).
#[derive(Shader)]
#[shader(module = "slosh::solver::implicit::scatter")]
pub struct WgImplicitScatter<B: Backend> {
    scatter_operator: GpuFunction<B>,
    scatter_residual: GpuFunction<B>,
}

/// Vector kernels and reductions of the CG, Newton and line search loops.
#[derive(Shader)]
#[shader(module = "slosh::solver::implicit::cg")]
pub struct WgImplicitCg<B: Backend> {
    cg_reset: GpuFunction<B>,
    prepare_particles: GpuFunction<B>,
    seed_from_grid: GpuFunction<B>,
    fixed_nodes: GpuFunction<B>,
    apply_boundary_conditions: GpuFunction<B>,
    cg_init_residual: GpuFunction<B>,
    cg_update_solution: GpuFunction<B>,
    cg_update_direction: GpuFunction<B>,
    ls_prepare: GpuFunction<B>,
    ls_trial: GpuFunction<B>,
    ls_finalize: GpuFunction<B>,
    energy_particles: GpuFunction<B>,
    cg_reduce_init: GpuFunction<B>,
    cg_reduce_alpha: GpuFunction<B>,
    cg_reduce_beta: GpuFunction<B>,
    newton_check: GpuFunction<B>,
    ls_reduce_prepare: GpuFunction<B>,
    ls_reduce_energy: GpuFunction<B>,
    ls_reduce_residual: GpuFunction<B>,
    init_energy: GpuFunction<B>,
}

/// Bindings of every implicit kernel; each kernel binds the subset it declares (slang-hal
/// matches by parameter name).
#[derive(ShaderArgs)]
struct ImplicitArgs<'a, B: Backend, GpuModel: GpuParticleModelData> {
    params: &'a GpuScalar<SimulationParams, B>,
    grid: &'a GpuScalar<GpuGridMetadata, B>,
    hmap_entries: &'a GpuVector<GpuGridHashMapEntry, B>,
    active_blocks: &'a GpuVector<GpuActiveBlockHeader, B>,
    nodes: &'a GpuVector<GpuGridNode, B>,
    node_collisions: &'a GpuVector<GpuNodeCollision, B>,
    body_materials: &'a GpuVector<GpuBoundaryCondition, B>,
    sorted_particle_ids: &'a GpuVector<u32, B>,
    particles_pos: &'a GpuVector<ParticlePosition, B>,
    particles_kin: &'a GpuVector<Kinematics, B>,
    particles_props: &'a GpuVector<ParticleProperties, B>,
    particles_def_grad: &'a GpuTensor<Matrix, B>,
    particles_model: &'a GpuTensor<GpuModel, B>,
    particles_len: &'a GpuScalar<u32, B>,
    cg_nodes: &'a GpuVector<GpuCgNode, B>,
    implicit_particles: &'a GpuVector<GpuImplicitParticle, B>,
    partials: &'a GpuVector<f32, B>,
    partials_particles: &'a GpuVector<f32, B>,
    scalars: &'a GpuScalar<CgScalars, B>,
    cg_params: &'a GpuScalar<CgParams, B>,
    cg_dispatch: &'a GpuScalar<[u32; 3], B>,
    newton_dispatch: &'a GpuScalar<[u32; 3], B>,
    ls_dispatch: &'a GpuScalar<[u32; 3], B>,
    ls_particle_dispatch: &'a GpuScalar<[u32; 3], B>,
}

/// GPU kernels of the implicit grid solve.
pub struct WgImplicitSolver<B: Backend> {
    gather: WgImplicitGather<B>,
    scatter: WgImplicitScatter<B>,
    cg: WgImplicitCg<B>,
}

impl<B: Backend> WgImplicitSolver<B> {
    /// Compiles the kernels, specializing the gather leg for the given model modules (see
    /// [`GpuParticleModelData::specialization_modules`]).
    pub fn with_specializations(
        backend: &B,
        compiler: &slang_hal::SlangCompiler,
        specializations: &[String],
    ) -> Result<Self, B::Error> {
        Ok(Self {
            gather: WgImplicitGather::with_specializations(backend, compiler, specializations)?,
            scatter: WgImplicitScatter::from_backend(backend, compiler)?,
            cg: WgImplicitCg::from_backend(backend, compiler)?,
        })
    }

    fn args<'a, GpuModel: GpuParticleModelData>(
        sim_params: &'a GpuSimulationParams<B>,
        grid: &'a GpuGrid<B>,
        particles: &'a GpuParticles<B, GpuModel>,
        body_materials: &'a GpuMaterials<B>,
        workspace: &'a ImplicitWorkspace<B>,
    ) -> ImplicitArgs<'a, B, GpuModel> {
        ImplicitArgs {
            params: &sim_params.params,
            grid: &grid.meta,
            hmap_entries: &grid.hmap_entries,
            active_blocks: &grid.active_blocks,
            nodes: &grid.nodes,
            node_collisions: &grid.node_collisions,
            body_materials: &body_materials.materials,
            sorted_particle_ids: particles.sorted_ids(),
            particles_pos: particles.positions(),
            particles_kin: &particles.kinematics,
            particles_props: &particles.properties,
            particles_def_grad: &particles.def_grad,
            particles_model: particles.models(),
            particles_len: particles.gpu_len(),
            cg_nodes: &workspace.cg_nodes,
            implicit_particles: &workspace.particles,
            partials: &workspace.partials,
            partials_particles: &workspace.partials_particles,
            scalars: &workspace.scalars,
            cg_params: &workspace.cg_params,
            cg_dispatch: &workspace.cg_dispatch,
            newton_dispatch: &workspace.newton_dispatch,
            ls_dispatch: &workspace.ls_dispatch,
            ls_particle_dispatch: &workspace.ls_particle_dispatch,
        }
    }

    /// Encodes the whole solve. Must run right after the grid update (built-in or fused into a
    /// P2G hook), with the particle update told to leave the stress out of the affine matrices
    /// (see [`GpuSimulationParams::set_implicit`]). On return the grid nodes hold the new
    /// velocities. The particles must not be empty.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_solve<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        solver_params: &ImplicitSolverParams,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        _bodies: &GpuBodySet<B>,
        body_materials: &GpuMaterials<B>,
        workspace: &mut ImplicitWorkspace<B>,
    ) -> Result<(), B::Error> {
        workspace.ensure_capacity(backend, grid, particles)?;
        workspace.set_params(backend, solver_params)?;
        let num_particles = particles.len() as u32;
        let args = Self::args(sim_params, grid, particles, body_materials, workspace);
        let grid_dispatch = grid.indirect_n_g2p_p2g_groups.buffer();
        let residual_criterion = solver_params.line_search == LineSearchCriterion::Residual;

        // Setup: scalars and dispatches, per-particle coefficients, right-hand side and initial
        // iterate from the explicit grid update, grid-level Dirichlet conditions.
        self.cg.cg_reset.launch(backend, pass, &args, [1, 1, 1])?;
        self.cg
            .prepare_particles
            .launch_capped(backend, pass, &args, num_particles)?;
        self.cg
            .seed_from_grid
            .launch_indirect(backend, pass, &args, grid_dispatch)?;
        self.cg
            .fixed_nodes
            .launch_capped(backend, pass, &args, num_particles)?;

        // Trial state and incremental potential of the initial iterate.
        self.gather
            .gather_residual
            .launch_indirect(backend, pass, &args, grid_dispatch)?;
        self.cg
            .ls_trial
            .launch_indirect(backend, pass, &args, grid_dispatch)?;
        self.cg
            .energy_particles
            .launch(backend, pass, &args, [num_particles, 1, 1])?;
        self.cg
            .init_energy
            .launch(backend, pass, &args, [256, 1, 1])?;

        for _ in 0..solver_params.max_newton_iters.max(1) {
            // Newton residual at the current iterate, and its norm.
            self.scatter.scatter_residual.launch_indirect(
                backend,
                pass,
                &args,
                workspace.newton_dispatch.buffer(),
            )?;
            self.cg
                .newton_check
                .launch(backend, pass, &args, [256, 1, 1])?;

            // CG solve of (M + dt^2 K + dt D) dx = -g, from dx = 0.
            self.cg.cg_init_residual.launch_indirect(
                backend,
                pass,
                &args,
                workspace.cg_dispatch.buffer(),
            )?;
            self.cg
                .cg_reduce_init
                .launch(backend, pass, &args, [256, 1, 1])?;
            for _ in 0..solver_params.max_cg_iters {
                self.gather.gather_operator.launch_indirect(
                    backend,
                    pass,
                    &args,
                    workspace.cg_dispatch.buffer(),
                )?;
                self.scatter.scatter_operator.launch_indirect(
                    backend,
                    pass,
                    &args,
                    workspace.cg_dispatch.buffer(),
                )?;
                self.cg
                    .cg_reduce_alpha
                    .launch(backend, pass, &args, [256, 1, 1])?;
                self.cg.cg_update_solution.launch_indirect(
                    backend,
                    pass,
                    &args,
                    workspace.cg_dispatch.buffer(),
                )?;
                self.cg
                    .cg_reduce_beta
                    .launch(backend, pass, &args, [256, 1, 1])?;
                self.cg.cg_update_direction.launch_indirect(
                    backend,
                    pass,
                    &args,
                    workspace.cg_dispatch.buffer(),
                )?;
            }

            // Backtracking line search along dx.
            self.cg.ls_prepare.launch_indirect(
                backend,
                pass,
                &args,
                workspace.newton_dispatch.buffer(),
            )?;
            self.cg
                .ls_reduce_prepare
                .launch(backend, pass, &args, [256, 1, 1])?;
            for _ in 0..solver_params.line_search_steps.max(1) {
                self.cg.ls_trial.launch_indirect(
                    backend,
                    pass,
                    &args,
                    workspace.ls_dispatch.buffer(),
                )?;
                self.gather.gather_residual.launch_indirect(
                    backend,
                    pass,
                    &args,
                    workspace.ls_dispatch.buffer(),
                )?;
                if residual_criterion {
                    self.scatter.scatter_residual.launch_indirect(
                        backend,
                        pass,
                        &args,
                        workspace.ls_dispatch.buffer(),
                    )?;
                    self.cg
                        .ls_reduce_residual
                        .launch(backend, pass, &args, [256, 1, 1])?;
                } else {
                    self.cg.energy_particles.launch_indirect(
                        backend,
                        pass,
                        &args,
                        workspace.ls_particle_dispatch.buffer(),
                    )?;
                    self.cg
                        .ls_reduce_energy
                        .launch(backend, pass, &args, [256, 1, 1])?;
                }
            }
            self.cg
                .ls_finalize
                .launch_indirect(backend, pass, &args, grid_dispatch)?;
        }

        // The contact boundary condition stays an explicit correction on the solved velocities.
        self.cg
            .apply_boundary_conditions
            .launch_indirect(backend, pass, &args, grid_dispatch)
    }

    /// Setup passes only (reset, coefficients, seed from the grid, Dirichlet nodes, trial state
    /// of the initial iterate). Exposed for the operator tests.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_prepare<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        body_materials: &GpuMaterials<B>,
        workspace: &mut ImplicitWorkspace<B>,
    ) -> Result<(), B::Error> {
        workspace.ensure_capacity(backend, grid, particles)?;
        let num_particles = particles.len() as u32;
        let args = Self::args(sim_params, grid, particles, body_materials, workspace);
        let grid_dispatch = grid.indirect_n_g2p_p2g_groups.buffer();
        self.cg.cg_reset.launch(backend, pass, &args, [1, 1, 1])?;
        self.cg
            .prepare_particles
            .launch_capped(backend, pass, &args, num_particles)?;
        self.cg
            .seed_from_grid
            .launch_indirect(backend, pass, &args, grid_dispatch)?;
        self.cg
            .fixed_nodes
            .launch_capped(backend, pass, &args, num_particles)?;
        self.gather
            .gather_residual
            .launch_indirect(backend, pass, &args, grid_dispatch)
    }

    /// Applies the operator to the CG search direction `p` (writes `ap` and the block partials
    /// of `p . ap`). Exposed for the operator tests; dispatched over every active block.
    pub fn launch_apply<GpuModel: GpuParticleModelData>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        sim_params: &GpuSimulationParams<B>,
        grid: &GpuGrid<B>,
        particles: &GpuParticles<B, GpuModel>,
        body_materials: &GpuMaterials<B>,
        workspace: &ImplicitWorkspace<B>,
    ) -> Result<(), B::Error> {
        let args = Self::args(sim_params, grid, particles, body_materials, workspace);
        let grid_dispatch = grid.indirect_n_g2p_p2g_groups.buffer();
        self.gather
            .gather_operator
            .launch_indirect(backend, pass, &args, grid_dispatch)?;
        self.scatter
            .scatter_operator
            .launch_indirect(backend, pass, &args, grid_dispatch)
    }
}
