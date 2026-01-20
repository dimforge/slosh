//! Impulse accumulation and application for MPM-rigid body coupling.

use crate::grid::grid::{GpuGrid, GpuGridMetadata};
use crate::solver::SimulationParams;
use crate::solver::params::GpuSimulationParams;
use encase::ShaderType;
use nexus::dynamics::{GpuBodySet, GpuMassProperties, GpuVelocity};
use nexus::math::GpuSim;
use rapier::math::{AngVector, Point, Vector};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{BufferUsages, Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};

/// GPU kernels for computing and applying impulses to rigid bodies from MPM.
///
/// Accumulates forces from MPM particles and applies them as impulses to
/// coupled rigid bodies for two-way interaction.
#[derive(Shader)]
#[shader(module = "slosh::solver::rigid_impulses")]
pub struct WgRigidImpulses<B: Backend> {
    /// Kernel for computing and applying impulses.
    pub update: GpuFunction<B>,
    /// Kernel for updating world-space mass properties.
    pub update_world_mass_properties: GpuFunction<B>,
}

/// Linear and angular impulse to apply to a rigid body.
///
/// Accumulated from MPM particle interactions during P2G.
#[derive(Copy, Clone, PartialEq, Debug, Default, ShaderType)]
#[repr(C)]
pub struct RigidImpulse {
    /// Center of mass (for convenience).
    pub com: Point<f32>,
    /// Linear impulse vector.
    pub linear: Vector<f32>,
    /// Angular impulse (torque).
    pub angular: AngVector<f32>,
}

/// GPU buffers for storing impulses from MPM to rigid bodies.
pub struct GpuImpulses<B: Backend> {
    /// Per-timestep incremental impulses.
    pub incremental_impulses: GpuVector<RigidImpulse, B>,
    /// Accumulated total impulses.
    pub total_impulses: GpuVector<RigidImpulse, B>,
    /// Staging buffer for CPU readback.
    pub total_impulses_staging: GpuVector<RigidImpulse, B>,
}

impl<B: Backend> GpuImpulses<B> {
    /// Creates impulse buffers for rigid bodies.
    ///
    /// Allocates space for up to 16 bodies (CPIC limitation).
    pub fn new(backend: &B) -> Result<Self, B::Error> {
        const MAX_BODY_COUNT: usize = 16; // CPIC doesnt support more.
        let impulses = [RigidImpulse::default(); MAX_BODY_COUNT];
        Ok(Self {
            incremental_impulses: GpuVector::vector_encased(
                backend,
                impulses,
                BufferUsages::STORAGE,
            )?,
            total_impulses: GpuVector::vector_encased(
                backend,
                impulses,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )?,
            total_impulses_staging: GpuVector::vector_encased(
                backend,
                impulses,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )?,
        })
    }
}

#[derive(ShaderArgs)]
struct RigidImpulsesArgs<'a, B: Backend> {
    params: Option<&'a GpuScalar<SimulationParams, B>>,
    grid: Option<&'a GpuVector<GpuGridMetadata, B>>,
    local_mprops: &'a GpuVector<GpuMassProperties, B>,
    poses: &'a GpuVector<GpuSim, B>,
    vels: &'a GpuVector<GpuVelocity, B>,
    mprops: &'a GpuVector<GpuMassProperties, B>,
    incremental_impulses: &'a GpuVector<RigidImpulse, B>,
}

impl<B: Backend> WgRigidImpulses<B> {
    /// Computes and applies impulses to rigid bodies from MPM grid.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `grid` - Grid containing accumulated momentum
    /// * `sim_params` - Simulation parameters
    /// * `impulses` - Impulse buffers to write
    /// * `bodies` - Target rigid bodies
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        grid: &GpuGrid<B>,
        sim_params: &GpuSimulationParams<B>,
        impulses: &GpuImpulses<B>,
        bodies: &GpuBodySet<B>,
    ) -> Result<(), B::Error> {
        if bodies.is_empty() {
            return Ok(());
        }

        let args = RigidImpulsesArgs {
            params: Some(&sim_params.params),
            grid: Some(&grid.meta),
            local_mprops: bodies.local_mprops(),
            poses: bodies.poses(),
            vels: bodies.vels(),
            mprops: bodies.mprops(),
            incremental_impulses: &impulses.incremental_impulses,
        };
        self.update.launch_grid(backend, pass, &args, 1)
    }

    /// Updates world-space mass properties for rigid bodies.
    ///
    /// Transforms local inertia tensors to world coordinates based on current poses.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `impulses` - Impulse buffers (unused in this kernel)
    /// * `bodies` - Bodies to update
    pub fn launch_update_world_mass_properties(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        impulses: &GpuImpulses<B>,
        bodies: &GpuBodySet<B>,
    ) -> Result<(), B::Error> {
        if bodies.is_empty() {
            return Ok(());
        }

        let args = RigidImpulsesArgs {
            params: None,
            grid: None,
            local_mprops: bodies.local_mprops(),
            poses: bodies.poses(),
            vels: bodies.vels(),
            mprops: bodies.mprops(),
            incremental_impulses: &impulses.incremental_impulses,
        };

        self.update_world_mass_properties
            .launch_grid(backend, pass, &args, 1)
    }
}
