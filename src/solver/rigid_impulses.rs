use crate::grid::grid::{GpuGrid, GpuGridMetadata};
use crate::solver::SimulationParams;
use crate::solver::params::GpuSimulationParams;
use encase::ShaderType;
use nexus::dynamics::{GpuBodySet, GpuMassProperties, GpuVelocity};
use nexus::math::GpuSim;
use rapier::math::{AngVector, Point, Vector};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuVector};
use wgpu::BufferUsages;

#[derive(Shader)]
#[shader(module = "slosh::solver::rigid_impulses")]
pub struct WgRigidImpulses<B: Backend> {
    pub update: GpuFunction<B>,
    pub update_world_mass_properties: GpuFunction<B>,
}

#[derive(Copy, Clone, PartialEq, Debug, Default, ShaderType)]
#[repr(C)]
pub struct RigidImpulse {
    pub com: Point<f32>, // For convenience, to reduce the number of bindings
    pub linear: Vector<f32>,
    pub angular: AngVector<f32>,
}

pub struct GpuImpulses<B: Backend> {
    pub incremental_impulses: GpuVector<RigidImpulse, B>,
    pub total_impulses: GpuVector<RigidImpulse, B>,
    pub total_impulses_staging: GpuVector<RigidImpulse, B>,
}

impl<B: Backend> GpuImpulses<B> {
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
