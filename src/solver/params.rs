use crate::math::Vector;
use bytemuck::{Pod, Zeroable};
use slang_hal::{BufferUsages, backend::Backend};
use stensor::tensor::{GpuScalar, GpuTensor};

/// Global simulation parameters applied to all particles.
///
/// These parameters control the time integration and external forces acting on
/// the simulation. They're uploaded to GPU memory once and accessed by
/// multiple kernels during each timestep.
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct SimulationParams {
    /// Gravitational acceleration vector (m/s²).
    pub gravity: Vector,
    /// Padding for GPU alignment (2D only).
    #[cfg(feature = "dim2")]
    pub padding: f32,
    /// Simulation timestep duration (seconds).
    pub dt: f32,
}

/// Time-integration flags read by the particle update.
///
/// Mirror of `IntegratorFlags` in `shaders/slosh/solver/params.slang`.
#[derive(Copy, Clone, PartialEq, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct IntegratorFlags {
    /// Non-zero when the grid update is the implicit solver, which then evaluates the forces
    /// itself: the particle update leaves the stress out of the APIC affine matrix.
    pub implicit: u32,
    pub padding0: u32,
    pub padding1: u32,
    pub padding2: u32,
}

/// GPU-resident simulation parameters.
///
/// Wraps [`SimulationParams`] in a GPU uniform buffer for efficient access
/// across compute shaders.
pub struct GpuSimulationParams<B: Backend> {
    /// Uniform buffer containing simulation parameters.
    pub params: GpuScalar<SimulationParams, B>,
    /// Uniform buffer of time-integration flags read by the particle update.
    pub integrator_flags: GpuScalar<IntegratorFlags, B>,
    implicit: bool,
}

impl<B: Backend> GpuSimulationParams<B> {
    /// Uploads simulation parameters to GPU memory.
    ///
    /// Creates a uniform buffer that can be bound to compute shaders.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend for buffer allocation
    /// * `params` - Simulation parameters to upload
    pub fn new(backend: &B, params: SimulationParams) -> Result<Self, B::Error> {
        Ok(Self {
            params: GpuTensor::scalar(
                backend,
                params,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
            integrator_flags: GpuTensor::scalar(
                backend,
                IntegratorFlags::default(),
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )?,
            implicit: false,
        })
    }

    /// Tells the particle update whether the grid update is the implicit solver (which then
    /// evaluates the forces itself). Uploads only on change.
    pub fn set_implicit(&mut self, backend: &B, implicit: bool) -> Result<(), B::Error> {
        if implicit != self.implicit {
            self.implicit = implicit;
            let flags = IntegratorFlags {
                implicit: implicit as u32,
                ..Default::default()
            };
            backend.write_buffer(self.integrator_flags.buffer_mut(), 0, &[flags])?;
        }
        Ok(())
    }

    /// Whether the particle update currently leaves the stress out of the affine matrices.
    pub fn implicit(&self) -> bool {
        self.implicit
    }
}
