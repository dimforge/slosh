use bytemuck::{Pod, Zeroable};
use slang_hal::BufferUsages;
use slang_hal::backend::Backend;
use stensor::tensor::GpuVector;

#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBoundaryCondition {
    pub ty: u32,
    pub friction: f32,
}

impl GpuBoundaryCondition {
    pub const STICK: u32 = 0u32;
    pub const SLIP: u32 = 1u32;
    pub const SEPARATE: u32 = 2u32;
    pub const FRICTION_Z_UP: u32 = 3u32;
    pub const NON_REFLECTING: u32 = 4u32;
    pub const DISABLED: u32 = 5u32;

    pub fn stick() -> GpuBoundaryCondition {
        Self {
            ty: Self::STICK,
            friction: 0.0,
        }
    }

    pub fn slip() -> GpuBoundaryCondition {
        Self {
            ty: Self::SLIP,
            friction: 0.0,
        }
    }

    pub fn separate(friction: f32) -> GpuBoundaryCondition {
        Self {
            ty: Self::SEPARATE,
            friction,
        }
    }

    pub fn friction_z_up(friction: f32) -> GpuBoundaryCondition {
        Self {
            ty: Self::FRICTION_Z_UP,
            friction,
        }
    }

    pub fn disabled() -> GpuBoundaryCondition {
        Self {
            ty: Self::DISABLED,
            friction: 0.0,
        }
    }

    pub fn non_reflecting() -> GpuBoundaryCondition {
        todo!();
    }
}

impl Default for GpuBoundaryCondition {
    fn default() -> Self {
        Self::separate(1.0)
    }
}

/// GPU buffers for storing impulses from MPM to rigid bodies.
pub struct GpuMaterials<B: Backend> {
    pub materials: GpuVector<GpuBoundaryCondition, B>,
}

impl<B: Backend> GpuMaterials<B> {
    /// Creates impulse buffers for rigid bodies.
    ///
    /// Allocates space for up to 16 bodies (CPIC limitation).
    pub fn new(backend: &B, materials: &[GpuBoundaryCondition]) -> Result<Self, B::Error> {
        assert!(
            materials.len() <= 16,
            "CPIC only supports up to 16 colliders"
        );
        Ok(Self {
            materials: GpuVector::vector(backend, materials, BufferUsages::STORAGE)?,
        })
    }
}
