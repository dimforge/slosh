use crate::models::ElasticCoefficients;
use bytemuck::{Pod, Zeroable};
use slang_hal::BufferUsages;
use slang_hal::backend::Backend;
use stensor::tensor::GpuVector;

/// Boundary condition applied to the grid nodes in contact with a collider.
///
/// The memory layout must match the shader-side `BoundaryCondition` struct in
/// `shaders/slosh/solver/boundary_condition.slang`.
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBoundaryCondition {
    pub ty: u32,
    pub friction: f32,
    /// Pressure (dilatational) wave speed of the material in contact with the boundary (m/s).
    ///
    /// Only read by the [`Self::NON_REFLECTING`] boundary condition.
    pub wave_speed_p: f32,
    /// Shear wave speed of the material in contact with the boundary (m/s).
    ///
    /// Only read by the [`Self::NON_REFLECTING`] boundary condition.
    pub wave_speed_s: f32,
}

impl GpuBoundaryCondition {
    pub const STICK: u32 = 0u32;
    pub const SLIP: u32 = 1u32;
    pub const SEPARATE: u32 = 2u32;
    pub const FRICTION_Z_UP: u32 = 3u32;
    pub const NON_REFLECTING: u32 = 4u32;
    pub const DISABLED: u32 = 5u32;

    pub fn stick() -> GpuBoundaryCondition {
        Self::new(Self::STICK, 0.0)
    }

    pub fn slip() -> GpuBoundaryCondition {
        Self::new(Self::SLIP, 0.0)
    }

    pub fn separate(friction: f32) -> GpuBoundaryCondition {
        Self::new(Self::SEPARATE, friction)
    }

    pub fn friction_z_up(friction: f32) -> GpuBoundaryCondition {
        Self::new(Self::FRICTION_Z_UP, friction)
    }

    pub fn disabled() -> GpuBoundaryCondition {
        Self::new(Self::DISABLED, 0.0)
    }

    /// An absorbing (non-reflecting) boundary based on Lysmer-Kuhlemeyer viscous dashpots.
    ///
    /// The boundary applies the traction a semi-infinite continuation of the material would,
    /// `-ρ·c_p·v_n` along the normal and `-ρ·c_s·v_t` along the tangent, so a wave at normal
    /// incidence is absorbed rather than reflected. Absorption degrades away from that incidence.
    ///
    /// The traction is graded over a band several cells deep (see `ABSORBING_LAYERS` in
    /// `shaders/slosh/solver/boundary_condition.slang`), which the domain must have room for.
    ///
    /// It is layered on top of the [`Self::separate`] contact response, so `friction` behaves as
    /// it does there. Note the shear dashpot damps tangential velocity all through the band, so
    /// material resting inside it is dragged to a halt; pass a zero `wave_speed_s` to leave
    /// sliding alone.
    ///
    /// # Arguments
    ///
    /// * `wave_speed_p` - Pressure wave speed `sqrt((λ + 2μ) / ρ)` of the material in contact (m/s)
    /// * `wave_speed_s` - Shear wave speed `sqrt(μ / ρ)` of the material in contact (m/s), or
    ///   zero to absorb the pressure wave only
    /// * `friction` - Coulomb friction coefficient of the contact response, as in [`Self::separate`]
    ///
    /// See [`Self::non_reflecting_for_material`] for computing the wave speeds from engineering
    /// parameters.
    pub fn non_reflecting(
        wave_speed_p: f32,
        wave_speed_s: f32,
        friction: f32,
    ) -> GpuBoundaryCondition {
        Self {
            ty: Self::NON_REFLECTING,
            friction,
            wave_speed_p,
            wave_speed_s,
        }
    }

    /// Same as [`Self::non_reflecting`], but derives the wave speeds from the elastic
    /// properties of the material in contact with the boundary.
    ///
    /// # Arguments
    ///
    /// * `young_modulus` - Young’s modulus E (Pa) of the material in contact
    /// * `poisson_ratio` - Poisson’s ratio ν of the material in contact
    /// * `density` - Density ρ of the material in contact (kg/m³, or kg/m² in 2D)
    /// * `friction` - Coulomb friction coefficient of the contact response
    pub fn non_reflecting_for_material(
        young_modulus: f32,
        poisson_ratio: f32,
        density: f32,
        friction: f32,
    ) -> GpuBoundaryCondition {
        let (p, s) = Self::wave_speeds(young_modulus, poisson_ratio, density);
        Self::non_reflecting(p, s, friction)
    }

    /// Pressure and shear wave speeds `(c_p, c_s)` of an isotropic linear elastic material.
    pub fn wave_speeds(young_modulus: f32, poisson_ratio: f32, density: f32) -> (f32, f32) {
        let coeffs = ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio);
        (
            ((coeffs.lambda + 2.0 * coeffs.mu) / density).sqrt(),
            (coeffs.mu / density).sqrt(),
        )
    }

    fn new(ty: u32, friction: f32) -> GpuBoundaryCondition {
        Self {
            ty,
            friction,
            wave_speed_p: 0.0,
            wave_speed_s: 0.0,
        }
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
