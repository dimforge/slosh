use crate::models::{DruckerPrager, DruckerPragerPlasticState, ElasticCoefficients};
use bytemuck::{NoUninit, Pod, Zeroable};

/// Material model for MPM particles.
///
/// Defines the constitutive behavior (how stress relates to deformation) for particles.
/// Supports both elastic and plastic materials with different strain energy formulations.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParticleModel {
    /// Linear elastic material (St. Venant-Kirchhoff).
    ElasticLinear(ElasticCoefficients),
    /// Neo-Hookean hyperelastic material (better for large deformations).
    ElasticNeoHookean(ElasticCoefficients),
    /// Sand/granular material with linear elasticity and Drucker-Prager plasticity.
    SandLinear(SandModel),
    /// Sand with Neo-Hookean elasticity and Drucker-Prager plasticity.
    SandNeoHookean(SandModel),
}

impl Default for ParticleModel {
    fn default() -> Self {
        Self::elastic(Self::DEFAULT_YOUNG_MODULUS, Self::DEFAULT_POISSON_RATIO)
    }
}

impl ParticleModel {
    /// Default Young's modulus for elastic materials (Pa).
    pub const DEFAULT_YOUNG_MODULUS: f32 = 1_000.0;
    /// Default Poisson's ratio for elastic materials (dimensionless).
    pub const DEFAULT_POISSON_RATIO: f32 = 0.2;

    /// Creates a linear elastic material model.
    ///
    /// # Arguments
    ///
    /// * `young_modulus` - Stiffness (Pa)
    /// * `poisson_ratio` - Ratio of lateral to axial strain (0.0 - 0.5)
    pub fn elastic(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::ElasticLinear(ElasticCoefficients::from_young_modulus(
            young_modulus,
            poisson_ratio,
        ))
    }

    pub fn elastic_neo_hookean(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::ElasticNeoHookean(ElasticCoefficients::from_young_modulus(
            young_modulus,
            poisson_ratio,
        ))
    }

    /// Creates a sand/granular material model with Drucker-Prager plasticity.
    ///
    /// # Arguments
    ///
    /// * `young_modulus` - Elastic stiffness (Pa)
    /// * `poisson_ratio` - Elastic Poisson's ratio (0.0 - 0.5)
    pub fn sand(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel::SandLinear(SandModel {
            plastic_state: DruckerPragerPlasticState::default(),
            plastic: DruckerPrager::new(young_modulus, poisson_ratio),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }

    /// Creates a sand/granular material model with Drucker-Prager plasticity and Neo-Hookean
    /// elasticity.
    ///
    /// # Arguments
    ///
    /// * `young_modulus` - Elastic stiffness (Pa)
    /// * `poisson_ratio` - Elastic Poisson's ratio (0.0 - 0.5)
    pub fn sand_neo_hookean(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel::SandNeoHookean(SandModel {
            plastic_state: DruckerPragerPlasticState::default(),
            plastic: DruckerPrager::new(young_modulus, poisson_ratio),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }
}

/// GPU-compatible version of [`ParticleModel`] with explicit padding.
///
/// This enum has the same variants as [`ParticleModel`] but includes padding
/// to satisfy alignment requirements for GPU buffers. The memory layout must
/// match the shader-side `SloshParticleModel` definition exactly.
#[derive(Copy, Clone, Debug, PartialEq, NoUninit)]
#[repr(u32)]
pub enum GpuParticleModel {
    /// Linear elastic model with padding for GPU alignment.
    ElasticLinear(ElasticCoefficients, [u32; 9]) = 0,
    /// Neo-Hookean elastic model with padding for GPU alignment.
    ElasticNeoHookean(ElasticCoefficients, [u32; 9]) = 1,
    /// Sand with linear elasticity and Drucker-Prager plasticity.
    SandLinear(SandModel) = 2,
    /// Sand with Neo-Hookean elasticity and Drucker-Prager plasticity.
    SandNeoHookean(SandModel) = 3,
}

// IMPORTANT: this assertions is here to reduce risks of `GpuParticleModel` from mismatching
//            `SloshParticleModel` in
static_assertions::assert_eq_size!(GpuParticleModel, [u8; 52]);

impl From<ParticleModel> for GpuParticleModel {
    fn from(val: ParticleModel) -> Self {
        match val {
            ParticleModel::ElasticLinear(elastic_linear) => {
                GpuParticleModel::ElasticLinear(elastic_linear, [0; _])
            }
            ParticleModel::ElasticNeoHookean(elastic_neo_hookean) => {
                GpuParticleModel::ElasticNeoHookean(elastic_neo_hookean, [0; _])
            }
            ParticleModel::SandLinear(sand_linear) => GpuParticleModel::SandLinear(sand_linear),
            ParticleModel::SandNeoHookean(sand_neo_hookean) => {
                GpuParticleModel::SandNeoHookean(sand_neo_hookean)
            }
        }
    }
}

impl From<GpuParticleModel> for ParticleModel {
    fn from(val: GpuParticleModel) -> Self {
        match val {
            GpuParticleModel::ElasticLinear(elastic_linear, _) => {
                ParticleModel::ElasticLinear(elastic_linear)
            }
            GpuParticleModel::ElasticNeoHookean(elastic_neo_hookean, _) => {
                ParticleModel::ElasticNeoHookean(elastic_neo_hookean)
            }
            GpuParticleModel::SandLinear(sand_linear) => ParticleModel::SandLinear(sand_linear),
            GpuParticleModel::SandNeoHookean(sand_neo_hookean) => {
                ParticleModel::SandNeoHookean(sand_neo_hookean)
            }
        }
    }
}

/// Combined elastic-plastic model for sand and granular materials.
///
/// Stores both elastic coefficients and Drucker-Prager plasticity state.
/// The plastic state tracks accumulated plastic deformation.
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct SandModel {
    /// Current plastic deformation state.
    pub plastic_state: DruckerPragerPlasticState,
    /// Drucker-Prager plasticity model parameters.
    pub plastic: DruckerPrager,
    /// Elastic coefficients (Lamé parameters).
    pub elastic: ElasticCoefficients,
}

/// Trait for types that can be used as GPU particle model data.
///
/// Implementors must provide conversion from CPU-side model representation
/// and specify shader specialization modules for link-time code generation.
pub trait GpuParticleModelData: NoUninit + Send + Sync {
    /// CPU-side material model type.
    type Model: Copy;
    /// Converts from CPU representation to GPU representation.
    fn from_model(model: Self::Model) -> Self;
    /// Returns Slang module paths for shader specialization.
    fn specialization_modules() -> Vec<String>;
}

impl GpuParticleModelData for GpuParticleModel {
    type Model = ParticleModel;

    fn specialization_modules() -> Vec<String> {
        // NOTE: we could have returned an empty `vec![]` here since the default specialization
        //       module is already set to that path. But we provide it here as an example.
        vec!["slosh::models::specializations".to_string()]
    }

    fn from_model(model: Self::Model) -> Self {
        model.into()
    }
}
