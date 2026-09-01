#[cfg(feature = "pml")]
use crate::math::Vector;
#[cfg(feature = "pml")]
use crate::models::PmlModel;
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
    /// Absorbing (perfectly-matched-layer) material for far-field boundaries.
    #[cfg(feature = "pml")]
    AbsorbingPml(PmlModel),
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

    /// Creates an absorbing (perfectly-matched-layer) material for a far-field boundary.
    ///
    /// Particles carrying this model form a layer around the region of interest that lets outgoing
    /// waves leave instead of reflecting them. The elastic parameters should match the surrounded
    /// material; [`crate::models::pml_stretch`] computes `stretch` from the layer's geometry.
    ///
    /// The stretch redirects energy but does not dissipate it, so pair it with
    /// [`crate::solver::ParticleDynamics::damping`] over the same particles (the paper uses
    /// `α_M = 1`; what matters is `α_M·L/c`, the attenuation per transit of the layer). Prefer
    /// [`crate::solver::ParticleDynamics::stiffness_damping`] when the domain itself moves, since
    /// mass-proportional damping would resist that motion.
    ///
    /// # Gravity
    ///
    /// The layer's inertia is `s_j²` times the real mass while gravity acts on the real one, so it
    /// falls `s_j²` times slower along a stretched axis instead of carrying that much weight.
    ///
    /// The stretched operator's static stiffness still differs from the elastic one, so a layer
    /// under sustained load settles more than the material would (about twice, for a 4 m column on
    /// a 2 m layer). The paper removes this with a geo-static pre-step, not implemented here;
    /// keeping absorbing layers out of the load path avoids it.
    ///
    /// # Arguments
    ///
    /// * `young_modulus` - Young's modulus E (Pa) of the surrounded material
    /// * `poisson_ratio` - Poisson's ratio ν of the surrounded material
    /// * `stretch` - Per-axis coordinate stretching `C'_j` (zero means plain linear elasticity)
    #[cfg(feature = "pml")]
    pub fn absorbing_pml(young_modulus: f32, poisson_ratio: f32, stretch: Vector) -> Self {
        #[cfg(feature = "dim2")]
        let stretch = [stretch.x, stretch.y, 0.0];
        #[cfg(feature = "dim3")]
        let stretch = [stretch.x, stretch.y, stretch.z];

        ParticleModel::AbsorbingPml(PmlModel {
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
            stretch,
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
    /// Absorbing (perfectly-matched-layer) material with padding for GPU alignment.
    #[cfg(feature = "pml")]
    AbsorbingPml(PmlModel, [u32; 6]) = 4,
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
            #[cfg(feature = "pml")]
            ParticleModel::AbsorbingPml(pml) => GpuParticleModel::AbsorbingPml(pml, [0; _]),
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
            #[cfg(feature = "pml")]
            GpuParticleModel::AbsorbingPml(pml, _) => ParticleModel::AbsorbingPml(pml),
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
