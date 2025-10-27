use crate::models::{DruckerPrager, DruckerPragerPlasticState, ElasticCoefficients};
use bytemuck::{NoUninit, Pod, Zeroable};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParticleModel {
    ElasticLinear(ElasticCoefficients),
    ElasticNeoHookean(ElasticCoefficients),
    SandLinear(SandModel),
    SandNeoHookean(SandModel),
}

impl Default for ParticleModel {
    fn default() -> Self {
        Self::elastic(Self::DEFAULT_YOUNG_MODULUS, Self::DEFAULT_POISSON_RATIO)
    }
}

impl ParticleModel {
    pub const DEFAULT_YOUNG_MODULUS: f32 = 1_000.0;
    pub const DEFAULT_POISSON_RATIO: f32 = 0.2;

    pub fn elastic(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::ElasticLinear(ElasticCoefficients::from_young_modulus(
            young_modulus,
            poisson_ratio,
        ))
    }

    pub fn sand(young_modulus: f32, poisson_ratio: f32) -> Self {
        ParticleModel::SandLinear(SandModel {
            plastic_state: DruckerPragerPlasticState::default(),
            plastic: DruckerPrager::new(young_modulus, poisson_ratio),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, NoUninit)]
#[repr(u32)]
/// This is the same as [`ParticleModel`] but with explicit padding for compatibility with `bytemuck::NoUninit`.
pub enum GpuParticleModel {
    ElasticLinear(ElasticCoefficients, [u32; 9]) = 0, // 3 floats + padding + 1 discriminant
    ElasticNeoHookean(ElasticCoefficients, [u32; 9]) = 1, // 3 floats + padding + 1 discriminant
    SandLinear(SandModel) = 2,                        // 12 floats + 1 discriminant
    SandNeoHookean(SandModel) = 3,                    // 12 floats + 1 discriminant
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

#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct SandModel {
    pub plastic_state: DruckerPragerPlasticState,
    pub plastic: DruckerPrager,
    pub elastic: ElasticCoefficients,
}

pub trait GpuParticleModelData: NoUninit + Send + Sync {
    type Model: Copy;
    fn from_model(model: Self::Model) -> Self;
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
