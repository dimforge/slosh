use crate::models::{DruckerPrager, DruckerPragerPlasticState, ElasticCoefficients};
use bytemuck::{NoUninit, Pod, Zeroable};
use slang_hal::backend::WebGpu;


#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DefaultParticleModel {
    ElasticLinear(ElasticCoefficients),
    ElasticNeoHookean(ElasticCoefficients),
    SandLinear(SandModel),
    SandNeoHookean(SandModel),
}

#[derive(Copy, Clone, Debug, PartialEq, NoUninit)]
#[repr(u32)]
/// This is the same as [`ParticleModel`] but with explicit padding for compatibility with `bytemuck::NoUninit`.
pub enum DefaultGpuParticleModel {
    ElasticLinear(ElasticCoefficients, [u32; 9]) = 0,     // 3 floats + padding + 1 discriminant
    ElasticNeoHookean(ElasticCoefficients, [u32; 9]) = 1, // 3 floats + padding + 1 discriminant
    SandLinear(SandModel) = 2,                            // 12 floats + 1 discriminant
    SandNeoHookean(SandModel) = 3,                        // 12 floats + 1 discriminant
}

// IMPORTANT: this assertions is here to reduce risks of `GpuParticleModel` from mismatching
//            `SloshParticleModel` in
static_assertions::assert_eq_size!(DefaultGpuParticleModel, [u8; 52]);

impl Into<DefaultGpuParticleModel> for DefaultParticleModel {
    fn into(self) -> DefaultGpuParticleModel {
        match self {
            DefaultParticleModel::ElasticLinear(elastic_linear) => {
                DefaultGpuParticleModel::ElasticLinear(elastic_linear, [0; _])
            }
            DefaultParticleModel::ElasticNeoHookean(elastic_neo_hookean) => {
                DefaultGpuParticleModel::ElasticNeoHookean(elastic_neo_hookean, [0; _])
            }
            DefaultParticleModel::SandLinear(sand_linear) => DefaultGpuParticleModel::SandLinear(sand_linear),
            DefaultParticleModel::SandNeoHookean(sand_neo_hookean) => {
                DefaultGpuParticleModel::SandNeoHookean(sand_neo_hookean)
            }
        }
    }
}

impl Into<DefaultParticleModel> for DefaultGpuParticleModel {
    fn into(self) -> DefaultParticleModel {
        match self {
            DefaultGpuParticleModel::ElasticLinear(elastic_linear, _) => {
                DefaultParticleModel::ElasticLinear(elastic_linear)
            }
            DefaultGpuParticleModel::ElasticNeoHookean(elastic_neo_hookean, _) => {
                DefaultParticleModel::ElasticNeoHookean(elastic_neo_hookean)
            }
            DefaultGpuParticleModel::SandLinear(sand_linear) => DefaultParticleModel::SandLinear(sand_linear),
            DefaultGpuParticleModel::SandNeoHookean(sand_neo_hookean) => {
                DefaultParticleModel::SandNeoHookean(sand_neo_hookean)
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


pub trait ParticleModel: Copy + From<DefaultParticleModel> {}

impl ParticleModel for DefaultParticleModel {}

pub trait GpuParticleModel: NoUninit + Send + Sync {
    type Model: ParticleModel;
    fn from_model(model: Self::Model) -> Self;
    fn specialization_modules() -> Vec<String>;
}

impl GpuParticleModel for DefaultGpuParticleModel {
    type Model = DefaultParticleModel;

    fn specialization_modules() -> Vec<String> {
        // NOTE: we could have returned an empty `vec![]` here since the default specialization
        //       module is already set to that path. But we provide it here as an example.
        vec!["slosh::models::specializations".to_string()]
    }

    fn from_model(model: Self::Model) -> Self {
        model.into()
    }
}