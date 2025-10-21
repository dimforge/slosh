use bytemuck::{AnyBitPattern, NoUninit, Pod, Zeroable};
use crate::models::{DruckerPrager, DruckerPragerPlasticState, ElasticCoefficients};
use crate::solver::GpuParticles;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParticleModel {
    ElasticLinear(ElasticCoefficients),
    ElasticNeoHookean(ElasticCoefficients),
    SandLinear(SandModel),
    SandNeoHookean(SandModel),
}

#[derive(Copy, Clone, Debug, PartialEq, NoUninit)]
#[repr(u32)]
/// This is the same as [`ParticleModel`] but with explicit padding for compatibility with `bytemuck::NoUninit`.
pub enum GpuParticleModel {
    ElasticLinear(ElasticCoefficients, [u32; 9]) = 0, // 3 floats + padding + 1 discriminant
    ElasticNeoHookean(ElasticCoefficients, [u32; 9]) = 1, // 3 floats + padding + 1 discriminant
    SandLinear(SandModel) = 2, // 12 floats + 1 discriminant
    SandNeoHookean(SandModel) = 3, // 12 floats + 1 discriminant
}

// IMPORTANT: this assertions is here to reduce risks of `GpuParticleModel` from mismatching
//            `SloshParticleModel` in
static_assertions::assert_eq_size!(GpuParticleModel, [u8; 52]);

impl ParticleModel {
    const ELASTIC_LINEAR: u32 = 0;
    const ELASTIC_NEO_HOOKEAN: u32 = 1;
    const SAND_LINEAR: u32 = 2;
    const SAND_NEO_HOOKEAN: u32 = 3;
}

impl Into<GpuParticleModel> for ParticleModel {
    fn into(self) -> GpuParticleModel {
        match self {
            ParticleModel::ElasticLinear(elastic_linear) => GpuParticleModel::ElasticLinear(elastic_linear, [0; _]),
            ParticleModel::ElasticNeoHookean(elastic_neo_hookean) => GpuParticleModel::ElasticNeoHookean(elastic_neo_hookean, [0; _]),
            ParticleModel::SandLinear(sand_linear) => GpuParticleModel::SandLinear(sand_linear),
            ParticleModel::SandNeoHookean(sand_neo_hookean) => GpuParticleModel::SandNeoHookean(sand_neo_hookean),
        }
    }
}

impl Into<ParticleModel> for GpuParticleModel {
    fn into(self) -> ParticleModel {
        match self {
            GpuParticleModel::ElasticLinear(elastic_linear, _) => ParticleModel::ElasticLinear(elastic_linear),
            GpuParticleModel::ElasticNeoHookean(elastic_neo_hookean, _) => ParticleModel::ElasticNeoHookean(elastic_neo_hookean),
            GpuParticleModel::SandLinear(sand_linear) => ParticleModel::SandLinear(sand_linear),
            GpuParticleModel::SandNeoHookean(sand_neo_hookean) => ParticleModel::SandNeoHookean(sand_neo_hookean),
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
