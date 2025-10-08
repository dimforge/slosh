use crate::solver::{Particle, ParticlePhase};
use bytemuck::{Pod, Zeroable};
pub use drucker_prager::{DruckerPrager, DruckerPragerPlasticState};
use slang_hal::backend::Backend;
use stensor::tensor::GpuTensor;
use wgpu::BufferUsages;

mod drucker_prager;

fn lame_lambda_mu(young_modulus: f32, poisson_ratio: f32) -> (f32, f32) {
    (
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
        shear_modulus(young_modulus, poisson_ratio),
    )
}

fn shear_modulus(young_modulus: f32, poisson_ratio: f32) -> f32 {
    young_modulus / (2.0 * (1.0 + poisson_ratio))
}

#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ElasticCoefficients {
    pub lambda: f32,
    pub mu: f32,
}

impl ElasticCoefficients {
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_lambda_mu(young_modulus, poisson_ratio);
        Self { lambda, mu }
    }
}
