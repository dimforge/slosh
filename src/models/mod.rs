use crate::solver::{Particle, ParticlePhase};
use bytemuck::{Pod, Zeroable};
pub use drucker_prager::{DruckerPrager, DruckerPragerPlasticState};
use slang_hal::backend::Backend;
use stensor::tensor::GpuTensor;
use wgpu::BufferUsages;

mod drucker_prager;

pub struct GpuModels<B: Backend> {
    pub linear_elasticity: GpuTensor<ElasticCoefficients, B>,
    pub drucker_prager_plasticity: GpuTensor<DruckerPrager, B>,
    pub drucker_prager_plastic_state: GpuTensor<DruckerPragerPlasticState, B>,
    pub phases: GpuTensor<ParticlePhase, B>,
}

impl<B: Backend> GpuModels<B> {
    pub fn from_particles(backend: &B, particles: &[Particle]) -> Result<Self, B::Error> {
        let models: Vec<_> = particles.iter().map(|p| p.model).collect();
        let plasticity: Vec<_> = particles
            .iter()
            .map(|p| p.plasticity.unwrap_or(DruckerPrager::new(-1.0, -1.0)))
            .collect();
        let plastic_states: Vec<_> = particles
            .iter()
            .map(|_| DruckerPragerPlasticState::default())
            .collect();
        let phases: Vec<_> = particles.iter().map(|p| p.phase).collect();
        Ok(Self {
            linear_elasticity: GpuTensor::vector(backend, &models, BufferUsages::STORAGE)?,
            drucker_prager_plasticity: GpuTensor::vector(
                backend,
                &plasticity,
                BufferUsages::STORAGE,
            )?,
            drucker_prager_plastic_state: GpuTensor::vector(
                backend,
                &plastic_states,
                BufferUsages::STORAGE,
            )?,
            phases: GpuTensor::vector(backend, &phases, BufferUsages::STORAGE)?,
        })
    }
}

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
