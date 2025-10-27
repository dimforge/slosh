
//! Material constitutive models for MPM particles.
//!
//! This module provides material models that define how particles respond to deformation:
//! - [`ElasticCoefficients`]: Linear elasticity using Lamé parameters
//! - [`DruckerPrager`]: Drucker-Prager plasticity model for granular materials (sand, soil)
//!
//! Material models are used by particles to compute stress from deformation gradients.

use bytemuck::{Pod, Zeroable};
pub use drucker_prager::{DruckerPrager, DruckerPragerPlasticState};

mod drucker_prager;

/// Computes Lamé parameters (λ, μ) from Young's modulus and Poisson's ratio.
///
/// Lamé parameters are used in linear elasticity for computing stress.
fn lame_lambda_mu(young_modulus: f32, poisson_ratio: f32) -> (f32, f32) {
    (
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
        shear_modulus(young_modulus, poisson_ratio),
    )
}

/// Computes shear modulus μ (also called G) from Young's modulus and Poisson's ratio.
fn shear_modulus(young_modulus: f32, poisson_ratio: f32) -> f32 {
    young_modulus / (2.0 * (1.0 + poisson_ratio))
}

/// Lamé parameters for linear elastic materials.
///
/// These coefficients define the relationship between stress and strain in
/// an isotropic linear elastic material. They're computed from engineering
/// parameters (Young's modulus E and Poisson's ratio ν).
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct ElasticCoefficients {
    /// Lamé's first parameter λ (bulk response).
    pub lambda: f32,
    /// Lamé's second parameter μ (shear modulus, also called G).
    pub mu: f32,
    /// CFL coefficient for timestep stability (default 0.5).
    pub cfl_coeff: f32,
}

impl ElasticCoefficients {
    /// Creates elastic coefficients from engineering parameters.
    ///
    /// # Arguments
    ///
    /// * `young_modulus` - Young's modulus E (Pa) - material stiffness
    /// * `poisson_ratio` - Poisson's ratio ν (0.0 - 0.5) - lateral to axial strain ratio
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_lambda_mu(young_modulus, poisson_ratio);
        Self {
            lambda,
            mu,
            cfl_coeff: 0.5,
        }
    }
}
