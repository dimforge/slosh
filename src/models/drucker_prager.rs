use crate::models::lame_lambda_mu;
use bytemuck::{Pod, Zeroable};

/// Drucker-Prager plasticity model for granular materials.
///
/// Models pressure-dependent plastic yielding typical of sand, soil, and granular
/// materials. The yield surface is a cone in principal stress space with hardening.
///
/// Parameters h0-h3 control friction angle, cohesion, and hardening behavior.
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DruckerPrager {
    /// Friction angle (radians) - controls yield surface slope.
    pub h0: f32,
    /// Dilation angle (radians) - controls volume change during plastic flow.
    pub h1: f32,
    /// Cohesion coefficient.
    pub h2: f32,
    /// Hardening parameter (radians).
    pub h3: f32,

    /// Elastic Lamé parameter λ.
    pub lambda: f32,
    /// Elastic shear modulus μ.
    pub mu: f32,
}

impl DruckerPrager {
    /// Creates a Drucker-Prager model with default sand parameters.
    ///
    /// # Arguments
    ///
    /// * `young_modulus` - Elastic Young's modulus (Pa)
    /// * `poisson_ratio` - Elastic Poisson's ratio (0.0 - 0.5)
    pub fn new(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = if young_modulus > 0.0 {
            lame_lambda_mu(young_modulus, poisson_ratio)
        } else {
            (-1.0, -1.0)
        };

        Self::from_lame(lambda, mu)
    }

    /// Creates a Drucker-Prager model from Lamé parameters with default plasticity settings.
    pub fn from_lame(lambda: f32, mu: f32) -> Self {
        Self {
            h0: 35.0f32.to_radians(),
            h1: 9.0f32.to_radians(),
            h2: 0.2,
            h3: 10.0f32.to_radians(),
            lambda,
            mu,
        }
    }

    /// Updates the elastic coefficients while preserving plasticity parameters.
    pub fn set_elastic_coefficients(&mut self, young_modulus: f32, poisson_ratio: f32) {
        let (lambda, mu) = if young_modulus > 0.0 {
            lame_lambda_mu(young_modulus, poisson_ratio)
        } else {
            (-1.0, -1.0)
        };

        self.lambda = lambda;
        self.mu = mu;
    }
}

/// Plastic deformation state for Drucker-Prager model.
///
/// Tracks the accumulated plastic deformation and hardening for each particle.
/// This state evolves when the material yields plastically.
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DruckerPragerPlasticState {
    /// Determinant of plastic deformation gradient (tracks volume change).
    plastic_deformation_gradient_det: f32,
    /// Hardening variable (increases with plastic deformation).
    plastic_hardening: f32,
    /// Logarithmic volumetric gain from plastic flow.
    log_vol_gain: f32,
}

impl Default for DruckerPragerPlasticState {
    fn default() -> Self {
        Self {
            plastic_deformation_gradient_det: 1.0,
            plastic_hardening: 1.0,
            log_vol_gain: 0.0,
        }
    }
}
