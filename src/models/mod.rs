//! Material constitutive models for MPM particles.
//!
//! This module provides material models that define how particles respond to deformation:
//! - [`ElasticCoefficients`]: Linear elasticity using Lamé parameters
//! - [`DruckerPrager`]: Drucker-Prager plasticity model for granular materials (sand, soil)
//! - `PmlModel`: Absorbing (perfectly-matched-layer) material for far-field boundaries,
//!   behind the `pml` feature
//!
//! Material models are used by particles to compute stress from deformation gradients.

#[cfg(feature = "pml")]
use crate::math::Vector;
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

/// Absorbing (perfectly-matched-layer) material, after Kurima, Chandra & Soga
/// ([arXiv:2407.02790](https://arxiv.org/abs/2407.02790)).
///
/// Linear elasticity with per-axis stretched coordinates, which slows outgoing waves instead of
/// reflecting them. The stretch does not dissipate: pair it with damping over the same region.
#[cfg(feature = "pml")]
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct PmlModel {
    /// Elastic coefficients, which should match the material the layer is absorbing for.
    pub elastic: ElasticCoefficients,
    /// Per-axis coordinate stretching `C'_j`, zero outside the absorbing layer. Always three
    /// entries so the GPU layout is identical in 2D and 3D; the third is unused in 2D.
    pub stretch: [f32; 3],
}

/// Maximum stretch `α` at the outer edge of an absorbing layer. The paper's parameter study
/// settles on 4, with little further gain beyond ~3.2.
#[cfg(feature = "pml")]
pub const DEFAULT_PML_MAX_STRETCH: f32 = 4.0;

/// Computes the PML coordinate stretching for a point inside an absorbing layer.
///
/// The layer wraps the box `[interior_mins, interior_maxs]`, `thickness` deep on every side, and
/// the stretch ramps linearly from zero at its inner boundary to `max_stretch` at the outer one.
#[cfg(feature = "pml")]
pub fn pml_stretch(
    position: Vector,
    interior_mins: Vector,
    interior_maxs: Vector,
    thickness: f32,
    max_stretch: f32,
) -> Vector {
    let below = (interior_mins - position).max(Vector::ZERO);
    let above = (position - interior_maxs).max(Vector::ZERO);
    let depth = ((below + above) / thickness.max(1.0e-6)).min(Vector::ONE);
    depth * max_stretch
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
