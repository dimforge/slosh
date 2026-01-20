//! Elastic Beam Cantilever Validation Test
//!
//! A horizontal beam fixed at one end deflects under gravity.
//! Compare tip deflection with analytical Euler-Bernoulli beam theory.
//!
//! Analytical solution for cantilever beam tip deflection:
//!   delta = (rho * g * L^4) / (8 * E * I)
//! where:
//!   rho = density
//!   g = gravity
//!   L = beam length
//!   E = Young's modulus
//!   I = second moment of area (for rectangular cross-section: I = b*h^3/12)

use crate::harness::{create_ground_plane, create_particle_block, MaterialParams, ScenarioConfig};
use nalgebra::{point, vector};
use rapier3d::prelude::{ColliderBuilder, ColliderSet, RigidBodyBuilder, RigidBodySet};
use slosh3d::solver::{GpuBoundaryCondition, ParticleModel};

/// Parameters for the elastic beam test.
#[derive(Clone, Debug)]
pub struct ElasticBeamParams {
    /// Beam length (m)
    pub length: f32,
    /// Beam width (m)
    pub width: f32,
    /// Beam height (m)
    pub height: f32,
    /// Young's modulus (Pa)
    pub young_modulus: f32,
    /// Poisson's ratio
    pub poisson_ratio: f32,
    /// Material density (kg/m^3)
    pub density: f32,
    /// Grid cell width (m)
    pub cell_width: f32,
    /// Gravity magnitude (m/s^2)
    pub gravity: f32,
}

impl Default for ElasticBeamParams {
    fn default() -> Self {
        Self {
            length: 10.0,
            width: 2.0,
            height: 2.0,
            young_modulus: 1.0e8,
            poisson_ratio: 0.3,
            density: 1000.0,
            cell_width: 0.5,
            gravity: 9.81,
        }
    }
}

impl ElasticBeamParams {
    /// Compute analytical tip deflection using Euler-Bernoulli beam theory.
    pub fn analytical_deflection(&self) -> f32 {
        let i = self.width * self.height.powi(3) / 12.0; // Second moment of area
        let q = self.density * self.gravity * self.width * self.height; // Load per unit length
                                                                        // delta = q * L^4 / (8 * E * I)
        q * self.length.powi(4) / (8.0 * self.young_modulus * i)
    }
}

/// Create an elastic beam cantilever scenario.
pub fn elastic_beam_scenario(params: ElasticBeamParams) -> ScenarioConfig {
    let model = ParticleModel::elastic(params.young_modulus, params.poisson_ratio);

    // Create beam particles - beam extends from x=0 to x=length, centered at y=height/2
    let center = point![
        params.length / 2.0,
        params.height / 2.0 + 5.0, // Elevated to leave room for deflection
        0.0
    ];
    let half_extents = vector![params.length / 2.0, params.height / 2.0, params.width / 2.0];

    let particles = create_particle_block(
        center,
        half_extents,
        params.cell_width,
        params.density,
        model,
    );

    // Create fixed constraint at x=0 (left end of beam)
    // In MPM, we'll use a fixed rigid body wall to clamp the beam
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    // Clamp wall at x=0
    let clamp_rb = RigidBodyBuilder::fixed().translation(vector![-0.5, center.y, 0.0]);
    let clamp_handle = bodies.insert(clamp_rb);
    let clamp_co = ColliderBuilder::cuboid(0.5, params.height, params.width);
    let wall = colliders.insert_with_parent(clamp_co, clamp_handle, &mut bodies);

    // Ground plane (below the beam)
    create_ground_plane(&mut bodies, &mut colliders, 0.0);

    ScenarioConfig {
        name: "elastic_beam_cantilever".to_string(),
        particles,
        bodies,
        colliders,
        materials: vec![(wall, GpuBoundaryCondition::stick())],
        gravity: vector![0.0, -params.gravity, 0.0],
        cell_width: params.cell_width,
        dt: 1.0 / 60.0,
        num_substeps: 20,
        total_steps: 600, // 10 seconds to reach equilibrium
        snapshot_interval: 10,
        grid_capacity: 30_000,
        material_params: MaterialParams {
            young_modulus: params.young_modulus,
            poisson_ratio: params.poisson_ratio,
            density: params.density,
            material_type: "neo_hookean".to_string(),
        },
    }
}
