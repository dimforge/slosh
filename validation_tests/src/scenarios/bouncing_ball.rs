//! Bouncing Elastic Ball Validation Test
//!
//! An elastic sphere drops onto a rigid surface and bounces.
//! Compare bounce height with theoretical coefficient of restitution.
//!
//! Key metrics:
//!   - Coefficient of restitution: e = sqrt(h_rebound / h_initial)
//!   - Energy conservation during bounce
//!   - Contact duration

use crate::harness::{
    create_ground_plane, create_particle_sphere, MaterialParams, ScenarioConfig,
};
use nalgebra::{point, vector};
use rapier3d::prelude::{ColliderSet, RigidBodySet};
use slosh3d::solver::{ParticleModel, GpuBoundaryCondition};

/// Parameters for the bouncing ball test.
#[derive(Clone, Debug)]
pub struct BouncingBallParams {
    /// Ball radius (m)
    pub radius: f32,
    /// Initial drop height from ball center (m)
    pub drop_height: f32,
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

impl Default for BouncingBallParams {
    fn default() -> Self {
        Self {
            radius: 2.0,
            drop_height: 10.0,
            young_modulus: 1.0e6,
            poisson_ratio: 0.3,
            density: 1000.0,
            cell_width: 0.2, // 25,
            gravity: 9.81,
        }
    }
}

impl BouncingBallParams {
    /// Compute expected impact velocity.
    pub fn impact_velocity(&self) -> f32 {
        // v = sqrt(2 * g * h) where h is distance from ball bottom to ground
        let h = self.drop_height - self.radius;
        (2.0 * self.gravity * h).sqrt()
    }

    /// Compute theoretical kinetic energy at impact.
    pub fn impact_kinetic_energy(&self) -> f32 {
        let v = self.impact_velocity();
        let volume = 4.0 / 3.0 * std::f32::consts::PI * self.radius.powi(3);
        let mass = volume * self.density;
        0.5 * mass * v * v
    }

    /// Compute expected bounce height for a given coefficient of restitution.
    pub fn expected_bounce_height(&self, coefficient_of_restitution: f32) -> f32 {
        let h = self.drop_height - self.radius;
        self.radius + h * coefficient_of_restitution.powi(2)
    }

    fn expected_displacement(&self) -> f32 {
        let coeff = 15.0 / 8.0;
        let volume = 4.0 / 3.0 * std::f32::consts::PI * self.radius.powi(3);
        let mass = volume * self.density;

        let numerator = mass * self.gravity * self.drop_height * (1.0 - self.poisson_ratio * self.poisson_ratio);
        let denom = self.young_modulus * self.radius.sqrt();
        (coeff * numerator / denom).powf(2.0 / 5.0)
    }
}

/// Create a bouncing ball scenario.
pub fn bouncing_ball_scenario(params: BouncingBallParams) -> ScenarioConfig {
    let model = ParticleModel::elastic_neo_hookean(params.young_modulus, params.poisson_ratio);

    // Create sphere particles
    let center = point![0.0, params.drop_height, 0.0];
    let particles = create_particle_sphere(
        center,
        params.radius,
        params.cell_width,
        params.density,
        model,
    );

    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    // Ground plane
    let ground_handle = create_ground_plane(&mut bodies, &mut colliders, 0.0);
    let disp = params.expected_displacement();
    println!("EXPECTED: {}", disp);

    ScenarioConfig {
        name: "bouncing_ball".to_string(),
        particles,
        bodies,
        colliders,
        materials: vec![(ground_handle, GpuBoundaryCondition::separate(0.0))],
        gravity: vector![0.0, -params.gravity, 0.0],
        cell_width: params.cell_width,
        dt: 1.0 / 60.0,
        num_substeps: 20,
        total_steps: 300, // 5 seconds for multiple bounces
        snapshot_interval: 2,
        grid_capacity: 30_000,
        material_params: MaterialParams {
            young_modulus: params.young_modulus,
            poisson_ratio: params.poisson_ratio,
            density: params.density,
            material_type: "neo_hookean".to_string(),
        },
    }
}
