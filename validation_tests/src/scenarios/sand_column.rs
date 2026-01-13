//! Collapsing Sand Column Validation Test
//!
//! A column of granular material (Drucker-Prager plasticity) collapses under gravity.
//! Compare runout distance and angle of repose with experimental data.
//!
//! Reference: Lajeunesse et al. (2005) "Granular slumping on a horizontal surface"
//! Key metrics:
//!   - Runout distance: R/R0 ~ (H0/R0)^alpha where alpha ~ 0.9 for tall columns
//!   - Angle of repose: typically 25-35 degrees for dry sand

use crate::harness::{
    create_ground_plane, MaterialParams, ScenarioConfig,
};
use nalgebra::{point, vector};
use rapier3d::prelude::{ColliderSet, RigidBodySet};
use slosh3d::solver::{ParticleModel, GpuBoundaryCondition};

/// Parameters for the sand column collapse test.
#[derive(Clone, Debug)]
pub struct SandColumnParams {
    /// Column radius (m)
    pub radius: f32,
    /// Column height (m)
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

impl Default for SandColumnParams {
    fn default() -> Self {
        Self {
            radius: 5.0,
            height: 20.0,
            young_modulus: 1.0e7,
            poisson_ratio: 0.2,
            density: 2700.0,
            cell_width: 0.5,
            gravity: 9.81,
        }
    }
}

impl SandColumnParams {
    /// Compute expected runout ratio based on aspect ratio.
    /// Based on Lajeunesse et al. (2005) empirical scaling.
    pub fn expected_runout_ratio(&self) -> f32 {
        let aspect_ratio = self.height / self.radius;
        if aspect_ratio > 3.0 {
            // Tall column regime: R/R0 ~ (H0/R0)^0.9
            aspect_ratio.powf(0.9)
        } else {
            // Short column regime: R/R0 ~ (H0/R0)^0.5
            aspect_ratio.powf(0.5)
        }
    }

    /// Expected angle of repose for dry sand (degrees).
    pub fn expected_angle_of_repose(&self) -> (f32, f32) {
        // Typical range for dry sand
        (25.0, 35.0)
    }
}

/// Create a sand column collapse scenario.
pub fn sand_column_scenario(params: SandColumnParams) -> ScenarioConfig {
    let model = ParticleModel::sand(params.young_modulus, params.poisson_ratio);

    // Create cylindrical column of particles
    let mut particles = vec![];
    let particle_spacing = params.cell_width / 2.0;
    let particle_radius = params.cell_width / 4.0;

    let nx = (params.radius * 2.0 / particle_spacing).ceil() as i32;
    let ny = (params.height / particle_spacing).ceil() as i32;
    let nz = (params.radius * 2.0 / particle_spacing).ceil() as i32;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = (i as f32 + 0.5) * particle_spacing - params.radius;
                let y = (j as f32 + 0.5) * particle_spacing;
                let z = (k as f32 + 0.5) * particle_spacing - params.radius;

                // Only include particles within cylinder
                if x * x + z * z <= params.radius * params.radius {
                    let position = point![x, y, z];
                    particles.push(slosh3d::solver::Particle::new(
                        position,
                        particle_radius,
                        params.density,
                        model,
                    ));
                }
            }
        }
    }

    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    // Ground plane
    let floor = create_ground_plane(&mut bodies, &mut colliders, 0.0);

    ScenarioConfig {
        name: "sand_column_collapse".to_string(),
        particles,
        bodies,
        colliders,
        materials: vec![(floor, GpuBoundaryCondition::stick())],
        gravity: vector![0.0, -params.gravity, 0.0],
        cell_width: params.cell_width,
        dt: 1.0 / 60.0,
        num_substeps: 20,
        total_steps: 600, // 10 seconds for settling
        snapshot_interval: 10,
        grid_capacity: 30_000,
        material_params: MaterialParams {
            young_modulus: params.young_modulus,
            poisson_ratio: params.poisson_ratio,
            density: params.density,
            material_type: "drucker_prager".to_string(),
        },
    }
}
