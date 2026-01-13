//! Dam Break Validation Test
//!
//! A column of fluid collapses horizontally after a virtual dam is removed.
//! Compare front position over time with experimental data.
//!
//! Reference: Martin & Moyce (1952) "Part IV. An experimental study of the
//! collapse of liquid columns on a rigid horizontal plane"
//!
//! Key metrics:
//!   - Front position vs time: dimensionless scaling x/a vs sqrt(2*g*t^2/a)
//!   - Height decay over time

use crate::harness::{
    create_ground_plane, create_particle_block, MaterialParams, ScenarioConfig,
};
use nalgebra::{point, vector};
use rapier3d::prelude::{ColliderBuilder, ColliderSet, RigidBodyBuilder, RigidBodySet};
use slosh3d::solver::{ParticleModel, GpuBoundaryCondition};

/// Parameters for the dam break test.
#[derive(Clone, Debug)]
pub struct DamBreakParams {
    /// Initial column width (m)
    pub width: f32,
    /// Initial column height (m)
    pub height: f32,
    /// Column depth (into page) (m)
    pub depth: f32,
    /// Young's modulus (Pa) - affects viscosity-like behavior
    pub young_modulus: f32,
    /// Poisson's ratio - near 0.5 for incompressible fluid
    pub poisson_ratio: f32,
    /// Material density (kg/m^3)
    pub density: f32,
    /// Grid cell width (m)
    pub cell_width: f32,
    /// Gravity magnitude (m/s^2)
    pub gravity: f32,
}

impl Default for DamBreakParams {
    fn default() -> Self {
        Self {
            width: 4.0,
            height: 8.0,
            depth: 4.0,
            young_modulus: 1.0e7,
            poisson_ratio: 0.2,
            density: 2700.0,
            cell_width: 0.15,
            gravity: 9.81,
        }
    }
}

impl DamBreakParams {
    /// Compute dimensionless time for Martin-Moyce comparison.
    /// tau = sqrt(2 * g / a) * t
    pub fn dimensionless_time(&self, t: f32) -> f32 {
        (2.0 * self.gravity / self.width).sqrt() * t
    }

    /// Expected front position (dimensionless) from Martin-Moyce data.
    /// For tau < 2: x/a ~ 1 + tau^2 / 4
    /// For tau > 2: x/a ~ tau (approximately linear)
    pub fn expected_front_position(&self, tau: f32) -> f32 {
        if tau < 2.0 {
            1.0 + tau * tau / 4.0
        } else {
            tau
        }
    }
}

/// Create a dam break scenario.
pub fn dam_break_scenario(params: DamBreakParams) -> ScenarioConfig {
    // Use elastic material with high Poisson ratio for fluid-like behavior
    let model = ParticleModel::sand(params.young_modulus, params.poisson_ratio);

    // Create fluid column particles at left side of domain
    let center = point![
        params.width / 2.0,
        params.height / 2.0,
        0.0
    ];
    let half_extents = vector![params.width / 2.0, params.height / 2.0, params.depth / 2.0];

    let particles =
        create_particle_block(center, half_extents, params.cell_width, params.density, model);

    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    // Ground plane
    let floor = create_ground_plane(&mut bodies, &mut colliders, 0.0);

    // Left wall (keeps fluid in place initially - can be removed in Taichi version)
    let left_rb = RigidBodyBuilder::fixed().translation(vector![-0.5, params.height / 2.0, 0.0]);
    let left_handle = bodies.insert(left_rb);
    let left_co = ColliderBuilder::cuboid(0.5, params.height, params.depth);
    let left = colliders.insert_with_parent(left_co, left_handle, &mut bodies);

    // Back and front walls
    let back_rb = RigidBodyBuilder::fixed().translation(vector![
        params.width * 2.0,
        params.height / 2.0,
        -params.depth / 2.0 - 0.5
    ]);
    let back_handle = bodies.insert(back_rb);
    let back_co = ColliderBuilder::cuboid(params.width * 4.0, params.height, 0.5);
    let back = colliders.insert_with_parent(back_co, back_handle, &mut bodies);

    let front_rb = RigidBodyBuilder::fixed().translation(vector![
        params.width * 2.0,
        params.height / 2.0,
        params.depth / 2.0 + 0.5
    ]);
    let front_handle = bodies.insert(front_rb);
    let front_co = ColliderBuilder::cuboid(params.width * 4.0, params.height, 0.5);
    let front = colliders.insert_with_parent(front_co, front_handle, &mut bodies);

    let materials = vec![
        (back, GpuBoundaryCondition::slip()),
        (front, GpuBoundaryCondition::slip()),
        (floor, GpuBoundaryCondition::default()),
        (left, GpuBoundaryCondition::slip()),
    ];

    ScenarioConfig {
        name: "dam_break".to_string(),
        particles,
        bodies,
        colliders,
        materials,
        gravity: vector![0.0, -params.gravity, 0.0],
        cell_width: params.cell_width,
        dt: 1.0 / 60.0,
        num_substeps: 20,
        total_steps: 600, // 5 seconds
        snapshot_interval: 5,
        grid_capacity: 30_000,
        material_params: MaterialParams {
            young_modulus: params.young_modulus,
            poisson_ratio: params.poisson_ratio,
            density: params.density,
            material_type: "neo_hookean".to_string(),
        },
    }
}
