//! Side-by-side comparison of a reflecting and a non-reflecting (absorbing) boundary. Two
//! identical elastic blocks get the same radial pulse at their centre; the left block has
//! [`GpuBoundaryCondition::slip`] walls, the right one Lysmer-Kuhlemeyer viscous dashpots
//! ([`GpuBoundaryCondition::non_reflecting_for_material`]) tuned to the material's wave speeds.
//!
//! A dashpot is only exact at normal incidence, so oblique arrivals and corners still send a
//! little energy back: the right block settles at roughly 1% of the left block’s residual motion.
//!
//! Switch the testbed’s render mode to **volume** to see the waves; they are well under 1% strain
//! and invisible in the default mode. Compression reads orange, dilation cyan, shear white.

use slosh_testbed2d::{RapierData, slosh};

use glam::{Vec4, vec2};
use rapier2d::prelude::{
    ColliderBuilder, ColliderHandle, ColliderSet, RigidBodyBuilder, RigidBodySet,
};
use slang_hal::backend::WebGpu;
use slosh::{
    pipeline::MpmData,
    solver::{Particle, SimulationParams},
};
use slosh_testbed2d::{AppState, PhysicsContext};
use slosh2d::solver::{GpuBoundaryCondition, ParticleModel};

#[allow(dead_code)]
fn main() {
    panic!("Run the `testbed2` example instead.");
}

/// Size of each elastic block.
const BLOCK_WIDTH: f32 = 14.0;
const BLOCK_HEIGHT: f32 = 10.0;
/// Horizontal gap between the two blocks.
const BLOCK_GAP: f32 = 5.0;

const CELL_WIDTH: f32 = 0.2;
const PARTICLES_PER_CELL_DIM: usize = 2;

const YOUNG_MODULUS: f32 = 8.0e5;
const POISSON_RATIO: f32 = 0.2;
const DENSITY: f32 = 1000.0;

/// Radius of the initial velocity pulse, and its peak velocity. The induced strain is roughly
/// `PULSE_VELOCITY / c_p` (~0.8% here), well within the linear elastic regime.
const PULSE_RADIUS: f32 = 1.5;
const PULSE_VELOCITY: f32 = 0.25;

pub fn non_reflecting_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let diameter = CELL_WIDTH / PARTICLES_PER_CELL_DIM as f32;
    let ni = (BLOCK_WIDTH / diameter).ceil() as usize;
    let nj = (BLOCK_HEIGHT / diameter).ceil() as usize;

    // The left block reflects, the right block absorbs. Everything else is identical.
    let reflecting_origin = vec2(0.0, 0.0);
    let absorbing_origin = vec2(BLOCK_WIDTH + BLOCK_GAP, 0.0);

    let mut particles = vec![];
    for origin in [reflecting_origin, absorbing_origin] {
        let center = origin + vec2(BLOCK_WIDTH, BLOCK_HEIGHT) / 2.0;

        for i in 0..ni {
            for j in 0..nj {
                let position = origin + vec2(i as f32 + 0.5, j as f32 + 0.5) * diameter;
                let model = ParticleModel::elastic(YOUNG_MODULUS, POISSON_RATIO);
                let mut particle = Particle::new(position, diameter / 2.0, DENSITY, model);

                // Radial pulse with a smooth (Hann) taper so it doesn’t inject high
                // frequencies the grid can’t resolve.
                let dpos = position - center;
                let dist = dpos.length();

                if dist < PULSE_RADIUS && dist > 1.0e-6 {
                    let taper = 0.5 * (1.0 + (std::f32::consts::PI * dist / PULSE_RADIUS).cos());
                    particle.dynamics.velocity = dpos / dist * PULSE_VELOCITY * taper;
                }

                particles.push(particle);
            }
        }
    }

    let reflecting_walls = build_walls(
        &mut rapier_data.bodies,
        &mut rapier_data.colliders,
        reflecting_origin,
    );
    let absorbing_walls = build_walls(
        &mut rapier_data.bodies,
        &mut rapier_data.colliders,
        absorbing_origin,
    );

    let absorbing =
        GpuBoundaryCondition::non_reflecting_for_material(YOUNG_MODULUS, POISSON_RATIO, DENSITY);
    let materials: Vec<_> = reflecting_walls
        .into_iter()
        .map(|handle| (handle, GpuBoundaryCondition::slip()))
        .chain(
            absorbing_walls
                .into_iter()
                .map(|handle| (handle, absorbing)),
        )
        .collect();

    if !app_state.restarting {
        // Fixed substepping: the CFL bound for this material is `0.5 * h / c_p` ≈ 3.3ms, so
        // 8 substeps per 1/60s frame leaves a comfortable margin.
        app_state.min_num_substeps = 8;
        app_state.max_num_substeps = 8;
        // A pure wave propagation test: a dashpot has no static stiffness to hold the blocks
        // up against gravity.
        app_state.gravity_factor = 0.0;
        app_state.initial_camera2d_at = Some([BLOCK_WIDTH + BLOCK_GAP / 2.0, BLOCK_HEIGHT / 2.0]);
        app_state.initial_camera2d_zoom = Some(20.0);
    }

    // Uniform base color so the two blocks look alike in the default render mode (the `volume`
    // mode derives its colors from the strain and ignores this).
    app_state.particle_colors = Some(vec![Vec4::new(0.2, 0.25, 0.3, 1.0); particles.len()]);

    let params = SimulationParams {
        gravity: vec2(0.0, -9.81) * app_state.gravity_factor,
        dt: 1.0 / 60.0,
        padding: 0.0,
    };

    let data = MpmData::new(
        backend,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        &materials,
        CELL_WIDTH,
        30_000,
    )
    .unwrap();

    PhysicsContext {
        data,
        rapier_data,
        callbacks: vec![],
        hooks_state: None,
    }
}

/// Encloses the block at `origin` in four fixed walls whose inner faces are flush with its bounds.
fn build_walls(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    origin: glam::Vec2,
) -> [ColliderHandle; 4] {
    const THICKNESS: f32 = 1.0;

    let center = origin + vec2(BLOCK_WIDTH, BLOCK_HEIGHT) / 2.0;
    let half_width = BLOCK_WIDTH / 2.0;
    let half_height = BLOCK_HEIGHT / 2.0;

    let walls = [
        // Bottom, top, left, right.
        (
            vec2(center.x, origin.y - THICKNESS),
            vec2(half_width + THICKNESS, THICKNESS),
        ),
        (
            vec2(center.x, origin.y + BLOCK_HEIGHT + THICKNESS),
            vec2(half_width + THICKNESS, THICKNESS),
        ),
        (
            vec2(origin.x - THICKNESS, center.y),
            vec2(THICKNESS, half_height + THICKNESS),
        ),
        (
            vec2(origin.x + BLOCK_WIDTH + THICKNESS, center.y),
            vec2(THICKNESS, half_height + THICKNESS),
        ),
    ];

    walls.map(|(translation, half_extents)| {
        let rb = RigidBodyBuilder::fixed().translation(translation);
        let rb_handle = bodies.insert(rb);
        let co = ColliderBuilder::cuboid(half_extents.x, half_extents.y);
        colliders.insert_with_parent(co, rb_handle, bodies)
    })
}
