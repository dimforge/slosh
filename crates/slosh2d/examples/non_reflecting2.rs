//! Three ways of terminating a domain, side by side. Three identical elastic blocks get the same
//! radial pulse at their centre; what differs is what happens when the wave reaches the edge:
//!
//! - **left**: [`GpuBoundaryCondition::slip`] walls, a perfect reflector, the reference.
//! - **middle**: [`GpuBoundaryCondition::non_reflecting_for_material`] walls, Lysmer-Kuhlemeyer
//!   viscous dashpots graded over a band a few cells deep. Costs nothing but the boundary itself.
//! - **right**: a perfectly-matched-layer skirt of [`ParticleModel::absorbing_pml`] particles
//!   enclosed by fixed walls, after Kurima, Chandra & Soga (arXiv:2407.02790). Pays for it in
//!   particles: the skirt more than doubles the block's particle count.
//!
//! Residual motion in the interior settles at roughly 1% of the reflecting block for the dashpot
//! and 0.3% for the PML.
//!
//! Switch the testbed's render mode to **volume** to see the waves; they are well under 1% strain
//! and invisible in the default mode. Compression reads orange, dilation cyan, shear white.

use slosh_testbed2d::{RapierData, slosh};

use glam::{Vec2, Vec4, vec2};
use rapier2d::prelude::{
    ColliderBuilder, ColliderHandle, ColliderSet, RigidBodyBuilder, RigidBodySet,
};
use slang_hal::backend::WebGpu;
use slosh::{
    pipeline::MpmData,
    solver::{Particle, SimulationParams},
};
use slosh_testbed2d::{AppState, PhysicsContext};
use slosh2d::models::{DEFAULT_PML_MAX_STRETCH, pml_stretch};
use slosh2d::solver::{GpuBoundaryCondition, ParticleModel};

#[allow(dead_code)]
fn main() {
    panic!("Run the `testbed2` example instead.");
}

/// Size of each block's region of interest.
const BLOCK_WIDTH: f32 = 14.0;
const BLOCK_HEIGHT: f32 = 10.0;
/// Horizontal gap between blocks.
const BLOCK_GAP: f32 = 5.0;

/// Depth of the absorbing skirt around the third block: 15 cells, enough for this small domain
/// (the paper uses layers tens of cells deep).
const PML_THICKNESS: f32 = 3.0;

/// Both halves of the Rayleigh damping applied over the skirt, ramped with the stretch so they
/// stay continuous at the interface. The two are complementary in frequency and beat either alone
/// here; raising either from these values makes absorption worse, and 0.010 stiffness damping
/// would already drag the timestep bound below this demo's fixed timestep.
const PML_DAMPING: f32 = 40.0;
const PML_STIFFNESS_DAMPING: f32 = 0.008;

const CELL_WIDTH: f32 = 0.2;
const PARTICLES_PER_CELL_DIM: usize = 2;

const YOUNG_MODULUS: f32 = 8.0e5;
const POISSON_RATIO: f32 = 0.2;
const DENSITY: f32 = 1000.0;

/// Radius of the initial velocity pulse, and its peak velocity. The induced strain is roughly
/// `PULSE_VELOCITY / c_p`, well within the linear elastic regime.
const PULSE_RADIUS: f32 = 1.5;
const PULSE_VELOCITY: f32 = 0.25;

#[derive(Copy, Clone, PartialEq)]
enum Termination {
    Reflecting,
    Dashpot,
    Pml,
}

impl Termination {
    /// Depth of absorbing material outside the region of interest.
    fn skirt(self) -> f32 {
        match self {
            Termination::Pml => PML_THICKNESS,
            _ => 0.0,
        }
    }

    fn wall_condition(self) -> GpuBoundaryCondition {
        match self {
            Termination::Reflecting => GpuBoundaryCondition::slip(),
            Termination::Dashpot => GpuBoundaryCondition::non_reflecting_for_material(
                YOUNG_MODULUS,
                POISSON_RATIO,
                DENSITY,
                0.0,
            ),
            // The paper encloses the absorbing layer in fixed displacement boundaries.
            Termination::Pml => GpuBoundaryCondition::stick(),
        }
    }
}

pub fn non_reflecting_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let diameter = CELL_WIDTH / PARTICLES_PER_CELL_DIM as f32;
    let mut particles = vec![];
    let mut materials = vec![];

    let mut x = 0.0;
    for termination in [
        Termination::Reflecting,
        Termination::Dashpot,
        Termination::Pml,
    ] {
        let skirt = termination.skirt();
        x += skirt;

        let mins = vec2(x, skirt);
        let maxs = mins + vec2(BLOCK_WIDTH, BLOCK_HEIGHT);
        let center = (mins + maxs) / 2.0;

        let ni = ((BLOCK_WIDTH + 2.0 * skirt) / diameter).ceil() as usize;
        let nj = ((BLOCK_HEIGHT + 2.0 * skirt) / diameter).ceil() as usize;

        for i in 0..ni {
            for j in 0..nj {
                let position =
                    mins - Vec2::splat(skirt) + vec2(i as f32 + 0.5, j as f32 + 0.5) * diameter;

                // Outside the region of interest, particles become absorbing: stretched more the
                // deeper into the skirt, and carrying the damping that dissipates the wave.
                let stretch =
                    pml_stretch(position, mins, maxs, PML_THICKNESS, DEFAULT_PML_MAX_STRETCH);
                let absorbing = termination == Termination::Pml && stretch.length_squared() > 0.0;

                let model = if absorbing {
                    ParticleModel::absorbing_pml(YOUNG_MODULUS, POISSON_RATIO, stretch)
                } else {
                    ParticleModel::elastic(YOUNG_MODULUS, POISSON_RATIO)
                };

                let mut particle = Particle::new(position, diameter / 2.0, DENSITY, model);
                if absorbing {
                    let ramp = stretch.max_element() / DEFAULT_PML_MAX_STRETCH;
                    particle.dynamics.damping = PML_DAMPING * ramp;
                    particle.dynamics.stiffness_damping = PML_STIFFNESS_DAMPING * ramp;
                }

                // Radial pulse with a smooth (Hann) taper so it doesn't inject high frequencies
                // the grid can't resolve.
                let dpos = position - center;
                let dist = dpos.length();

                if dist < PULSE_RADIUS && dist > 1.0e-6 {
                    let taper = 0.5 * (1.0 + (std::f32::consts::PI * dist / PULSE_RADIUS).cos());
                    particle.dynamics.velocity = dpos / dist * PULSE_VELOCITY * taper;
                }

                particles.push(particle);
            }
        }

        let walls = build_walls(
            &mut rapier_data.bodies,
            &mut rapier_data.colliders,
            mins - Vec2::splat(skirt),
            maxs + Vec2::splat(skirt),
        );
        let condition = termination.wall_condition();
        materials.extend(walls.into_iter().map(|handle| (handle, condition)));

        x += BLOCK_WIDTH + skirt + BLOCK_GAP;
    }

    let total_width = x - BLOCK_GAP;

    if !app_state.restarting {
        // Fixed substepping: the CFL bound for this material is `0.5 * h / c_p` ≈ 3.3ms, so
        // 8 substeps per 1/60s frame leaves a comfortable margin.
        app_state.min_num_substeps = 8;
        app_state.max_num_substeps = 8;
        // A pure wave propagation test: neither absorbing termination has the static stiffness to
        // hold the blocks up against gravity.
        app_state.gravity_factor = 0.0;
        app_state.initial_camera2d_at = Some([total_width / 2.0, BLOCK_HEIGHT / 2.0]);
        app_state.initial_camera2d_zoom = Some(950.0 / total_width);
    }

    // Uniform base color so the blocks look alike in the default render mode (the `volume` mode
    // derives its colors from the strain and ignores this).
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

/// Encloses the box `[mins, maxs]` in four fixed walls whose inner faces are flush with it.
fn build_walls(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    mins: Vec2,
    maxs: Vec2,
) -> [ColliderHandle; 4] {
    const THICKNESS: f32 = 1.0;

    let center = (mins + maxs) / 2.0;
    let half = (maxs - mins) / 2.0;

    let walls = [
        // Bottom, top, left, right.
        (
            vec2(center.x, mins.y - THICKNESS),
            vec2(half.x + THICKNESS, THICKNESS),
        ),
        (
            vec2(center.x, maxs.y + THICKNESS),
            vec2(half.x + THICKNESS, THICKNESS),
        ),
        (
            vec2(mins.x - THICKNESS, center.y),
            vec2(THICKNESS, half.y + THICKNESS),
        ),
        (
            vec2(maxs.x + THICKNESS, center.y),
            vec2(THICKNESS, half.y + THICKNESS),
        ),
    ];

    walls.map(|(translation, half_extents)| {
        let rb = RigidBodyBuilder::fixed().translation(translation);
        let rb_handle = bodies.insert(rb);
        let co = ColliderBuilder::cuboid(half_extents.x, half_extents.y);
        colliders.insert_with_parent(co, rb_handle, bodies)
    })
}
