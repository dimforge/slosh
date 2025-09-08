use slosh_testbed2d::{slosh, RapierData};

use nalgebra::{vector, Vector2};
use rapier2d::prelude::{ColliderBuilder, RigidBodyBuilder};
use slang_hal::backend::WebGpu;
use slosh::solver::ParticlePhase;
use slosh::{
    models::ElasticCoefficients,
    pipeline::MpmData,
    solver::{Particle, SimulationParams},
};
use slosh2d::solver::ParticleDynamics;
use slosh_testbed2d::{AppState, PhysicsContext};

#[allow(dead_code)]
fn main() {
    panic!("Run the `testbed3` example instead.");
}

pub fn elasticity_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let offset_y = 10.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position =
                vector![i as f32 + 0.5 + (i / 50) as f32 * 2.0, j as f32 + 0.5] * cell_width / 2.0
                    + Vector2::y() * offset_y;
            let density = 1000.0;
            let radius = cell_width / 4.0;
            particles.push(Particle {
                position,
                dynamics: ParticleDynamics::with_density(radius, density),
                model: ElasticCoefficients::from_young_modulus(5_000_000.0, 0.2),
                plasticity: None,
                phase: Some(ParticlePhase {
                    phase: 1.0,
                    max_stretch: f32::MAX,
                }),
            });
        }
    }

    if !app_state.restarting {
        app_state.num_substeps = 15;
        app_state.gravity_factor = 2.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
        padding: 0.0,
    };

    let rb = RigidBodyBuilder::fixed().translation(vector![0.0, -1.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1000.0, 1.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed()
        .translation(vector![-20.0, 0.0])
        .rotation(0.5);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 60.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed()
        .translation(vector![90.0, 0.0])
        .rotation(-0.5);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(1.0, 60.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let data = MpmData::new(
        backend,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        cell_width,
        60_000,
    )
    .unwrap();
    PhysicsContext {
        data,
        rapier_data,
        particles,
    }
}
