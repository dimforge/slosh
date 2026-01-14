use slosh_testbed2d::{RapierData, slosh};

use nalgebra::{Vector2, point, vector};
use rapier2d::prelude::{ColliderBuilder, RigidBodyBuilder};
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

pub fn beam_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let width = 10.0;
    let height = 2.0;
    let fixed_part = 1.0;
    let cell_width = 0.2;
    let particle_per_cell_dim = 2;
    let young_modulus = 1.0e8;
    let poisson_ratio = 0.3;

    let diameter = cell_width / particle_per_cell_dim as f32;
    let ni = ((width + fixed_part) / diameter).ceil() as usize;
    let nj = (height / diameter).ceil() as usize;

    let mut particles = vec![];
    for i in 0..ni {
        for j in 0..nj {
            let position =
                point![i as f32, j as f32] * diameter;
            let density = 1000.0;
            let radius = diameter / 2.0;
            let model = ParticleModel::elastic_neo_hookean(young_modulus, poisson_ratio);
            particles.push(Particle::new(position, radius, density, model));
        }
    }

    if !app_state.restarting {
        app_state.min_num_substeps = 150;
        app_state.max_num_substeps = 150;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81] * app_state.gravity_factor,
        dt: 1.0 / 60.0,
        padding: 0.0,
    };

    let rb = RigidBodyBuilder::fixed().translation(vector![0.0, height / 2.0]).build();
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(fixed_part, height);
    let ground = rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let data = MpmData::new(
        backend,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        &[(ground, GpuBoundaryCondition::stick())],
        cell_width,
        30_000,
    )
        .unwrap();
    PhysicsContext {
        data,
        rapier_data,
        callbacks: vec![],
        hooks_state: None
    }
}
