use slosh_testbed3d::{RapierData, slosh, PhysicsState};

use nalgebra::{Vector3, point, vector};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder};
use slang_hal::backend::WebGpu;
use slosh::{
    pipeline::MpmData,
    solver::{Particle, SimulationParams, GpuBoundaryCondition},
};
use slosh_testbed3d::{AppState, PhysicsContext};
use slosh3d::solver::ParticleModel;

#[allow(dead_code)]
fn main() {
    panic!("Run the `testbed3` example instead.");
}

pub fn beam_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let width = 10.0;
    let height = 2.0;
    let fixed_part = 1.0;
    let cell_width = 0.2;
    let particle_per_cell_dim = 2;
    let young_modulus = 1.0e7;
    let poisson_ratio = 0.3;

    let diameter = cell_width / particle_per_cell_dim as f32;
    let ni = ((width + fixed_part) / diameter).ceil() as usize;
    let njk = (height / diameter).ceil() as usize;

    let mut particles = vec![];
    for i in 0..ni {
        for j in 0..njk {
            for k in 0..njk {
                let position =
                    point![i as f32, j as f32, k as f32] * diameter;
                let density = 1000.0;
                let radius = diameter / 2.0;
                let model = ParticleModel::elastic_neo_hookean(young_modulus, poisson_ratio);
                let mut particle = Particle::new(position, radius, density, model);
                particle.dynamics.set_damping(2.0);
                particles.push(particle);
            }
        }
    }

    if !app_state.restarting {
        app_state.min_num_substeps = 20;
        app_state.max_num_substeps = 20;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: 1.0 / 60.0,
    };

    let rb = RigidBodyBuilder::fixed().translation(vector![0.0, height / 2.0, height / 2.0]).build();
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(fixed_part, height, height);
    let co_handle = rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    let co_boundary_condition = [(co_handle, GpuBoundaryCondition::stick())];


    let data = MpmData::new(
        backend,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        &co_boundary_condition,
        cell_width,
        60_000,
    )
        .unwrap();


    let mut all_time_max = 0.0;
    let callback = move |state: &mut PhysicsState| {
        let mut max_diff = 0.0;
        for (init, now) in particles.iter().zip(state.results.instances.iter()) {
            let diff = (init.position.y - now.position.y).abs();
            max_diff = diff.max(max_diff);
        }
        all_time_max = max_diff.max(all_time_max);
        println!("max diff: {} (all time: {})", max_diff, all_time_max);
    };

    PhysicsContext {
        data,
        rapier_data,
        callbacks: vec![Box::new(callback)],
        hooks_state: None
    }
}
