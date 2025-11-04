use slosh_testbed2d::{RapierData, slosh};

use nalgebra::{Vector2, point, vector};
use rapier2d::prelude::{ColliderBuilder, RigidBodyBuilder};
use slang_hal::backend::WebGpu;
use slosh::{
    pipeline::MpmData,
    solver::{Particle, SimulationParams},
};
use slosh_testbed2d::{AppState, PhysicsContext};
use slosh2d::solver::ParticleModel;

#[allow(dead_code)]
fn main() {
    panic!("Run the `testbed3` example instead.");
}

pub fn elastic_cut_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let offset_y = 46.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position =
                point![i as f32 + 0.5, j as f32 + 0.5] * cell_width / 2.0 + Vector2::y() * offset_y;
            let density = 1000.0;
            let radius = cell_width / 4.0;
            let model = ParticleModel::elastic(5.0e6, 0.2);
            particles.push(Particle::new(position, radius, density, model));
        }
    }

    if !app_state.restarting {
        app_state.max_num_substeps = 15;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81] * app_state.gravity_factor,
        dt: 1.0 / 60.0,
        padding: 0.0,
    };

    // const ANGVEL: f32 = 1.0; // 2.0;

    /*
     * Static platforms.
     */
    let rb = RigidBodyBuilder::fixed().translation(vector![35.0, 20.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(70.0, 1.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let mut polyline = vec![];
    let subdivs = 100;
    let length = 84.0;
    let start = point![35.0, 70.0] - vector![length / 2.0, 0.0];

    for i in 0..=subdivs {
        let step = length / (subdivs as f32);
        let dx = i as f32 * step;
        polyline.push(start + vector![dx, dx.sin()])
    }

    let rb = RigidBodyBuilder::fixed();
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::polyline(polyline, None).build();
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    for k in 0..6 {
        let rb = RigidBodyBuilder::fixed();
        let rb_handle = rapier_data.bodies.insert(rb);
        let co = ColliderBuilder::polyline(
            vec![
                point![0.0 + k as f32 * 15.0, 20.0],
                point![-10.0 + k as f32 * 15.0, 45.0],
            ],
            None,
        )
        .build();
        rapier_data
            .colliders
            .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    }

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
        callbacks: vec![],
    }
}
