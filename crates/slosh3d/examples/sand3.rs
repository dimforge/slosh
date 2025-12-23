use slosh_testbed3d::{PhysicsState, RapierData, slosh};

use nalgebra::{point, vector};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder};
use slang_hal::backend::WebGpu;
use slosh::{
    pipeline::MpmData,
    solver::{Particle, ParticleModel, SimulationParams},
};
use slosh_testbed3d::{AppState, PhysicsContext};

#[allow(dead_code)]
fn main() {
    panic!("Run the `testbed3` example instead.");
}

// const YOUNG_MODULUS: f32 = 2.0e5;
// const DENSITY: f32 = 400.0;
const DENSITY: f32 = 2700.0;
const YOUNG_MODULUS: f32 = 2.0e9;
const POISSON_RATIO: f32 = 0.2;

pub fn sand_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let nxz = 45;
    let cell_width = 1.0;
    let mut particles = vec![];
    for i in 0..nxz {
        for j in 0..100 {
            for k in 0..nxz {
                let position = point![
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 10.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0
                ] * cell_width
                    / 2.0;
                let radius = cell_width / 4.0;
                let model = ParticleModel::sand(YOUNG_MODULUS, POISSON_RATIO);
                particles.push(Particle::new(position, radius, DENSITY, model));
            }
        }
    }

    // let nxz = 2; // 45;
    // let cell_width = 1.0;
    // let mut particles = vec![];
    // for i in 0..nxz {
    //     for j in 0..1 {
    //         for k in 0..nxz {
    //             let position = point![
    //                 i as f32 * 4.0 + 0.5 - nxz as f32 / 2.0,
    //                 j as f32 + 0.5 + 10.0,
    //                 k as f32 * 4.0 + 0.5 - nxz as f32 / 2.0
    //             ] * cell_width
    //                 / 2.0;
    //             let radius = cell_width / 4.0;
    //             particles.push(
    //                 ParticleBuilder::new(position, radius, DENSITY)
    //                     .sand(YOUNG_MODULUS, POISSON_RATIO)
    //                     .build()
    //             );
    //         }
    //     }
    // }

    if !app_state.restarting {
        // app_state.min_num_substeps = 10;
        // app_state.max_num_substeps = 40;
        app_state.min_num_substeps = 20;
        app_state.max_num_substeps = 20;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: 1.0 / 60.0,
    };

    let rb = RigidBodyBuilder::fixed().translation(vector![0.0, -4.0, 0.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(100.0, 4.0, 100.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::fixed().translation(vector![0.0, 5.0, -35.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(35.0, 5.0, 0.5);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    let rb = RigidBodyBuilder::fixed().translation(vector![0.0, 5.0, 35.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(35.0, 5.0, 0.5);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    let rb = RigidBodyBuilder::fixed().translation(vector![-35.0, 5.0, 0.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(0.5, 5.0, 35.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    let rb = RigidBodyBuilder::fixed().translation(vector![35.0, 5.0, 0.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(0.5, 5.0, 35.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    let rb = RigidBodyBuilder::kinematic_velocity_based()
        .translation(vector![0.0, 2.0, 0.0])
        .rotation(vector![0.0, 0.0, -0.5])
        .angvel(vector![0.0, -1.0, 0.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(0.5, 2.0, 30.0);
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

    let callback = move |phx: &mut PhysicsState| {
        if phx.step_id().is_multiple_of(50) {
            let mut particles = vec![];
            for i in 0..nxz {
                for k in 0..nxz {
                    let position = point![
                        i as f32 + 0.5 - nxz as f32 / 2.0,
                        110.0,
                        k as f32 + 0.5 - nxz as f32 / 2.0
                    ] * cell_width
                        / 2.0;
                    let radius = cell_width / 4.0;
                    let model = ParticleModel::sand(YOUNG_MODULUS, 0.2);
                    particles.push(Particle::new(position, radius, DENSITY, model));
                }
            }
            phx.add_particles(&particles);
        }
    };

    PhysicsContext {
        data,
        rapier_data,
        callbacks: vec![],
        hooks_state: None,
        // callbacks: vec![Box::new(callback)],
    }
}
