use slosh_testbed3d::{RapierData, slosh};

use nalgebra::{point, vector};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder};
use slang_hal::backend::WebGpu;
use slosh::models::DruckerPrager;
use slosh::{
    models::ElasticCoefficients,
    pipeline::MpmData,
    solver::{Particle, ParticleDynamics, SimulationParams, ParticleBuilder},
};
use slosh_testbed3d::{AppState, PhysicsContext};

#[allow(dead_code)]
fn main() {
    panic!("Run the `testbed3` example instead.");
}

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
                let density = 2700.0;
                let radius = cell_width / 4.0;
                particles.push(
                    ParticleBuilder::new(position, radius, density)
                        .sand(2.0e9, 0.2)
                        .build()
                );
            }
        }
    }

    if !app_state.restarting {
        app_state.num_substeps = 20;
        app_state.gravity_factor = 1.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
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
    PhysicsContext {
        data,
        rapier_data,
        particles,
    }
}
