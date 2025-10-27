use slosh_testbed3d::{RapierData, slosh};

use nalgebra::{DMatrix, Isometry3, point, vector};
use rapier3d::geometry::HeightField;
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

pub fn elastic_cut_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let nxz = 50;
    let cell_width = 1.0;
    let mut particles = vec![];
    for i in 0..nxz {
        for j in 0..30 {
            for k in 0..nxz {
                let position = point![
                    i as f32 + 0.5 - nxz as f32 / 2.0,
                    j as f32 + 0.5 + 60.0,
                    k as f32 + 0.5 - nxz as f32 / 2.0
                ] * cell_width
                    / 2.0;
                let density = 2700.0;
                let radius = cell_width / 4.0;
                let model = ParticleModel::elastic(1.0e7, 0.2);
                particles.push(Particle::new(position, radius, density, model));
            }
        }
    }

    if !app_state.restarting {
        app_state.num_substeps = 20;
        app_state.gravity_factor = 4.0;
    };

    let params = SimulationParams {
        gravity: vector![0.0, -9.81, 0.0] * app_state.gravity_factor,
        dt: (1.0 / 60.0) / (app_state.num_substeps as f32),
    };

    let rb = RigidBodyBuilder::fixed().translation(vector![0.0, -4.0, 0.0]);
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(100.0, 1.0, 100.0);
    rapier_data
        .colliders
        .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);

    // TODO: use only two rectangle per cutting tool.
    //       We can’t right now since we don’t really sample the triangles.
    for k in 0..3 {
        let heights = DMatrix::zeros(10, 10);
        let heightfield = HeightField::new(heights, vector![35.0, 1.0, 10.0]);
        let (mut vtx, idx) = heightfield.to_trimesh();
        vtx.iter_mut().for_each(|pt| {
            *pt = Isometry3::rotation(vector![1.3, 0.0, 0.0]) * *pt
                + vector![0.0, 10.0, k as f32 * 10.0 - 10.0]
        });
        let rb = RigidBodyBuilder::fixed();
        let rb_handle = rapier_data.bodies.insert(rb);
        let co = ColliderBuilder::trimesh(vtx, idx).unwrap();
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
