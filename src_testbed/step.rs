use crate::prep_readback::ReadbackData;
use crate::{RunState, Stage};
use nexus::rapier::na;
use slang_hal::backend::Backend;

#[derive(Default)]
pub struct SimulationTimes {
    pub total_step_time: f32,
    pub encoding_time: f32,
    pub readback_time: f32,
}

#[derive(Default)]
pub struct SimulationStepResult {
    pub instances: Vec<ReadbackData>,
    pub timings: SimulationTimes,
}

impl Stage {
    // TODO PERF: don’t reallocate the result buffer each time.
    pub fn step_simulation(&mut self) -> bool {
        if self.app_state.run_state == RunState::Paused {
            return false;
        }

        // Run the simulation.
        let physics = &mut self.physics;

        let t_total = std::time::Instant::now();
        let t_encoding = std::time::Instant::now();
        let mut encoder = self.gpu.begin_encoding();

        // Send updated bodies information to the gpu.
        // PERF: don’t reallocate the buffers at each step.
        // let poses_data: Vec<GpuSim> = physics
        //     .data
        //     .coupling()
        //     .iter()
        //     .map(|coupling| {
        //         let c = &physics.rapier_data.colliders[coupling.collider];
        //         #[cfg(feature = "dim2")]
        //         return (*c.position()).into();
        //         #[cfg(feature = "dim3")]
        //         return GpuSim::from_isometry(*c.position(), 1.0);
        //     })
        //     .collect();
        // println!("poses: {:?}", poses_data);
        // compute_queue.write_buffer(
        //     physics.data.bodies.poses().buffer(),
        //     0,
        //     bytemuck::cast_slice(&poses_data),
        // );

        // let divisor = 1.0; // self.app_state.num_substeps as f32;
        // let gravity = Vector::y() * -9.81;
        // let vels_data: Vec<_> = physics
        //     .data
        //     .coupling()
        //     .iter()
        //     .map(|coupling| {
        //         let rb = &physics.rapier_data.bodies[coupling.body];
        //         GpuVelocity {
        //             linear: *rb.linvel()
        //                 + gravity * physics.rapier_data.params.dt * (rb.is_dynamic() as u32 as f32)
        //                 / (self.app_state.num_substeps as f32),
        //             #[allow(clippy::clone_on_copy)] // Needed for the 2d/3d switch.
        //             angular: rb.angvel().clone(),
        //         }
        //     })
        //     .collect();
        //
        // let mut vels_bytes = vec![];
        // let mut buffer = StorageBuffer::new(&mut vels_bytes);
        // buffer.write(&vels_data).unwrap();
        // compute_queue.write_buffer(physics.data.bodies.vels().buffer(), 0, &vels_bytes);

        //// Step the simulation.

        for _ in 0..self.app_state.num_substeps {
            self.app_state
                .pipeline
                .launch_step(&self.gpu, &mut encoder, &mut physics.data)
                .unwrap();
        }

        // physics
        //     .data
        //     .poses_staging
        //     .copy_from(&mut encoder, physics.data.bodies.poses());
        // physics
        //     .data
        //     .particles_pos_staging
        //     .copy_from(&mut encoder, &physics.data.particles.positions);

        // Prepare the vertex buffer for rendering the particles.
        /*
        if let Ok(instances_buffer) = particles.get_single() {
            queue.clear();
            self.app_state.prep_vertex_buffer.queue(
                &mut queue,
                &self.app_state.gpu_render_config,
                &physics.data.particles,
                &physics.data.rigid_particles,
                &physics.data.grid,
                &physics.data.sim_params,
                &instances_buffer.buffer.buffer,
                rigid_particles
                    .get_single()
                    .ok()
                    .map(|b| &**b.buffer.buffer),
            );
            queue.encode(&mut encoder, None);
        }
         */

        // Submit.
        self.readback_shader
            .launch(
                &self.gpu,
                &mut encoder,
                &mut self.readback,
                &physics.data.sim_params,
                &physics.data.grid,
                &physics.data.particles,
            )
            .unwrap();

        self.gpu.submit(encoder).unwrap();
        let t_encoding = t_encoding.elapsed().as_secs_f32() * 1000.0;

        self.gpu.synchronize().unwrap();
        let t_total_step = t_total.elapsed().as_secs_f32() * 1000.0;

        // TODO: reuse the `physics.data.particles_pos_staging` buffer.
        let t_readback = std::time::Instant::now();
        futures::executor::block_on(self.gpu.read_buffer(
            self.readback.instances_staging.buffer(),
            self.step_result.instances.as_mut_slice(),
        ))
        .unwrap();
        let t_readback = t_readback.elapsed().as_secs_f32() * 1000.0;
        // Step rapier to update kinematic bodies.
        let rapier = &mut self.physics.rapier_data;
        rapier.physics_pipeline.step(
            &na::zero(),
            &rapier.params,
            &mut rapier.islands,
            &mut rapier.broad_phase,
            &mut rapier.narrow_phase,
            &mut rapier.bodies,
            &mut rapier.colliders,
            &mut rapier.impulse_joints,
            &mut rapier.multibody_joints,
            &mut rapier.ccd_solver,
            &(),
            &(),
        );

        if self.app_state.run_state == RunState::Step {
            self.app_state.run_state = RunState::Paused;
        }

        self.step_result.timings = SimulationTimes {
            total_step_time: t_total_step,
            encoding_time: t_encoding,
            readback_time: t_readback,
        };

        true
    }
}
