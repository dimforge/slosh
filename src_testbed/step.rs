use crate::prep_readback::{GpuReadbackData, ReadbackData};
use crate::{PhysicsState, RunState, Stage};
use slang_hal::backend::{Backend, WebGpu};
use slang_hal::BufferUsages;
#[cfg(feature = "webgpu")]
use slang_hal::GpuTimingResult;
use slosh::solver::{GpuParticleModelData, SimulationParams};
use stensor::tensor::GpuTensor;

/// Byte stride of one `Mat<f32>` element in the GPU def-grad buffer, matching
/// the slang/WGSL structured-buffer layout. In 3D, `mat3x3<f32>` has three
/// `vec4`-aligned columns for a total of 48 bytes (not the tightly-packed 36
/// bytes of a CPU `glam::Mat3`). In 2D, `mat2x2<f32>` is already
/// tightly packed at 16 bytes, so there is no mismatch.
#[cfg(feature = "dim2")]
pub const GPU_DEF_GRAD_STRIDE_BYTES: usize = 16;
#[cfg(feature = "dim3")]
pub const GPU_DEF_GRAD_STRIDE_BYTES: usize = 48;

/// Same stride, expressed in number of `f32` elements. 4 in 2D, 12 in 3D.
pub const GPU_DEF_GRAD_STRIDE_F32: usize = GPU_DEF_GRAD_STRIDE_BYTES / 4;

#[derive(Default)]
pub struct SimulationTimes {
    pub total_step_time: f32,
    pub encoding_time: f32,
    pub readback_time: f32,
    #[cfg(feature = "webgpu")]
    pub gpu_passes: Vec<GpuTimingResult>,
}

#[derive(Default)]
pub struct SimulationStepResult {
    pub instances: Vec<ReadbackData>,
    pub timings: SimulationTimes,
    /// Raw model data read back from GPU, stored as u32 words.
    /// Can be cast to the concrete model type using `bytemuck::cast_slice`.
    pub model_data_raw: Vec<u32>,
    /// Raw deformation gradient data read back from GPU, stored as f32 values.
    /// Stride per particle is `GPU_DEF_GRAD_STRIDE_F32`: 4 in 2D (a
    /// `mat2x2<f32>`), 12 in 3D (a `mat3x3<f32>` with `vec4`-aligned columns).
    /// In 3D only the first three entries of each column are meaningful; the
    /// fourth entry of each column is slang padding.
    pub def_grad_raw: Vec<f32>,
}

impl<GpuModel: GpuParticleModelData> Stage<GpuModel> {
    // TODO PERF: don’t reallocate the result buffer each time.
    pub async fn step_simulation(&mut self) -> bool {
        if self.app_state.run_state == RunState::Paused {
            return false;
        }

        // Run the simulation.
        let physics = &mut self.physics;
        let prev_particle_count = physics.data.particles.len();
        for callback in &mut physics.callbacks {
            let mut phx = PhysicsState {
                backend: &self.gpu,
                data: &mut physics.data,
                results: &self.step_result,
                step_id: self.step_id,
            };
            callback.update(&mut phx);
        }

        // Check if the particle size changed. If it did, adjust the instance buffers.
        let new_particle_count = physics.data.particles.len();
        if prev_particle_count != new_particle_count {
            // TODO: resize buffers instead of recreating.
            // Custom colors from init() don't apply after dynamic particle
            // count changes, so fall back to the default palette here.
            self.readback =
                GpuReadbackData::new(&self.gpu, new_particle_count, self.render_mode, None)
                    .unwrap();
            let model_u32_count = new_particle_count * std::mem::size_of::<GpuModel>() / 4;
            self.model_staging = GpuTensor::<u32, WebGpu>::vector_uninit(
                &self.gpu,
                model_u32_count as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )
            .unwrap();
            self.step_result
                .instances
                .resize(new_particle_count, ReadbackData::default());
            let def_grad_f32_count = new_particle_count * GPU_DEF_GRAD_STRIDE_F32;
            self.def_grad_staging = GpuTensor::<f32, WebGpu>::vector_uninit(
                &self.gpu,
                def_grad_f32_count as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            )
            .unwrap();
            self.step_result.model_data_raw.resize(model_u32_count, 0);
            self.step_result
                .def_grad_raw
                .resize(def_grad_f32_count, 0.0);
            println!("Adjust readback buffers: {}", new_particle_count);
        }

        let t_total = web_time::Instant::now();
        let base_dt = physics.data.base_dt;
        let prev_num_substeps = self.app_state.num_substeps;

        if self.app_state.min_num_substeps < self.app_state.max_num_substeps {
            // Adaptive stepping.
            let bounds = self
                .app_state
                .pipeline
                .timestep_bounds
                .compute_bounds(
                    &self.gpu,
                    &physics.data.grid,
                    &physics.data.particles,
                    &physics.data.timestep_bounds,
                    &mut physics.data.timestep_bounds_staging,
                )
                .await
                .unwrap();

            let num_substeps_estimated = (base_dt / bounds).ceil() as u32;
            let num_substeps = num_substeps_estimated.clamp(
                self.app_state.min_num_substeps,
                self.app_state.max_num_substeps,
            );
            self.app_state.num_substeps = num_substeps;

            // println!(
            //     "Found timestep bounds: {:?}. Estimated substeps: {}. Actual: {}",
            //     bounds, num_substeps_estimated, num_substeps
            // );
        } else if self.app_state.num_substeps != self.app_state.max_num_substeps {
            // No adaptive stepping, but we need to update the number of substeps on the gpu.
            self.app_state.num_substeps = self.app_state.max_num_substeps;
        }

        if prev_num_substeps != self.app_state.num_substeps {
            let gravity = physics.data.gravity;
            let params = SimulationParams {
                gravity,
                dt: base_dt / self.app_state.num_substeps as f32,
                #[cfg(feature = "dim2")]
                padding: 0.0,
            };
            println!("Updated GPU sim params to: {:?}", params);
            let gpu_params = physics.data.sim_params.params.buffer_mut();
            self.gpu.write_buffer(gpu_params, 0, &[params]).unwrap();
        }

        let t_encoding = web_time::Instant::now();
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

        let mut no_state = Box::new(());
        let hooks_state = physics.hooks_state.as_deref_mut().unwrap_or(&mut no_state);

        #[cfg(feature = "webgpu")]
        let mut timestamps = slang_hal::GpuTimestamps::new(
            self.gpu.device(),
            self.gpu.queue(),
            self.app_state.num_substeps * 10,
        );

        for _ in 0..self.app_state.num_substeps {
            self.app_state
                .pipeline
                .launch_step(
                    &self.gpu,
                    &mut encoder,
                    &mut physics.data,
                    &mut *self.hooks,
                    hooks_state,
                    #[cfg(feature = "webgpu")]
                    Some(&mut timestamps),
                    #[cfg(not(feature = "webgpu"))]
                    None,
                )
                .await
                .unwrap();
        }

        #[cfg(feature = "webgpu")]
        timestamps.resolve(&mut encoder);

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

        // Copy model buffer to staging for readback.
        {
            let model_buf = physics.data.particles.models().buffer();
            let staging_buf = self.model_staging.buffer();
            let bytes =
                physics.data.particles.len() as u64 * std::mem::size_of::<GpuModel>() as u64;
            wgpu::CommandEncoder::copy_buffer_to_buffer(
                &mut encoder,
                model_buf,
                0,
                staging_buf,
                0,
                bytes,
            );
        }

        // Copy deformation gradient buffer to staging for readback. We use the
        // GPU-side stride (see `GPU_DEF_GRAD_STRIDE_BYTES`) rather than
        // `size_of::<Matrix>()`, which is only correct in 2D. In 3D the
        // GPU mat3x3 has vec4-aligned columns (48 bytes), not the tightly
        // packed 36 bytes of a CPU `glam::Mat3`.
        {
            let def_grad_buf = physics.data.particles.def_grad.buffer();
            let staging_buf = self.def_grad_staging.buffer();
            let bytes = physics.data.particles.len() as u64 * GPU_DEF_GRAD_STRIDE_BYTES as u64;
            wgpu::CommandEncoder::copy_buffer_to_buffer(
                &mut encoder,
                def_grad_buf,
                0,
                staging_buf,
                0,
                bytes,
            );
        }

        self.gpu.submit(encoder).unwrap();
        let t_encoding = t_encoding.elapsed().as_secs_f32() * 1000.0;

        self.gpu.synchronize().unwrap();
        let t_total_step = t_total.elapsed().as_secs_f32() * 1000.0;

        // TODO: reuse the `physics.data.particles_pos_staging` buffer.
        let t_readback = web_time::Instant::now();
        self.gpu
            .read_buffer(
                self.readback.instances_staging.buffer(),
                self.step_result.instances.as_mut_slice(),
            )
            .await
            .unwrap();
        self.gpu
            .read_buffer(
                self.model_staging.buffer(),
                self.step_result.model_data_raw.as_mut_slice(),
            )
            .await
            .unwrap();
        self.gpu
            .read_buffer(
                self.def_grad_staging.buffer(),
                self.step_result.def_grad_raw.as_mut_slice(),
            )
            .await
            .unwrap();
        let t_readback = t_readback.elapsed().as_secs_f32() * 1000.0;
        // Step rapier to update kinematic bodies.
        let rapier = &mut self.physics.rapier_data;
        rapier.physics_pipeline.step(
            slosh::math::Vector::ZERO,
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

        #[cfg(feature = "webgpu")]
        let gpu_passes = timestamps
            .read_results(self.gpu.device())
            .await
            .unwrap_or_default();

        self.step_result.timings = SimulationTimes {
            total_step_time: t_total_step,
            encoding_time: t_encoding,
            readback_time: t_readback,
            #[cfg(feature = "webgpu")]
            gpu_passes,
        };
        self.step_id += 1;

        true
    }
}
