//! Rigid body particle transformation kernels.

use crate::sampling::GpuSampleIds;
use crate::solver::GpuRigidParticles;
use nexus::dynamics::GpuBodySet;
use nexus::math::{GpuSim, Point};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::GpuTensor;

/// GPU kernels for updating rigid body particle positions.
///
/// Transforms surface-sampled particles from local to world coordinates
/// as rigid bodies move.
#[derive(Shader)]
#[shader(module = "slosh::solver::rigid_particle_update")]
pub struct WgRigidParticleUpdate<B: Backend> {
    /// Kernel for transforming sample points.
    pub transform_sample_points: GpuFunction<B>,
    /// Kernel for transforming shape vertex points.
    pub transform_shape_points: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct RigidParticleUpdateArgs<'a, B: Backend> {
    vertex_collider_ids: Option<&'a GpuTensor<u32, B>>,
    rigid_particle_indices: Option<&'a GpuTensor<GpuSampleIds, B>>,
    poses: &'a GpuTensor<GpuSim, B>,
    local_pts: &'a GpuTensor<Point<f32>, B>,
    world_pts: &'a GpuTensor<Point<f32>, B>,
}

impl<B: Backend> WgRigidParticleUpdate<B> {
    /// Transforms rigid body particles from local to world space.
    ///
    /// Updates surface particle positions based on current rigid body poses.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `bodies` - Rigid bodies with current poses
    /// * `rigid_particles` - Particles to transform
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        bodies: &GpuBodySet<B>,
        rigid_particles: &GpuRigidParticles<B>,
    ) -> Result<(), B::Error> {
        if rigid_particles.sample_points.is_empty() {
            return Ok(());
        }

        let args = RigidParticleUpdateArgs {
            vertex_collider_ids: None,
            rigid_particle_indices: Some(&rigid_particles.sample_ids),
            poses: bodies.poses(),
            local_pts: &rigid_particles.local_sample_points,
            world_pts: &rigid_particles.sample_points,
        };
        self.transform_sample_points.launch(
            backend,
            pass,
            &args,
            [rigid_particles.local_sample_points.len() as u32, 1, 1],
        )?;

        let args = RigidParticleUpdateArgs {
            vertex_collider_ids: Some(bodies.shapes_vertex_collider_id()),
            rigid_particle_indices: None,
            poses: bodies.poses(),
            local_pts: bodies.shapes_local_vertex_buffers(),
            world_pts: bodies.shapes_vertex_buffers(),
        };
        self.transform_shape_points.launch(
            backend,
            pass,
            &args,
            [bodies.shapes_vertex_buffers().len() as u32, 1, 1],
        )
    }
}
