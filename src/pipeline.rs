use std::marker::PhantomData;
use crate::grid::grid::{GpuGrid, WgGrid};
use crate::grid::prefix_sum::{PrefixSumWorkspace, WgPrefixSum};
use crate::grid::sort::WgSort;
use crate::solver::{GpuImpulses, GpuParticleModelData, GpuParticles, GpuRigidParticles, GpuSimulationParams, Particle, SimulationParams, WgG2P, WgG2PCdf, WgGridUpdate, WgGridUpdateCdf, WgP2G, WgP2GCdf, WgParticleUpdate, WgRigidImpulses, WgRigidParticleUpdate};
use nexus::dynamics::GpuBodySet;
use nexus::dynamics::body::{BodyCoupling, BodyCouplingEntry};
use nexus::math::GpuSim;
use rapier::dynamics::RigidBodySet;
use rapier::geometry::ColliderSet;
use slang_hal::Shader;
use slang_hal::backend::{Backend, Encoder};
use slang_hal::re_exports::minislang::SlangCompiler;
use stensor::tensor::GpuVector;
use wgpu::BufferUsages;

pub struct MpmPipeline<B: Backend, GpuModel: GpuParticleModelData> {
    grid: WgGrid<B>,
    prefix_sum: WgPrefixSum<B>,
    sort: WgSort<B>,
    p2g: WgP2G<B>,
    p2g_cdf: WgP2GCdf<B>,
    grid_update_cdf: WgGridUpdateCdf<B>,
    grid_update: WgGridUpdate<B>,
    particles_update: WgParticleUpdate<B>,
    g2p: WgG2P<B>,
    g2p_cdf: WgG2PCdf<B>,
    rigid_particles_update: WgRigidParticleUpdate<B>,
    pub impulses: WgRigidImpulses<B>,
    _phantom: PhantomData<GpuModel>,
}

pub struct MpmData<B: Backend, GpuModel: GpuParticleModelData> {
    pub sim_params: GpuSimulationParams<B>,
    pub grid: GpuGrid<B>,
    pub particles: GpuParticles<B, GpuModel>, // TODO: keep private?
    pub rigid_particles: GpuRigidParticles<B>,
    pub bodies: GpuBodySet<B>,
    pub impulses: GpuImpulses<B>,
    pub poses_staging: GpuVector<GpuSim, B>,
    prefix_sum: PrefixSumWorkspace<B>,
    coupling: Vec<BodyCouplingEntry>,
}

// Defines module paths for specializing parts of the MPM pipeline leveraging Slang’s
// link-time specialization feature.
pub struct MpmSpecializations {
    pub particle_model: Vec<String>,
}

impl<B: Backend, GpuModel: GpuParticleModelData> MpmData<B, GpuModel> {
    pub fn new(
        backend: &B,
        params: SimulationParams,
        particles: &[Particle<GpuModel::Model>],
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, B::Error> {
        let coupling: Vec<_> = colliders
            .iter()
            .filter_map(|(co_handle, co)| {
                let rb_handle = co.parent()?;
                Some(BodyCouplingEntry {
                    body: rb_handle,
                    collider: co_handle,
                    mode: BodyCoupling::OneWay,
                })
            })
            .collect();
        Self::with_select_coupling(
            backend,
            params,
            particles,
            bodies,
            colliders,
            coupling,
            cell_width,
            grid_capacity,
        )
    }

    pub fn with_select_coupling(
        backend: &B,
        params: SimulationParams,
        particles: &[Particle<GpuModel::Model>],
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        coupling: Vec<BodyCouplingEntry>,
        cell_width: f32,
        grid_capacity: u32,
    ) -> Result<Self, B::Error> {
        let sampling_step = cell_width; // TODO: * 1.5 ?
        let bodies = GpuBodySet::from_rapier(backend, bodies, colliders, &coupling)?;
        let sim_params = GpuSimulationParams::new(backend, params)?;
        let particles = GpuParticles::from_particles(backend, particles)?;
        let rigid_particles =
            GpuRigidParticles::from_rapier(backend, colliders, &bodies, &coupling, sampling_step)?;
        let grid = GpuGrid::with_capacity(backend, grid_capacity, cell_width)?;
        let prefix_sum = PrefixSumWorkspace::with_capacity(backend, grid_capacity)?;
        let impulses = GpuImpulses::new(backend)?;
        let poses_staging = GpuVector::vector_uninit(
            backend,
            bodies.len(),
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;

        Ok(Self {
            sim_params,
            particles,
            rigid_particles,
            bodies,
            impulses,
            grid,
            prefix_sum,
            poses_staging,
            coupling,
        })
    }

    pub fn coupling(&self) -> &[BodyCouplingEntry] {
        &self.coupling
    }
}

impl<B: Backend, GpuModel: GpuParticleModelData> MpmPipeline<B, GpuModel> {
    pub fn new(backend: &B, compiler: &SlangCompiler) -> Result<Self, B::Error> {
        Ok(Self {
            grid: WgGrid::from_backend(backend, compiler)?,
            prefix_sum: WgPrefixSum::from_backend(backend, compiler)?,
            sort: WgSort::from_backend(backend, compiler)?,
            p2g: WgP2G::from_backend(backend, compiler)?,
            p2g_cdf: WgP2GCdf::from_backend(backend, compiler)?,
            grid_update: WgGridUpdate::from_backend(backend, compiler)?,
            grid_update_cdf: WgGridUpdateCdf::from_backend(backend, compiler)?,
            particles_update: WgParticleUpdate::with_specializations(backend, compiler, &GpuModel::specialization_modules())?,
            rigid_particles_update: WgRigidParticleUpdate::from_backend(backend, compiler)?,
            g2p: WgG2P::from_backend(backend, compiler)?,
            g2p_cdf: WgG2PCdf::from_backend(backend, compiler)?,
            impulses: WgRigidImpulses::from_backend(backend, compiler)?,
            _phantom: PhantomData,
        })
    }

    pub fn launch_step(
        &self,
        backend: &B,
        encoder: &mut B::Encoder,
        data: &mut MpmData<B, GpuModel>,
        // mut timestamps: Option<&mut GpuTimestamps>,
    ) -> Result<(), B::Error> {
        {
            let mut pass = encoder.begin_pass(); // "update rigid particles", timestamps.as_deref_mut());
            self.impulses.launch_update_world_mass_properties(
                backend,
                &mut pass,
                &data.impulses,
                &data.bodies,
            )?;
            self.rigid_particles_update.launch(
                backend,
                &mut pass,
                &data.bodies,
                &data.rigid_particles,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("grid sort", timestamps.as_deref_mut());
            self.grid.launch_sort(
                backend,
                &mut pass,
                &data.particles,
                &data.rigid_particles,
                &data.grid,
                &mut data.prefix_sum,
                &self.sort,
                &self.prefix_sum,
            )?;
            self.sort.launch_sort_rigid_particles(
                backend,
                &mut pass,
                &data.rigid_particles,
                &data.grid,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("grid_update_cdf", timestamps.as_deref_mut());
            self.grid_update_cdf
                .launch(backend, &mut pass, &data.grid, &data.bodies)?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("p2g_cdf", timestamps.as_deref_mut());
            self.p2g_cdf.launch(
                backend,
                &mut pass,
                &data.grid,
                &data.rigid_particles,
                &data.bodies,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("g2p_cdf", timestamps.as_deref_mut());
            self.g2p_cdf.launch(
                backend,
                &mut pass,
                &data.sim_params,
                &data.grid,
                &data.particles,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("p2g", timestamps.as_deref_mut());
            self.p2g.launch(
                backend,
                &mut pass,
                &data.grid,
                &data.particles,
                &data.impulses,
                &data.bodies,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("grid_update", timestamps.as_deref_mut());
            self.grid_update
                .launch(backend, &mut pass, &data.sim_params, &data.grid)?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("g2p", timestamps.as_deref_mut());
            self.g2p.launch(
                backend,
                &mut pass,
                &data.sim_params,
                &data.grid,
                &data.particles,
                &data.bodies,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("particles_update", timestamps.as_deref_mut());
            self.particles_update.launch(
                backend,
                &mut pass,
                &data.sim_params,
                &data.grid,
                &data.particles,
                &data.bodies,
            )?;
        }

        {
            let mut pass = encoder.begin_pass(); // ("integrate_bodies", timestamps.as_deref_mut());
            // TODO: should this be in a separate pipeline? Within impulse probably?
            self.impulses.launch(
                backend,
                &mut pass,
                &data.grid,
                &data.sim_params,
                &data.impulses,
                &data.bodies,
            )?;
        }

        Ok(())
    }
}

/*
#[cfg(test)]
#[cfg(feature = "dim3")]
mod test {
    use crate::models::ElasticCoefficients;
    use crate::pipeline::{MpmData, MpmPipeline};
    use crate::solver::{Particle, ParticleDynamics, SimulationParams};
    use nalgebra::vector;
    use rapier::prelude::{ColliderSet, RigidBodySet};
    use slang_hal::gpu::GpuInstance;
    use slang_hal::kernel::KernelInvocationQueue;
    use wgpu::Maintain;

    #[futures_test::test]
    #[serial_test::serial]
    async fn pipeline_queue_step() {
        let gpu = GpuInstance::new().await.unwrap();
        let pipeline = MpmPipeline::new(gpu.backend()).unwrap();

        let cell_width = 1.0;
        let mut cpu_particles = vec![];
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let position = vector![i as f32, j as f32, k as f32] / cell_width / 2.0;
                    cpu_particles.push(Particle {
                        position,
                        dynamics: ParticleDynamics::with_density(cell_width / 4.0, 1.0),
                        model: ElasticCoefficients::from_young_modulus(100_000.0, 0.33),
                        plasticity: None,
                        phase: None,
                    });
                }
            }
        }

        let params = SimulationParams {
            gravity: vector![0.0, -9.81, 0.0],
            dt: (1.0 / 60.0) / 10.0,
        };
        let mut data = MpmData::new(
            gpu.backend(),
            params,
            &cpu_particles,
            &RigidBodySet::default(),
            &ColliderSet::default(),
            cell_width,
            100_000,
        );
        let mut queue = KernelInvocationQueue::new(gpu.backend());
        pipeline.queue_step(&mut data, &mut queue, false);

        for _ in 0..3 {
            let mut encoder = gpu.backend().create_command_encoder(&Default::default());
            queue.encode(&mut encoder, None);
            let t0 = std::time::Instant::now();
            gpu.queue().submit(Some(encoder.finish()));
            gpu.backend().poll(Maintain::Wait);
            println!("Sim step time: {}", t0.elapsed().as_secs_f32());
        }
    }
}
 */
