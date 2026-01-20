//! Headless simulation harness for validation tests.
//!
//! Provides utilities for running MPM simulations without rendering
//! and extracting particle data for comparison with reference implementations.

use nalgebra::{vector, Point3, Vector3};
use rapier3d::geometry::ColliderHandle;
use rapier3d::prelude::{ColliderBuilder, ColliderSet, RigidBodyBuilder, RigidBodySet};
use serde::{Deserialize, Serialize};
use slang_hal::backend::{Backend, WebGpu};
use slang_hal::{BufferUsages, SlangCompiler};
use slosh3d::{
    pipeline::{MpmData, MpmPipeline},
    solver::{
        GpuBoundaryCondition, GpuParticleModel, Particle, ParticleModel, ParticlePosition,
        SimulationParams,
    },
};
use std::any::Any;
use std::path::Path;
use std::time::Instant;
use stensor::tensor::GpuTensor;
use wgpu::Limits;

/// Recorded state of a single particle at a point in time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParticleState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
}

/// Recorded state of the entire simulation at a point in time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationSnapshot {
    pub time: f32,
    pub step: usize,
    pub particles: Vec<ParticleState>,
}

/// Complete trajectory of a simulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationTrajectory {
    pub name: String,
    pub dt: f32,
    pub num_substeps: u32,
    pub snapshots: Vec<SimulationSnapshot>,
    pub metadata: SimulationMetadata,
}

/// Metadata about the simulation setup.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationMetadata {
    pub num_particles: usize,
    pub cell_width: f32,
    pub gravity: [f32; 3],
    pub material: MaterialParams,
}

/// Material parameters for comparison.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialParams {
    pub young_modulus: f32,
    pub poisson_ratio: f32,
    pub density: f32,
    pub material_type: String,
}

impl SimulationTrajectory {
    /// Export trajectory to JSON file.
    pub fn export_json(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Export trajectory to CSV file (one file per timestep, positions only).
    pub fn export_csv(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;

        for snapshot in &self.snapshots {
            let filename = format!("step_{:06}.csv", snapshot.step);
            let path = dir.join(filename);

            let mut wtr = csv::Writer::from_path(path)?;
            wtr.write_record(["particle_id", "x", "y", "z", "vx", "vy", "vz"])?;

            for (i, p) in snapshot.particles.iter().enumerate() {
                wtr.write_record([
                    i.to_string(),
                    p.position[0].to_string(),
                    p.position[1].to_string(),
                    p.position[2].to_string(),
                    p.velocity[0].to_string(),
                    p.velocity[1].to_string(),
                    p.velocity[2].to_string(),
                ])?;
            }
            wtr.flush()?;
        }
        Ok(())
    }

    /// Load trajectory from JSON file.
    pub fn load_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let trajectory: SimulationTrajectory = serde_json::from_str(&json)?;
        Ok(trajectory)
    }
}

/// Configuration for a validation test scenario.
#[derive(Clone, Debug)]
pub struct ScenarioConfig {
    pub name: String,
    pub particles: Vec<Particle<ParticleModel>>,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub materials: Vec<(ColliderHandle, GpuBoundaryCondition)>,
    pub gravity: Vector3<f32>,
    pub cell_width: f32,
    pub dt: f32,
    pub num_substeps: u32,
    pub total_steps: usize,
    pub snapshot_interval: usize,
    pub grid_capacity: u32,
    pub material_params: MaterialParams,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            particles: vec![],
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            materials: vec![],
            gravity: vector![0.0, -9.81, 0.0],
            cell_width: 1.0,
            dt: 1.0 / 60.0,
            num_substeps: 10,
            total_steps: 300,
            snapshot_interval: 1,
            grid_capacity: 30_000,
            material_params: MaterialParams {
                young_modulus: 1.0e6,
                poisson_ratio: 0.3,
                density: 1000.0,
                material_type: "elastic".to_string(),
            },
        }
    }
}

/// Headless simulation runner for validation tests.
pub struct ValidationHarness {
    gpu: WebGpu,
    compiler: SlangCompiler,
}

impl ValidationHarness {
    /// Create a new validation harness with GPU backend.
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let limits = Limits {
            max_storage_buffers_per_shader_stage: 11,
            max_compute_workgroup_storage_size: 32768,
            ..Limits::default()
        };
        let gpu = WebGpu::new(Default::default(), limits).await?;

        let mut compiler = SlangCompiler::default();
        #[cfg(feature = "runtime")]
        {
            slosh3d::register_shaders(&mut compiler);
            compiler.set_global_macro("DIM", 3);
        }

        Ok(Self { gpu, compiler })
    }

    /// Run a simulation scenario and record the trajectory.
    pub async fn run_scenario(
        &self,
        config: ScenarioConfig,
    ) -> Result<SimulationTrajectory, Box<dyn std::error::Error>> {
        let pipeline: MpmPipeline<WebGpu, GpuParticleModel> =
            MpmPipeline::new(&self.gpu, &self.compiler)?;

        let params = SimulationParams {
            gravity: config.gravity,
            dt: config.dt / config.num_substeps as f32,
        };

        let mut data = MpmData::new(
            &self.gpu,
            params,
            &config.particles,
            &config.bodies,
            &config.colliders,
            &config.materials,
            config.cell_width,
            config.grid_capacity,
        )?;

        // Create staging buffer for readback
        let num_particles = config.particles.len();
        let mut positions_staging: GpuTensor<ParticlePosition, WebGpu> = GpuTensor::vector_uninit(
            &self.gpu,
            num_particles as u32,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )?;

        let mut snapshots = Vec::new();
        let mut hooks: () = ();
        let mut hooks_state: Box<dyn Any> = Box::new(());

        // Record initial state
        let initial_snapshot = self
            .extract_snapshot(&data, &mut positions_staging, 0, 0.0)
            .await?;
        snapshots.push(initial_snapshot);

        // Run simulation
        let mut current_time = 0.0;
        for step in 1..=config.total_steps {
            let step_start = Instant::now();
            let mut encoder = self.gpu.begin_encoding();

            for _ in 0..config.num_substeps {
                pipeline
                    .launch_step(
                        &self.gpu,
                        &mut encoder,
                        &mut data,
                        &mut hooks,
                        &mut *hooks_state,
                    )
                    .await?;
            }

            self.gpu.submit(encoder)?;
            self.gpu.synchronize()?;
            let step_time = step_start.elapsed();

            current_time += config.dt;

            // Record snapshot at intervals
            if step % config.snapshot_interval == 0 {
                let snapshot = self
                    .extract_snapshot(&data, &mut positions_staging, step, current_time)
                    .await?;
                snapshots.push(snapshot);

                // Print progress with timing
                println!(
                    "  Step {}/{} ({:.2}ms/step)",
                    step,
                    config.total_steps,
                    step_time.as_secs_f64() * 1000.0
                );
            }
        }

        Ok(SimulationTrajectory {
            name: config.name,
            dt: config.dt,
            num_substeps: config.num_substeps,
            snapshots,
            metadata: SimulationMetadata {
                num_particles: config.particles.len(),
                cell_width: config.cell_width,
                gravity: config.gravity.into(),
                material: config.material_params,
            },
        })
    }

    /// Extract current particle states from GPU.
    async fn extract_snapshot(
        &self,
        data: &MpmData<WebGpu, GpuParticleModel>,
        positions_staging: &mut GpuTensor<ParticlePosition, WebGpu>,
        step: usize,
        time: f32,
    ) -> Result<SimulationSnapshot, Box<dyn std::error::Error>> {
        let num_particles = data.particles.len();

        // Copy positions from GPU storage buffer to staging buffer
        let mut encoder = self.gpu.begin_encoding();
        positions_staging.copy_from_view(&mut encoder, data.particles.positions())?;
        self.gpu.submit(encoder)?;
        self.gpu.synchronize()?;

        // Read positions from staging buffer
        let mut positions = vec![ParticlePosition::origin(); num_particles];
        self.gpu
            .read_buffer(positions_staging.buffer(), &mut positions)
            .await?;

        let particles: Vec<ParticleState> = positions
            .iter()
            .map(|p| ParticleState {
                position: [p.x, p.y, p.z],
                velocity: [0.0, 0.0, 0.0], // Velocity extraction would need a custom readback shader
            })
            .collect();

        Ok(SimulationSnapshot {
            time,
            step,
            particles,
        })
    }
}

/// Helper to create a block of particles in a grid pattern.
pub fn create_particle_block(
    center: Point3<f32>,
    half_extents: Vector3<f32>,
    cell_width: f32,
    density: f32,
    model: ParticleModel,
) -> Vec<Particle<ParticleModel>> {
    let mut particles = vec![];
    let particle_spacing = cell_width / 2.0;
    let radius = cell_width / 4.0;

    let nx = (half_extents.x * 2.0 / particle_spacing).ceil() as i32;
    let ny = (half_extents.y * 2.0 / particle_spacing).ceil() as i32;
    let nz = (half_extents.z * 2.0 / particle_spacing).ceil() as i32;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let offset = vector![
                    (i as f32 + 0.5) * particle_spacing - half_extents.x,
                    (j as f32 + 0.5) * particle_spacing - half_extents.y,
                    (k as f32 + 0.5) * particle_spacing - half_extents.z
                ];
                let position = center + offset;
                particles.push(Particle::new(position, radius, density, model));
            }
        }
    }

    particles
}

/// Helper to create a sphere of particles.
pub fn create_particle_sphere(
    center: Point3<f32>,
    radius: f32,
    cell_width: f32,
    density: f32,
    model: ParticleModel,
) -> Vec<Particle<ParticleModel>> {
    let mut particles = vec![];
    let particle_spacing = cell_width / 2.0;
    let particle_radius = cell_width / 4.0;

    let n = (radius * 2.0 / particle_spacing).ceil() as i32;

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let offset = vector![
                    (i as f32 + 0.5) * particle_spacing - radius,
                    (j as f32 + 0.5) * particle_spacing - radius,
                    (k as f32 + 0.5) * particle_spacing - radius
                ];
                if offset.norm() <= radius {
                    let position = center + offset;
                    particles.push(Particle::new(position, particle_radius, density, model));
                }
            }
        }
    }

    particles
}

/// Create a ground plane collider.
pub fn create_ground_plane(
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    y_position: f32,
) -> ColliderHandle {
    let hy = 1.0;
    let rb = RigidBodyBuilder::fixed().translation(vector![0.0, y_position - hy, 0.0]);
    let rb_handle = bodies.insert(rb);
    let co = ColliderBuilder::cuboid(100.0, hy, 100.0);
    colliders.insert_with_parent(co, rb_handle, bodies)
}
