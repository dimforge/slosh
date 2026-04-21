use crate::step::SimulationStepResult;
use slang_hal::backend::WebGpu;
use slosh::pipeline::{MpmData, MpmPipeline};
use slosh::rapier::prelude::{
    CCDSolver, ColliderSet, DefaultBroadPhase, ImpulseJointSet, IntegrationParameters,
    IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, RigidBodySet,
};
use slosh::solver::{GpuParticleModel, GpuParticleModelData, Particle};
use std::any::Any;

pub struct AppState<GpuModel: GpuParticleModelData = GpuParticleModel> {
    pub run_state: RunState,
    // pub render_config: RenderConfig,
    // pub gpu_render_config: GpuRenderConfig,
    pub pipeline: MpmPipeline<WebGpu, GpuModel>,
    // pub prep_vertex_buffer: WgPrepVertexBuffer,
    pub min_num_substeps: u32,
    pub max_num_substeps: u32,
    pub num_substeps: u32,
    pub gravity_factor: f32,
    pub restarting: bool,
    // pub hot_reload: HotReloadState,
    pub show_rigid_particles: bool,
    pub cell_width: f32,
    /// Optional per-particle RGBA colors. If `Some` and the length matches
    /// the particle count, these override the default index-cycled palette
    /// in the readback color buffer. Scene builders set this field on
    /// `&mut AppState` during `init()` to color particles by material.
    pub particle_colors: Option<Vec<nalgebra::Vector4<f32>>>,
    /// Enable the axis-aligned cutting box (3D only). When false, no filtering
    /// is applied and all particles render.
    #[cfg(feature = "dim3")]
    pub render_aabb_enabled: bool,
    /// Current cutting-box bounds (3D only). Particles outside this box are
    /// culled before being sent to the GPU.
    #[cfg(feature = "dim3")]
    pub render_aabb_min: nalgebra::Vector3<f32>,
    #[cfg(feature = "dim3")]
    pub render_aabb_max: nalgebra::Vector3<f32>,
    /// Slider range for the cutting-box sliders (3D only). Scene builders
    /// set this from their particle positions at init time so the sliders
    /// span the actual scene extent.
    #[cfg(feature = "dim3")]
    pub render_aabb_slider_min: nalgebra::Vector3<f32>,
    #[cfg(feature = "dim3")]
    pub render_aabb_slider_max: nalgebra::Vector3<f32>,
    /// Optional initial camera eye position (3D only). Set by the scene
    /// builder during `init()` to override the testbed's default arc-ball
    /// camera placement.
    #[cfg(feature = "dim3")]
    pub initial_camera_eye: Option<[f32; 3]>,
    /// Optional initial camera look-at target (3D only).
    #[cfg(feature = "dim3")]
    pub initial_camera_target: Option<[f32; 3]>,
    /// Optional initial 2D side-scroll camera center.
    #[cfg(feature = "dim2")]
    pub initial_camera2d_at: Option<[f32; 2]>,
    /// Optional initial 2D side-scroll camera zoom factor (higher = more
    /// zoomed in; default kiss3d zoom is 1.0).
    #[cfg(feature = "dim2")]
    pub initial_camera2d_zoom: Option<f32>,
}

#[derive(Default)]
pub struct RapierData {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub params: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub narrow_phase: NarrowPhase,
    pub broad_phase: DefaultBroadPhase,
    pub ccd_solver: CCDSolver,
    pub islands: IslandManager,
}

pub trait PhysicsCallback<GpuModel: GpuParticleModelData> {
    fn update(&mut self, state: &mut PhysicsState<'_, GpuModel>);
}

impl<GpuModel: GpuParticleModelData, F: FnMut(&mut PhysicsState<GpuModel>)>
    PhysicsCallback<GpuModel> for F
{
    fn update(&mut self, state: &mut PhysicsState<'_, GpuModel>) {
        (*self)(state);
    }
}

pub struct PhysicsState<'a, GpuModel: GpuParticleModelData = GpuParticleModel> {
    pub backend: &'a WebGpu,
    pub data: &'a mut MpmData<WebGpu, GpuModel>,
    pub results: &'a SimulationStepResult,
    pub(crate) step_id: usize,
}

impl<'a, GpuModel: GpuParticleModelData> PhysicsState<'a, GpuModel> {
    pub fn step_id(&self) -> usize {
        self.step_id
    }

    pub fn add_particles(&mut self, particles: &[Particle<GpuModel::Model>]) {
        self.data
            .particles
            .append(self.backend, particles)
            .expect("Failed to add particles.");
    }
}

pub struct PhysicsContext<GpuModel: GpuParticleModelData = GpuParticleModel> {
    pub data: MpmData<WebGpu, GpuModel>,
    pub rapier_data: RapierData,
    pub callbacks: Vec<Box<dyn PhysicsCallback<GpuModel>>>,
    pub hooks_state: Option<Box<dyn Any>>,
}

// #[derive(Default)]
// pub struct RenderContext {
//     pub instanced_materials: InstancedMaterials,
//     pub prefab_meshes: HashMap<ShapeType, Handle<Mesh>>,
//     pub rigid_entities: Vec<EntityWithGraphics>,
// }

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RunState {
    Running,
    Paused,
    Step,
}
