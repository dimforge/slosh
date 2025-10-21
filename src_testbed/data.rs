use slang_hal::backend::{Backend, WebGpu};
use slosh::pipeline::{MpmData, MpmPipeline};
use slosh::rapier::prelude::{
    CCDSolver, ColliderSet, DefaultBroadPhase, ImpulseJointSet, IntegrationParameters,
    IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, RigidBodySet,
};
use slosh::solver::{DefaultGpuParticleModel, GpuParticleModel, Particle};

pub struct AppState<GpuModel: GpuParticleModel = DefaultGpuParticleModel> {
    pub run_state: RunState,
    // pub render_config: RenderConfig,
    // pub gpu_render_config: GpuRenderConfig,
    pub pipeline: MpmPipeline<WebGpu, GpuModel>,
    // pub prep_vertex_buffer: WgPrepVertexBuffer,
    pub num_substeps: usize,
    pub gravity_factor: f32,
    pub restarting: bool,
    // pub hot_reload: HotReloadState,
    pub show_rigid_particles: bool,
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

pub trait PhysicsCallback<GpuModel: GpuParticleModel> {
    fn update(&mut self, state: &mut PhysicsState<'_, GpuModel>);
}

impl<GpuModel: GpuParticleModel, F: FnMut(&mut PhysicsState<GpuModel>)> PhysicsCallback<GpuModel> for F {
    fn update(&mut self, state: &mut PhysicsState<'_, GpuModel>) {
        (*self)(state);
    }
}

pub struct PhysicsState<'a, GpuModel: GpuParticleModel = DefaultGpuParticleModel> {
    pub(crate) backend: &'a WebGpu,
    pub(crate) data: &'a mut MpmData<WebGpu, GpuModel>,
    pub(crate) step_id: usize,
}

impl<'a, GpuModel: GpuParticleModel> PhysicsState<'a, GpuModel> {
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

pub struct PhysicsContext<GpuModel: GpuParticleModel = DefaultGpuParticleModel> {
    pub data: MpmData<WebGpu, GpuModel>,
    pub rapier_data: RapierData,
    pub callbacks: Vec<Box<dyn PhysicsCallback<GpuModel>>>,
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
