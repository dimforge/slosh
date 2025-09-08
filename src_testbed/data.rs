use slang_hal::backend::WebGpu;
use slosh::pipeline::{MpmData, MpmPipeline};
use slosh::rapier::prelude::{
    CCDSolver, ColliderSet, DefaultBroadPhase, ImpulseJointSet, IntegrationParameters,
    IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, RigidBodySet,
};
use slosh::solver::Particle;

pub struct AppState {
    pub run_state: RunState,
    // pub render_config: RenderConfig,
    // pub gpu_render_config: GpuRenderConfig,
    pub pipeline: MpmPipeline<WebGpu>,
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

pub struct PhysicsContext {
    pub data: MpmData<WebGpu>,
    pub rapier_data: RapierData,
    pub particles: Vec<Particle>,
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
