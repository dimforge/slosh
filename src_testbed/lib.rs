#[cfg(feature = "dim2")]
pub extern crate nexus2d as nexus;
#[cfg(feature = "dim3")]
pub extern crate nexus3d as nexus;
#[cfg(feature = "dim2")]
pub extern crate slosh2d as slosh;
#[cfg(feature = "dim3")]
pub extern crate slosh3d as slosh;

pub use data::*;
use slang_hal::Shader;
use std::cell::RefCell;
use std::collections::HashMap;

mod data;
mod prep_readback;
mod step;

pub const SLANG_SRC_DIR: include_dir::Dir<'_> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../shaders_testbed");

#[cfg(feature = "runtime")]
pub fn register_shaders(compiler: &mut SlangCompiler) {
    slosh::register_shaders(compiler);
    compiler.add_dir(SLANG_SRC_DIR.clone());
}

use crate::prep_readback::{GpuReadbackData, PrepReadback, RenderMode};
use crate::step::SimulationStepResult;
use kiss3d::egui;
use kiss3d::planar_camera::Sidescroll;
use kiss3d::prelude::*;
#[cfg(feature = "dim3")]
use nalgebra::Vector3;
use nexus::rapier::geometry::ShapeType;
use regex::Regex;
use slang_hal::backend::{Backend, WebGpu};
use slang_hal::re_exports::include_dir;
use slang_hal::SlangCompiler;
use slosh::pipeline::{MpmPipeline, MpmPipelineHooks};
use slosh::rapier::geometry::Shape;
use slosh::rapier::prelude::ColliderHandle;
use slosh::solver::GpuParticleModelData;
use std::rc::Rc;
use wgpu::Limits;

type SceneBuilders<GpuModel> = Vec<(String, SceneBuildFn<GpuModel>)>;
type SceneBuildFn<GpuModel> = fn(&WebGpu, &mut AppState<GpuModel>) -> PhysicsContext<GpuModel>;

#[cfg(feature = "dim2")]
type RenderNode = PlanarSceneNode;
#[cfg(feature = "dim3")]
type RenderNode = kiss3d::scene::SceneNode;

struct Stage<GpuModel: GpuParticleModelData> {
    gpu: WebGpu,

    selected_demo: usize,
    builders: SceneBuilders<GpuModel>,
    physics: PhysicsContext<GpuModel>,
    hooks: Box<dyn MpmPipelineHooks<WebGpu, GpuModel>>,
    app_state: AppState<GpuModel>,
    step_id: usize,
    render_mode: RenderMode,

    step_result: SimulationStepResult,
    readback_shader: PrepReadback<WebGpu>,
    readback: GpuReadbackData<WebGpu>,
    #[cfg(feature = "dim2")]
    instances: Vec<PlanarInstanceData>,
    #[cfg(feature = "dim3")]
    instances: Vec<InstanceData>,
}

impl<GpuModel: GpuParticleModelData> Stage<GpuModel> {
    pub async fn new(
        #[allow(unused_mut)] // mut is needed for the `runtime` feature.
        mut compiler: SlangCompiler,
        hooks: impl FnOnce(&WebGpu, &SlangCompiler) -> Box<dyn MpmPipelineHooks<WebGpu, GpuModel>>,
        builders: SceneBuilders<GpuModel>,
    ) -> Stage<GpuModel> {
        let limits = Limits {
            max_storage_buffers_per_shader_stage: 11,
            max_compute_workgroup_storage_size: 32768, // Why do we need this if wgsparkl didn’t?
            max_buffer_size: 1_000_000_000,
            max_storage_buffer_binding_size: 1_000_000_000,
            ..Limits::default()
        };
        let mut gpu = WebGpu::new(Default::default(), limits).await.unwrap();
        // TODO: this is a terrible, horrible, hack, to work around the fact that slang isn’t giving us access to
        //       `exch.exchanged` to properly handle the _weak_ nature of `atomicCompareExchangeWeak̀
        let reg =
            Regex::new(r"(?<out>var.*)(?<exch>atomicCompareExchangeWeak.*).old_value;").unwrap();
        let replace = "\
            var exch = $exch;
            while (!exch.exchanged && exch.old_value == u32(4294967295)) {
                exch = $exch;
            }
            $out exch.old_value;
        ";
        gpu.append_hack(reg, replace.to_string());

        #[cfg(feature = "runtime")]
        {
            crate::register_shaders(&mut compiler);
            compiler.set_global_macro("DIM", nexus::math::DIM);
        }

        let mpm_pipeline = MpmPipeline::new(&gpu, &compiler).unwrap();
        let mut app_state = AppState {
            pipeline: mpm_pipeline,
            run_state: RunState::Paused,
            max_num_substeps: 1,
            min_num_substeps: 1,
            num_substeps: 1,
            gravity_factor: 1.0,
            restarting: false,
            show_rigid_particles: false,
        };
        let hooks = hooks(&gpu, &compiler);
        let physics = (builders[0].1)(&gpu, &mut app_state);
        app_state.num_substeps = 0; // Ensures it will be updated at the next step.

        let readback_shader = PrepReadback::from_backend(&gpu, &compiler).unwrap();
        let readback =
            GpuReadbackData::new(&gpu, physics.data.particles.len(), RenderMode::Default).unwrap();
        let mut step_result = SimulationStepResult::default();
        step_result
            .instances
            .resize(physics.data.particles.len(), Default::default());

        Stage {
            builders,
            instances: vec![],
            render_mode: RenderMode::Default,
            readback,
            readback_shader,
            gpu,
            physics,
            hooks,
            app_state,
            step_result,
            step_id: 0,
            selected_demo: 0,
        }
    }

    pub fn set_demo(&mut self, demo_id: usize) {
        self.selected_demo = demo_id;
        self.physics = (self.builders[demo_id]).1(&self.gpu, &mut self.app_state);
        self.readback = GpuReadbackData::new(
            &self.gpu,
            self.physics.data.particles.len(),
            self.render_mode,
        )
        .unwrap();
        self.app_state.num_substeps = 1; // Reset so it gets reinitialized automatically if needed.
        self.step_result
            .instances
            .resize(self.physics.data.particles.len(), Default::default());
    }

    async fn update(&mut self) {
        if !self.step_simulation().await {
            return;
        }

        self.instances.clear();
        #[cfg(feature = "dim2")]
        self.instances.extend(
            self.step_result
                .instances
                .iter()
                .map(|d| PlanarInstanceData {
                    position: kiss3d::nalgebra::Point2::new(d.position.x, d.position.y),
                    color: d.color.into(),
                    #[rustfmt::skip]
                    deformation: kiss3d::nalgebra::Matrix2::new(
                        d.deformation.m11, d.deformation.m12,
                        d.deformation.m21, d.deformation.m22
                    ),
                }),
        );
        #[cfg(feature = "dim3")]
        self.instances
            .extend(self.step_result.instances.iter().map(|d| InstanceData {
                position: kiss3d::nalgebra::Point3::new(d.position.x, d.position.y, d.position.z),
                color: d.color.into(),
                #[rustfmt::skip]
                    deformation: kiss3d::nalgebra::Matrix3::new(
                        d.deformation.m11, d.deformation.m12, d.deformation.m13,
                        d.deformation.m21, d.deformation.m22, d.deformation.m23,
                        d.deformation.m31, d.deformation.m32, d.deformation.m33,
                    ),
            }));
    }
}

pub async fn run<GpuModel: GpuParticleModelData>(scene_builders: SceneBuilders<GpuModel>) {
    run_with_compiler(SlangCompiler::default(), scene_builders).await
}

pub async fn run_with_compiler<GpuModel: GpuParticleModelData>(
    compiler: SlangCompiler,
    scene_builders: SceneBuilders<GpuModel>,
) {
    #[cfg(feature = "dim2")]
    {
        run_with_hooks(compiler, |_, _| Box::new(()), scene_builders).await;
    }
    #[cfg(feature = "dim3")]
    {
        run_with_hooks(compiler, |_, _| Box::new(()), scene_builders, Vector3::y()).await;
    }
}

pub async fn run_with_hooks<GpuModel: GpuParticleModelData>(
    compiler: SlangCompiler,
    hooks: impl FnOnce(&WebGpu, &SlangCompiler) -> Box<dyn MpmPipelineHooks<WebGpu, GpuModel>>,
    scene_builders: SceneBuilders<GpuModel>,
    #[cfg(feature = "dim3")] up_axis: Vector3<f32>,
) {
    let mut colliders_gfx = HashMap::new();
    let mut stage = Stage::new(compiler, hooks, scene_builders).await;
    let mut window = Window::new("slosh - 3D testbed");
    render_colliders(&mut window, &stage.physics, &mut colliders_gfx);

    window.set_light(Light::StickToCamera);

    #[cfg(feature = "dim2")]
    let mut c = window.add_rectangle(1.0, 1.0);
    #[cfg(feature = "dim3")]
    let mut c = window.add_cube(1.0, 1.0, 1.0);

    #[cfg(feature = "dim2")]
    let mut camera3d = FixedView::new();
    #[cfg(feature = "dim3")]
    let mut camera3d = ArcBall::new_with_frustum(
        std::f32::consts::PI / 4.0,
        0.1,
        1000.0,
        [40.0, 40.0, 40.0].into(),
        [0.0; 3].into(),
    );
    #[cfg(feature = "dim3")]
    {
        camera3d.set_up_axis(up_axis);
    }
    let mut camera2d = Sidescroll::new();

    while !window.should_close() {
        let mut new_selected_demo = None;

        /*
         * Step simulation.
         */
        stage.update().await;

        /*
         * Update rendering.
         */
        update_colliders(&mut window, &stage.physics, &mut colliders_gfx);
        c.data_mut()
            .get_object_mut()
            .set_instances(&stage.instances);

        /*
         * UI
         */
        // TODO: refactor in a different file
        window.draw_ui(|ctx| {
            kiss3d::egui::Window::new("Settings").show(ctx, |ui| {
                let mut changed = false;
                kiss3d::egui::ComboBox::from_label("selected sample")
                    .selected_text(&stage.builders[stage.selected_demo].0)
                    .show_ui(ui, |ui| {
                        for (i, (name, _)) in stage.builders.iter().enumerate() {
                            changed = ui
                                .selectable_value(&mut stage.selected_demo, i, name)
                                .changed()
                                || changed;
                        }
                    });
                if changed {
                    new_selected_demo = Some(stage.selected_demo);
                }

                let mut changed = false;
                egui::ComboBox::from_label("render mode")
                    .selected_text(stage.render_mode.text())
                    .show_ui(ui, |ui| {
                        for i in 0..6 {
                            let mode_i = RenderMode::from_u32(i);
                            changed = ui
                                .selectable_value(&mut stage.render_mode, mode_i, mode_i.text())
                                .changed()
                                || changed;
                        }
                    });

                if changed {
                    stage
                        .gpu
                        .write_buffer(
                            stage.readback.mode.buffer_mut(),
                            0,
                            bytemuck::bytes_of(&(stage.render_mode as u32)),
                        )
                        .unwrap();
                }

                ui.label(format!(
                    "total: {:.1}ms (encoding: {:.1}ms)",
                    stage.step_result.timings.total_step_time,
                    stage.step_result.timings.encoding_time
                ));
                ui.label(format!(
                    "readback: {:.1}ms",
                    stage.step_result.timings.readback_time
                ));
                ui.label(format!("particles: {}", stage.physics.data.particles.len()));

                ui.horizontal(|ui| {
                    let play_pause_label = if stage.app_state.run_state == RunState::Running {
                        "Pause"
                    } else {
                        "Play"
                    };
                    if ui.button(play_pause_label).clicked() {
                        if stage.app_state.run_state == RunState::Running {
                            stage.app_state.run_state = RunState::Paused;
                        } else {
                            stage.app_state.run_state = RunState::Running;
                        }
                    }
                    if ui.button("Step").clicked() {
                        stage.app_state.run_state = RunState::Step;
                    }
                    if ui.button("Restart").clicked() {
                        new_selected_demo = Some(stage.selected_demo);
                    }
                });
            });
        });

        if let Some(demo) = new_selected_demo {
            stage.set_demo(demo);
            render_colliders(&mut window, &stage.physics, &mut colliders_gfx);
        }

        /*
         * Render
         */
        window
            .render_with_cameras(&mut camera3d, &mut camera2d)
            .await;
    }
}

fn update_colliders<GpuModel: GpuParticleModelData>(
    window: &mut Window,
    physics: &PhysicsContext<GpuModel>,
    colliders: &mut HashMap<ColliderHandle, RenderNode>,
) {
    for (handle, node) in colliders {
        if let Some(collider) = physics.rapier_data.colliders.get(*handle) {
            let pose = collider.position();

            #[cfg(feature = "dim3")]
            {
                // TODO: here we are converting between nalgebra versions.
                //       This can be simplified once kiss3d is updated to the latest nalgebra.
                let tra = pose.translation.vector;
                let rot = pose.rotation.into_inner();
                node.set_local_translation([tra.x, tra.y, tra.z].into());
                node.set_local_rotation(kiss3d::nalgebra::Unit::new_unchecked(
                    kiss3d::nalgebra::Quaternion::new(rot.w, rot.i, rot.j, rot.k),
                ));
            }
            #[cfg(feature = "dim2")]
            {
                let tra = pose.translation.vector;
                let rot = pose.rotation.into_inner();
                node.set_local_translation([tra.x, tra.y].into());
                node.set_local_rotation(kiss3d::nalgebra::Unit::new_unchecked(
                    kiss3d::nalgebra::Complex::new(rot.re, rot.im),
                ));
            }
        } else {
            #[cfg(feature = "dim2")]
            window.remove_planar_node(node);
            #[cfg(feature = "dim3")]
            window.remove_node(node);
        }
    }
}

pub fn render_colliders<GpuModel: GpuParticleModelData>(
    window: &mut Window,
    physics: &PhysicsContext<GpuModel>,
    colliders: &mut HashMap<ColliderHandle, RenderNode>,
) {
    for (_, mut node) in colliders.drain() {
        #[cfg(feature = "dim2")]
        window.remove_planar_node(&mut node);
        #[cfg(feature = "dim3")]
        window.remove_node(&mut node);
    }

    for (handle, collider) in physics.rapier_data.colliders.iter() {
        if let Some(mesh) = generate_collider_mesh(collider.shape()) {
            #[cfg(feature = "dim2")]
            let node = window.add_planar_mesh(Rc::new(RefCell::new(mesh)), [1.0; 2].into());
            #[cfg(feature = "dim3")]
            let node = window.add_mesh(Rc::new(RefCell::new(mesh)), [1.0; 3].into());
            colliders.insert(handle, node);
        }
    }
}

#[cfg(feature = "dim2")]
fn generate_collider_mesh(co_shape: &dyn Shape) -> Option<PlanarMesh> {
    let mesh = match co_shape.shape_type() {
        ShapeType::Cuboid => {
            let cuboid = co_shape.as_cuboid().unwrap();
            kiss3d_mesh_from_polyline(cuboid.to_polyline())
        }
        ShapeType::Ball => {
            let ball = co_shape.as_ball().unwrap();
            kiss3d_mesh_from_polyline(ball.to_polyline(40))
        }
        ShapeType::Capsule => {
            let capsule = co_shape.as_capsule().unwrap();
            kiss3d_mesh_from_polyline(capsule.to_polyline(40))
        }
        ShapeType::Triangle => {
            let tri = co_shape.as_triangle().unwrap();
            kiss3d_mesh_from_polyline(vec![tri.a, tri.b, tri.c])
        }
        ShapeType::TriMesh => {
            let trimesh = co_shape.as_trimesh().unwrap();
            kiss3d_mesh((trimesh.vertices().to_vec(), trimesh.indices().to_vec()))
        }
        ShapeType::Voxels => {
            let mut vtx = vec![];
            let mut idx = vec![];
            let voxels = co_shape.as_voxels().unwrap();
            let sz = voxels.voxel_size() / 2.0;
            for vox in voxels.voxels() {
                if !vox.state.is_empty() {
                    let bid = vtx.len() as u32;
                    let center = nalgebra::point![vox.center.x, vox.center.y];
                    vtx.push(center + nalgebra::vector![sz.x, sz.y]);
                    vtx.push(center + nalgebra::vector![-sz.x, sz.y]);
                    vtx.push(center + nalgebra::vector![-sz.x, -sz.y]);
                    vtx.push(center + nalgebra::vector![sz.x, -sz.y]);
                    idx.push([bid, bid + 1, bid + 2]);
                    idx.push([bid + 2, bid + 3, bid]);
                }
            }

            kiss3d_mesh((vtx, idx))
        }
        // ShapeType::Polyline => {
        //     let polyline = co_shape.as_polyline().unwrap();
        //     bevy_polyline((
        //         polyline.vertices().to_vec(),
        //         Some(polyline.indices().to_vec()),
        //     ))
        // }
        // ShapeType::HeightField => {
        //     let heightfield = co_shape.as_heightfield().unwrap();
        //     let vertices: Vec<_> = heightfield
        //         .segments()
        //         .flat_map(|s| vec![s.a, s.b])
        //         .collect();
        //     bevy_polyline((vertices, None))
        // }
        ShapeType::ConvexPolygon => {
            let poly = co_shape.as_convex_polygon().unwrap();
            kiss3d_mesh_from_polyline(poly.points().to_vec())
        }
        ShapeType::RoundConvexPolygon => {
            let poly = co_shape.as_round_convex_polygon().unwrap();
            kiss3d_mesh_from_polyline(poly.inner_shape.points().to_vec())
        }
        _ => return None,
    };

    Some(mesh)
}

#[cfg(feature = "dim2")]
fn kiss3d_mesh_from_polyline(vertices: Vec<nalgebra::Point2<f32>>) -> PlanarMesh {
    let n = vertices.len();
    let idx = (1..n as u32 - 1).map(|i| [0, i, i + 1]).collect();
    kiss3d_mesh((vertices, idx))
}

#[cfg(feature = "dim3")]
fn generate_collider_mesh(co_shape: &dyn Shape) -> Option<GpuMesh> {
    let mesh = match co_shape.shape_type() {
        ShapeType::Ball => {
            let ball = co_shape.as_ball().unwrap();
            kiss3d_mesh(ball.to_trimesh(10, 10))
        }
        ShapeType::Cuboid => {
            let cuboid = co_shape.as_cuboid().unwrap();
            kiss3d_mesh(cuboid.to_trimesh())
        }
        ShapeType::Capsule => {
            let capsule = co_shape.as_capsule().unwrap();
            kiss3d_mesh(capsule.to_trimesh(20, 10))
        }
        ShapeType::Triangle => {
            let tri = co_shape.as_triangle().unwrap();
            kiss3d_mesh((vec![tri.a, tri.b, tri.c], vec![[0, 1, 2], [0, 2, 1]]))
        }
        ShapeType::TriMesh => {
            let trimesh = co_shape.as_trimesh().unwrap();
            kiss3d_mesh((trimesh.vertices().to_vec(), trimesh.indices().to_vec()))
        }
        ShapeType::HeightField => {
            let heightfield = co_shape.as_heightfield().unwrap();
            kiss3d_mesh(heightfield.to_trimesh())
        }
        ShapeType::ConvexPolyhedron => {
            let poly = co_shape.as_convex_polyhedron().unwrap();
            kiss3d_mesh(poly.to_trimesh())
        }
        ShapeType::RoundConvexPolyhedron => {
            let poly = co_shape.as_round_convex_polyhedron().unwrap();
            kiss3d_mesh(poly.inner_shape.to_trimesh())
        }
        ShapeType::Voxels => {
            let voxels = co_shape.as_voxels().unwrap();
            kiss3d_mesh(voxels.to_trimesh())
        }
        _ => return None,
    };

    Some(mesh)
}

#[cfg(feature = "dim3")]
fn kiss3d_mesh(buffers: (Vec<nalgebra::Point3<f32>>, Vec<[u32; 3]>)) -> kiss3d::resource::GpuMesh {
    let (vtx, idx) = buffers;
    let kiss_vtx: Vec<_> = vtx
        .into_iter()
        .map(|pt| kiss3d::nalgebra::Point3::new(pt.x, pt.y, pt.z))
        .collect();
    let kiss_idx: Vec<_> = idx
        .into_iter()
        .map(|idx| kiss3d::nalgebra::Point3::new(idx[0], idx[1], idx[2]))
        .collect();
    GpuMesh::new(kiss_vtx, kiss_idx, None, None, false)
}

#[cfg(feature = "dim2")]
fn kiss3d_mesh(
    buffers: (Vec<nalgebra::Point2<f32>>, Vec<[u32; 3]>),
) -> kiss3d::resource::PlanarMesh {
    let (vtx, idx) = buffers;
    let kiss_vtx: Vec<_> = vtx
        .into_iter()
        .map(|pt| kiss3d::nalgebra::Point2::new(pt.x, pt.y))
        .collect();
    let kiss_idx: Vec<_> = idx
        .into_iter()
        .map(|idx| kiss3d::nalgebra::Point3::new(idx[0], idx[1], idx[2]))
        .collect();
    PlanarMesh::new(kiss_vtx, kiss_idx, None, false)
}
