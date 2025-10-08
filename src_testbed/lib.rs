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
pub fn register_shaders(compiler: &mut SlangCompiler) {
    slosh::register_shaders(compiler);
    compiler.add_dir(SLANG_SRC_DIR.clone());
}

use crate::prep_readback::{GpuReadbackData, PrepReadback};
use crate::step::SimulationStepResult;
use kiss3d::planar_camera::Sidescroll;
use kiss3d::prelude::*;
use nexus::math::DIM;
use nexus::rapier::geometry::ShapeType;
use regex::Regex;
use slang_hal::backend::WebGpu;
use slang_hal::re_exports::include_dir;
use slang_hal::re_exports::minislang::SlangCompiler;
use slosh::pipeline::MpmPipeline;
use slosh::rapier::geometry::Shape;
use slosh::rapier::prelude::ColliderHandle;
use std::rc::Rc;
use wgpu::Limits;

type SceneBuilders = Vec<(String, SceneBuildFn)>;
type SceneBuildFn = fn(&WebGpu, &mut AppState) -> PhysicsContext;

#[cfg(feature = "dim2")]
type RenderNode = PlanarSceneNode;
#[cfg(feature = "dim3")]
type RenderNode = kiss3d::scene::SceneNode;

struct Stage {
    gpu: WebGpu,

    builders: SceneBuilders,
    physics: PhysicsContext,
    app_state: AppState,
    step_id: usize,

    step_result: SimulationStepResult,
    readback_shader: PrepReadback<WebGpu>,
    readback: GpuReadbackData<WebGpu>,
    #[cfg(feature = "dim2")]
    instances: Vec<PlanarInstanceData>,
    #[cfg(feature = "dim3")]
    instances: Vec<InstanceData>,
}

impl Stage {
    pub fn new(builders: SceneBuilders) -> Stage {
        let limits = Limits {
            max_storage_buffers_per_shader_stage: 10, // Did the default get bumped down to 8???!!
            ..Limits::default()
        };
        let mut gpu = futures::executor::block_on(WebGpu::new(Default::default(), limits)).unwrap();
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

        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);

        compiler.set_global_macro("DIM", DIM);

        let mpm_pipeline = MpmPipeline::new(&gpu, &compiler).unwrap();
        let mut app_state = AppState {
            pipeline: mpm_pipeline,
            run_state: RunState::Running,
            num_substeps: 1,
            gravity_factor: 1.0,
            restarting: false,
            show_rigid_particles: false,
        };
        let physics = (builders[0].1)(&gpu, &mut app_state);

        let readback_shader = PrepReadback::from_backend(&gpu, &compiler).unwrap();
        let readback = GpuReadbackData::new(&gpu, physics.data.particles.len()).unwrap();
        let mut step_result = SimulationStepResult::default();
        step_result
            .instances
            .resize(physics.data.particles.len(), Default::default());

        Stage {
            builders,
            instances: vec![],
            readback,
            readback_shader,
            gpu,
            physics,
            app_state,
            step_result,
            step_id: 0,
        }
    }

    pub fn set_demo(&mut self, demo_id: usize) {
        self.physics = (self.builders[demo_id]).1(&self.gpu, &mut self.app_state);
        self.readback = GpuReadbackData::new(&self.gpu, self.physics.data.particles.len()).unwrap();
        self.step_result
            .instances
            .resize(self.physics.data.particles.len(), Default::default());
    }

    fn update(&mut self) {
        if !self.step_simulation() {
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

pub fn run(scene_builders: SceneBuilders) {
    let mut colliders_gfx = HashMap::new();
    let mut stage = Stage::new(scene_builders);
    let mut window = Window::new("slosh - 3D testbed");
    render_colliders(&mut window, &stage.physics, &mut colliders_gfx);

    let font = Font::default();
    let font_sz = 40.0;

    window.set_light(Light::StickToCamera);

    #[cfg(feature = "dim2")]
    let mut c = window.add_rectangle(0.1, 0.1);
    #[cfg(feature = "dim3")]
    let mut c = window.add_cube(0.5, 0.5, 0.5);

    #[cfg(feature = "dim2")]
    let mut camera3d = FixedView::new();
    #[cfg(feature = "dim3")]
    let mut camera3d = ArcBall::new_with_frustum(
        std::f32::consts::PI / 4.0,
        0.1,
        1000.0,
        [40.0, 40.0, 40.0].into(), [0.0; 3].into()
    );
    let mut camera2d = Sidescroll::new();
    println!("CLIP planes: {:?}", camera3d.clip_planes());

    while !window.should_close() {
        /*
         * Step simulation.
         */
        stage.update();

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
        let mut offset = 0.0;
        for (i, builder) in stage.builders.iter().enumerate() {
            window.draw_text(
                &format!("{} - {} demo", i + 1, builder.0),
                &[0.0, offset].into(),
                font_sz,
                &font,
                &[0.8, 0.8, 0.8].into(),
            );
            offset += font_sz;
        }

        window.draw_text(
            &format!(
                "total: {:.1}ms (encoding: {:.1}ms)",
                stage.step_result.timings.total_step_time, stage.step_result.timings.encoding_time
            ),
            &[0.0, offset].into(),
            font_sz / 2.0,
            &font,
            &[0.8, 0.8, 0.8].into(),
        );
        offset += font_sz / 2.0;
        window.draw_text(
            &format!("readback: {:.1}ms", stage.step_result.timings.readback_time),
            &[0.0, offset].into(),
            font_sz / 2.0,
            &font,
            &[0.8, 0.8, 0.8].into(),
        );
        offset += font_sz / 2.0;
        window.draw_text(
            &format!("particles: {}", stage.physics.data.particles.len()),
            &[0.0, offset].into(),
            font_sz / 2.0,
            &font,
            &[0.8, 0.8, 0.8].into(),
        );

        /*
         * Handle events
         */
        for event in window.events().iter() {
            let mut new_selected_demo = None;

            if let WindowEvent::Key(button, Action::Release, _) = event.value {
                match button {
                    Key::Key1 => new_selected_demo = Some(0),
                    Key::Key2 => new_selected_demo = Some(1),
                    Key::Key3 => new_selected_demo = Some(2),
                    _ => {}
                }
            }

            if let Some(demo) = new_selected_demo {
                stage.set_demo(demo as usize);
                render_colliders(&mut window, &stage.physics, &mut colliders_gfx);
            }
        }

        /*
         * Render
         */
        window.render_with_cameras(&mut camera3d, &mut camera2d);
    }
}

fn update_colliders(
    window: &mut Window,
    physics: &PhysicsContext,
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

pub fn render_colliders(
    window: &mut Window,
    physics: &PhysicsContext,
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
fn generate_collider_mesh(co_shape: &dyn Shape) -> Option<Mesh> {
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
fn kiss3d_mesh(buffers: (Vec<nalgebra::Point3<f32>>, Vec<[u32; 3]>)) -> kiss3d::resource::Mesh {
    let (vtx, idx) = buffers;
    let kiss_vtx: Vec<_> = vtx
        .into_iter()
        .map(|pt| kiss3d::nalgebra::Point3::new(pt.x, pt.y, pt.z))
        .collect();
    let kiss_idx: Vec<_> = idx
        .into_iter()
        .map(|idx| kiss3d::nalgebra::Point3::new(idx[0], idx[1], idx[2]))
        .collect();
    Mesh::new(kiss_vtx, kiss_idx, None, None, false)
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
