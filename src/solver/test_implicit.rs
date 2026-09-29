//! Headless checks of the implicit grid solver.
//!
//! The operator test applies `A = M + dt^2 K` to random grid vectors through the gather/scatter
//! kernels and checks symmetry and positive definiteness, which is what the conjugate gradient
//! relies on. The stability test drops a stiff jelly cube on a floor at a substep length far
//! beyond the sound-speed CFL bound, with the explicit and the implicit integrators. Run with
//! `cargo test -p slosh3d --features runtime,webgpu test_implicit -- --nocapture --ignored`.

#[cfg(feature = "dim3")]
mod scenes {
    use crate::pipeline::{MpmData, MpmPipeline, MpmPipelineKernels};
    use crate::rapier::prelude::*;
    use crate::solver::implicit::GpuCgNode;
    use crate::solver::{
        CgScalars, GpuBoundaryCondition, GpuParticleModel, ImplicitSolverParams, MpmIntegrator,
        Particle, ParticleModel, ParticlePosition, SimulationParams,
    };
    use glam::{Vec3, vec3};
    use regex::Regex;
    use slang_hal::backend::{Backend, Encoder, WebGpu};
    use slang_hal::{BufferUsages, SlangCompiler};
    use stensor::tensor::GpuTensor;
    use wgpu::Limits;

    type Data = MpmData<WebGpu, GpuParticleModel>;

    async fn gpu_and_compiler() -> (WebGpu, SlangCompiler) {
        let limits = Limits {
            max_storage_buffers_per_shader_stage: 13,
            max_compute_workgroup_storage_size: 32768,
            max_buffer_size: 4_000_000_000,
            max_storage_buffer_binding_size: 4_000_000_000,
            ..Limits::default()
        };
        let mut gpu = WebGpu::new(Default::default(), limits).await.unwrap();
        // Same workaround as the testbed for the weak compare-exchange of the grid hashmap.
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
        let mut compiler = SlangCompiler::default();
        crate::register_shaders(&mut compiler);
        compiler.set_global_macro("DIM", "3");
        (gpu, compiler)
    }

    const CELL_WIDTH: f32 = 0.5;
    const CUBE_SIDE: f32 = 3.0;
    const DENSITY: f32 = 1000.0;

    /// A cube of the given material hovering above a fixed floor.
    fn build_data(gpu: &WebGpu, model: ParticleModel, dt: f32) -> Data {
        let radius = CELL_WIDTH / 4.0;
        let spacing = radius * 2.0;
        let n = (CUBE_SIDE / spacing) as i32;
        let mut particles = vec![];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let pos = vec3(i as f32, j as f32, k as f32) * spacing
                        + vec3(-CUBE_SIDE / 2.0, 0.5, -CUBE_SIDE / 2.0)
                        + Vec3::splat(spacing / 2.0);
                    particles.push(Particle::new(pos, radius, DENSITY, model));
                }
            }
        }

        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let floor = bodies.insert(RigidBodyBuilder::fixed().translation(vec3(0.0, -1.0, 0.0)));
        let floor_collider = colliders.insert_with_parent(
            ColliderBuilder::cuboid(10.0, 1.0, 10.0),
            floor,
            &mut bodies,
        );
        let params = SimulationParams {
            gravity: vec3(0.0, -9.81, 0.0),
            dt,
        };
        MpmData::new(
            gpu,
            params,
            &particles,
            &bodies,
            &colliders,
            &[(floor_collider, GpuBoundaryCondition::separate(0.5))],
            CELL_WIDTH,
            4096,
        )
        .unwrap()
    }

    fn pipeline(gpu: &WebGpu, compiler: &SlangCompiler) -> MpmPipeline<WebGpu, GpuParticleModel> {
        let kernels = MpmPipelineKernels {
            implicit: true,
            ..MpmPipelineKernels::default()
        };
        MpmPipeline::new_with_kernels(gpu, compiler, kernels).unwrap()
    }

    async fn step(gpu: &WebGpu, pipeline: &MpmPipeline<WebGpu, GpuParticleModel>, data: &mut Data) {
        let mut encoder = gpu.begin_encoding();
        let mut hooks = ();
        let mut hooks_state: Box<dyn std::any::Any> = Box::new(());
        pipeline
            .launch_step(gpu, &mut encoder, data, &mut hooks, &mut *hooks_state, None)
            .await
            .unwrap();
        gpu.submit(encoder).unwrap();
    }

    async fn read_positions(gpu: &WebGpu, data: &Data) -> Vec<Vec3> {
        let n = data.particles.len();
        let mut staging: GpuTensor<ParticlePosition, WebGpu> = GpuTensor::vector_uninit(
            gpu,
            n as u32,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        )
        .unwrap();
        let mut encoder = gpu.begin_encoding();
        staging
            .copy_from_view(&mut encoder, data.particles.positions())
            .unwrap();
        gpu.submit(encoder).unwrap();
        gpu.synchronize().unwrap();
        let mut positions = vec![ParticlePosition::ZERO; n];
        gpu.read_buffer(staging.buffer(), &mut positions)
            .await
            .unwrap();
        positions.iter().map(|p| vec3(p.x, p.y, p.z)).collect()
    }

    async fn read_cg_nodes(gpu: &WebGpu, data: &Data) -> Vec<GpuCgNode> {
        gpu.synchronize().unwrap();
        gpu.slow_read_vec(data.implicit.cg_nodes.buffer())
            .await
            .unwrap()
    }

    /// Tiny deterministic generator for the test vectors.
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn dot(a: &[Vec3], b: &[Vec3]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x.dot(*y) as f64).sum()
    }

    /// Uploads `dir` as the search direction, applies the operator, and reads back `A dir`.
    async fn apply_operator(
        gpu: &WebGpu,
        pipeline: &MpmPipeline<WebGpu, GpuParticleModel>,
        data: &mut Data,
        template: &[GpuCgNode],
        dir: &[Vec3],
    ) -> Vec<Vec3> {
        let mut nodes = template.to_vec();
        for (node, d) in nodes.iter_mut().zip(dir) {
            node.p = *d;
        }
        gpu.write_buffer(data.implicit.cg_nodes.buffer_mut(), 0, &nodes)
            .unwrap();

        let mut encoder = gpu.begin_encoding();
        {
            let mut pass = encoder.begin_pass("apply", None);
            pipeline
                .implicit_solver()
                .unwrap()
                .launch_apply(
                    gpu,
                    &mut pass,
                    &data.sim_params,
                    &data.grid,
                    &data.particles,
                    &data.body_materials,
                    &data.implicit,
                )
                .unwrap();
        }
        gpu.submit(encoder).unwrap();
        gpu.synchronize().unwrap();
        let out = read_cg_nodes(gpu, data).await;
        out[..dir.len()].iter().map(|n| n.ap).collect()
    }

    #[futures_test::test]
    #[serial_test::serial]
    #[ignore]
    async fn test_implicit_operator() {
        let (gpu, compiler) = gpu_and_compiler().await;
        let pipeline = pipeline(&gpu, &compiler);
        let model = ParticleModel::elastic_neo_hookean(1.0e6, 0.3);
        let mut data = build_data(&gpu, model, 1.0 / 60.0);
        // One explicit step sorts the particles and fills the grid; the implicit flag keeps the
        // stress out of the affine matrices so the nodes hold the force-free velocity.
        data.integrator = MpmIntegrator::Implicit(ImplicitSolverParams::default());
        data.sim_params.set_implicit(&gpu, true).unwrap();
        data.integrator = MpmIntegrator::Explicit;
        step(&gpu, &pipeline, &mut data).await;
        gpu.synchronize().unwrap();

        let mut encoder = gpu.begin_encoding();
        {
            let mut pass = encoder.begin_pass("prepare", None);
            pipeline
                .implicit_solver()
                .unwrap()
                .launch_prepare(
                    &gpu,
                    &mut pass,
                    &data.sim_params,
                    &data.grid,
                    &data.particles,
                    &data.body_materials,
                    &mut data.implicit,
                )
                .unwrap();
        }
        gpu.submit(encoder).unwrap();
        gpu.synchronize().unwrap();

        let num_nodes = data.grid.num_active_blocks(&gpu).await as usize * 64;
        let template = read_cg_nodes(&gpu, &data).await;
        assert!(num_nodes <= template.len());
        let template = &template[..num_nodes];
        let num_massive = template.iter().filter(|n| n.mass > 0.0).count();
        assert!(num_massive > 100, "only {num_massive} massive nodes");

        let mut rng = Lcg(7);
        let random_dir = |rng: &mut Lcg| -> Vec<Vec3> {
            template
                .iter()
                .map(|n| {
                    if n.mass > 0.0 {
                        vec3(rng.next_f32(), rng.next_f32(), rng.next_f32())
                    } else {
                        Vec3::ZERO
                    }
                })
                .collect()
        };
        let u = random_dir(&mut rng);
        let v = random_dir(&mut rng);
        let au = apply_operator(&gpu, &pipeline, &mut data, template, &u).await;
        let av = apply_operator(&gpu, &pipeline, &mut data, template, &v).await;

        let uav = dot(&u, &av);
        let vau = dot(&v, &au);
        let scale = uav.abs().max(vau.abs());
        assert!(
            (uav - vau).abs() <= 1.0e-3 * scale,
            "operator is not symmetric: <u, A v> = {uav}, <v, A u> = {vau}"
        );
        let umu: f64 = u
            .iter()
            .zip(template)
            .map(|(x, n)| (x.dot(*x) * n.mass) as f64)
            .sum();
        let uau = dot(&u, &au);
        assert!(
            uau >= umu * (1.0 - 1.0e-4),
            "<u, A u> = {uau} < <u, M u> = {umu}"
        );
        assert!(
            uau > umu * 1.01,
            "stiffness contributes nothing: {uau} vs {umu}"
        );
        println!(
            "OK: nodes={num_nodes} massive={num_massive} <u,Av>={uav:.6e} <v,Au>={vau:.6e} <u,Au>/<u,Mu>={:.3}",
            uau / umu
        );
    }

    /// Drops the cube and reports the largest distance from the cube's initial center, whether all
    /// positions are finite, and the last solver scalars.
    async fn run_drop(
        gpu: &WebGpu,
        pipeline: &MpmPipeline<WebGpu, GpuParticleModel>,
        model: ParticleModel,
        integrator: MpmIntegrator,
        substeps: u32,
        frames: u32,
    ) -> (f32, bool, CgScalars) {
        let dt = 1.0 / 60.0 / substeps as f32;
        let mut data = build_data(gpu, model, dt);
        data.integrator = integrator;
        for _ in 0..frames * substeps {
            step(gpu, pipeline, &mut data).await;
        }
        gpu.synchronize().unwrap();
        let positions = read_positions(gpu, &data).await;
        let center = vec3(0.0, 0.5 + CUBE_SIDE / 2.0, 0.0);
        let finite = positions.iter().all(|p| p.is_finite());
        let extent = positions
            .iter()
            .filter(|p| p.is_finite())
            .map(|p| (*p - center).length())
            .fold(0.0f32, f32::max);
        let scalars = data.implicit.read_scalars(gpu).await.unwrap();
        (extent, finite, scalars)
    }

    #[futures_test::test]
    #[serial_test::serial]
    #[ignore]
    async fn test_implicit_stability() {
        let (gpu, compiler) = gpu_and_compiler().await;
        let pipeline = pipeline(&gpu, &compiler);
        let newton = MpmIntegrator::Implicit(ImplicitSolverParams {
            max_cg_iters: 40,
            ..ImplicitSolverParams::default()
        });
        // 2 substeps at E = 2e7 Pa puts the substep ~3x past the sound-speed CFL bound.
        for (name, model) in [
            (
                "neo-hookean",
                ParticleModel::elastic_neo_hookean(2.0e7, 0.3),
            ),
            ("corotated", ParticleModel::elastic(2.0e7, 0.3)),
            ("sand", ParticleModel::sand(1.0e7, 0.3)),
        ] {
            let (extent, finite, sc) = run_drop(&gpu, &pipeline, model, newton, 2, 60).await;
            println!(
                "{name} implicit: extent={extent:.3} finite={finite} newton_iters={} |g|/|g0|={:.3e} cg_iters={} energy={:.4e}",
                sc.newton_iters,
                (sc.g_sq / sc.g0_sq.max(f32::MIN_POSITIVE)).sqrt(),
                sc.cg_iters,
                sc.energy
            );
            assert!(finite, "{name}: non-finite positions");
            assert!(extent < 4.0, "{name}: blew up, extent {extent}");
            assert!(sc.newton_iters >= 1, "{name}: the solve never iterated");
            assert!(sc.energy.is_finite(), "{name}: non-finite energy");
        }

        let (extent, finite, _) = run_drop(
            &gpu,
            &pipeline,
            ParticleModel::elastic_neo_hookean(2.0e7, 0.3),
            MpmIntegrator::Explicit,
            2,
            60,
        )
        .await;
        println!("neo-hookean explicit: extent={extent:.3} finite={finite}");
    }
}

use slang_hal::SlangCompiler;

/// Prints the resource bindings of every implicit kernel as compiled to WGSL, to catch
/// parameters the reflection lists but the generated shader never references.
#[futures_test::test]
#[ignore]
async fn test_implicit_dump_bindings() {
    let mut compiler = SlangCompiler::default();
    crate::register_shaders(&mut compiler);
    compiler.set_global_macro("DIM", crate::math::DIM);
    let spec = ["slosh/models/specializations".to_string()];
    for (module, entries) in [
        (
            "slosh/solver/implicit/gather",
            vec!["gather_operator", "gather_residual"],
        ),
        (
            "slosh/solver/implicit/scatter",
            vec!["scatter_operator", "scatter_residual"],
        ),
        (
            "slosh/solver/implicit/cg",
            vec![
                "cg_reset",
                "prepare_particles",
                "seed_from_grid",
                "fixed_nodes",
                "apply_boundary_conditions",
                "cg_init_residual",
                "cg_update_solution",
                "cg_update_direction",
                "ls_prepare",
                "ls_trial",
                "ls_finalize",
                "energy_particles",
                "cg_reduce_init",
                "cg_reduce_alpha",
                "cg_reduce_beta",
                "newton_check",
                "ls_reduce_prepare",
                "ls_reduce_energy",
                "ls_reduce_residual",
                "init_energy",
            ],
        ),
    ] {
        for entry in entries {
            let program = compiler.compile(
                module,
                slang_hal::backend::CompileTarget::Wgsl.into(),
                Some(entry),
                &spec,
                &[],
            );
            let code = program.target_code(0).unwrap();
            let wgsl = String::from_utf8_lossy(code.as_slice());
            let bindings: Vec<_> = wgsl
                .lines()
                .filter(|l| l.contains("@binding"))
                .map(|l| l.trim().to_string())
                .collect();
            let params = program
                .layout(0)
                .unwrap()
                .find_entry_point_by_name(entry)
                .unwrap()
                .parameters()
                .filter(|p| p.semantic_name().is_none())
                .count();
            println!(
                "{entry}: reflection={params} wgsl_bindings={}",
                bindings.len()
            );
            for b in bindings {
                println!("    {b}");
            }
        }
    }
}
