# Unreleased
- Add an implicit grid solver, selected per simulation with `MpmData::integrator =
  MpmIntegrator::Implicit(ImplicitSolverParams { .. })` on a pipeline built with
  `MpmPipelineKernels { implicit: true, .. }` (off by default). It replaces the explicit
  momentum-to-velocity update by backward Euler solved with Newton's method on the incremental
  potential, entirely on the GPU: each Newton iteration evaluates the trial state of every particle
  (deformation gradient projected for plastic models, stress, energy), solves for the direction with
  a matrix-free, mass-preconditioned conjugate gradient on the definiteness-fixed Hessian, and picks
  the step with a backtracking line search (on the incremental potential, or on the residual norm
  for models without an energy). This lifts the sound-speed bound on the substep length: stiff
  elastics run with a handful of substeps per frame. `ImplicitSolverParams::semi_implicit()` gives
  the one-linearization variant. Grid-level stick/slip conditions and fixed particles are Dirichlet
  conditions inside the solve; separating contacts stay explicit corrections.
- Constitutive models take part in the implicit solve through the new `IImplicitParticleModel`
  slang interface (`slosh/models/implicit_interfaces.slang`): a trial-state evaluation and a stress
  differential. The default models implement it, and `LinearElasticModel` / `NeoHookeanModel`
  gained an `energy_density`. A custom model specialization must export an
  `ImplicitParticleModel` alongside `ParticleModel` to compile the implicit kernels.
- The particle update reads a new `IntegratorFlags` uniform (`GpuSimulationParams::integrator_flags`)
  and leaves the stress out of the APIC affine matrix when the implicit solver is active. A `run_g2p`
  hook that fuses the particle update must do the same.
- The testbeds compile the implicit kernels and expose the integrator and its parameters in the
  settings window, with the last solve's Newton and CG statistics.
- The testbed settings window exposes the substep count: a fixed number, or the adaptive
  `[min, max]` range (scenes still set their defaults on restart).

# v0.8.0
- Add the `GpuBoundaryCondition::non_reflecting` (absorbing) boundary condition, based on
  Lysmer-Kuhlemeyer viscous dashpots. It lets outgoing elastic waves leave the domain instead of
  being reflected back into it, emulating an unbounded medium. See the new `non_reflecting2` 2D
  demo for a side-by-side comparison with a reflecting boundary.
- Add `ParticleModel::absorbing_pml`, a perfectly-matched-layer absorbing material after
  Kurima, Chandra & Soga (arXiv:2407.02790). Particles carrying it form a layer around the region
  of interest whose coordinates are stretched, so outgoing waves slow and spread instead of
  returning; pair it with `ParticleDynamics::damping` over the same particles to dissipate them.
  `models::pml_stretch` computes the per-particle stretch from the layer geometry. It absorbs
  better than the dashpot boundary above (~0.2% vs ~1% of a reflecting wall's residual motion on
  a 2D impulse test) at the cost of the extra particles the layer needs.
- The PML and the two items below sit behind the new `pml` cargo feature, off by default, mirrored
  in the shaders as `SLOSH_PML`. It shifts `DefaultParticleModelType::LEN` (5 on, 4 off).
- With `pml` on, grid nodes carry a per-direction mass (`Node.directional_mass`), so a material can
  rescale its own inertia per axis via `ModelUpdateResult::mass_scale`. The momentum update divides
  by it while gravity keeps acting on the real mass. Ordinary materials report a scale of one. A
  P2G hook replacing the built-in transfer must write this field too.
- Add `ParticleDynamics::stiffness_damping`, the stiffness-proportional half of Rayleigh damping.
  It adds a viscous stress `a_K * C : sym(grad v)` from each material's own elastic tensor, so
  unlike the mass-proportional `damping` it is blind to rigid-body motion. Supported by every
  built-in model. It tightens the explicit stability bound, which `WgTimestepBounds` accounts for.
- Grid nodes now get a collision reported up to `COLLISION_REPORT_CELLS` (6.5) cells from a
  collider instead of 1.5, so the absorbing boundary above can grade its damping over a band
  several cells deep. The contact boundary conditions gate themselves on the distance to the
  surface and are unaffected.
- Update to Rapier 0.32. This migrates most public APIs and internals to use `glam` instead of `nalgebra`.
- Fix a GPU validation error / panic on simulations with more than ~4.19M particles, caused by
  compute kernels dispatching more than 65535 workgroups along a single dimension. The affected 
  kernels now clamp the dispatch and grid-stride over the particles.

# v0.2.0 (27 Oct. 2025)
- Add support for dynamic particle insertion.
- Add support for specializing the particle update logic using slang’s link-time specializaiton feature.
- Update dependencies.