# Unreleased
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
- Grid nodes now carry a per-direction mass (`Node.directional_mass`) alongside the scalar one, so
  a material can rescale its own inertia per axis via `ModelUpdateResult::mass_scale`. The
  momentum update divides by it while gravity keeps acting on the real mass. Ordinary materials
  report a scale of one and are unaffected. This is what lets the absorbing PML layer above hold
  itself up under gravity. A P2G hook replacing the built-in transfer must write this field too.
- Add `ParticleDynamics::stiffness_damping`, the stiffness-proportional half of Rayleigh damping.
  It adds a viscous stress `a_K * C : sym(grad v)` using each material's own elastic tensor, so
  unlike the existing mass-proportional `damping` it is blind to rigid-body motion: a body in free
  flight or a domain shaken at its base keeps moving while its vibrations are damped. Supported by
  every built-in model. It tightens the explicit stability bound, which `WgTimestepBounds` now
  accounts for.
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