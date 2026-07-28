# Unreleased
- Add the `GpuBoundaryCondition::non_reflecting` (absorbing) boundary condition, based on
  Lysmer-Kuhlemeyer viscous dashpots. It lets outgoing elastic waves leave the domain instead of
  being reflected back into it, emulating an unbounded medium. See the new `non_reflecting2` 2D
  demo for a side-by-side comparison with a reflecting boundary.
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