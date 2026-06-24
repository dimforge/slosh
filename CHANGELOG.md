# Unreleased
- Fix a GPU validation error / panic on simulations with more than ~4.19M particles, caused by
  compute kernels dispatching more than 65535 workgroups along a single dimension. The affected 
  kernels now clamp the dispatch and grid-stride over the particles.

# v0.2.0 (27 Oct. 2025)
- Add support for dynamic particle insertion.
- Add support for specializing the particle update logic using slang’s link-time specializaiton feature.
- Update dependencies.