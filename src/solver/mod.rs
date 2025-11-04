//! Core MPM solver algorithms and GPU kernels.
//!
//! This module implements the Material Point Method algorithm stages that run on the GPU:
//!
//! # MPM Algorithm Overview
//!
//! The MPM algorithm alternates between two representations of the simulation:
//! - **Particles** (Lagrangian): Track material points with mass, position, velocity, and deformation
//! - **Grid** (Eulerian): Temporary background grid for computing forces and velocities
//!
//! Each simulation step executes these phases:
//!
//! 1. **Particle-to-Grid (P2G)**: Transfer particle momentum and mass to nearby grid nodes
//!    - Implemented by [`WgP2G`] and [`WgP2GCdf`] (for rigid body coupling)
//! 2. **Grid Update**: Compute forces, update velocities, apply boundary conditions
//!    - Implemented by [`WgGridUpdate`] and [`WgGridUpdateCdf`]
//! 3. **Grid-to-Particle (G2P)**: Interpolate grid velocities back to particles
//!    - Implemented by [`WgG2P`] and [`WgG2PCdf`]
//! 4. **Particle Update**: Integrate positions, update deformation gradients
//!    - Implemented by [`WgParticleUpdate`]
//!
//! # Rigid Body Coupling
//!
//! The "CDF" (Collision Detection Field) variants handle two-way coupling with rigid bodies:
//! - [`WgRigidParticleUpdate`]: Updates particles sampled from rigid body surfaces
//! - [`WgRigidImpulses`]: Accumulates and applies forces from MPM to rigid bodies
//!
//! # Key Types
//!
//! - [`Particle`]: CPU-side particle data (position, velocity, material model)
//! - [`GpuParticles`]: GPU buffers storing all particle data
//! - [`ParticleDynamics`]: Physical state (velocity, deformation gradient, mass)
//! - [`ParticleModel`]: Material model (elastic, sand, etc.)
//! - [`SimulationParams`]: Global parameters (gravity, timestep)

pub use g2p::WgG2P;
pub use g2p_cdf::WgG2PCdf;
pub use p2g::WgP2G;
pub use p2g_cdf::WgP2GCdf;
pub use params::{GpuSimulationParams, SimulationParams};
pub use particle::*;
pub use particle_model::*;
// pub use particle_update::WgParticleUpdate;
pub use grid_update::WgGridUpdate;
pub use grid_update_cdf::WgGridUpdateCdf;
pub use particle_update::WgParticleUpdate;
pub use rigid_impulses::{GpuImpulses, RigidImpulse, WgRigidImpulses};
pub use rigid_particle_update::WgRigidParticleUpdate;
pub use timestep_bound::{WgTimestepBounds, TimestepBoundsArgs, GpuTimestepBounds};

mod g2p;
mod g2p_cdf;
mod p2g;
mod p2g_cdf;
mod params;
mod particle_update;
mod rigid_impulses;
mod rigid_particle_update;

mod grid_update;
mod grid_update_cdf;
mod particle;
mod particle_model;
mod timestep_bound;
