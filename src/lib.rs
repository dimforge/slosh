//! Slosh: GPU-accelerated Material Point Method (MPM) physics simulation.
//!
//! Slosh provides a high-performance implementation of the Material Point Method for
//! simulating materials like fluids, sand, snow, and elastic solids. The simulation
//! runs entirely on the GPU using compute shaders, achieving real-time performance
//! for large particle systems.
//!
//! # Overview
//!
//! The MPM algorithm works by transferring data between particles (Lagrangian representation)
//! and a background grid (Eulerian representation):
//! 1. **P2G (Particle-to-Grid)**: Transfer particle mass and momentum to grid nodes
//! 2. **Grid Update**: Solve momentum equations on the grid
//! 3. **G2P (Grid-to-Particle)**: Transfer velocities back to particles and update positions
//!
//! Slosh also supports two-way coupling with rigid bodies via the Rapier physics engine.
//!
//! # Features
//!
//! - `dim2`: Enable 2D simulation mode (mutually exclusive with `dim3`)
//! - `dim3`: Enable 3D simulation mode (default, mutually exclusive with `dim2`)
//!
//! # Example
//!
//! ```ignore
//! use slosh::pipeline::{MpmPipeline, MpmData};
//! use slosh::solver::{Particle, SimulationParams};
//!
//! // Create GPU pipeline
//! let pipeline = MpmPipeline::new(&backend, &compiler)?;
//!
//! // Initialize simulation data
//! let mut data = MpmData::new(
//!     &backend,
//!     params,
//!     &particles,
//!     &bodies,
//!     &colliders,
//!     cell_width,
//!     grid_capacity,
//! )?;
//!
//! // Run simulation step
//! pipeline.launch_step(&backend, &mut encoder, &mut data)?;
//! ```
//!
//! # Module Organization
//!
//! - [`pipeline`]: High-level MPM simulation orchestration
//! - [`solver`]: Core MPM algorithm implementations (P2G, G2P, grid updates, particle updates)
//! - [`grid`]: Spatial grid data structures and operations
//! - [`models`]: Material models (elastic, sand, Drucker-Prager plasticity)

#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_inception)]
#![warn(missing_docs)]

#[cfg(feature = "dim2")]
pub extern crate nexus2d as nexus;
#[cfg(feature = "dim3")]
pub extern crate nexus3d as nexus;
#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

use slang_hal::re_exports::include_dir;

#[cfg(feature = "runtime")]
use slang_hal::re_exports::minislang::SlangCompiler;

pub mod grid;
pub mod models;
pub mod pipeline;
pub(crate) mod sampling;
pub mod solver;
pub mod trimesh;

/// Embedded directory containing Slang shader source files.
pub const SLANG_SRC_DIR: include_dir::Dir<'_> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../shaders");

/// Registers all Slosh shader modules with the Slang compiler.
///
/// This must be called before creating any [`pipeline::MpmPipeline`] to ensure
/// all compute shaders are available for compilation.
///
/// # Arguments
///
/// * `compiler` - The Slang compiler instance to register shaders with
#[cfg(feature = "runtime")]
pub fn register_shaders(compiler: &mut SlangCompiler) {
    nexus::register_shaders(compiler);
    compiler.add_dir(SLANG_SRC_DIR.clone());
}

/// Re-exports of commonly used dependencies for convenience.
pub mod re_exports {
    pub use nexus;
    pub use slang_hal;
    pub use slang_hal::re_exports::*;
    pub use stensor;
}
