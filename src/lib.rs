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
//! - `dim3`: Enable 3D simulation mode (mutually exclusive with `dim2`)
//! - `rapier-f32`: Support coupling with Rapier using `f32` precision (mutually exclusive with `rapier-f64`)
//! - `rapier-f64`: Support coupling with Rapier using `f64` precision (mutually exclusive with `rapier-f32`)
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
#![allow(missing_docs)]

#[cfg(all(feature = "rapier-f32", feature = "rapier-f64"))]
compile_error!(
    "Features `rapier-f32` and `rapier-f64` are mutually exclusive. Please enable only one of them."
);
#[cfg(all(
    feature = "rapier",
    not(any(feature = "rapier-f32", feature = "rapier-f64"))
))]
compile_error!("Feature `rapier` requires either `rapier-f32` or `rapier-f64` to be enabled.");

#[cfg(all(feature = "dim2", feature = "rapier-f32"))]
pub extern crate rapier2d as rapier;
#[cfg(all(feature = "dim2", feature = "rapier-f64"))]
pub extern crate rapier2d_f64 as rapier;
#[cfg(all(feature = "dim3", feature = "rapier-f32"))]
pub extern crate rapier3d as rapier;
#[cfg(all(feature = "dim3", feature = "rapier-f64"))]
pub extern crate rapier3d_f64 as rapier;

use slang_hal::re_exports::include_dir;

#[cfg(feature = "runtime")]
use slang_hal::re_exports::minislang::SlangCompiler;

pub mod grid;
pub mod models;
pub mod pipeline;
pub mod rbd;
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
    stensor::register_shaders(compiler);
    compiler.add_dir(SLANG_SRC_DIR.clone());
    // Mirror the cpic and node_particle_lists cargo features into shader macros so the shader
    // Node layout and sort kernels match GpuGridNode and GridArgs on the Rust side. slosh sets
    // these itself (unlike DIM, which the consumer picks) since they're crate-layout invariants.
    compiler.set_global_macro("SLOSH_CPIC", if cfg!(feature = "cpic") { 1 } else { 0 });
    compiler.set_global_macro(
        "SLOSH_NODE_PARTICLE_LISTS",
        if cfg!(feature = "node_particle_lists") {
            1
        } else {
            0
        },
    );
}

/// Mathematical types and utilities for physics simulation.
///
/// Re-exports Rapier's math types and defines dimension-specific type aliases
/// for GPU simulation and angular inertia calculations.
pub mod math {
    /// Scalar type used by the simulation.
    pub type Real = f32;

    /// Spatial point type.
    #[cfg(feature = "dim2")]
    pub type Point = glam::Vec2;
    /// Spatial point type.
    #[cfg(feature = "dim3")]
    pub type Point = glam::Vec3;

    /// Spatial vector type.
    #[cfg(feature = "dim2")]
    pub type Vector = glam::Vec2;
    /// Spatial vector type.
    #[cfg(feature = "dim3")]
    pub type Vector = glam::Vec3;

    /// Square matrix type.
    #[cfg(feature = "dim2")]
    pub type Matrix = glam::Mat2;
    /// Square matrix type.
    #[cfg(feature = "dim3")]
    pub type Matrix = glam::Mat3;

    /// Angular vector type.
    #[cfg(feature = "dim2")]
    pub type AngVector = f32;
    /// Angular vector type.
    #[cfg(feature = "dim3")]
    pub type AngVector = glam::Vec3;

    /// Spatial dimension.
    #[cfg(feature = "dim2")]
    pub const DIM: usize = 2;
    /// Spatial dimension.
    #[cfg(feature = "dim3")]
    pub const DIM: usize = 3;

    /// GPU similarity transformation for 2D simulations (translation + rotation).
    #[cfg(feature = "dim2")]
    pub type GpuSim = stensor::geometry::GpuSim2;
    /// GPU similarity transformation for 3D simulations (translation + rotation).
    #[cfg(feature = "dim3")]
    pub type GpuSim = stensor::geometry::GpuSim3;

    /// Angular inertia type for 2D simulations (scalar).
    #[cfg(feature = "dim2")]
    pub type AngularInertia = f32;
    /// Angular inertia type for 3D simulations (3x3 matrix).
    #[cfg(feature = "dim3")]
    pub type AngularInertia = glam::Mat3;

    /// Conversions from Rapier's math types to the simulation's `f32` types.
    ///
    /// This is needed to support using Rapier with `f64` precision.
    #[cfg(feature = "rapier")]
    pub use rapier_convert::*;

    #[cfg(feature = "rapier")]
    mod rapier_convert {
        use super::{Matrix, Vector};

        /// Converts a Rapier scalar into the simulation's `f32` scalar.
        #[cfg(feature = "rapier-f32")]
        #[inline]
        pub fn real(x: rapier::math::Real) -> f32 {
            x
        }
        /// Converts a Rapier scalar into the simulation's `f32` scalar.
        #[cfg(feature = "rapier-f64")]
        #[inline]
        pub fn real(x: rapier::math::Real) -> f32 {
            x as f32
        }

        /// Converts a Rapier vector into the simulation's `f32` vector.
        #[cfg(feature = "rapier-f32")]
        #[inline]
        pub fn vector(v: rapier::math::Vector) -> Vector {
            v
        }
        /// Converts a Rapier vector into the simulation's `f32` vector.
        #[cfg(all(feature = "rapier-f64", feature = "dim2"))]
        #[inline]
        pub fn vector(v: rapier::math::Vector) -> Vector {
            v.as_vec2()
        }
        /// Converts a Rapier vector into the simulation's `f32` vector.
        #[cfg(all(feature = "rapier-f64", feature = "dim3"))]
        #[inline]
        pub fn vector(v: rapier::math::Vector) -> Vector {
            v.as_vec3()
        }

        /// Converts a Rapier matrix into the simulation's `f32` matrix.
        #[cfg(feature = "rapier-f32")]
        #[inline]
        pub fn matrix(m: rapier::math::Matrix) -> Matrix {
            m
        }
        /// Converts a Rapier matrix into the simulation's `f32` matrix.
        #[cfg(all(feature = "rapier-f64", feature = "dim2"))]
        #[inline]
        pub fn matrix(m: rapier::math::Matrix) -> Matrix {
            m.as_mat2()
        }
        /// Converts a Rapier matrix into the simulation's `f32` matrix.
        #[cfg(all(feature = "rapier-f64", feature = "dim3"))]
        #[inline]
        pub fn matrix(m: rapier::math::Matrix) -> Matrix {
            m.as_mat3()
        }
    }
}

/// Re-exports of commonly used dependencies for convenience.
pub mod re_exports {
    pub use crate::rbd;
    pub use slang_hal;
    pub use slang_hal::re_exports::*;
    pub use stensor;
}
