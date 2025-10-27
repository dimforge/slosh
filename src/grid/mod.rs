//! Spatial grid data structures and operations.
//!
//! The grid is the Eulerian (fixed spatial frame) component of MPM. Particles
//! transfer their momentum to grid nodes, forces are computed on the grid, and
//! velocities are interpolated back to particles.
//!
//! # Modules
//!
//! - [`grid`]: Grid cell data structure and management
//! - [`sort`]: Spatial sorting of particles into grid cells
//! - [`prefix_sum`]: Parallel prefix sum for compacting sparse grid data

/// Grid cell data structure and management.
pub mod grid;
/// Parallel prefix sum for compacting sparse grid data.
pub mod prefix_sum;
/// Spatial sorting of particles into grid cells.
pub mod sort;
