//! Validation test infrastructure for comparing Slosh MPM simulations with reference implementations.
//!
//! This module provides utilities for:
//! - Running headless MPM simulations
//! - Recording particle trajectories over time
//! - Exporting results to CSV/JSON for comparison with Genesis
//! - Computing error metrics between simulations
//!
//! # Usage
//!
//! ```bash
//! # Run Slosh simulations
//! cargo run -p slosh_validation_tests --features "webgpu runtime" -- --scenario all
//!
//! # Run Genesis reference simulations
//! python validation_tests/genesis/elastic_beam.py
//! python validation_tests/genesis/sand_column.py
//! # ... etc
//!
//! # Compare results (compute error metrics)
//! cargo run -p slosh_validation_tests -- --compare
//! ```

pub mod harness;
pub mod metrics;
pub mod scenarios;

#[cfg(feature = "render")]
pub mod viewer;

pub use harness::*;
pub use metrics::*;

#[cfg(feature = "render")]
pub use viewer::*;
