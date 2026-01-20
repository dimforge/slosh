//! Validation test scenarios for comparing Slosh with Taichi Elements.
//!
//! Each scenario defines a specific physics setup that can be run in both
//! Slosh and Taichi for comparison.

pub mod bouncing_ball;
pub mod dam_break;
pub mod elastic_beam;
pub mod sand_column;

pub use bouncing_ball::{bouncing_ball_scenario, BouncingBallParams};
pub use dam_break::{dam_break_scenario, DamBreakParams};
pub use elastic_beam::{elastic_beam_scenario, ElasticBeamParams};
pub use sand_column::{sand_column_scenario, SandColumnParams};
