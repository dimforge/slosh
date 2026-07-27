//! Rigid-body dynamics (forces, velocities, etc.)

#[cfg(feature = "rapier")]
pub use body::BodyCouplingEntry;
pub use body::{BodyCoupling, BodyDesc, GpuBodySet, GpuForce, GpuMassProperties, GpuVelocity};

/// Rigid body definitions and GPU body set management.
pub mod body;
// /// Physics integration routines (position, velocity updates).
// pub mod integrate;
