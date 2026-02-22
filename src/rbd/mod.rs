use slang_hal::re_exports::include_dir;

#[cfg(feature = "runtime")]
use slang_hal::re_exports::minislang::SlangCompiler;

/// GPU-accelerated rigid body dynamics simulation.
///
/// This module provides structures and methods for managing physics bodies
/// on the GPU, including body state, integration, and coupling with colliders.
pub mod dynamics;
/// GPU-compatible shape representations.
///
/// This module defines shape types and utilities for converting Rapier/Parry shapes
/// to GPU-friendly formats with vertex buffers.
pub mod shapes;
