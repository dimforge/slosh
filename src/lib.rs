#![allow(clippy::too_many_arguments)]
#![allow(clippy::module_inception)]

#[cfg(feature = "dim2")]
pub extern crate nexus2d as nexus;
#[cfg(feature = "dim3")]
pub extern crate nexus3d as nexus;
#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

use slang_hal::re_exports::include_dir;
use slang_hal::re_exports::minislang::SlangCompiler;

pub mod grid;
pub mod models;
pub mod pipeline;
pub(crate) mod sampling;
pub mod solver;

pub const SLANG_SRC_DIR: include_dir::Dir<'_> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../shaders");
pub fn register_shaders(compiler: &mut SlangCompiler) {
    nexus::register_shaders(compiler);
    compiler.add_dir(SLANG_SRC_DIR.clone());
}
