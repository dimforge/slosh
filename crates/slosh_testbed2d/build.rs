#[cfg(not(feature = "comptime"))]
pub fn main() {}

#[cfg(feature = "comptime")]
pub fn main() {
    use slang_hal_build::ShaderCompiler;
    use std::env;

    const SLANG_SRC_DIR: include_dir::Dir<'_> =
        include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../shaders");

    let out_dir = env::var("OUT_DIR").expect("Couldn't determine output directory.");
    let mut compiler = ShaderCompiler::new(vec![], &out_dir);
    compiler.add_dir(nexus2d::re_exports::stensor::SLANG_SRC_DIR);
    compiler.add_dir(nexus2d::SLANG_SRC_DIR);
    compiler.add_dir(SLANG_SRC_DIR);
    compiler.set_global_macro("DIM", "2");
    compiler.set_global_macro("COMPTIME", "1");

    // Compile all shaders.
    // Note: slang-hal-build will automatically detect which backends to compile for
    // based on the cargo features enabled during the build.
    compiler
        .compile_shaders_dir("../../shaders_testbed", &[])
        .expect("Failed to compile shaders");
}
