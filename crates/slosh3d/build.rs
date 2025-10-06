use minislang::{SlangCompiler, shader_slang::CompileTarget};
use std::path::PathBuf;
use std::str::FromStr;

pub fn main() {
    println!("cargo:rerun-if-changed=../../shaders");

    let mut slang = SlangCompiler::new(vec![PathBuf::from_str("../../shaders").unwrap()]);
    nexus3d::register_shaders(&mut slang);

    let targets = [CompileTarget::Wgsl, CompileTarget::CudaSource];

    for target in targets {
        slang.compile_all(
            target,
            "../../shaders",
            "../../src/autogen3d",
            &[("DIM".to_string(), "3".to_string())],
        );
    }
}
