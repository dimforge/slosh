#!/bin/sh

cargo build --example testbed2 --release --target wasm32-unknown-unknown --features dim2
cargo build --example testbed3 --release --target wasm32-unknown-unknown --features dim3
wasm-bindgen --no-typescript --target web --out-dir dist2d --out-name testbed2 ./target/wasm32-unknown-unknown/release/examples/testbed2.wasm
wasm-bindgen --no-typescript --target web --out-dir dist3d --out-name testbed3 ./target/wasm32-unknown-unknown/release/examples/testbed3.wasm
wasm-opt -Oz -o ./dist2d/opt.wasm ./dist2d/testbed2_bg.wasm && mv ./dist2d/opt.wasm ./dist2d/testbed2_bg.wasm
wasm-opt -Oz -o ./dist3d/opt.wasm ./dist3d/testbed3_bg.wasm && mv ./dist3d/opt.wasm ./dist3d/testbed3_bg.wasm

brotli ./dist2d/testbed2_bg.wasm && mv ./dist2d/testbed2_bg.wasm.br ./dist2d/testbed2_bg.wasm
brotli ./dist3d/testbed3_bg.wasm && mv ./dist3d/testbed3_bg.wasm.br ./dist3d/testbed3_bg.wasm