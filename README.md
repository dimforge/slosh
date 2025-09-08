# Slosh: cross-platform MPM GPU physics simulation

<p align="center">
  <img src="assets/slosh-logo.png" alt="crates.io" width="400px">
</p>
<p align="center">
    <a href="https://discord.gg/vt9DJSW">
        <img src="https://img.shields.io/discord/507548572338880513.svg?logo=discord&colorB=7289DA">
    </a>
</p>

# Running the examples

1. Download the Slang compiler libraries for your platform: https://github.com/shader-slang/slang/releases/tag/v2025.16
2. Unzip the downloaded directory, and use its path as value to the `SLANG_DIR` environment variable: `SLANG_DIR=/path/to/slang`.
   Note that the variable must point to the root of the slang installation (i.e. the directory that contains `bin` and `lib`).
3. For the 2D examples, run `cargo run --release --example testbed2`
4. For the 3D examples, run `cargo run --release --example testbed3`
