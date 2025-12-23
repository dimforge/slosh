# Slosh MPM Validation Tests

This module provides validation tests to compare Slosh MPM simulation results against:
- **Genesis/Taichi MPM** - Another GPU-accelerated MPM implementation
- **FEniCSx FEM** - Finite Element Method reference solutions
- **Analytical solutions** - Euler-Bernoulli beam theory, etc.

## Validation Scenarios

| Scenario | Description | Key Metrics |
|----------|-------------|-------------|
| **elastic_beam** | Cantilever beam deflects under gravity | Tip deflection vs. analytical Euler-Bernoulli |
| **sand_column** | Granular column collapse (Drucker-Prager) | Runout distance, angle of repose |
| **dam_break** | Fluid column collapse | Front position vs. Martin-Moyce data |
| **bouncing_ball** | Elastic sphere bounces on rigid surface | Coefficient of restitution, energy |
| **twisting_cube** | Torsion test on elastic cube | Strain energy, volume conservation |
| **rigid_coupling** | Rigid body impacts deformable bed | Momentum transfer, penetration depth |

---

## Quick Start

```bash
# 1. Run Slosh simulation + generate Genesis scripts
cargo run -p slosh_validation_tests --features "webgpu runtime" -- --scenario elastic_beam

# 2. Run Genesis/Taichi reference (requires Genesis installed)
python validation_tests/genesis/elastic_beam.py

# 3. Run FEniCSx FEM reference (requires conda fenicsx env)
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate fenicsx
cd validation_results/fenics_scripts
python elastic_beam.py

# 4. Visualize results
cargo run -p slosh_validation_tests --features "render" -- --file validation_results/slosh_results/elastic_beam.json
```

---

## Command Reference

### Slosh Validation Runner

The main validation test runner. Runs Slosh simulations and generates reference scripts.

#### Run All Scenarios
```bash
cargo run -p slosh_validation_tests --features "webgpu runtime"
```

#### Run Specific Scenario
```bash
cargo run -p slosh_validation_tests --features "webgpu runtime" -- --scenario <NAME>
```

Available scenarios: `elastic_beam`, `sand_column`, `dam_break`, `bouncing_ball`, `twisting_cube`, `rigid_coupling`, `all`

#### Run and Visualize Results
```bash
cargo run -p slosh_validation_tests --features "webgpu runtime render" -- --scenario elastic_beam --render
```

#### Visualize Existing Results
```bash
# Visualize Slosh results
cargo run -p slosh_validation_tests --features "render" -- --render-only --scenario elastic_beam

# Visualize any JSON trajectory file (Slosh, Genesis, or FEniCSx)
cargo run -p slosh_validation_tests --features "render" -- --file <path_to_json>
```

#### Custom Output Directory
```bash
cargo run -p slosh_validation_tests --features "webgpu runtime" -- --output-dir my_results
```

#### Show Help
```bash
cargo run -p slosh_validation_tests --features "webgpu runtime" -- --help
```

### All Options Summary

| Option | Description |
|--------|-------------|
| `--scenario <NAME>` | Run specific scenario (default: `all`) |
| `--output-dir <PATH>` | Output directory (default: `validation_results`) |
| `--render` | Render results after simulation (requires `render` feature) |
| `--render-only` | Render existing results without simulation |
| `--file <PATH>` | Render specific JSON file (implies `--render-only`) |
| `--help`, `-h` | Show help message |

---

## Genesis Reference Simulations

Genesis is a GPU-accelerated physics engine built on Taichi. The generated scripts use Genesis MPM.

### Requirements
```bash
pip install genesis-world numpy
```

### Run Individual Scenarios
```bash
python genesis/elastic_beam.py      # Cantilever beam
python genesis/sand_column.py       # Granular collapse
python genesis/dam_break.py         # Fluid column
python genesis/bouncing_ball.py     # Elastic bounce
```

### Output Location
Results are saved to `validation_results/<scenario>_genesis.json`

---

## FEniCSx FEM Reference Simulations

FEniCSx provides high-accuracy FEM solutions for validation. Currently available for elastic beam.

### Setup (One-Time)
```bash
# If not already installed:
brew install miniforge
conda create -n fenicsx python=3.11
conda activate fenicsx
conda install -c conda-forge fenics-dolfinx mpich petsc4py
```

### Run FEniCSx Validation
```bash
# Activate environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate fenicsx

# Run elastic beam FEM solution
cd validation_results/fenics_scripts
python elastic_beam.py
python elastic_beam.py --output results/custom_output.json  # Custom output path
```

### Output Location
Results are saved to `validation_results/fenics_scripts/results/fenics_elastic_beam.json`

---

## Viewer Controls

When using `--render`, `--render-only`, or `--file`:

| Key | Action |
|-----|--------|
| `Space` | Play/Pause animation |
| `←` / `→` | Step through snapshots |
| `R` | Reset to beginning |
| Mouse drag | Rotate camera |
| Scroll | Zoom in/out |

---

## JSON Trajectory Format

All tools export results in a compatible JSON format:

```json
{
  "name": "elastic_beam_cantilever",
  "dt": 0.0167,
  "num_substeps": 20,
  "snapshots": [
    {
      "time": 0.0,
      "step": 0,
      "particles": [
        {"position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0, 0.0]},
        ...
      ]
    },
    ...
  ],
  "metadata": {
    "num_particles": 1000,
    "cell_width": 0.5,
    "gravity": [0.0, -9.81, 0.0],
    "material": {
      "young_modulus": 1000000.0,
      "poisson_ratio": 0.3,
      "density": 1000.0,
      "material_type": "neo_hookean"
    }
  }
}
```

---

## Example Workflows

### Full Validation Workflow (Elastic Beam)

```bash
# Step 1: Run Slosh simulation
cargo run -p slosh_validation_tests --features "webgpu runtime" -- --scenario elastic_beam

# Step 2: Run Genesis reference
cd validation_results/genesis_scripts
python elastic_beam.py
cd ../..

# Step 3: Run FEniCSx FEM reference
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate fenicsx
cd validation_results/fenics_scripts
python elastic_beam.py
cd ../..

# Step 4: Compare results visually
cargo run -p slosh_validation_tests --features "render" -- --file validation_results/slosh_results/elastic_beam.json
cargo run -p slosh_validation_tests --features "render" -- --file validation_results/genesis_scripts/results/genesis_elastic_beam.json
```

### Quick Visualization of Any Result
```bash
# Slosh result
cargo run -p slosh_validation_tests --features "render" -- --file validation_results/dam_break_slosh.json

# Taichi result
cargo run -p slosh_validation_tests --features "render" -- --file validation_results/bouncing_ball_genesis.json

# FEniCSx result (static, shows final deformed state)
cargo run -p slosh_validation_tests --features "render" -- --file validation_results/fenics_scripts/results/fenics_elastic_beam.json
```

### Run All Tests for CI
```bash
# Generate scripts and run all Slosh scenarios
cargo run -p slosh_validation_tests --features "webgpu runtime" -- --scenario all

# Or just generate scripts (no GPU needed)
cargo run -p slosh_validation_tests --features "runtime" -- --generate-genesis
```

---

## Comparison Metrics

When comparing Slosh vs. reference simulations:

| Metric | Description | Target |
|--------|-------------|--------|
| Position Error | Euclidean distance between particles | < 0.1 × cell_width |
| RMSE | Root mean squared position error | < 0.5 × cell_width |
| COM Error | Center of mass trajectory divergence | < 0.5 × cell_width |
| Tip Deflection | For beam tests, vs. analytical | < 5% error |

---

## Customizing Parameters

Each scenario has configurable parameters:

```rust
use slosh_validation::scenarios::*;

let params = ElasticBeamParams {
    length: 15.0,           // Beam length (m)
    young_modulus: 2.0e6,   // Stiffer material (Pa)
    ..Default::default()
};

let config = elastic_beam_scenario(params);
```

---

## References

- Stomakhin et al. (2013) "A Material Point Method for Snow Simulation"
- Klár et al. (2016) "Drucker-Prager Elastoplasticity for Sand Animation"
- Martin & Moyce (1952) "An experimental study of the collapse of liquid columns"
- Lajeunesse et al. (2005) "Granular slumping on a horizontal surface"
- FEniCSx Documentation: https://docs.fenicsproject.org/
