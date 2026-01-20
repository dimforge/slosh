# FEniCSx Validation Scripts

FEM reference solutions using FEniCSx for validating Slosh MPM simulations.

## Installation

FEniCSx is already installed in the `fenicsx` conda environment.

```bash
# Activate the environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate fenicsx

# Run the validation
cd /Users/sebcrozet/work/slosh/validation_results/fenics_scripts
python elastic_beam.py
```

## Scripts

### elastic_beam.py

Cantilever beam under self-weight. Solves 3D linear elasticity with FEM.

**Parameters:**
- Length: 10m, Width: 2m, Height: 2m
- E = 1 MPa, ν = 0.3, ρ = 1000 kg/m³
- Gravity: 9.81 m/s²

**Results:**
| Method | Tip Deflection | Error vs Analytical |
|--------|----------------|---------------------|
| Euler-Bernoulli (1D) | 36.79 m | - |
| FEniCSx FEM (3D) | 37.16 m | +1.0% |

The 1% difference is expected: Euler-Bernoulli neglects shear deformation, which matters for thick beams (L/H=5).

## Output Format

Results are saved to `results/fenics_elastic_beam.json`:

```json
{
  "static_solution": {
    "analytical_deflection": 36.7875,
    "fem_avg_tip_deflection": 37.156,
    "particles": [
      {
        "initial_position": [x, y, z],
        "position": [x', y', z'],
        "displacement": [dx, dy, dz]
      }
    ]
  }
}
```

## Comparing with Slosh/Genesis MPM

The output uses the same coordinate system as Slosh:
- Beam elevated at y=5 (above ground plane)
- Sample resolution: 0.5m (matches MPM cell width)
- 525 sample points for direct particle-to-particle comparison
