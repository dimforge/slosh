# Sand Column Collapse - Genesis MPM Reference
# Compare with Slosh MPM simulation. Uses Sand material (Drucker-Prager plasticity).
# Uses Genesis physics engine (https://github.com/Genesis-Embodied-AI/Genesis)

import genesis as gs
import numpy as np
import json
import os
import math
import time

# Material parameters (matching Slosh)
YOUNG_MODULUS = 10000000
POISSON_RATIO = 0.2
DENSITY = 2700

# Column geometry
RADIUS = 5
HEIGHT = 20
CELL_WIDTH = 0.5

# Simulation parameters
GRAVITY = 9.81
DT = 1.0 / 60.0
N_SUBSTEPS = 20
TOTAL_FRAMES = 600
SNAPSHOT_INTERVAL = 10

def sanitize_value(val):
    """Replace NaN and Inf with 0.0 for JSON compatibility."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return 0.0
    return val

def sanitize_array(arr):
    """Sanitize an array of values."""
    return [sanitize_value(float(v)) for v in arr]

def run_simulation():
    # Initialize Genesis
    gs.init(precision="32", logging_level="warning")

    # Compute domain bounds - sand will spread outward
    margin = 2.0
    spread = HEIGHT * 2  # Rough estimate of how far sand might spread
    lower_bound = (-20.0, -5.0, -20.0)
    upper_bound = (20.0, 25.0, 20.0)

    # Create scene with MPM solver
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,  # Genesis divides by substeps automatically
            substeps=N_SUBSTEPS,
            gravity=(0.0, -GRAVITY, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            grid_density=int(1.0 / CELL_WIDTH),
            particle_size=CELL_WIDTH / 2.0,  # Match Slosh particle spacing
        ),
        show_viewer=False,
    )

    # Add ground plane
    plane = scene.add_entity(
        morph=gs.morphs.Plane(pos=(0,0,0), euler=(-90, 0, 0)),
        material=gs.materials.Rigid(coup_friction=1.0),
    )

    # Add sand column (cylinder approximated by box for simplicity)
    # Genesis MPM.Sand uses Drucker-Prager plasticity
    sand = scene.add_entity(
        material=gs.materials.MPM.Sand(
            E=YOUNG_MODULUS,
            nu=POISSON_RATIO,
            rho=DENSITY,
            friction_angle=35,
            sampler="regular",
        ),
        morph=gs.morphs.Cylinder(
            pos=(0.0, HEIGHT/2, 0.0),
            radius=RADIUS,
            height=HEIGHT,
            euler=(90, 0, 0),
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.7, 0.4),
            vis_mode='particle',
        ),
    )

    # Build the scene
    scene.build()

    snapshots = []
    n_particles = sand.n_particles

    # Record initial state
    state = sand.get_state()
    positions = state.pos.cpu().numpy().squeeze(0)  # Remove batch dimension
    velocities = np.zeros_like(positions)

    snapshots.append({
        "time": 0.0,
        "step": 0,
        "particles": [
            {"position": sanitize_array(pos.tolist()), "velocity": sanitize_array(vel.tolist())}
            for pos, vel in zip(positions, velocities)
        ]
    })

    print(f"Number of particles: {n_particles}")

    prev_positions = positions.copy()

    for frame in range(1, TOTAL_FRAMES + 1):
        step_start = time.time()
        scene.step()
        step_time = (time.time() - step_start) * 1000  # Convert to ms

        if frame % SNAPSHOT_INTERVAL == 0:
            state = sand.get_state()
            positions = state.pos.cpu().numpy().squeeze(0)  # Remove batch dimension
            # Estimate velocities from position change
            velocities = (positions - prev_positions) / (DT * SNAPSHOT_INTERVAL)
            prev_positions = positions.copy()

            snapshots.append({
                "time": frame * DT,
                "step": frame,
                "particles": [
                    {"position": sanitize_array(pos.tolist()), "velocity": sanitize_array(vel.tolist())}
                    for pos, vel in zip(positions, velocities)
                ]
            })
            print(f"Frame {frame}/{TOTAL_FRAMES} ({step_time:.2f}ms/step)")

    # Save results
    result = {
        "name": "sand_column_collapse",
        "dt": DT,
        "num_substeps": N_SUBSTEPS,
        "snapshots": snapshots,
        "metadata": {
            "num_particles": n_particles,
            "cell_width": CELL_WIDTH,
            "gravity": [0.0, -GRAVITY, 0.0],
            "material": {
                "young_modulus": YOUNG_MODULUS,
                "poisson_ratio": POISSON_RATIO,
                "density": DENSITY,
                "material_type": "drucker_prager"
            }
        }
    }

    os.makedirs("validation_results", exist_ok=True)
    with open("validation_results/sand_column_genesis.json", "w") as f:
        json.dump(result, f)

    print("Simulation complete. Results saved to validation_results/sand_column_genesis.json")

if __name__ == "__main__":
    run_simulation()
