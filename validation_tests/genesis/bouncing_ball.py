# Bouncing Elastic Ball - Genesis MPM Reference
# Compare with Slosh MPM simulation.
# Uses Genesis physics engine (https://github.com/Genesis-Embodied-AI/Genesis)

import genesis as gs
import numpy as np
import json
import os
import math
import time

# Material parameters
YOUNG_MODULUS = 1000000
POISSON_RATIO = 0.3
DENSITY = 1000

# Ball geometry
RADIUS = 2
DROP_HEIGHT = 10
CELL_WIDTH = 0.2

# Simulation parameters
GRAVITY = 9.81
DT = 1.0 / 60.0
N_SUBSTEPS = 20
TOTAL_FRAMES = 300
SNAPSHOT_INTERVAL = 2

def sanitize_value(val):
    """Replace NaN and Inf with 0.0 for JSON compatibility."""
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return 0.0
    return val

def sanitize_list(lst):
    """Recursively sanitize a list of values."""
    result = []
    for v in lst:
        if isinstance(v, (list, tuple)):
            result.append(sanitize_list(v))
        else:
            result.append(sanitize_value(float(v)))
    return result

def run_simulation():
    # Initialize Genesis
    gs.init(precision="32", logging_level="warning")

    # Compute domain bounds
    margin = 2.0
    lower_bound = (-RADIUS - margin, -5.0, -RADIUS - margin)
    upper_bound = (RADIUS + margin, DROP_HEIGHT + RADIUS + margin, RADIUS + margin)

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

    # Add elastic sphere
    ball = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=YOUNG_MODULUS,
            nu=POISSON_RATIO,
            rho=DENSITY,
            sampler="regular",
        ),
        morph=gs.morphs.Sphere(
            pos=(0.0, DROP_HEIGHT, 0.0),
            radius=RADIUS,
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.6, 0.9),
            vis_mode='particle',
        ),
    )

    # Build the scene
    scene.build()

    snapshots = []
    n_particles = ball.n_particles

    # Record initial state
    state = ball.get_state()
    positions = state.pos.cpu().numpy().squeeze(0)  # Remove batch dimension
    velocities = np.zeros_like(positions)

    snapshots.append({
        "time": 0.0,
        "step": 0,
        "particles": [
            {"position": sanitize_list(pos.tolist()), "velocity": sanitize_list(vel.tolist())}
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
            state = ball.get_state()
            positions = state.pos.cpu().numpy().squeeze(0)  # Remove batch dimension
            # Estimate velocities from position change
            velocities = (positions - prev_positions) / (DT * (SNAPSHOT_INTERVAL if frame > SNAPSHOT_INTERVAL else 1))
            prev_positions = positions.copy()

            snapshots.append({
                "time": frame * DT,
                "step": frame,
                "particles": [
                    {"position": sanitize_list(pos.tolist()), "velocity": sanitize_list(vel.tolist())}
                    for pos, vel in zip(positions, velocities)
                ]
            })
            print(f"Frame {frame}/{TOTAL_FRAMES} ({step_time:.2f}ms/step)")

    # Save results
    result = {
        "name": "bouncing_ball",
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
                "material_type": "neo_hookean"
            }
        }
    }

    os.makedirs("validation_results", exist_ok=True)
    with open("validation_results/bouncing_ball_genesis.json", "w") as f:
        json.dump(result, f)

    print("Simulation complete. Results saved to validation_results/bouncing_ball_genesis.json")

if __name__ == "__main__":
    run_simulation()
