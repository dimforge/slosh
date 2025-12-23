# Elastic Beam Cantilever - FEniCSx FEM Reference
# Solves the cantilever beam problem using finite elements with dynamic analysis.
# Uses Newmark-β time integration to match Slosh MPM simulation.
#
# Install: conda install -c conda-forge fenics-dolfinx mpich petsc4py

import numpy as np
import json
import os
import argparse
import time as time_module

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

# Material parameters (matching Slosh)
YOUNG_MODULUS = 1.0e7  # Pa
POISSON_RATIO = 0.3
DENSITY = 1000.0  # kg/m³

# Beam geometry
LENGTH = 10.0  # m
WIDTH = 2.0    # m (z-direction)
HEIGHT = 2.0   # m (y-direction)

# Simulation parameters (matching Slosh)
GRAVITY = 9.81  # m/s²
DT = 1.0 / 60.0  # Time step (frame rate)
NUM_SUBSTEPS = 20  # Substeps per frame
TOTAL_FRAMES = 600  # Total frames (10 seconds)
SNAPSHOT_INTERVAL = 10  # Save every N frames

# Mesh resolution (elements per unit length)
MESH_DENSITY = 8

# Newmark-β parameters (average acceleration method - unconditionally stable)
BETA = 0.25
GAMMA = 0.5


def compute_analytical_deflection():
    """Euler-Bernoulli beam theory for cantilever under uniform load."""
    I = WIDTH * HEIGHT**3 / 12.0  # Second moment of area
    q = DENSITY * GRAVITY * WIDTH * HEIGHT  # Load per unit length (N/m)
    # Maximum deflection at free end: delta = q * L^4 / (8 * E * I)
    delta = q * LENGTH**4 / (8.0 * YOUNG_MODULUS * I)
    return delta


def lame_parameters(E, nu):
    """Convert Young's modulus and Poisson ratio to Lamé parameters."""
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lmbda, mu


def epsilon(u):
    """Strain tensor (symmetric gradient)."""
    return ufl.sym(ufl.grad(u))


def sigma(u, lmbda, mu):
    """Stress tensor (linear elasticity)."""
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


def solve_dynamic():
    """Solve dynamic linear elasticity problem with Newmark-β integration."""
    print("=" * 60)
    print("FEniCSx Dynamic Elasticity Solution (Newmark-β)")
    print("=" * 60)

    # Effective time step (accounting for substeps)
    dt = DT / NUM_SUBSTEPS
    total_steps = TOTAL_FRAMES * NUM_SUBSTEPS

    # Create 3D box mesh
    nx = int(LENGTH * MESH_DENSITY)
    ny = int(HEIGHT * MESH_DENSITY)
    nz = int(WIDTH * MESH_DENSITY)

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, -WIDTH/2]), np.array([LENGTH, HEIGHT, WIDTH/2])],
        [nx, ny, nz],
        cell_type=mesh.CellType.hexahedron
    )

    # Define function space (vector-valued for displacement)
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    # Lamé parameters
    lmbda_val, mu_val = lame_parameters(YOUNG_MODULUS, POISSON_RATIO)
    lmbda = fem.Constant(domain, default_scalar_type(lmbda_val))
    mu = fem.Constant(domain, default_scalar_type(mu_val))

    # Density and body force
    rho = fem.Constant(domain, default_scalar_type(DENSITY))
    f = fem.Constant(domain, default_scalar_type((0.0, -DENSITY * GRAVITY, 0.0)))

    # Boundary conditions: fix displacement at x=0 (clamped end)
    def clamped_boundary(x):
        return np.isclose(x[0], 0.0)

    clamped_dofs = fem.locate_dofs_geometrical(V, clamped_boundary)
    u_zero = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_zero, clamped_dofs, V)

    # Functions for time stepping
    u = ufl.TrialFunction(V)  # Displacement at n+1 (unknown)
    v = ufl.TestFunction(V)

    # State variables (displacement, velocity, acceleration at time n)
    u_n = fem.Function(V, name="u_n")      # Displacement at n
    v_n = fem.Function(V, name="v_n")      # Velocity at n
    a_n = fem.Function(V, name="a_n")      # Acceleration at n
    u_new = fem.Function(V, name="u_new")  # Solution at n+1

    # Newmark-β coefficients
    dt_const = fem.Constant(domain, default_scalar_type(dt))
    beta_const = fem.Constant(domain, default_scalar_type(BETA))
    gamma_const = fem.Constant(domain, default_scalar_type(GAMMA))

    # Newmark predictor (what u would be if a_{n+1} = 0)
    # u_pred = u_n + dt*v_n + dt²*(0.5 - β)*a_n
    # Then: a_{n+1} = (u_{n+1} - u_pred) / (β * dt²)

    # Effective mass coefficient: 1 / (β * dt²)
    a1 = 1.0 / (BETA * dt**2)
    a2 = 1.0 / (BETA * dt)
    a3 = (0.5 - BETA) / BETA

    a1_const = fem.Constant(domain, default_scalar_type(a1))

    # Bilinear form: M * a_{n+1} + K * u_{n+1} = F
    # With Newmark: a_{n+1} = a1 * (u_{n+1} - u_n - dt*v_n - (0.5-β)*dt²*a_n)
    # Substituting: (a1*M + K) * u_{n+1} = F + M * a1 * (u_n + dt*v_n + (0.5-β)*dt²*a_n)

    # Left-hand side: effective stiffness matrix
    a_form = (
        a1_const * rho * ufl.inner(u, v) * ufl.dx  # Mass term with Newmark coefficient
        + ufl.inner(sigma(u, lmbda, mu), epsilon(v)) * ufl.dx  # Stiffness term
    )

    # Right-hand side: effective force vector (updated each step)
    # We'll create this as a form and update u_n, v_n, a_n each step
    L_form = (
        ufl.inner(f, v) * ufl.dx  # External force
        + a1_const * rho * ufl.inner(u_n, v) * ufl.dx  # From Newmark
        + a1_const * rho * dt * ufl.inner(v_n, v) * ufl.dx  # From Newmark
        + a1_const * rho * (0.5 - BETA) * dt**2 * ufl.inner(a_n, v) * ufl.dx  # From Newmark
    )

    # Assemble once (matrix doesn't change)
    print(f"Mesh: {nx} x {ny} x {nz} hexahedral elements")
    print(f"DOFs: {V.dofmap.index_map.size_global * 3}")
    print(f"Time step: {dt:.6f} s (effective), {DT:.6f} s (frame)")
    print(f"Total simulation time: {TOTAL_FRAMES * DT:.2f} s")
    print(f"Newmark-β: β={BETA}, γ={GAMMA}")

    # Create the linear problem
    problem = LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-8},
        petsc_options_prefix="dynamic_"
    )

    # For sampling particle positions
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    tree = bb_tree(domain, domain.topology.dim)

    sample_resolution = 0.2  # Same as MPM cell width
    nx_sample = int(LENGTH / sample_resolution) + 1
    ny_sample = int(HEIGHT / sample_resolution) + 1
    nz_sample = int(WIDTH / sample_resolution) + 1

    sample_points = []
    for ix in range(nx_sample):
        for iy in range(ny_sample):
            for iz in range(nz_sample):
                x = min(ix * sample_resolution, LENGTH - 0.001)
                y = min(iy * sample_resolution, HEIGHT - 0.001)
                z = max(min(-WIDTH/2 + iz * sample_resolution, WIDTH/2 - 0.001), -WIDTH/2 + 0.001)
                sample_points.append([x, y, z])
    sample_points = np.array(sample_points)

    cell_candidates = compute_collisions_points(tree, sample_points)
    colliding_cells = compute_colliding_cells(domain, cell_candidates, sample_points)

    # Find valid sample points
    valid_samples = []
    for i, point in enumerate(sample_points):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            valid_samples.append((i, point, cells[0]))

    print(f"Sample points: {len(valid_samples)}")

    # Storage for snapshots
    snapshots = []

    # Record initial state
    def record_snapshot(step, time_val, u_func, v_func):
        particles = []
        for i, point, cell in valid_samples:
            disp = u_func.eval(point, cell)
            vel = v_func.eval(point, cell)
            particles.append({
                "position": [
                    float(point[0] + disp[0]),
                    float(point[1] + disp[1] + 5.0),  # Shift to match Slosh coordinates
                    float(point[2] + disp[2])
                ],
                "velocity": [float(vel[0]), float(vel[1]), float(vel[2])]
            })
        snapshots.append({
            "time": float(time_val),
            "step": step,
            "particles": particles
        })

    # Initial snapshot
    record_snapshot(0, 0.0, u_n, v_n)

    # Time stepping
    print("\nTime stepping...")
    start_time = time_module.time()

    current_time = 0.0
    for frame in range(1, TOTAL_FRAMES + 1):
        frame_start = time_module.time()

        for substep in range(NUM_SUBSTEPS):
            # Solve for u_{n+1}
            u_new = problem.solve()

            # Update acceleration: a_{n+1} = a1 * (u_{n+1} - u_n - dt*v_n) - a3 * a_n
            # Update velocity: v_{n+1} = v_n + dt * ((1-γ)*a_n + γ*a_{n+1})
            with u_new.x.petsc_vec.localForm() as u_new_local, \
                 u_n.x.petsc_vec.localForm() as u_n_local, \
                 v_n.x.petsc_vec.localForm() as v_n_local, \
                 a_n.x.petsc_vec.localForm() as a_n_local:

                u_new_arr = np.asarray(u_new_local)
                u_n_arr = np.asarray(u_n_local)
                v_n_arr = np.asarray(v_n_local)
                a_n_arr = np.asarray(a_n_local)

                # New acceleration
                a_new_arr = a1 * (u_new_arr - u_n_arr - dt * v_n_arr) - a3 * a_n_arr

                # New velocity
                v_new_arr = v_n_arr + dt * ((1.0 - GAMMA) * a_n_arr + GAMMA * a_new_arr)

                # Update for next step
                u_n_arr[:] = u_new_arr
                v_n_arr[:] = v_new_arr
                a_n_arr[:] = a_new_arr

            current_time += dt

        frame_time = (time_module.time() - frame_start) * 1000

        # Record snapshot
        if frame % SNAPSHOT_INTERVAL == 0:
            record_snapshot(frame, current_time, u_n, v_n)

            # Compute tip deflection for progress
            tip_y_displacements = []
            for i, point, cell in valid_samples:
                if point[0] > LENGTH - 0.5:  # Near the tip
                    disp = u_n.eval(point, cell)
                    tip_y_displacements.append(disp[1])

            if tip_y_displacements:
                avg_tip = np.mean(tip_y_displacements)
                print(f"Frame {frame}/{TOTAL_FRAMES} (t={current_time:.3f}s): "
                      f"tip_deflection={-avg_tip:.4f}m, {frame_time:.1f}ms/frame")

    total_time = time_module.time() - start_time
    print(f"\nSimulation complete in {total_time:.1f}s")

    # Analytical solution for reference
    analytical = compute_analytical_deflection()
    print(f"\nAnalytical static deflection (Euler-Bernoulli): {analytical:.6f} m")

    return {
        "snapshots": snapshots,
        "analytical_deflection": float(analytical),
        "mesh_info": {
            "nx": nx, "ny": ny, "nz": nz,
            "num_elements": nx * ny * nz,
            "num_dofs": V.dofmap.index_map.size_global * 3
        }
    }


def main():
    parser = argparse.ArgumentParser(description="FEniCSx elastic beam validation (dynamic)")
    parser.add_argument("--output", "-o", default="validation_results/fenics_elastic_beam.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    # Solve dynamic problem
    result = solve_dynamic()

    # Save results in viewer-compatible format (SimulationTrajectory)
    output = {
        "name": "elastic_beam_cantilever_fenics_dynamic",
        "dt": DT,
        "num_substeps": NUM_SUBSTEPS,
        "snapshots": result["snapshots"],
        "metadata": {
            "num_particles": len(result["snapshots"][0]["particles"]) if result["snapshots"] else 0,
            "cell_width": 0.2,
            "gravity": [0.0, -GRAVITY, 0.0],
            "material": {
                "young_modulus": YOUNG_MODULUS,
                "poisson_ratio": POISSON_RATIO,
                "density": DENSITY,
                "material_type": "linear_elastic"
            }
        },
        "fenics_data": {
            "solver": "FEniCSx (Newmark-β dynamic)",
            "version": "0.10.x",
            "analytical_deflection": result["analytical_deflection"],
            "time_integration": {
                "method": "Newmark-beta",
                "beta": BETA,
                "gamma": GAMMA
            },
            "mesh_info": result["mesh_info"],
            "parameters": {
                "length": LENGTH,
                "width": WIDTH,
                "height": HEIGHT,
                "mesh_density": MESH_DENSITY,
                "dt": DT,
                "num_substeps": NUM_SUBSTEPS,
                "total_frames": TOTAL_FRAMES
            }
        }
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results saved to {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
