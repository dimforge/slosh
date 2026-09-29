//! Buffers shared by the implicit solver kernels.

module types;

#include "slosh/features.h"

import slosh.aliases;

/// No grid-level boundary condition on this node.
public static const uint BC_NONE = 0;
/// The node sticks to a collider: its velocity increment is zero.
public static const uint BC_STICK = 1;
/// The node slides along a collider: its velocity increment is tangential.
public static const uint BC_SLIP = 2;

/// Homogeneous part of a grid-level boundary condition: the projection the velocity increments
/// (and the residual) live in.
public func project_bc(bc: uint, normal: Vect<float>, v: Vect<float>) -> Vect<float> {
    if (bc == BC_STICK) {
        return Vect<float>(0.0);
    } else if (bc == BC_SLIP) {
        return v - normal * dot(v, normal);
    } else {
        return v;
    }
}

/// Per-node state of the Newton / conjugate gradient solve on the grid velocities.
///
/// Holds the right-hand side, the Newton iterate and direction, the CG work vectors, and copies
/// of the node mass and grid-level boundary condition so the operator kernels only bind this
/// buffer. The current trial velocity `v` is mirrored into the grid node for the final G2P.
/// Keep in lockstep with `GpuCgNode` on the Rust side.
public struct CgNode {
    /// `M v_free`: the mass times the explicit (force-free) velocity.
    public Vect<float> rhs;
    /// Base velocity of the current line search (the accepted Newton iterate).
    public Vect<float> x;
    /// Newton direction, accumulated by the CG solve.
    public Vect<float> dx;
    /// Current trial velocity, the one the residual pass evaluates.
    public Vect<float> v;
    /// Newton residual (gradient of the incremental potential) at `v`.
    public Vect<float> g;
    /// CG residual.
    public Vect<float> r;
    /// Preconditioned CG residual.
    public Vect<float> z;
    /// CG search direction; the operator kernels read it and write `ap`.
    public Vect<float> p;
    /// Operator applied to the search direction.
    public Vect<float> ap;
    /// Boundary normal (inward), when `bc` is not `BC_NONE`.
    public Vect<float> normal;
    /// Node mass.
    public float mass;
    /// Grid-level boundary condition (`BC_*`).
    public uint bc;

    public func project(v: Vect<float>) -> Vect<float> {
        return project_bc(this.bc, this.normal, v);
    }
}

/// Per-particle scratch of the implicit solver. Keep in lockstep with `GpuImplicitParticle`.
public struct ImplicitParticle {
    /// Kirchhoff stress at the trial state (residual pass) or its differential (operator pass),
    /// scaled by `coeff`; consumed by the scatter kernels. Stored layout, like `Kinematics.affine`.
    public Mat<float> stress;
    /// Deformation gradient at the trial velocity, projected for plastic models: the state the
    /// stress, energy and Hessian refer to.
    public Mat<float> trial_def_grad;
    /// `dt * V0 * inv_d`, or zero when the particle is disabled.
    public float coeff;
    /// Reference volume `V0`.
    public float init_volume;
    /// Incremental energy `V0 psi` at the trial state.
    public float energy;
    public uint padding;
}

/// Line search state: no search in progress.
public static const uint LS_IDLE = 0;
/// Line search state: trials running, none accepted yet.
public static const uint LS_SEARCHING = 1;
/// Line search state: a trial was accepted.
public static const uint LS_ACCEPTED = 2;

/// Scalars of the Newton / conjugate gradient iteration, kept on the GPU. Keep in lockstep with
/// `CgScalars` on the Rust side.
public struct CgScalars {
    public float rz;
    public float rz0;
    public float p_ap;
    public float alpha;
    public float beta;
    public uint cg_converged;
    public uint cg_iters;
    public float g_sq;
    public float g0_sq;
    public float energy;
    public float energy_trial;
    public float gdx;
    public float ls_alpha;
    public uint ls_state;
    public uint ls_trial;
    public uint newton_iters;
    public uint newton_done;
    public uint padding;
}

/// Line search on the incremental potential (needs the models' energy densities).
public static const uint LS_CRITERION_ENERGY = 0;
/// Line search on the residual norm (only needs the stresses).
public static const uint LS_CRITERION_RESIDUAL = 1;

/// Uniform parameters of the implicit solve. Keep in lockstep with `CgParams` on the Rust side.
public struct CgParams {
    public float cg_rel_tol_sq;
    public float newton_rel_tol_sq;
    public float armijo;
    public uint line_search_steps;
    public uint criterion;
    public uint padding0;
    public uint padding1;
    public uint padding2;
}
