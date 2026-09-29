//! Model interface of the implicit grid solver, plus the shared machinery for linearizing
//! isotropic constitutive models.
//!
//! For an energy density written in the singular values of `F = U diag(s) Vt`, the Hessian is
//! block diagonal in the SVD basis: one `DIM x DIM` block coupling the diagonal entries, plus
//! one 2x2 block per off-diagonal pair whose eigenvectors are the symmetric and antisymmetric
//! parts of that pair. `apply_isotropic_hessian` applies such a Hessian to a differential of `F`
//! and clamps the pair eigenvalues so the operator stays positive semi-definite (the definiteness
//! fix of Stomakhin et al. 2013). The diagonal block is left as is: it only turns indefinite
//! under extreme compression.
//!
//! Matrix convention: slosh stores the transpose of the mathematical matrices (a velocity
//! gradient is accumulated as `outer_product(v, dpt)` with `v` along the rows, and the transfers
//! contract them as `mul(dpt, M)`). Every formula below is written for that stored layout.

module implicit_interfaces;

#include "slosh/features.h"

import slosh.aliases;
import slosh.models.interfaces;
import stensor.utils.mat;

/// Stress and incremental energy density of a model evaluated at a trial deformation gradient.
public struct TrialState {
    public Mat<float> kirchoff_stress;
    public float energy_density;

    public __init(Mat<float> stress, float energy) {
        this.kirchoff_stress = stress;
        this.energy_density = energy;
    }
}

/// What a constitutive model must provide for the implicit grid solver.
///
/// Both methods must only read the model buffer (it is bound read-write for uniformity with
/// `IParticleModel`): the Newton iterations and the line search evaluate them at every trial
/// velocity, and the stored plastic/damage state is only advanced by the regular particle update
/// once the substep's velocity is known.
public interface IImplicitParticleModel {
    /// Evaluates the model at a trial deformation gradient. On input `def_grad` is the trial
    /// gradient; on output it is the projected one for plastic models (unchanged otherwise),
    /// which is the state the returned stress and energy, and the Hessian, refer to.
    /// `data.velocity_gradient` is the trial velocity gradient, for rate-dependent terms.
    public func trial_state(
        models: RWByteAddressBuffer,
        data: ParticleUpdateData,
        inout def_grad: Mat<float>,
    ) -> TrialState;

    /// Differential of the Kirchhoff stress for a change `d_vel_grad` of the velocity gradient
    /// over the substep `data.dt`, at the (projected) trial gradient `def_grad`. Returns
    /// `dP F^T` with `P` the first Piola-Kirchhoff stress: the reference gradient is held fixed.
    /// `data.velocity_gradient` is the trial velocity gradient the model was evaluated at.
    public func stress_differential(
        models: RWByteAddressBuffer,
        data: ParticleUpdateData,
        def_grad: Mat<float>,
        d_vel_grad: Mat<float>,
    ) -> Mat<float>;
}

/// Singular values are clamped to this before any division.
public static const float MIN_SINGULAR_VALUE = 1.0e-4;

/// Product of the components of a vector, i.e. the determinant of `diag(v)`.
public func component_product(v: Vect<float>) -> float {
#if DIM == 2
    return v.x * v.y;
#else
    return v.x * v.y * v.z;
#endif
}

/// Symmetric outer product `a a^T`.
public func self_outer_product(a: Vect<float>) -> Mat<float> {
#if DIM == 2
    return float2x2(a * a.x, a * a.y);
#else
    return float3x3(a * a.x, a * a.y, a * a.z);
#endif
}

/// Pairwise products of the components of `s`, one per off-diagonal pair: `(s0 s1, s0 s2, s1 s2)`
/// in 3D, `(s0 s1, 1)` in 2D (the unused lane is 1 so that dividing by it is harmless).
public func pair_products(s: Vect<float>) -> Vect<float> {
#if DIM == 2
    return float2(s.x * s.y, 1.0);
#else
    return float3(s.x * s.y, s.x * s.z, s.y * s.z);
#endif
}

/// Pairwise sums of the components of `s`, in the same order as `pair_products`.
public func pair_sums(s: Vect<float>) -> Vect<float> {
#if DIM == 2
    return float2(s.x + s.y, 1.0);
#else
    return float3(s.x + s.y, s.x + s.z, s.y + s.z);
#endif
}

/// Assembles the diagonal block of an isotropic Hessian:
/// `H = ones_coeff * inv_s inv_s^T + diag(diag_extra)`.
public func diag_block(inv_s: Vect<float>, ones_coeff: float, diag_extra: Vect<float>) -> Mat<float> {
    return self_outer_product(inv_s) * ones_coeff + diag(diag_extra);
}

/// Applies an isotropic Hessian, given in the SVD basis of the deformation gradient, to the
/// differential `d_def_grad` (stored layout). `svd_u` and `svd_vt` are the factors of the SVD of
/// the stored gradient; `diag_block` holds the second derivatives of the energy density with
/// respect to the singular values; `sym` and `asym` hold, for each pair `(0, 1), (0, 2), (1, 2)`
/// (only `(0, 1)` in 2D), the eigenvalues acting on the symmetric and antisymmetric parts of that
/// pair. Negative pair eigenvalues are clamped to zero. Returns the differential of the first
/// Piola-Kirchhoff stress, in the stored layout.
public func apply_isotropic_hessian(
    svd_u: Mat<float>,
    svd_vt: Mat<float>,
    diag_block: Mat<float>,
    sym: Vect<float>,
    asym: Vect<float>,
    d_def_grad: Mat<float>,
) -> Mat<float> {
    // The stored gradient is the transpose of the mathematical one, so the SVD factors swap
    // roles: the mathematical `U^T dF V` is the stored `U_s^T dF_s Vt_s^T`, and `w[i][j]` is the
    // mathematical entry `(j, i)`.
    let w = mul(mul(transpose(svd_u), d_def_grad), transpose(svd_vt));
    let sym_c = max(sym, Vect<float>(0.0));
    let asym_c = max(asym, Vect<float>(0.0));

#if DIM == 2
    let dd = mul(diag_block, float2(w[0][0], w[1][1]));
    let w01 = w[1][0];
    let w10 = w[0][1];
    let s01 = (w01 + w10) * 0.5;
    let a01 = (w01 - w10) * 0.5;
    let dp01 = sym_c.x * s01 + asym_c.x * a01;
    let dp10 = sym_c.x * s01 - asym_c.x * a01;
    // Stored layout: each row holds a mathematical column of dP_hat.
    let dp_hat = float2x2(float2(dd.x, dp10), float2(dp01, dd.y));
#else
    let dd = mul(diag_block, float3(w[0][0], w[1][1], w[2][2]));
    let w01 = w[1][0];
    let w10 = w[0][1];
    let w02 = w[2][0];
    let w20 = w[0][2];
    let w12 = w[2][1];
    let w21 = w[1][2];
    let s01 = (w01 + w10) * 0.5;
    let a01 = (w01 - w10) * 0.5;
    let s02 = (w02 + w20) * 0.5;
    let a02 = (w02 - w20) * 0.5;
    let s12 = (w12 + w21) * 0.5;
    let a12 = (w12 - w21) * 0.5;
    let dp01 = sym_c.x * s01 + asym_c.x * a01;
    let dp10 = sym_c.x * s01 - asym_c.x * a01;
    let dp02 = sym_c.y * s02 + asym_c.y * a02;
    let dp20 = sym_c.y * s02 - asym_c.y * a02;
    let dp12 = sym_c.z * s12 + asym_c.z * a12;
    let dp21 = sym_c.z * s12 - asym_c.z * a12;
    let dp_hat = float3x3(
        float3(dd.x, dp10, dp20),
        float3(dp01, dd.y, dp21),
        float3(dp02, dp12, dd.z),
    );
#endif
    // Mathematical `U dP_hat V^T` is the stored `U_s dP_hat_s Vt_s`.
    return mul(mul(svd_u, dp_hat), svd_vt);
}

/// `dP F^T` in the stored layout, given the stored `dP` and `F`.
public func piola_differential_to_kirchhoff(def_grad: Mat<float>, dp: Mat<float>) -> Mat<float> {
    return mul(transpose(def_grad), dp);
}
