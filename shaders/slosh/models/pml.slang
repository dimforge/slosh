//! Perfectly-matched-layer absorbing particle, after Kurima, Chandra & Soga, "Absorbing boundary
//! conditions in material point method adopting perfectly matched layer theory" (arXiv:2407.02790).
//!
//! The real coordinate stretch only slows the layer-normal wave; the energy is removed by damping
//! applied within the layer (`ParticleDynamics::damping`).

module pml;

import slosh.aliases;
import slosh.models.interfaces;
import slosh.models.linear_elasticity;
import stensor.utils.mat;

public struct PmlModel {
    public LinearElasticModel elastic;
    // Per-axis coordinate stretching C'_j, zero outside the absorbing layer. The third entry is
    // unused in 2D, kept so the layout matches the host side and avoids vector alignment padding.
    public float stretch_x;
    public float stretch_y;
    public float stretch_z;

    public func stretch() -> Vect<float> {
    #if DIM == 2
        return float2(this.stretch_x, this.stretch_y);
    #else
        return float3(this.stretch_x, this.stretch_y, this.stretch_z);
    #endif
    }

    // Kirchhoff stress of the PML medium, transposed (the result is not symmetric, and p2g
    // applies `M^T`). Implements the paper's Eq. 11; the inertia scaling is left to `mass_scale`.
    public func kirchoff_stress(def_grad: Mat<float>) -> Mat<float> {
        // `grad_u[k][j]` is du_j/dx_k. Small strain, like the paper: the layer stays near-rigid.
        return this.stress_from_gradient(def_grad - diag(Vect<float>(1.0)));
    }

    // Rayleigh stiffness-proportional damping stress: the elastic operator applied to the strain
    // rate. Blind to rigid-body motion, like `viscous_stress` in models/utils.slang.
    public func viscous_stress(velocity_gradient: Mat<float>, stiffness_damping: float) -> Mat<float> {
        if (stiffness_damping <= 0.0) {
            return Mat<float>(0.0);
        }

        return this.stress_from_gradient(velocity_gradient * stiffness_damping);
    }

    func stress_from_gradient(grad_u: Mat<float>) -> Mat<float> {
        let s = Vect<float>(1.0) + this.stretch();
        let lambda = this.elastic.lambda;
        let mu = this.elastic.mu;

        var trace = 0.0;
        [[ForceUnroll]]
        for (var k = 0; k < DIM; k++) {
            trace += grad_u[k][k];
        }

        var result = Mat<float>(0.0);

        [[ForceUnroll]]
        for (var a = 0; a < DIM; a++) {
            // result[a][a] = sig_aa.
            result[a][a] = (lambda + 2.0 * mu) * grad_u[a][a]
                + s[a] * lambda * (trace - grad_u[a][a]);

            [[ForceUnroll]]
            for (var b = 0; b < DIM; b++) {
                if (a != b) {
                    // result[a][b] = sig_ba, with g_ab = grad_u[b][a] and g_ba = grad_u[a][b].
                    result[a][b] = mu * (s[b] * grad_u[b][a] + s[b] * s[b] * grad_u[a][b]);
                }
            }
        }

        return result;
    }

    // The layer's direction-dependent inertia, `rho s_j^2` (the paper's Eq. 16). Gravity keeps
    // acting on the real mass, so the stretched layer falls slower instead of getting heavier.
    public func mass_scale() -> Vect<float> {
        let s = Vect<float>(1.0) + this.stretch();
        return s * s;
    }

    public func timestep_bound(
        particle_density0: float,
        particle_velocity: Vect<float>,
        def_grad_det: float,
        cell_width: float,
    ) -> float {
        // The stretch scales inertia at least as fast as stiffness, so every PML wave speed is
        // bounded by the unstretched one and the elastic bound is conservative here.
        return this.elastic.timestep_bound(
            particle_density0, particle_velocity, def_grad_det, 1.0, cell_width);
    }
}
