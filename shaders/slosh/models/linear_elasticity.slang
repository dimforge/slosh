//! Linear (corotated) elasticity model.

module linear_elasticity;

import stensor.geometry.svd2;
import stensor.geometry.svd_glam.svd3;
import slosh.aliases;
import slosh.models.utils;
import slosh.models.interfaces;
import slosh.models.implicit_interfaces;

// Positive elastic energy density and Kirchhoff stress computed from a single shared SVD.
public struct ElasticEnergyStress {
    public float pos_energy;
    public Mat<float> kirchoff_stress;
}

public struct LinearElasticModel {
    public float lambda;
    public float mu;
    public float cfl_coeff;

    // TODO: have these be methods of ElasticCoefficient
    #if DIM == 2
    public func kirchoff_stress(deformation_gradient: float2x2) -> float2x2 {
        var svd = Svd2(deformation_gradient);
        let j = svd.S.x * svd.S.y;

        svd.S -= float2(1.0);

        let diag = this.lambda * (j - 1.0) * j;
        var result = mul(transpose(deformation_gradient), svd.recompose()) * (2.0 * this.mu);
        result[0].x += diag;
        result[1].y += diag;

        return result;
    }

    // Same as kirchoff_stress but using a pre-computed SVD instead of computing a new one.
    public func kirchoff_stress_from_svd(deformation_gradient: float2x2, svd_u: float2x2, svd_s: float2, svd_vt: float2x2) -> float2x2 {
        let j = svd_s.x * svd_s.y;
        let s_minus_1 = svd_s - float2(1.0);
        let svd = Svd2(svd_u, s_minus_1, svd_vt);

        let diag = this.lambda * (j - 1.0) * j;
        var result = mul(transpose(deformation_gradient), svd.recompose()) * (2.0 * this.mu);
        result[0].x += diag;
        result[1].y += diag;

        return result;
    }

    // Response to only the compressive part of the deformation.
    public func neg_kirchoff_stress(deformation_gradient: float2x2) -> float2x2 {
        var svd = Svd2(deformation_gradient);
        let j = svd.S.x * svd.S.y;
        svd.S = min(svd.S - float2(1.0), float2(0.0)); // Only keep the negative part.

        let deviatoric_stress = this.lambda * (j - 1.0) * j;
        var result = mul(transpose(deformation_gradient), svd.recompose()) * (2.0 * this.mu);

        if (j < 1.0) {
            result[0].x += deviatoric_stress;
            result[1].y += deviatoric_stress;
        }

        return result;
    }
    #else
    public func kirchoff_stress(deformation_gradient: float3x3) -> float3x3 {
        var svd = Svd3(deformation_gradient);
        let j = svd.S.x * svd.S.y * svd.S.z;

        svd.S -= float3(1.0);

        let diag = this.lambda * (j - 1.0) * j;
        var result = mul(transpose(deformation_gradient), svd.recompose()) * (2.0 * this.mu);
        result[0].x += diag;
        result[1].y += diag;
        result[2].z += diag;

        return result;
    }

    // Same as kirchoff_stress but using a pre-computed SVD instead of computing a new one.
    public func kirchoff_stress_from_svd(deformation_gradient: float3x3, svd_u: float3x3, svd_s: float3, svd_vt: float3x3) -> float3x3 {
        let j = svd_s.x * svd_s.y * svd_s.z;
        let s_minus_1 = svd_s - float3(1.0);
        let svd = Svd3(svd_u, s_minus_1, svd_vt);
        let diag = this.lambda * (j - 1.0) * j;
        var result = mul(transpose(deformation_gradient), svd.recompose()) * (2.0 * this.mu);
        result[0].x += diag;
        result[1].y += diag;
        result[2].z += diag;

        return result;
    }

    // Response to only the compressive part of the deformation.
    public func neg_kirchoff_stress(deformation_gradient: float3x3) -> float3x3 {
        var svd = Svd3(deformation_gradient);
        let j = svd.S.x * svd.S.y * svd.S.z;
        svd.S = min(svd.S - float3(1.0), float3(0.0)); // Only keep the negative part.

        let deviatoric_stress = this.lambda * (j - 1.0) * j;
        var result = mul(transpose(deformation_gradient), svd.recompose()) * (2.0 * this.mu);

        if (j < 1.0) {
            result[0].x += deviatoric_stress;
            result[1].y += deviatoric_stress;
            result[2].z += deviatoric_stress;
        }

        return result;
    }
    #endif

    /// Elastic energy density `mu sum (s_i - 1)^2 + lambda / 2 (J - 1)^2`, per unit reference volume.
    public func energy_density(def_grad: Mat<float>) -> float {
        #if DIM == 2
        let svd = Svd2(def_grad);
        #else
        let svd = Svd3(def_grad);
        #endif
        let j = component_product(svd.S);
        let dev = svd.S - Vect<float>(1.0);
        return this.mu * dot(dev, dev) + 0.5 * this.lambda * (j - 1.0) * (j - 1.0);
    }

    /// Differential of the Kirchhoff stress for a change `d_def_grad` of the deformation gradient,
    /// with the reference gradient held fixed (`dP F^T`, stored layout). The off-diagonal blocks
    /// of the Hessian are clamped to be positive semi-definite, which the implicit solver's
    /// conjugate gradient needs.
    public func stress_differential(def_grad: Mat<float>, d_def_grad: Mat<float>) -> Mat<float> {
        #if DIM == 2
        let svd = Svd2(def_grad);
        #else
        let svd = Svd3(def_grad);
        #endif
        let s = max(svd.S, Vect<float>(MIN_SINGULAR_VALUE));
        let inv_s = Vect<float>(1.0) / s;
        let j = component_product(s);
        let mu2 = 2.0 * this.mu;
        // lambda (J - 1) J: the volumetric derivative shared by every block.
        let vol = this.lambda * (j - 1.0) * j;

        // Second derivatives w.r.t. the singular values:
        // H_ii = 2 mu + lambda J^2 / s_i^2, H_ij = lambda (2J - 1) J / (s_i s_j).
        let ones_coeff = this.lambda * (2.0 * j - 1.0) * j;
        let diag_extra = Vect<float>(mu2) - inv_s * inv_s * vol;
        let h = diag_block(inv_s, ones_coeff, diag_extra);

        // Pair eigenvalues (psi_i - psi_j) / (s_i - s_j) and (psi_i + psi_j) / (s_i + s_j).
        let pp = pair_products(s);
        let ps = pair_sums(s);
        let sym = Vect<float>(mu2) - Vect<float>(vol) / pp;
        let asym = Vect<float>(mu2) * (Vect<float>(1.0) - Vect<float>(2.0) / ps) + Vect<float>(vol) / pp;

        let dp = apply_isotropic_hessian(svd.U, svd.Vt, h, sym, asym, d_def_grad);
        return piola_differential_to_kirchhoff(def_grad, dp);
    }

    public func timestep_bound(
        particle_density0: float,
        particle_velocity: Vect<float>,
        particle_def_grad_det: float,
        elastic_hardening: float,
        cell_width: float,
    ) -> float {
        let bulk_modulus = bulk_modulus_from_lame(this.lambda, this.mu);
        let shear_modulus = shear_modulus_from_lame(this.lambda, this.mu);

        let bound = ElasticitySoundSpeedTimestepBound(
            this.cfl_coeff,
            bulk_modulus * elastic_hardening,
            shear_modulus * elastic_hardening,
        );
        return bound.timestep_bound(particle_density0, particle_def_grad_det, particle_velocity, cell_width);
    }

    public func pos_energy_density(in def_grad: Mat<float>, elastic_hardening: float) -> float {
        let j = determinant(def_grad);

        #if DIM == 2
        let svd = Svd2(def_grad);
        #else
        let svd = Svd3(def_grad);
        #endif

        return this.pos_energy_density_from_svd(svd.S, j, elastic_hardening);
    }

    // Same as pos_energy_density but using a pre-computed SVD (singular values) and Jacobian.
    public func pos_energy_density_from_svd(svd_s: Vect<float>, j: float, elastic_hardening: float) -> float {
        let sig = max(svd_s - Vect<float>(1.0), Vect<float>(0.0));
        let pos_dev_part = this.mu * elastic_hardening * dot(sig, sig);
        let spherical_part = this.lambda * elastic_hardening * 0.5 * (j - 1.0) * (j - 1.0);

        if (j < 1.0) {
            return pos_dev_part;
        } else {
            return pos_dev_part + spherical_part;
        }
    }

    // Computes both the positive elastic energy density and the Kirchhoff stress while sharing a
    // single SVD, instead of computing one SVD per quantity.
    public func pos_energy_and_kirchoff_stress(in def_grad: Mat<float>, elastic_hardening: float) -> ElasticEnergyStress {
        let j = determinant(def_grad);

        #if DIM == 2
        let svd = Svd2(def_grad);
        #else
        let svd = Svd3(def_grad);
        #endif

        let energy = this.pos_energy_density_from_svd(svd.S, j, elastic_hardening);
        let stress = this.kirchoff_stress_from_svd(def_grad, svd.U, svd.S, svd.Vt);
        return ElasticEnergyStress(energy, stress);
    }
}
