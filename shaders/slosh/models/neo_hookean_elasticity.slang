//! Neo-hookean elasticity model.

module neo_hookean_elasticity;

import stensor.geometry.svd2;
import stensor.geometry.svd_glam.svd3;
import slosh.aliases;
import slosh.models.utils;
import slosh.models.interfaces;
import slosh.models.implicit_interfaces;
import stensor.utils.mat;

public struct NeoHookeanModel {
    public float lambda;
    public float mu;
    public float cfl_coeff;

    public func kirchoff_stress(deformation_gradient: matrix<float, DIM, DIM>) -> matrix<float, DIM, DIM> {
        let j = max(determinant(deformation_gradient), 1.0e-10);
        let diag = this.lambda * log(j) - this.mu;
        var stress = this.mu * mul(transpose(deformation_gradient), deformation_gradient);
        stress[0].x += diag;
        stress[1].y += diag;
        #if DIM == 3
        stress[2].z += diag;
        #endif

        return stress;
    }

    /// Elastic energy density `mu / 2 (tr(F^T F) - DIM) - mu ln J + lambda / 2 (ln J)^2`, per unit
    /// reference volume.
    public func energy_density(def_grad: Mat<float>) -> float {
        let j = max(determinant(def_grad), 1.0e-10);
        let log_j = log(j);
        let sq_norm = trace(mul(transpose(def_grad), def_grad));
        return 0.5 * this.mu * (sq_norm - float(DIM)) - this.mu * log_j + 0.5 * this.lambda * log_j * log_j;
    }

    /// Differential of the Kirchhoff stress for a change `d_def_grad` of the deformation gradient,
    /// with the reference gradient held fixed (`dP F^T`, stored layout). The off-diagonal blocks
    /// of the Hessian are clamped to be positive semi-definite.
    public func stress_differential(def_grad: Mat<float>, d_def_grad: Mat<float>) -> Mat<float> {
        #if DIM == 2
        let svd = Svd2(def_grad);
        #else
        let svd = Svd3(def_grad);
        #endif
        let s = max(svd.S, Vect<float>(MIN_SINGULAR_VALUE));
        let inv_s = Vect<float>(1.0) / s;
        let j = max(component_product(s), 1.0e-10);
        let log_j = log(j);
        // mu - lambda ln J: the coefficient of the 1 / (s_i s_j) terms.
        let c = this.mu - this.lambda * log_j;

        // Second derivatives w.r.t. the singular values:
        // H_ii = mu + (mu + lambda (1 - ln J)) / s_i^2, H_ij = lambda / (s_i s_j).
        let diag_extra = Vect<float>(this.mu) + inv_s * inv_s * c;
        let h = diag_block(inv_s, this.lambda, diag_extra);

        // Pair eigenvalues (psi_i - psi_j) / (s_i - s_j) and (psi_i + psi_j) / (s_i + s_j).
        let inv_pp = Vect<float>(1.0) / pair_products(s);
        let sym = Vect<float>(this.mu) + inv_pp * c;
        let asym = Vect<float>(this.mu) - inv_pp * c;

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

        let sig = max(svd.S - Vect<float>(1.0), Vect<float>(0.0));
        let pos_dev_part = this.mu * elastic_hardening * dot(sig, sig);
        let spherical_part = this.lambda * elastic_hardening * 0.5 * (j - 1.0) * (j - 1.0);

        if (j < 1.0) {
            return pos_dev_part;
        } else {
            return pos_dev_part + spherical_part;
        }
    }
}
