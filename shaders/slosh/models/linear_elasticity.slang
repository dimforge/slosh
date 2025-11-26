//! Linear (corotated) elasticity model.

module linear_elasticity;

import stensor.geometry.svd2;
import stensor.geometry.svd3;
import nexus.aliases;
import slosh.models.utils;
import slosh.models.interfaces;

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
    #endif

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
