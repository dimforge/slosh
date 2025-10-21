//! Neo-hookean elasticity model.

module neo_hookean_elasticity;

import nexus.aliases;
import slosh.models.utils;
import slosh.solver.timestep_bound;

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

    public func timestep_bound(
        particle_density0: float,
        particle_velocity: Vect<float>,
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
        return bound.timestep_bound(particle_density0, particle_velocity, cell_width);
    }
}
