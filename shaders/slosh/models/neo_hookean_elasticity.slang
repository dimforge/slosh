//! Neo-hookean elasticity model.

module neo_hookean_elasticity;

public struct NeoHookeanElasticCoefficients {
    public float lambda;
    public float mu;

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
}
