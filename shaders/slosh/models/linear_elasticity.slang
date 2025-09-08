//! Linear (corotated) elasticity model.

module linear_elasticity;

import stensor.geometry.svd2;
import stensor.geometry.svd3;


public struct LinearElasticCoefficients {
    public float lambda;
    public float mu;

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
}
