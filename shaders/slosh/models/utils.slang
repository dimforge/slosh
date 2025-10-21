module utils;

import stensor.utils.mat;
import nexus.aliases;

/// Computes the Lamé parameters (lambda, mu) from the young modulus and poisson ratio.
public func lame_lambda_mu(young_modulus: float, poisson_ratio: float) -> float2 {
    return float2(
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
        hook_to_shear_modulus(young_modulus, poisson_ratio),
    );
}

public func hook_to_shear_modulus(young_modulus: float, poisson_ratio: float) -> float {
    return young_modulus / (2.0 * (1.0 + poisson_ratio));
}

public func hook_to_bulk_modulus(young_modulus: float, poisson_ratio: float) -> float {
    return young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));
}

public func shear_modulus_from_lame(_lambda: float, mu: float) -> float {
    return mu;
}

public func bulk_modulus_from_lame(lambda: float, mu: float) -> float {
    return lambda + 2.0 * mu / 3.0;
}

public func solve_quadratic(a: float, b: float, c: float) -> float2 {
    let discr_sqr = sqrt(b * b - 4.0 * a * c);
    return float2((-b + discr_sqr) / (2.0 * a), (-b - discr_sqr) / (2.0 * a));
}

//public func min_componentwise_quadratic_solve<const R: usize, const C: usize>(
//    a: &SMatrix<float, R, C>,
//    b: &SMatrix<float, R, C>,
//    c: float,
//    sol_half_range: (float, float),
//) -> float {
//    return a.zip_map(b, |a, b| {
//        let (mut s0, mut s1) = solve_quadratic(a, b, c);
//        if s0 <= sol_half_range.0 {
//            s0 = float::MAX;
//        }
//        if s1 <= sol_half_range.0 {
//            s1 = float::MAX;
//        }
//
//        s0.min(s1)
//    })
//    .min()
//    .min(sol_half_range.1);
//}

public func spin_tensor(velocity_gradient: Mat<float>) -> Mat<float> {
    return (velocity_gradient - transpose(velocity_gradient)) * 0.5;
}

public func strain_rate(velocity_gradient: Mat<float>) -> Mat<float> {
    return (velocity_gradient + transpose(velocity_gradient)) * 0.5;
}

public func deviatoric_part(tensor: Mat<float>) -> Mat<float> {
    return DecomposedTensor(tensor).deviatoric_part;
}
public func spherical_part(tensor: Mat<float>) -> float {
    return trace(tensor) / float(DIM);
}

public struct DecomposedTensor {
    public Mat<float> deviatoric_part;
    public float spherical_part;

    public __init(Mat<float> tensor) {
        let spherical_part = trace(tensor) / float(DIM);
        var deviatoric_part = tensor;

        [[ForceUnroll]]
        for (var i = 0; i < DIM; i++) {
            deviatoric_part[i][i] -= spherical_part;
        }

        this.deviatoric_part = deviatoric_part;
        this.spherical_part = spherical_part;
    }

    public func recompose() -> Mat<float> {
        var result = this.deviatoric_part;

        [[ForceUnroll]]
        for (var i = 0; i < DIM; i++) {
            result[i][i] += this.spherical_part;
        }

        return result;
    }
}