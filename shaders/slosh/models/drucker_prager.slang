//! Drucker-Prager plasticity model

module drucker_prager;

import nexus.aliases;
import stensor.geometry.svd2;
import stensor.geometry.svd3;

// TODO: should be this in the same buffer as the plasticity parameters themselves?
public struct DruckerPragerPlasticState {
    public float plastic_deformation_gradient_det;
    public float plastic_hardening;
    public float log_vol_gain;
}

public struct DruckerPragerResult {
    public DruckerPragerPlasticState state;
    public Mat deformation_gradient;
}

public struct DruckerPragerProjectionResult {
    public Vect singular_values;
    public float plastic_hardening;
    public bool valid;
}

public struct DruckerPragerPlasticity {
    // TODO: change to (h0, h1, h2, h3) since slang supports it (the field
    //       names ending with digits was an issue with naga-oil).
    public float ha;
    public float hb;
    public float hc;
    public float hd;
    public float lambda;
    public float mu;


    public func alpha(q: float) -> float {
        let angle = this.ha + (this.hb * q - this.hd) * exp(-this.hc * q);
        let s_angle = sin(angle);
        return sqrt(2.0 / 3.0) * (2.0 * s_angle) / (3.0 - s_angle);
    }


    #if DIM == 2

    func project_deformation_gradient(singular_values: float2, log_vol_gain: float, alpha: float) -> DruckerPragerProjectionResult {
        let d = 2.0; // NOTE: this is 3 in 2D
        let strain = log(singular_values) + float2(log_vol_gain / d);
        let strain_trace = strain.x + strain.y;
        let deviatoric_strain = strain - float2(strain_trace / d);

        if (strain_trace > 0.0 || all(deviatoric_strain == float2(0.0))) {
            return DruckerPragerProjectionResult(float2(1.0), length(strain), true);
        }

        let deviatoric_strain_norm = length(deviatoric_strain);
        let gamma = deviatoric_strain_norm
            + (d * this.lambda + 2.0 * this.mu) / (2.0 * this.mu) * strain_trace * alpha;
        if (gamma <= 0.0) {
            return DruckerPragerProjectionResult(float2(0.0), 0.0, false);
        }

        let h = strain - deviatoric_strain * (gamma / deviatoric_strain_norm);
        return DruckerPragerProjectionResult(exp(h), gamma, true);
    }

    public func project(state: DruckerPragerPlasticState, deformation_gradient: float2x2) -> DruckerPragerResult {
    //    if (true) {
    //        let snowvd = Svd2(deformation_gradient);
    //        let yield_surface = exp(1.0 - 0.95);
    //        let j = snowvd.S.x * snowvd.S.y;
    //        var new_sigvals = clamp(snowvd.S, float2(1.0 / yield_surface), float2(yield_surface));
    //        let new_j = new_sigvals.x * new_sigvals.y;
    //        new_sigvals *= sqrt(j / new_j);
    //        let new_def_grad = Svd2:(snowvd.U, new_sigvals, snowvd.Vt).recompose();
    //        return DruckerPragerResult(state, new_def_grad);
    //    }
        if (this.lambda == 0) {
            // Plasticity is disable on this particle.
            return DruckerPragerResult(state, deformation_gradient);
        }

        let svd = Svd2(deformation_gradient);
        let alpha = this.alpha(state.plastic_hardening);
        let projection = this.project_deformation_gradient(svd.S, state.log_vol_gain, alpha);

        if (projection.valid) {
            let prev_det = svd.S.x * svd.S.y;
            let new_det = projection.singular_values.x * projection.singular_values.y;

            let new_plastic_deformation_gradient_det = state.plastic_deformation_gradient_det * prev_det / new_det;
            let new_log_vol_gain = state.log_vol_gain + log(prev_det) - log(new_det);
            let new_plastic_hardening = state.plastic_hardening + projection.plastic_hardening;
            let new_deformation_gradient = Svd2(svd.U, projection.singular_values, svd.Vt).recompose();
            return DruckerPragerResult(
                DruckerPragerPlasticState(new_plastic_deformation_gradient_det, new_plastic_hardening, new_log_vol_gain),
                new_deformation_gradient,
            );
        } else {
            return DruckerPragerResult(state, deformation_gradient);
        }
    }
    #else

    func project_deformation_gradient(singular_values: float3, log_vol_gain: float, alpha: float) -> DruckerPragerProjectionResult {
        let d = 3.0; // NOTE: this is 2 in 2D
        let strain = log(singular_values) + float3(log_vol_gain / d);
        let strain_trace = strain.x + strain.y + strain.z;
        let deviatoric_strain = strain - float3(strain_trace / d);

        if (strain_trace > 0.0 || all(deviatoric_strain == float3(0.0))) {
            return DruckerPragerProjectionResult(float3(1.0), length(strain), true);
        }

        let deviatoric_strain_norm = length(deviatoric_strain);
        let gamma = deviatoric_strain_norm
            + (d * this.lambda + 2.0 * this.mu) / (2.0 * this.mu) * strain_trace * alpha;
        if (gamma <= 0.0) {
            return DruckerPragerProjectionResult(float3(0.0), 0.0, false);
        }

        let h = strain - deviatoric_strain * (gamma / deviatoric_strain_norm);
        return DruckerPragerProjectionResult(exp(h), gamma, true);
    }

    public func project(state: DruckerPragerPlasticState, deformation_gradient: float3x3) -> DruckerPragerResult {
        if (this.lambda == 0) {
            // Plasticity is disable on this particle.
            return DruckerPragerResult(state, deformation_gradient);
        }

        let svd = Svd3(deformation_gradient);
        let alpha = this.alpha(state.plastic_hardening);
        let projection = this.project_deformation_gradient(svd.S, state.log_vol_gain, alpha);

        if (projection.valid) {
            let prev_det = svd.S.x * svd.S.y * svd.S.z;
            let new_det = projection.singular_values.x * projection.singular_values.y * projection.singular_values.z;

            let new_plastic_deformation_gradient_det = state.plastic_deformation_gradient_det * prev_det / new_det;
            let new_log_vol_gain = state.log_vol_gain + log(prev_det) - log(new_det);
            let new_plastic_hardening = state.plastic_hardening + projection.plastic_hardening;
            let new_deformation_gradient = Svd3(svd.U, projection.singular_values, svd.Vt).recompose();
            return DruckerPragerResult(
                DruckerPragerPlasticState(new_plastic_deformation_gradient_det, new_plastic_hardening, new_log_vol_gain),
                new_deformation_gradient,
            );
        } else {
            return DruckerPragerResult(state, deformation_gradient);
        }
    }
    #endif
}
