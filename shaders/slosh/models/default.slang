module default_model;

#include "slosh/features.h"

import slosh.models.interfaces;
import slosh.models.linear_elasticity;
import slosh.models.neo_hookean_elasticity;
import slosh.models.drucker_prager;
#if SLOSH_PML
import slosh.models.pml;
#endif
import slosh.solver.particle;
import slosh.models.utils;
import slosh.aliases;


public enum class DefaultParticleModelType {
    ElasticLinear = 0,
    ElasticNeoHookean = 1,
    SandLinear = 2,
    SandNeoHookean = 3,
#if SLOSH_PML
    // Absorbing (perfectly-matched-layer) particle; see models/pml.slang.
    AbsorbingPml = 4,
#endif
    // Provided to help downstream users extend with their own enum.
#if SLOSH_PML
    LEN = 5,
#else
    LEN = 4,
#endif
}

public struct SandModel<ElasticModel> {
    // NOTE: if `plastic_state` is move, its offset in `SandModel::state_offset` needs to be adjusted.
    DruckerPragerPlasticState plastic_state;
    DruckerPragerPlasticity plastic;
    ElasticModel elastic;

    func state_offset() -> uint {
        return 0; // 0 since `plastics_state` is the first field.
    }
}

// TODO: hard-coding this is quite fragile as is must match the size of
//       the host-side enum. Should it be configured by the host?
// This **MUST** match the byte size of `GpuParticleModel` on the rust side.
public static const uint DEFAULT_MODEL_BYTES_STRIDE = 52;

// NOTE: the BYTES_STRIDE parameter exists so that third-party models extending this default model
//       can specify their own stride if it differ from the default one (i.e. if the extended union
//       type is bigger than the default union type).
public struct DefaultParticleModel<let BYTES_STRIDE: uint = DEFAULT_MODEL_BYTES_STRIDE>: IParticleModel {
    uint dummy; // Dummy field because naga doesn’t accept empty structs.

    public __init() {
        this.dummy = 0;
    }

    public func model_flags(models: RWByteAddressBuffer, particle_id: uint) -> ModelFlags {
        // NOTE: we don’t have any fluid model yet.
        return ModelFlags::NONE;
    }

    public func update(
        models: RWByteAddressBuffer,
        data: ParticleUpdateData,
        inout def_grad: Mat<float>,
    ) -> ModelUpdateResult {
        let tag_offset = data.particle_id * BYTES_STRIDE;
        let model_offset = tag_offset + 4;
        let model_tag = models.Load(tag_offset);

        switch (DefaultParticleModelType(model_tag)) {
            case DefaultParticleModelType::ElasticLinear: {
                let model = models.Load<LinearElasticModel>(model_offset);
                var stress = model.kirchoff_stress(def_grad);
            #if SLOSH_PML
                stress += viscous_stress(model.lambda, model.mu, data.velocity_gradient, data.stiffness_damping);
            #endif
                return ModelUpdateResult(stress);
            }
            case DefaultParticleModelType::ElasticNeoHookean: {
                let model = models.Load<NeoHookeanModel>(model_offset);
                var stress = model.kirchoff_stress(def_grad);
            #if SLOSH_PML
                stress += viscous_stress(model.lambda, model.mu, data.velocity_gradient, data.stiffness_damping);
            #endif
                return ModelUpdateResult(stress);
            }
            case DefaultParticleModelType::SandLinear: {
                let model = models.Load<SandModel<LinearElasticModel>>(model_offset);
                let projection = model.plastic.project(model.plastic_state, def_grad);
                let state_offset = model_offset + model.state_offset();
                models.Store<DruckerPragerPlasticState>(state_offset, projection.state);
                var stress = model.elastic.kirchoff_stress(projection.deformation_gradient);
            #if SLOSH_PML
                stress += viscous_stress(model.elastic.lambda, model.elastic.mu, data.velocity_gradient, data.stiffness_damping);
            #endif
                def_grad = projection.deformation_gradient;
                return ModelUpdateResult(stress);
            }
            case DefaultParticleModelType::SandNeoHookean: {
                let model = models.Load<SandModel<NeoHookeanModel>>(model_offset);
                let projection = model.plastic.project(model.plastic_state, def_grad);
                let state_offset = model_offset + model.state_offset();
                models.Store<DruckerPragerPlasticState>(state_offset, projection.state);
                var stress = model.elastic.kirchoff_stress(projection.deformation_gradient);
            #if SLOSH_PML
                stress += viscous_stress(model.elastic.lambda, model.elastic.mu, data.velocity_gradient, data.stiffness_damping);
            #endif
                def_grad = projection.deformation_gradient;
                return ModelUpdateResult(stress);
            }
        #if SLOSH_PML
            case DefaultParticleModelType::AbsorbingPml: {
                let model = models.Load<PmlModel>(model_offset);
                let stress = model.kirchoff_stress(def_grad)
                    + model.viscous_stress(data.velocity_gradient, data.stiffness_damping);
                return ModelUpdateResult(stress, -1.0, model.mass_scale());
            }
        #endif
            default:
                return ModelUpdateResult(Mat<float>(0.0));
        }
    }

    public func timestep_bound(
        models: ByteAddressBuffer,
        particle_id: uint,
        particle_density0: float,
        def_grad: Mat<float>,
        particle_velocity: Vect<float>,
        cell_width: float,
    ) -> float {
        let tag_offset = particle_id * BYTES_STRIDE;
        let model_offset = tag_offset + 4;
        let model_tag = models.Load(tag_offset);
        let def_grad_det = determinant(def_grad);

        switch (DefaultParticleModelType(model_tag)) {
            case DefaultParticleModelType::ElasticLinear: {
                let model = models.Load<LinearElasticModel>(model_offset);
                return model.timestep_bound(particle_density0, particle_velocity, def_grad_det, 1.0, cell_width);
            }
            case DefaultParticleModelType::ElasticNeoHookean: {
                let model = models.Load<NeoHookeanModel>(model_offset);
                return model.timestep_bound(particle_density0, particle_velocity, def_grad_det, 1.0, cell_width);
            }
            case DefaultParticleModelType::SandLinear: {
                let model = models.Load<SandModel<LinearElasticModel>>(model_offset);
                return model.elastic.timestep_bound(particle_density0, particle_velocity, def_grad_det, 1.0, cell_width);
            }
            case DefaultParticleModelType::SandNeoHookean: {
                let model = models.Load<SandModel<NeoHookeanModel>>(model_offset);
                return model.elastic.timestep_bound(particle_density0, particle_velocity, def_grad_det, 1.0, cell_width);
            }
        #if SLOSH_PML
            case DefaultParticleModelType::AbsorbingPml: {
                let model = models.Load<PmlModel>(model_offset);
                return model.timestep_bound(particle_density0, particle_velocity, def_grad_det, cell_width);
            }
        #endif
            default:
                return 0.0;
        }
    }
}