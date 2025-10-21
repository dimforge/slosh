module default_model;

import slosh.models.interfaces;
import slosh.models.linear_elasticity;
import slosh.models.neo_hookean_elasticity;
import slosh.models.drucker_prager;
import slosh.solver.particle;
import nexus.aliases;


public enum class DefaultParticleModelType {
    ElasticLinear = 0,
    ElasticNeoHookean = 1,
    SandLinear = 2,
    SandNeoHookean = 3,
    // Provided to help downstream users extend with their own enum.
    LEN = 4,
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

public struct DefaultParticleModel {
    uint dummy; // naga doesn’t allow empty structs so we add an unused field.

    // TODO: hard-coding this is quite fragile as is must match the size of
    //       the host-side enum. Should it be configured by the host?
    // This **MUST** match the byte size of `GpuParticleModel` on the rust side.
    public static const uint BYTES_STRIDE = 52;

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
        in vel_grad: Mat<float>,
        inout def_grad: Mat<float>,
    ) -> ModelUpdateResult {
        let tag_offset = data.particle_id * BYTES_STRIDE;
        let model_offset = tag_offset + 4;
        let model_tag = models.Load(tag_offset);

        switch (DefaultParticleModelType(model_tag)) {
            case DefaultParticleModelType::ElasticLinear: {
                let model = models.Load<LinearElasticModel>(model_offset);
                let stress = model.kirchoff_stress(def_grad);
                return ModelUpdateResult(stress);
            }
            case DefaultParticleModelType::ElasticNeoHookean: {
                let model = models.Load<NeoHookeanModel>(model_offset);
                let stress = model.kirchoff_stress(def_grad);
                return ModelUpdateResult(stress);
            }
            case DefaultParticleModelType::SandLinear: {
                let model = models.Load<SandModel<LinearElasticModel>>(model_offset);
                let projection = model.plastic.project(model.plastic_state, def_grad);
                let state_offset = model_offset + model.state_offset();
                models.Store<DruckerPragerPlasticState>(state_offset, projection.state);
                let stress = model.elastic.kirchoff_stress(projection.deformation_gradient);
                // TODO PERF: have plasticity modify def_grad in-place?
                def_grad = projection.deformation_gradient;
                return ModelUpdateResult(stress);
            }
            case DefaultParticleModelType::SandNeoHookean: {
                let model = models.Load<SandModel<NeoHookeanModel>>(model_offset);
                let projection = model.plastic.project(model.plastic_state, def_grad);
                let state_offset = model_offset + model.state_offset();
                models.Store<DruckerPragerPlasticState>(state_offset, projection.state);
                let stress = model.elastic.kirchoff_stress(projection.deformation_gradient);
                // TODO PERF: have plasticity modify def_grad in-place?
                def_grad = projection.deformation_gradient;
                return ModelUpdateResult(stress);
            }
            default:
                return ModelUpdateResult(Mat<float>(0.0));
        }
    }
}