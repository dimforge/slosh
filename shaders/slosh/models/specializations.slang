module specializations;

import slosh.models.interfaces;
import slosh.models.linear_elasticity;
import slosh.models.neo_hookean_elasticity;
import slosh.models.drucker_prager;
import slosh.solver.particle;
import nexus.aliases;


public export struct ParticleModel: IParticleModel = DefaultParticleModel;

public enum class DefaultParticleModelType {
    ElasticLinear = 0,
    ElasticNeoHookean = 1,
    SandLinear = 2,
    SandNeoHookean = 3,
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
    static const uint BYTES_STRIDE = 52;

    public __init() {
        this.dummy = 0;
    }

    public static func update(
        models: RWByteAddressBuffer,
        particle_id: uint,
        in def_grad: Mat<float>,
    ) -> ModelUpdateResult {
        let tag_offset = particle_id * BYTES_STRIDE;
        let model_offset = tag_offset + 4;
        let model_tag = models.Load(tag_offset);

        switch (DefaultParticleModelType(model_tag)) {
            case DefaultParticleModelType::ElasticLinear: {
                let model = models.Load<LinearElasticModel>(model_offset);
                let stress = model.kirchoff_stress(def_grad);
                return ModelUpdateResult(stress, def_grad);
            }
            case DefaultParticleModelType::ElasticNeoHookean: {
                let model = models.Load<NeoHookeanModel>(model_offset);
                let stress = model.kirchoff_stress(def_grad);
                return ModelUpdateResult(stress, def_grad);
            }
            case DefaultParticleModelType::SandLinear: {
                let model = models.Load<SandModel<LinearElasticModel>>(model_offset);
                let projection = model.plastic.project(model.plastic_state, def_grad);
                let state_offset = model_offset + model.state_offset();
                models.Store<DruckerPragerPlasticState>(state_offset, projection.state);
                let stress = model.elastic.kirchoff_stress(projection.deformation_gradient);
                return ModelUpdateResult(stress, projection.deformation_gradient);
            }
            case DefaultParticleModelType::SandNeoHookean: {
                let model = models.Load<SandModel<NeoHookeanModel>>(model_offset);
                let projection = model.plastic.project(model.plastic_state, def_grad);
                let state_offset = model_offset + model.state_offset();
                models.Store<DruckerPragerPlasticState>(state_offset, projection.state);
                let stress = model.elastic.kirchoff_stress(projection.deformation_gradient);
                return ModelUpdateResult(stress, projection.deformation_gradient);
            }
            default:
                return ModelUpdateResult(Mat<float>(0.0), def_grad);
        }
    }
}