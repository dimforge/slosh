module interfaces;

import nexus.aliases;

public struct ModelUpdateResult {
    public Mat<float> kirchoff_stress;
    public Mat<float> deformation_gradient;

    public __init(Mat stress, Mat def_grad) {
        this.kirchoff_stress = stress;
        this.deformation_gradient = def_grad;
    }
}

// TODO: keep separate traits for constitutive, plastic, fracture models,
//       or combine everything into a single trait for simplicity?
public interface IParticleModel {
    public func update(
            models: RWByteAddressBuffer,
            particle_id: uint,
            in def_grad: Mat<float>
    ) -> ModelUpdateResult;
}
