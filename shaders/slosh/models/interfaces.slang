module interfaces;

import nexus.aliases;
import slosh.solver.particle;

public struct ModelUpdateResult {
    public Mat<float> kirchoff_stress;

    public __init(Mat stress) {
        this.kirchoff_stress = stress;
    }
}

public enum class ModelFlags {
    NONE = 0,
    FLUID = 1 << 0
}

public struct ParticleUpdateData {
    public float dt;
    public float cell_width;
    public uint particle_id;

    public __init(float dt, float cell_width, uint particle_id) {
        this.dt = dt;
        this.cell_width = cell_width;
        this.particle_id = particle_id;
    }
}

// TODO: keep separate traits for constitutive, plastic, fracture models,
//       or combine everything into a single trait for simplicity?
public interface IParticleModel {
    public func model_flags(models: RWByteAddressBuffer, particle_id: uint) -> ModelFlags;
    public func update(
        models: RWByteAddressBuffer,
        data: ParticleUpdateData,
        inout dynamics: Dynamics,
    ) -> ModelUpdateResult;
    public func timestep_bound(
        models: ByteAddressBuffer,
        particle_id: uint,
        particle_density0: float,
        def_grad: Mat<float>,
        particle_velocity: Vect<float>,
        cell_width: float,
    ) -> float;
}
