module interfaces;

import nexus.aliases;

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
    public float init_volume;

    public __init(float dt, float cell_width, uint particle_id, float init_volume) {
        this.dt = dt;
        this.cell_width = cell_width;
        this.particle_id = particle_id;
        this.init_volume = init_volume;
    }
}

// TODO: keep separate traits for constitutive, plastic, fracture models,
//       or combine everything into a single trait for simplicity?
public interface IParticleModel {
    public func model_flags(models: RWByteAddressBuffer, particle_id: uint) -> ModelFlags;
    public func update(
        models: RWByteAddressBuffer,
        data: ParticleUpdateData,
        // TODO PERF: define as inout to ensure it’s passed by-ref (to avoid useless copy)?
        in vel_grad: Mat<float>,
        inout def_grad: Mat<float>
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
