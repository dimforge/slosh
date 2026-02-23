module interfaces;

import slosh.aliases;
import slosh.solver.particle;

public struct ModelUpdateResult {
    public Mat<float> kirchoff_stress;
    // Negative means unchanged. If >= 0, the model modified init_volume (e.g., swell).
    public float init_volume;

    public __init(Mat stress) {
        this.kirchoff_stress = stress;
        this.init_volume = -1.0;
    }

    public __init(Mat stress, float init_volume) {
        this.kirchoff_stress = stress;
        this.init_volume = init_volume;
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
    public matrix<float, DIM, DIM> velocity_gradient;
    public float init_volume;
    public float phase;

    public __init(float dt, float cell_width, uint particle_id, matrix<float, DIM, DIM> velocity_gradient, float init_volume, float phase) {
        this.dt = dt;
        this.cell_width = cell_width;
        this.particle_id = particle_id;
        this.velocity_gradient = velocity_gradient;
        this.init_volume = init_volume;
        this.phase = phase;
    }
}

// TODO: keep separate traits for constitutive, plastic, fracture models,
//       or combine everything into a single trait for simplicity?
public interface IParticleModel {
    public func model_flags(models: RWByteAddressBuffer, particle_id: uint) -> ModelFlags;
    public func update(
        models: RWByteAddressBuffer,
        data: ParticleUpdateData,
        inout def_grad: Mat<float>,
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
