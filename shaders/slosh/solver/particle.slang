module particle;

public struct Position {
    public vector<float, DIM> pt;
}

/// Kinematic state for APIC particle-grid transfers.
///
/// Contains the fields needed by P2G and G2P kernels: velocity, mass, affine matrix,
/// external forces, and particle status. Separated from deformation and material
/// properties to reduce GPU memory bandwidth in transfer kernels.
public struct Kinematics {
    // NOTE: when running the particle_update shader, this contains the
    //       particle's velocity gradient. After particle_update, this
    //       contains the affine matrix for APIC transfer.
    public matrix<float, DIM, DIM> affine;
    public vector<float, DIM> velocity;
    // Additional user-defined force applied indirectly to the particle.
    // Reset at each particle_update invocation.
    // NOTE: we store the force multiplied by dt instead of just
    //       the force so we don't have to have access to dt in p2g
    //       (adding that access would exceed 10 bindings…)
    public vector<float, DIM> force_dt;
    public float vel_grad_det;
    public float mass;
    public uint enabled;
}

/// Static per-particle properties that are read-only on the GPU.
///
/// These fields are set once during particle creation and never modified by any
/// GPU shader. Storing them separately allows the GPU to cache this buffer
/// more aggressively and avoids unnecessary write-back bandwidth.
public struct ParticleProperties {
    public float init_volume;
    public float init_radius;
    public float damping;
    public float phase;
    public uint fixed;
}

public struct RigidParticleIndices {
#if DIM == 2
    public vector<uint, DIM> segment;
#else
    public vector<uint, DIM> triangle;
#endif
    public uint collider;
}

public struct Cdf {
    // Should we pack this?
    public vector<float, DIM> normal;
    public vector<float, DIM> rigid_vel;
    public float signed_distance;
    public uint affinity;
//    // Index to the closest collider.
//    closest_id: u32,

    public __init() {
        this.normal = vector<float, DIM>(0.0);
        this.rigid_vel = vector<float, DIM>(0.0);
        this.signed_distance = 0.0;
        this.affinity = 0;
    }

    public __init(vector<float, DIM> normal, vector<float, DIM> rigid_vel, float signed_distance, uint affinity) {
        this.normal = normal;
        this.rigid_vel = rigid_vel;
        this.signed_distance = signed_distance;
        this.affinity = affinity;
    }
}

public func closest_grid_pos(part_pos: Position, cell_width: float) -> vector<float, DIM> {
    return round(part_pos.pt / cell_width) * cell_width;
}

public func associated_cell_index_in_block_off_by_one(part_pos: Position, cell_width: float) -> vector<uint, DIM> {
    let assoc_cell = round(part_pos.pt / cell_width) - 1.0;
#if DIM == 2
    let assoc_block = floor(assoc_cell / 8.0) * 8;
#else
    let assoc_block = floor(assoc_cell / 4.0) * 4;
#endif
    return vector<uint, DIM>(assoc_cell - assoc_block); // Will always be positive.
}

public func associated_grid_pos(part_pos: Position, cell_width: float) -> vector<float, DIM> {
    return (round(part_pos.pt / cell_width) - vector<float, DIM>(1.0)) * cell_width;
}

public func dir_to_closest_grid_node(part_pos: Position, cell_width: float) -> vector<float, DIM> {
    return closest_grid_pos(part_pos, cell_width) - part_pos.pt;
}

public func dir_to_associated_grid_node(part_pos: Position, cell_width: float) -> vector<float, DIM> {
    return associated_grid_pos(part_pos, cell_width) - part_pos.pt;
}