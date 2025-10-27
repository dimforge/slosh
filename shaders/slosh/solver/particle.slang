module particle;

public struct Position {
    public vector<float, DIM> pt;
}

public struct Dynamics {
    // NOTE: with this arrangement, we have
    //       only a 4-bytes padding at the end
    //       of the struct in 2D.
    public vector<float, DIM> velocity;
    public matrix<float, DIM, DIM> def_grad;
    public matrix<float, DIM, DIM> affine;
    public float vel_grad_det;
    public Cdf cdf;
    public float init_volume;
    public float init_radius;
    public float mass;
    public uint enabled;
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
        this.rigid_vel = rigid_vel,
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