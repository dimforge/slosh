module params;

public struct SimulationParams {
    public vector<float, DIM> gravity;

#if DIM == 2
    public float padding; // Due to uniform size limits
#endif

    public float dt;
}
/// Time-integration flags read by the particle update. A separate uniform rather than a field of
/// `SimulationParams`, so the public struct literal users write stays unchanged.
public struct IntegratorFlags {
    /// Non-zero when the grid update is the implicit solver. The particle update then leaves
    /// the stress out of the APIC affine matrix: the solver evaluates every force itself at the
    /// trial state.
    public uint implicit;
    public uint padding0;
    public uint padding1;
    public uint padding2;
}
