module params;

public struct SimulationParams {
    public vector<float, DIM> gravity;

#if DIM == 2
    public float padding; // Due to uniform size limits
#endif

    public float dt;
    /// Rayleigh mass-proportional damping coefficient (1/s).
    public float damping;
}