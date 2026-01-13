module params;

public struct SimulationParams {
    public vector<float, DIM> gravity;

#if DIM == 2
    public float padding; // Due to uniform size limits
#endif

    public float dt;
}