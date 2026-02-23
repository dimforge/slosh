#if DIM == 3

module prep_readback3;

import slosh.solver.particle;
import slosh.grid.grid;
import slosh.solver.params;
import stensor.geometry.svd3;
import stensor.utils.mat;
import slosh.aliases;

struct ReadbackData {
    float4 color;
    float3x3 deformation;
    float3 position;
}

struct RenderConfig {
    uint mode;
}

static const uint DEFAULT = 0;
static const uint VOLUME = 1;
static const uint VELOCITY = 2;
static const uint PHASE = 3;
static const uint CDF_NORMALS = 4;
static const uint CDF_DISTANCES = 5;
static const uint CDF_SIGNS = 6;

[shader("compute")]
[numthreads(64, 1, 1)]
func prep_readback(
    uint3 invocation_id: SV_DispatchThreadID,
    RWStructuredBuffer<ReadbackData> instances,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<Kinematics> particles_kin,
    StructuredBuffer<Cdf> particles_cdf,
    StructuredBuffer<Mat<float>> particles_def_grad,
    StructuredBuffer<ParticleProperties> particles_props,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<float4> base_colors,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<RenderConfig> config,
) {
    let particle_id = invocation_id.x;

    if (particle_id < instances.getCount()) {
        let failed = particles_kin[particle_id].enabled == 0;
        let def_grad = particles_def_grad[particle_id];
        let init_deformation = diag(Vect<float>(particles_props[particle_id].init_radius * 2.0));
        instances[particle_id].deformation = mul(init_deformation, clamp(def_grad, Mat<float>(-4.0), Mat<float>(4.0)));
        instances[particle_id].position = particles_pos[particle_id].pt;

        let color = base_colors[particle_id];
        let cell_width = grid[0].cell_width;
        let mode = config[0].mode;
        let dt = params.dt;
        let max_vel = cell_width / dt;

        if (mode == DEFAULT) {
            instances[particle_id].color = color;
        } else if (mode == VELOCITY) {
            let vel = particles_kin[particle_id].velocity;
            instances[particle_id].color = float4(abs(vel) * dt * 100.0 + float3(0.2), color.w);
        } else if (mode == VOLUME) {
            let svd = Svd3(def_grad);
            let color_xyz = (float3(1.0) - svd.S) / 0.005 + float3(0.2);
            instances[particle_id].color = float4(color_xyz, color.w);
        } else if (mode == PHASE) {
            let phase = particles_props[particle_id].phase;
            instances[particle_id].color = float4(0.0, 0.4 * phase, 0.4 * (1.0 - phase), color.w);
        } else if (mode == CDF_NORMALS) {
            let particle_normal = particles_cdf[particle_id].normal;
            if (all(particle_normal == float3(0.0))) {
                instances[particle_id].color = float4(0.0, 0.0, 0.0, color.w);
            } else {
                let n = (particle_normal + float3(1.0)) / 2.0;
                instances[particle_id].color = float4(n.x, n.y, 0.0, color.w);
            }
        } else if (mode == CDF_DISTANCES) {
            let d = particles_cdf[particle_id].signed_distance / (cell_width * 1.5);
            if (d > 0.0) {
                instances[particle_id].color = float4(0.0, abs(d), 0.0, color.w);
            } else {
                instances[particle_id].color = float4(abs(d), 0.0, 0.0, color.w);
            }
        } else if (mode == CDF_SIGNS) {
             let d = particles_cdf[particle_id].affinity;
             let a = (d >> 16) & (d & 0x0000ffff);
             if (d == 0) {
                 instances[particle_id].color = float4(0.0, 0.0, 0.0, color.w);
             } else if (a == 0) {
                 instances[particle_id].color = float4(0.0, 1.0, 0.0, color.w);
             } else {
                 instances[particle_id].color = float4(1.0, 0.0, 0.0, color.w);
             }
         }

         if (failed) {
            instances[particle_id].color = float4(1.0, 0.0, 0.0, 1.0);
         }
    }
}

#endif
