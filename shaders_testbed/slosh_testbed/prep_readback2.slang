module prep_readback2;

import slosh.solver.particle;
import slosh.grid.grid;
import slosh.solver.params;
import stensor.geometry.svd2;
import stensor.utils.mat;
import nexus.aliases;

struct ReadbackData {
    float4 color;
    float2x2 deformation;
    float2 position;
}

struct RenderConfig {
    uint mode;
}

static const uint DEFAULT = 0;
static const uint VOLUME = 1;
static const uint VELOCITY = 2;
static const uint CDF_NORMALS = 3;
static const uint CDF_DISTANCES = 4;
static const uint CDF_SIGNS = 5;

[shader("compute")]
[numthreads(64, 1, 1)]
func prep_readback(
    uint3 invocation_id: SV_DispatchThreadID,
    RWStructuredBuffer<ReadbackData> instances,
    StructuredBuffer<Position> particles_pos,
    StructuredBuffer<Dynamics> particles_dyn,
    StructuredBuffer<Grid> grid,
    StructuredBuffer<float4> base_colors,
    ConstantBuffer<SimulationParams> params,
    StructuredBuffer<RenderConfig> config,
) {
    let particle_id = invocation_id.x;

    if (particle_id < instances.getCount()) {
        let def_grad = particles_dyn[particle_id].def_grad;
        let init_deformation = diag(Vect<float>(particles_dyn[particle_id].init_radius * 2.0));
        instances[particle_id].deformation = def_grad * init_deformation;
        instances[particle_id].position = float2(particles_pos[particle_id].pt);

        let color = base_colors[particle_id];
        let cell_width = grid[0].cell_width;
        let mode = config[0].mode;
        let dt = params.dt;
        let max_vel = cell_width / dt;

        if (mode == DEFAULT) {
            instances[particle_id].color = color;
        } else if (mode == VELOCITY) {
            let vel = particles_dyn[particle_id].velocity;
            instances[particle_id].color = float4(abs(vel) * dt * 100.0 + float2(0.2), color.z, color.w);
        } else if (mode == VOLUME) {
            let svd = Svd2(def_grad);
            let color_xy = (float2(1.0) - svd.S) / 0.005 + float2(0.2);
            instances[particle_id].color = float4(color_xy, color.z, color.w);
        } else if (mode == CDF_NORMALS) {
            let particle_normal = particles_dyn[particle_id].cdf.normal;
            if (all(particle_normal == float2(0.0))) {
                instances[particle_id].color = float4(0.0, 0.0, 0.0, color.w);
            } else {
                let n = (particle_normal + float2(1.0)) / 2.0;
                instances[particle_id].color = float4(n.x, n.y, 0.0, color.w);
            }
        } else if (mode == CDF_DISTANCES) {
            let d = particles_dyn[particle_id].cdf.signed_distance / (cell_width * 1.5);
            if (d > 0.0) {
                instances[particle_id].color = float4(0.0, abs(d), 0.0, color.w);
            } else {
                instances[particle_id].color = float4(abs(d), 0.0, 0.0, color.w);
            }
        } else if (mode == CDF_SIGNS) {
             let d = particles_dyn[particle_id].cdf.affinity;
             let a = (d >> 16) & (d & 0x0000ffff);
             if (d == 0) {
                 instances[particle_id].color = float4(0.0, 0.0, 0.0, color.w);
             } else if (a == 0) {
                 instances[particle_id].color = float4(0.0, 1.0, 0.0, color.w);
             } else {
                 instances[particle_id].color = float4(1.0, 0.0, 0.0, color.w);
             }
         }
    }
}

//@compute @workgroup_size(64, 1, 1)
//fn main_rigid_particles(
//    @builtin(global_invocation_id) tid: uint3,
//) {
//    let particle_id = tid.x;
//
//    if particle_id < arrayLength(&instances) {
//        instances[particle_id].deformation = float3x3f(
//            0.4, 0.0, 0.0,
//            0.0, 0.4, 0.0,
//            0.0, 0.0, 0.4,
//        );
//        instances[particle_id].position = float3(particles_pos[particle_id].pt, 0.0);
//        instances[particle_id].color = instances[particle_id].base_color;
//    }
//}