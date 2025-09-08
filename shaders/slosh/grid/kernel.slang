//! Quadratic kernel.

module kernel;

#if DIM == 2
public static const uint NBH_LEN = 9;
public static const Array<uint2, NBH_LEN> NBH_SHIFTS = Array<uint2, NBH_LEN>(
    uint2(2, 2),
    uint2(2, 0),
    uint2(2, 1),
    uint2(0, 2),
    uint2(0, 0),
    uint2(0, 1),
    uint2(1, 2),
    uint2(1, 0),
    uint2(1, 1),
);
public static const Array<uint, NBH_LEN> NBH_SHIFTS_SHARED = Array<uint, NBH_LEN>(
    22, 2, 12, 20, 0, 10, 21, 1, 11,
);
#else
public static const uint NBH_LEN = 27;
public static const Array<uint3, NBH_LEN> NBH_SHIFTS = Array<uint3, NBH_LEN>(
    uint3(2, 2, 2),
    uint3(2, 0, 2),
    uint3(2, 1, 2),
    uint3(0, 2, 2),
    uint3(0, 0, 2),
    uint3(0, 1, 2),
    uint3(1, 2, 2),
    uint3(1, 0, 2),
    uint3(1, 1, 2),
    uint3(2, 2, 0),
    uint3(2, 0, 0),
    uint3(2, 1, 0),
    uint3(0, 2, 0),
    uint3(0, 0, 0),
    uint3(0, 1, 0),
    uint3(1, 2, 0),
    uint3(1, 0, 0),
    uint3(1, 1, 0),
    uint3(2, 2, 1),
    uint3(2, 0, 1),
    uint3(2, 1, 1),
    uint3(0, 2, 1),
    uint3(0, 0, 1),
    uint3(0, 1, 1),
    uint3(1, 2, 1),
    uint3(1, 0, 1),
    uint3(1, 1, 1),
);
public static const Array<uint, NBH_LEN> NBH_SHIFTS_SHARED = Array<uint, NBH_LEN>(
    86, 74, 80, 84, 72, 78, 85, 73, 79, 14, 2, 8, 12, 0, 6, 13, 1, 7, 50, 38, 44, 48, 36, 42, 49, 37, 43
);
#endif

public struct QuadraticKernel {
    public static func inv_d(cell_width: float) -> float {
        return 4.0 / (cell_width * cell_width);
    }

    public static func eval_all(x: float) -> float3 {
        return float3(
            0.5 * (1.5 - x) * (1.5 - x),
            0.75 - (x - 1.0) * (x - 1.0),
            0.5 * (x - 0.5) * (x - 0.5)
        );
    }

    public static func eval(x: float) -> float {
        let x_abs = abs(x);
        let part1 = 0.75 - x_abs * x_abs;
        let part2 = 0.5 * (1.5 - x_abs) * (1.5 - x_abs);
        let part3 = 0.0;
        return select(x_abs < 0.5, part1, select(x_abs < 1.5, part2, part3));
    }

    public static func eval_derivative(x: float) -> float {
        let x_abs = abs(x);
        let part1 = -2.0 * sign(x) * x_abs;
        let part2 = -sign(x) * (1.5 - x_abs);
        let part3 = 0.0;
        return select(x_abs < 0.5, part1, select(x_abs < 1.5, part2, part3));
    }

    #if DIM == 2
    public static func precompute_weights(
        ref_elt_pos_minus_particle_pos: float2,
        h: float,
    ) -> float2x3 {
        return float2x3(
            eval_all(-ref_elt_pos_minus_particle_pos.x / h),
            eval_all(-ref_elt_pos_minus_particle_pos.y / h),
        );
    }
    #else
    public static func precompute_weights(
        ref_elt_pos_minus_particle_pos: float3,
        h: float,
    ) -> float3x3 {
        return float3x3(
            eval_all(-ref_elt_pos_minus_particle_pos.x / h),
            eval_all(-ref_elt_pos_minus_particle_pos.y / h),
            eval_all(-ref_elt_pos_minus_particle_pos.z / h),
        );
    }
    #endif
}