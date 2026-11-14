#include <optix_device.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

extern "C" __global__ void __raygen__main() {
    const unsigned int i = optixGetLaunchIndex().x;
    if (i >= params.count) return;
    unsigned int id = 0xffffffffu;
    unsigned int distance = __float_as_uint(1e37f);
    // Closest hit, two-sided triangles, no early-termination or face filtering.
    optixTrace(params.handle, params.origins[0], params.directions[i],
        0.0f, 1e37f, 0.0f, 255,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, id, distance);
    params.ids[i] = static_cast<int>(id);
    params.distances[i] = __uint_as_float(distance);
}

extern "C" __global__ void __miss__main() {
    optixSetPayload_0(0xffffffffu);
    optixSetPayload_1(__float_as_uint(1e37f));
}

extern "C" __global__ void __closesthit__main() {
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}
