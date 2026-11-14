#pragma once
#include <optix.h>
#include <vector_types.h>
struct LaunchParams {
    OptixTraversableHandle handle;
    const float3* origins;
    const float3* directions;
    int* ids;
    float* distances;
    unsigned int count;
};
