#define NOMINMAX
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "params.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static thread_local std::string last_error;
static void cuda_check(CUresult value, const char* operation) {
    if (value == CUDA_SUCCESS) return;
    const char *name=nullptr, *description=nullptr;
    cuGetErrorName(value,&name); cuGetErrorString(value,&description);
    throw std::runtime_error(std::string(operation)+": "+(name?name:"")+" "+(description?description:""));
}
static void optix_check(OptixResult value, const char* operation) {
    if (value != OPTIX_SUCCESS)
        throw std::runtime_error(std::string(operation)+": OptiX error "+std::to_string(value));
}
#define CU(call) cuda_check((call), #call)
#define OX(call) optix_check((call), #call)
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

struct Scene {
    CUdevice device=0;
    CUcontext cuda_context=nullptr;
    CUstream stream=nullptr;
    OptixDeviceContext context=nullptr;
    OptixModule module=nullptr;
    OptixPipeline pipeline=nullptr;
    OptixProgramGroup groups[3]={};
    OptixShaderBindingTable sbt={};
    OptixTraversableHandle handle=0;
    CUdeviceptr gas=0,records[3]={};
    struct DirectSlot { LaunchParams* host=nullptr; CUdeviceptr gpu=0; CUevent done=nullptr; };
    DirectSlot slots[64];
    unsigned int next_slot=0;


    ~Scene() {
        if (cuda_context) cuCtxSetCurrent(cuda_context);
        if (cuda_context) cuCtxSynchronize();
        for (auto& slot:slots) {
            if (slot.host) cuMemFreeHost(slot.host);
            if (slot.gpu) cuMemFree(slot.gpu);
            if (slot.done) cuEventDestroy(slot.done);
        }
        if (stream) cuStreamSynchronize(stream);
        for (CUdeviceptr pointer : {gas,records[0],records[1],records[2]})
            if (pointer) cuMemFree(pointer);
        if (pipeline) optixPipelineDestroy(pipeline);
        for (auto group:groups) if (group) optixProgramGroupDestroy(group);
        if (module) optixModuleDestroy(module);
        if (context) optixDeviceContextDestroy(context);
        if (stream) cuStreamDestroy(stream);
        if (cuda_context) cuDevicePrimaryCtxRelease(device);
    }

    void init(int device_index,const float* vertices,unsigned int vertex_count,const unsigned int* triangles,
              unsigned int triangle_count,const char* ptx_path,const char* cache_dir) {
        CU(cuInit(0)); CU(cuDeviceGet(&device,device_index));
        CU(cuDevicePrimaryCtxRetain(&cuda_context,device)); CU(cuCtxSetCurrent(cuda_context));
        CU(cuStreamCreate(&stream,CU_STREAM_DEFAULT));
        OX(optixInit());
        OptixDeviceContextOptions options={};
        OX(optixDeviceContextCreate(cuda_context,&options,&context));
        OX(optixDeviceContextSetCacheLocation(context,cache_dir));
        OX(optixDeviceContextSetCacheEnabled(context,1));
        std::ifstream input(std::filesystem::u8path(ptx_path),std::ios::binary);
        if (!input) throw std::runtime_error("Cannot read PTX");
        std::string ptx((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
        OptixModuleCompileOptions module_options={};
        module_options.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_options.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        OptixPipelineCompileOptions pipeline_options={};
        pipeline_options.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_options.numPayloadValues=2;
        pipeline_options.numAttributeValues=2;
        pipeline_options.pipelineLaunchParamsVariableName="params";
        pipeline_options.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        char log[8192]={}; size_t log_size=sizeof(log);
        OptixResult result=optixModuleCreate(context,&module_options,&pipeline_options,
            ptx.data(),ptx.size(),log,&log_size,&module);
        if (result!=OPTIX_SUCCESS) throw std::runtime_error(std::string("optixModuleCreate: ")+log);
        OptixProgramGroupOptions group_options={};
        OptixProgramGroupDesc descriptions[3]={};
        descriptions[0].kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descriptions[0].raygen.module=module;
        descriptions[0].raygen.entryFunctionName="__raygen__main";
        descriptions[1].kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
        descriptions[1].miss.module=module;
        descriptions[1].miss.entryFunctionName="__miss__main";
        descriptions[2].kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptions[2].hitgroup.moduleCH=module;
        descriptions[2].hitgroup.entryFunctionNameCH="__closesthit__main";
        log_size=sizeof(log);
        OX(optixProgramGroupCreate(context,descriptions,3,&group_options,log,&log_size,groups));
        OptixPipelineLinkOptions link={}; link.maxTraceDepth=1;
        log_size=sizeof(log);
        OX(optixPipelineCreate(context,&pipeline_options,&link,groups,3,log,&log_size,&pipeline));
        OptixStackSizes stack={};
        for (auto group:groups) OX(optixUtilAccumulateStackSizes(group,&stack,pipeline));
        unsigned int dc_traversal=0,dc_state=0,continuation=0;
        OX(optixUtilComputeStackSizes(&stack,1,0,0,&dc_traversal,&dc_state,&continuation));
        OX(optixPipelineSetStackSize(pipeline,dc_traversal,dc_state,continuation,1));
        for (int i=0;i<3;++i) {
            Record record={}; OX(optixSbtRecordPackHeader(groups[i],&record));
            CU(cuMemAlloc(&records[i],sizeof(record))); CU(cuMemcpyHtoD(records[i],&record,sizeof(record)));
        }
        sbt.raygenRecord=records[0]; sbt.missRecordBase=records[1];
        sbt.missRecordStrideInBytes=sizeof(Record); sbt.missRecordCount=1;
        sbt.hitgroupRecordBase=records[2]; sbt.hitgroupRecordStrideInBytes=sizeof(Record); sbt.hitgroupRecordCount=1;
        CUdeviceptr d_vertices=0,d_triangles=0,temporary=0;
        try {
            CU(cuMemAlloc(&d_vertices,size_t(vertex_count)*12));
            CU(cuMemAlloc(&d_triangles,size_t(triangle_count)*12));
            CU(cuMemcpyHtoD(d_vertices,vertices,size_t(vertex_count)*12));
            CU(cuMemcpyHtoD(d_triangles,triangles,size_t(triangle_count)*12));
            unsigned int geometry_flags=OPTIX_GEOMETRY_FLAG_NONE;
            OptixBuildInput build={}; build.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            build.triangleArray.vertexBuffers=&d_vertices;
            build.triangleArray.numVertices=vertex_count;
            build.triangleArray.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;
            build.triangleArray.vertexStrideInBytes=12;
            build.triangleArray.indexBuffer=d_triangles;
            build.triangleArray.numIndexTriplets=triangle_count;
            build.triangleArray.indexFormat=OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build.triangleArray.indexStrideInBytes=12;
            build.triangleArray.flags=&geometry_flags;
            build.triangleArray.numSbtRecords=1;
            OptixAccelBuildOptions build_options={};
            build_options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            build_options.operation=OPTIX_BUILD_OPERATION_BUILD;
            OptixAccelBufferSizes sizes={};
            OX(optixAccelComputeMemoryUsage(context,&build_options,&build,1,&sizes));
            CU(cuMemAlloc(&temporary,sizes.tempSizeInBytes));
            CU(cuMemAlloc(&gas,sizes.outputSizeInBytes));
            OX(optixAccelBuild(context,stream,&build_options,&build,1,temporary,sizes.tempSizeInBytes,
                gas,sizes.outputSizeInBytes,&handle,nullptr,0));
            CU(cuStreamSynchronize(stream));
        } catch (...) {
            if (temporary) cuMemFree(temporary); if (d_vertices) cuMemFree(d_vertices); if (d_triangles) cuMemFree(d_triangles);
            throw;
        }
        CU(cuMemFree(temporary)); CU(cuMemFree(d_vertices)); CU(cuMemFree(d_triangles));
    }

    void direct(CUdeviceptr origin, CUdeviceptr dirs, unsigned int count,
                CUdeviceptr output_ids, CUdeviceptr output_t, CUstream caller_stream) {
        if (!count) return;
        CU(cuCtxSetCurrent(cuda_context));
        auto& slot=slots[next_slot++ % 64];
        if (!slot.host) {
            CU(cuMemHostAlloc(reinterpret_cast<void**>(&slot.host),sizeof(LaunchParams),0));
            CU(cuMemAlloc(&slot.gpu,sizeof(LaunchParams)));
            CU(cuEventCreate(&slot.done,CU_EVENT_DISABLE_TIMING));
        } else CU(cuEventSynchronize(slot.done));
        *slot.host={handle,reinterpret_cast<const float3*>(origin),reinterpret_cast<const float3*>(dirs),
            reinterpret_cast<int*>(output_ids),reinterpret_cast<float*>(output_t),count};
        CU(cuMemcpyHtoDAsync(slot.gpu,slot.host,sizeof(LaunchParams),caller_stream));
        OX(optixLaunch(pipeline,caller_stream,slot.gpu,sizeof(LaunchParams),&sbt,count,1,1));
        CU(cuEventRecord(slot.done,caller_stream));
    }


};

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
// Restore the calling thread's previous CUDA context on every exported operation.
struct ContextGuard {
    CUcontext previous=nullptr;
    ContextGuard() { cuCtxGetCurrent(&previous); }
    ~ContextGuard() { cuCtxSetCurrent(previous); }
};
EXPORT const char* gpu_last_error() { return last_error.c_str(); }
EXPORT int gpu_abi_version() { return 1; }
EXPORT int gpu_optix_version() { return OPTIX_VERSION; }
EXPORT int gpu_create(int device,const float* vertices,unsigned int nv,const unsigned int* indices,unsigned int nf,
                      const char* ptx,const char* cache,void** result) {
    ContextGuard guard;
    try { auto state=std::make_unique<Scene>(); state->init(device,vertices,nv,indices,nf,ptx,cache); *result=state.release(); return 0; }
    catch (const std::exception& e) { last_error=e.what(); *result=nullptr; return -1; }
}
EXPORT int gpu_direct(void* handle,unsigned long long origin,unsigned long long directions,unsigned int count,
    unsigned long long ids,unsigned long long distances,unsigned long long stream) {
    ContextGuard guard;
    try { static_cast<Scene*>(handle)->direct(origin,directions,count,ids,distances,reinterpret_cast<CUstream>(stream)); return 0; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
}
EXPORT void gpu_destroy(void* handle) { ContextGuard guard; delete static_cast<Scene*>(handle); }
