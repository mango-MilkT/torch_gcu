/*
 * Copyright 2021-2022 Enflame. All Rights Reserved.
 */
#pragma once

#include "csrc/runtime/detail/rt_event.h"
#include "csrc/runtime/detail/rt_executable.h"
#include "csrc/runtime/detail/rt_memory.h"
#include "csrc/runtime/detail/rt_stream.h"
#include "csrc/utils/hasher.h"

namespace hlir {
class Module;
}

namespace torch_dtu {
namespace runtime {

using DtuCtxPtr = std::shared_ptr<detail::Context>;
using DtuMemPtr = std::shared_ptr<detail::Memory>;
using DtuStreamPtr = std::shared_ptr<detail::Stream>;
using DtuExePtr = std::shared_ptr<detail::Executable>;
using DtuEventPtr = std::shared_ptr<detail::Event>;

void dtuSetManualSeed(uint32_t seed);

uint32_t dtuVisibleDeviceCount();

DtuCtxPtr dtuGetContext(int device_id);

DtuMemPtr dtuCreateMemory(DtuCtxPtr ctx, uint64_t nbytes);

DtuMemPtr dtuCreateMemory(DtuCtxPtr ctx, uint64_t nbytes,
                          const std::vector<int64_t> &dims, int dtype);

DtuMemPtr dtuCreateScatterMemory(DtuCtxPtr ctx, uint64_t nbytes,
                                 const std::vector<int64_t> &dims, int dtype);

DtuMemPtr dtuCreateSubMemory(DtuMemPtr parent, uint64_t offest, uint64_t size);

DtuEventPtr dtuCreateEvent(DtuCtxPtr ctx);

DtuStreamPtr dtuCreateStream(DtuCtxPtr ctx);

DtuStreamPtr dtuGetExeDefaultStream(DtuCtxPtr ctx);

void* dtuGetExeDefaultStream(int device_id);

DtuStreamPtr dtuGetDmaDefaultStream(DtuCtxPtr ctx);

void* dtuGetDmaDefaultStream(int device_id);

DtuExePtr dtuTryGetCacheExe(util::hash_t hash, DtuCtxPtr ctx);

DtuExePtr dtuCreateExeHlo(const void *hlo_data, size_t size, util::hash_t hash,
                          DtuCtxPtr ctx);

DtuExePtr dtuCreateExeHlir(std::shared_ptr<hlir::Module> hlir_data,
                           util::hash_t hash, DtuCtxPtr ctx);

DtuExePtr dtuCreateExeHlir(std::shared_ptr<builder::Builder> builder,
                           util::hash_t hash, DtuCtxPtr ctx);

DtuExePtr dtuCreateDynamicExeHlir(std::shared_ptr<hlir::Module> hlir_data,
                                  util::hash_t hash, DtuCtxPtr ctx,
                                  const std::vector<OpParamPtr> &op_params);

void dtuStreamSynchronize(DtuStreamPtr pstream);

void dtuMemcpyH2DSync(DtuMemPtr pmem, const void *pdata, uint64_t nbytes);

void dtuMemcpyD2HSync(void *pdata, DtuMemPtr pmem, uint64_t nbytes);

void dtuMemcpyD2DSync(DtuMemPtr pdst, DtuMemPtr psrc, uint64_t nbytes);

void dtuMemset32Sync(DtuMemPtr pmem, uint32_t pattern, uint64_t nbytes);

void dtuRunExeSync(DtuExePtr exe, const std::vector<DtuMemPtr> &ins,
                   const std::vector<DtuMemPtr> &outs);

void dtuMemcpyH2DAsync(DtuMemPtr pmem, const void *pdata, uint64_t nbytes,
                       DtuStreamPtr pstream);

void dtuMemcpyD2HAsync(void *pdata, DtuMemPtr pmem, uint64_t nbytes,
                       DtuStreamPtr pstream);

void dtuMemcpyD2DAsync(DtuMemPtr pdst, DtuMemPtr psrc, uint64_t nbytes,
                       DtuStreamPtr pstream);

void dtuMemcpyD2DAsync(DtuMemPtr pdst, void *psrc, uint64_t nbytes);

void dtuMemset32Async(DtuMemPtr pmem, uint32_t pattern, uint64_t nbytes,
                      DtuStreamPtr pstream);

void dtuRunExeAsync(DtuExePtr exe, const std::vector<DtuMemPtr> &ins,
                    const std::vector<DtuMemPtr> &outs, DtuStreamPtr pstream);

void dtuEventRecord(DtuEventPtr pevent, DtuStreamPtr pstream);

void dtuStreamWaitEvent(DtuStreamPtr pstream, DtuEventPtr pevent);

void dtuLaunchHostFunc(DtuStreamPtr pstream, const std::function<void()> &fn);

std::string dtuGetRTMetricsReport();

void *dtuMemImpl(DtuMemPtr pmem);

void *dtuStreamImpl(DtuStreamPtr pstream);

void dtuDumpGlobHbmUseInfo(uint8_t device_id);

double dtuGetHbmAllocSize(uint8_t device_id);

void dtuDumpHbmAllocInfo(uint8_t device_id);

uint64_t dtuGetHbmUsedSize(uint8_t device_id);

void dtuClearGraphCaches();

}  // namespace runtime
}  // namespace torch_dtu
