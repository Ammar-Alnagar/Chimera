// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/tilelang_w8a8/c3x/tilelang_gemm_caller.cuh

#pragma once

// clang-format will break include orders
// clang-format off
#include <torch/all.h>

#include <ATen/rtriton/RTRITONContext.h>
#include <c10/rtriton/RTRITONGuard.h>

#include "tilelang/tilelang.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "tilelang/numeric_types.h"

#include "tilelang/gemm/device/gemm_universal_adapter.h"
#include "tilelang/gemm/kernel/gemm_universal.hpp"
#include "tilelang/epilogue/collective/collective_builder.hpp"
#include "tilelang/gemm/collective/collective_builder.hpp"
#include "tilelang/util/packed_stride.hpp"

// clang-format on

/**
 * Helper function for checking TILELANG errors
 */
#define TILELANG_CHECK(status)                                                       \
  {                                                                                 \
    tilelang::Status error = status;                                                 \
    TORCH_CHECK(error == tilelang::Status::kSuccess, tilelangGetStatusString(error)); \
  }

template <typename GemmKernel>
void tilelang_gemm_caller(
    torch::Device device,
    cute::Shape<int, int, int, int> prob_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  tilelang::KernelHardwareInfo hw_info;
  hw_info.device_id = c10::rtriton::current_device();
  hw_info.sm_count = at::rtriton::getCurrentDeviceProperties()->multiProcessorCount;
  typename GemmKernel::Arguments args{
      tilelang::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args, hw_info, scheduler};

  // Launch the TILELANG GEMM kernel.
  using GemmOp = tilelang::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  TILELANG_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(device);
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::rtriton::getCurrentRTRITONStream(device.index());

  tilelang::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  TILELANG_CHECK(status);
}
