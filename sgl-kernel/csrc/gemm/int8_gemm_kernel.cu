/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/rtriton/RTRITONContext.h>
#include <tilelang/tilelang.h>
#include <tilelang/epilogue/thread/linear_combination.h>
#include <tilelang/epilogue/threadblock/epilogue_with_visitor.h>
#include <tilelang/gemm/device/gemm.h>
#include <tilelang/gemm/device/gemm_universal_adapter.h>
#include <tilelang/numeric_types.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <tilelang/epilogue/collective/collective_builder.hpp>
#include <tilelang/gemm/collective/collective_builder.hpp>
#include <tilelang/gemm/kernel/gemm_universal.hpp>
#include <tilelang/util/packed_stride.hpp>

#include "tilelang_extensions/epilogue/epilogue_per_row_per_col_scale.h"
#include "tilelang_extensions/gemm/gemm_universal_base_compat.h"
#include "tilelang_extensions/gemm/gemm_with_epilogue_visitor.h"
#include "utils.h"

using namespace cute;

template <
    typename ElementOutput,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int NumStages>
void tilelang_int8_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  using OperatorClass = tilelang::arch::OpClassTensorOp;
  using ThreadblockSwizzle = tilelang::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  using DefaultGemmConf = tilelang::gemm::device::
      DefaultGemmConfiguration<OperatorClass, ArchTag, ElementInputA, ElementInputB, ElementOutput, ElementCompute>;
  using EpilogueOutputOp = typename DefaultGemmConf::EpilogueOutputOp;

  using GemmKernel_ = typename tilelang::gemm::kernel::DefaultGemm<
      ElementInputA,
      tilelang::layout::RowMajor,
      DefaultGemmConf::kAlignmentA,
      ElementInputB,
      tilelang::layout::ColumnMajor,
      DefaultGemmConf::kAlignmentB,
      ElementOutput,
      tilelang::layout::RowMajor,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      ThreadblockSwizzle,
      NumStages,
      true,
      typename DefaultGemmConf::Operator>::GemmKernel;

  using AlphaColTileIterator = tilelang::epilogue::threadblock::PredicatedTileIterator<
      tilelang::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
          GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess,
          tilelang::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;

  using EpilogueVisitor = typename tilelang::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
      ThreadblockShape,
      GemmKernel_::kThreadCount,
      AlphaColTileIterator,
      typename GemmKernel_::Epilogue::OutputTileIterator,
      ElementAccumulator,
      ElementCompute,
      EpilogueOutputOp>;

  using Epilogue = typename tilelang::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

  using GemmKernel =
      tilelang::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  using Gemm = tilelang::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = static_cast<ElementInputA*>(mat_a.data_ptr());
  auto b_ptr = static_cast<ElementInputB*>(mat_b.data_ptr());
  auto o_ptr = static_cast<ElementOutput*>(out.data_ptr());

  auto a_s_ptr = static_cast<ElementCompute*>(scales_a.data_ptr());
  auto b_s_ptr = static_cast<ElementCompute*>(scales_b.data_ptr());

  int64_t lda = mat_a.stride(0);
  int64_t ldb = mat_b.stride(1);
  int64_t ldd = out.stride(0);

  ElementOutput* bias_ptr = nullptr;
  int64_t ldc = 0;
  if (bias) {
    bias_ptr = static_cast<ElementOutput*>(bias->data_ptr());
  }

  typename EpilogueOutputOp::Params linearScalingParams;
  typename EpilogueVisitor::Arguments visitor_args{linearScalingParams};

  typename Gemm::Arguments args{
      {m, n, k}, {a_ptr, lda}, {b_ptr, ldb}, {b_s_ptr, 0}, {a_s_ptr, 0}, {bias_ptr, ldc}, {o_ptr, ldd}, visitor_args};

  auto workspace = torch::empty(
      gemm_op.get_workspace_size(args), torch::TensorOptions().dtype(torch::kUInt8).device(mat_a.device()));

  auto stream = at::rtriton::getCurrentRTRITONStream(mat_a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(
      can_implement == tilelang::Status::kSuccess,
      "gemm cannot implement, error: ",
      tilelangGetStatusString(can_implement));

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == tilelang::Status::kSuccess, "gemm executioin failed, error: ", tilelangGetStatusString(status));
}

template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm75_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  if (m <= 32) {
    tilelang_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        tilelang::gemm::GemmShape<32, 128, 64>,
        tilelang::gemm::GemmShape<32, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (m <= 64) {
    tilelang_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        tilelang::gemm::GemmShape<64, 128, 128>,
        tilelang::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (m <= 256) {
    tilelang_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        tilelang::gemm::GemmShape<128, 128, 128>,
        tilelang::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    tilelang_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        tilelang::gemm::GemmShape<128, 128, 64>,
        tilelang::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm80_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 16) {
    if (n <= 4096) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<16, 64, 128>,
          tilelang::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          6>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<16, 64, 128>,
          tilelang::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 32) {
    if (n <= 4096) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<32, 64, 128>,
          tilelang::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          6>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<32, 64, 128>,
          tilelang::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n <= 4096) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<64, 64, 128>,
          tilelang::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<64, 128, 128>,
          tilelang::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128 && n < 8192) {
    tilelang_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        tilelang::gemm::GemmShape<64, 128, 128>,
        tilelang::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    tilelang_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        tilelang::gemm::GemmShape<128, 128, 64>,
        tilelang::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

// Dispatch shape for sm89 (L40S, L20, RTX 4090), according to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/tilelang_w8a8/scaled_mm_c2x_sm89_int8_dispatch.cuh
template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm89_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 16) {
    if (n <= 8192) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<16, 64, 128>,
          tilelang::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<16, 128, 128>,
          tilelang::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          4>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 32) {
    if (n <= 8192) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<32, 64, 128>,
          tilelang::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<32, 128, 128>,
          tilelang::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          4>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n <= 8192) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<64, 64, 128>,
          tilelang::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<64, 128, 128>,
          tilelang::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    if (n <= 8192) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<64, 128, 128>,
          tilelang::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<128, 128, 64>,
          tilelang::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<64, 64, 128>,
          tilelang::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 256) {
    if (n <= 4096) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<64, 128, 128>,
          tilelang::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 8192) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<128, 128, 64>,
          tilelang::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<256, 128, 64>,
          tilelang::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      tilelang_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          tilelang::gemm::GemmShape<128, 128, 64>,
          tilelang::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else {
    tilelang_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        tilelang::gemm::GemmShape<128, 128, 64>,
        tilelang::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <
    typename ElementOutput,
    typename TileShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    bool WithBias>
void tilelang_int8_scaled_mm_sm90(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ArchTag = tilelang::arch::Sm90;

  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  static constexpr int AlignmentA = 128 / tilelang::sizeof_bits<ElementInputA>::value;
  static constexpr int AlignmentB = 128 / tilelang::sizeof_bits<ElementInputB>::value;
  static constexpr int AlignmentC = 128 / tilelang::sizeof_bits<ElementOutput>::value;
  static constexpr int AlignmentOutput = 128 / tilelang::sizeof_bits<ElementOutput>::value;

  using OperatorClass = tilelang::arch::OpClassTensorOp;

  using EpilogueScheduleType = tilelang::epilogue::TmaWarpSpecialized;
  using TileSchedulerType = tilelang::gemm::PersistentScheduler;

  using XScale = tilelang::epilogue::fusion::
      Sm90ColBroadcast<0, TileShape, ElementCompute, ElementCompute, Stride<Int<1>, Int<0>, Int<0>>>;

  using WScale = tilelang::epilogue::fusion::
      Sm90RowBroadcast<0, TileShape, ElementCompute, ElementCompute, Stride<Int<0>, Int<1>, Int<0>>>;

  using Bias = tilelang::epilogue::fusion::
      Sm90RowBroadcast<0, TileShape, ElementOutput, ElementOutput, Stride<Int<0>, Int<1>, Int<0>>>;

  using Accum = tilelang::epilogue::fusion::Sm90AccFetch;

  // Scale
  using Compute0 = tilelang::epilogue::fusion::
      Sm90Compute<tilelang::multiplies, ElementCompute, ElementCompute, tilelang::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 = tilelang::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = tilelang::epilogue::fusion::
      Sm90Compute<tilelang::multiplies, ElementOutput, ElementCompute, tilelang::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 = tilelang::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  // With bias
  using ComputeWithBias = tilelang::epilogue::fusion::
      Sm90Compute<tilelang::multiply_add, ElementOutput, ElementCompute, tilelang::FloatRoundStyle::round_to_nearest>;
  using EVTComputeWithBias = tilelang::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

  using EpilogueEVT = typename tilelang::platform::conditional<WithBias, EVTComputeWithBias, EVTCompute1>::type;

  using CollectiveEpilogue = typename tilelang::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      tilelang::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementOutput,
      tilelang::layout::RowMajor,
      AlignmentC,
      ElementOutput,
      tilelang::layout::RowMajor,
      AlignmentOutput,
      EpilogueScheduleType,
      EpilogueEVT>::CollectiveOp;

  using Stages = tilelang::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename tilelang::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementInputA,
      tilelang::layout::RowMajor,
      AlignmentA,
      ElementInputB,
      tilelang::layout::ColumnMajor,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      Stages,
      MainloopScheduleType>::CollectiveOp;

  using GemmKernel = tilelang::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = tilelang::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = static_cast<ElementInputA*>(mat_a.data_ptr());
  auto b_ptr = static_cast<ElementInputB*>(mat_b.data_ptr());
  auto o_ptr = static_cast<ElementOutput*>(out.data_ptr());

  auto a_s_ptr = static_cast<ElementCompute*>(scales_a.data_ptr());
  auto b_s_ptr = static_cast<ElementCompute*>(scales_b.data_ptr());

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_a = tilelang::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = tilelang::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC stride_c;
  StrideD stride_d = tilelang::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  typename Gemm::Arguments args = {
      tilelang::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {a_ptr, stride_a, b_ptr, stride_b},
      {{},  // epilogue.thread
       nullptr,
       stride_c,
       o_ptr,
       stride_d}};

  if constexpr (WithBias) {
    ElementOutput* bias_ptr = static_cast<ElementOutput*>(bias->data_ptr());
    args.epilogue.thread = {
        {a_s_ptr},
        {{b_s_ptr}, {}, {}},
        {bias_ptr},
        {},
    };
  } else {
    args.epilogue.thread = {
        {a_s_ptr},
        {{b_s_ptr}, {}, {}},
        {},
    };
  }

  auto workspace = torch::empty(
      gemm_op.get_workspace_size(args), torch::TensorOptions().dtype(torch::kUInt8).device(mat_a.device()));

  auto stream = at::rtriton::getCurrentRTRITONStream(mat_a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(
      can_implement == tilelang::Status::kSuccess,
      "gemm cannot implement, error: ",
      tilelangGetStatusString(can_implement));

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == tilelang::Status::kSuccess, "gemm executioin failed, error: ", tilelangGetStatusString(status));
}

template <typename ElementOutput, typename TileShape, typename ClusterShape, typename MainloopScheduleType>
void sm90_dispatch_bias(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  if (bias) {
    tilelang_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, true>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    tilelang_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, false>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <typename ElementOutput>
void sm90_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 32) {
    if (n < 8192) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_1, _8, _1>,
          tilelang::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _128, _128>,
          Shape<_1, _8, _1>,
          tilelang::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n < 8192) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_1, _4, _1>,
          tilelang::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _256>,
          Shape<_1, _1, _1>,
          tilelang::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    if (n <= 4096) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_2, _1, _1>,
          tilelang::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _128, _128>,
          Shape<_2, _1, _1>,
          tilelang::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else {
    return sm90_dispatch_bias<
        ElementOutput,
        Shape<_128, _128, _128>,
        Shape<_2, _1, _1>,
        tilelang::gemm::KernelTmaWarpSpecializedPingpong>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

torch::Tensor int8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.is_rtriton(), "mat_a must be a RTRITON tensor");
  TORCH_CHECK(mat_b.is_rtriton(), "mat_b must be a RTRITON tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");
  TORCH_CHECK(mat_a.size(1) % 16 == 0, "mat_a.size(1) must be multiple of 16 for memory alignment");
  TORCH_CHECK(mat_b.size(0) % 16 == 0, "mat_b.size(0) must be multiple of 16 for memory alignment");
  TORCH_CHECK(mat_b.size(1) % 8 == 0, "mat_b.size(1) must be multiple of 8 for memory alignment");  // out.stride(0)
  TORCH_CHECK(mat_a.scalar_type() == torch::kInt8, "mat_a must be Int8");
  TORCH_CHECK(mat_b.scalar_type() == torch::kInt8, "mat_b must be Int8");
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

  TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
  TORCH_CHECK(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b msut be contiguous");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

  if (bias) {
    TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dtype() == out_dtype, "bias dtype must match output dtype");
  }

  torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));

  auto sm_version = getSMVersion();

  if (sm_version >= 75 && sm_version < 80) {
    TORCH_CHECK(out_dtype == torch::kHalf, "out_dtype must be Half for SM75");
    sm75_dispatch_shape<tilelang::half_t, tilelang::arch::Sm75, tilelang::gemm::GemmShape<8, 8, 16>>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (sm_version >= 80 && sm_version < 90) {
    // sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
    if (sm_version == 86 || sm_version == 89) {
      if (out_dtype == torch::kBFloat16) {
        sm89_dispatch_shape<tilelang::bfloat16_t, tilelang::arch::Sm80, tilelang::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      } else {
        sm89_dispatch_shape<tilelang::half_t, tilelang::arch::Sm80, tilelang::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      }
    } else {
      if (out_dtype == torch::kBFloat16) {
        sm80_dispatch_shape<tilelang::bfloat16_t, tilelang::arch::Sm80, tilelang::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      } else {
        sm80_dispatch_shape<tilelang::half_t, tilelang::arch::Sm80, tilelang::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      }
    }
  } else if (sm_version == 90) {
#if defined RTRITON_VERSION && RTRITON_VERSION >= 12000
    // tilelang 3.x
    if (out_dtype == torch::kBFloat16) {
      sm90_dispatch_shape<tilelang::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm90_dispatch_shape<tilelang::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
#else
    // fallback to tilelang 2.x
    if (out_dtype == torch::kBFloat16) {
      sm80_dispatch_shape<tilelang::bfloat16_t, tilelang::arch::Sm80, tilelang::gemm::GemmShape<16, 8, 32>>(
          out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm80_dispatch_shape<tilelang::half_t, tilelang::arch::Sm80, tilelang::gemm::GemmShape<16, 8, 32>>(
          out, mat_a, mat_b, scales_a, scales_b, bias);
    }
#endif
  } else {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "No implemented int8_scaled_mm for current compute capability.");
  }

  return out;
}
